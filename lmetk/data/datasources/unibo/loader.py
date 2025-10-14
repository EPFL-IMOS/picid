import copy
import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, override

import awkward as ak
import numpy as np

from lmetk.data.data_objects.data import SplitDatasetContainer
from lmetk.data.datasources.base.interfaces import AbstractDataSourceLoader
from lmetk.data.datasources.nb14.prepare_rul_data import RulHandler
from lmetk.data.datasources.nb14.utils import flatten_cycles, table_to_ak_array
from lmetk.data.datasources.unibo.model_data_handler import ModelDataHandler
from lmetk.data.datasources.unibo.unibo_powertools_data import (
    CycleCols,
    UniboPowertoolsData,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def convert_outer_list_to_inner(data_list):
    if not data_list:
        return {}

    # Ensure all dictionaries have the same keys
    keys = data_list[0].keys()
    for d in data_list:
        if d.keys() != keys:
            raise ValueError("Not all dicts have the same keys!")

    # Concatenate values for each key
    stacked = {}
    for key in keys:
        stacked[key] = [d[key] for d in data_list]

    # Check that all keys in the stacked have the same length
    lengths = [len(stacked[key]) for key in stacked]
    if len(set(lengths)) != 1:
        raise ValueError("All keys in the stacked dict must have the same length.")

    return stacked


train_names = [
    "000-DM-3.0-4019-S",  # minimum capacity 1.48
    "001-DM-3.0-4019-S",  # minimum capacity 1.81
    "002-DM-3.0-4019-S",  # minimum capacity 2.06
    "009-DM-3.0-4019-H",  # minimum capacity 1.41
    "010-DM-3.0-4019-H",  # minimum capacity 1.44
    "014-DM-3.0-4019-P",  # minimum capacity 1.7
    "015-DM-3.0-4019-P",  # minimum capacity 1.76
    "016-DM-3.0-4019-P",  # minimum capacity 1.56
    "017-DM-3.0-4019-P",  # minimum capacity 1.29
    #'047-DM-3.0-4019-P',#new 1.98
    #'049-DM-3.0-4019-P',#new 2.19
    "007-EE-2.85-0820-S",  # 2.5
    "008-EE-2.85-0820-S",  # 2.49
    "042-EE-2.85-0820-S",  # 2.51
    "043-EE-2.85-0820-H",  # 2.31
    "018-DP-2.00-1320-S",  # minimum capacity 1.82
    #'019-DP-2.00-1320-S',#minimum capacity 1.61
    "036-DP-2.00-1720-S",  # minimum capacity 1.91
    "037-DP-2.00-1720-S",  # minimum capacity 1.84
    "038-DP-2.00-2420-S",  # minimum capacity 1.854 (to 0)
    "050-DP-2.00-4020-S",  # new 1.81
    "051-DP-2.00-4020-S",  # new 1.866
    "040-DM-4.00-2320-S",  # minimum capacity 3.75, cycles 188
]

test_names = [
    "003-DM-3.0-4019-S",  # minimum capacity 1.84
    "011-DM-3.0-4019-H",  # minimum capacity 1.36
    "013-DM-3.0-4019-P",  # minimum capacity 1.6
    "006-EE-2.85-0820-S",  # 2.621
    "044-EE-2.85-0820-H",  # 2.43
    "039-DP-2.00-2420-S",  # minimum capacity 1.93
    "041-DM-4.00-2320-S",  # minimum capacity 3.76, cycles 190
]


class UNIBO21Loader(AbstractDataSourceLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_path = kwargs["data_dir"]

        self._is_isplitted = False
        self._is_loaded = False
        self.data_dict = None
        self.meta_data = {}

        # Instantiate multisource_data_splitter if is not none
        assert (
            kwargs.get("multisource_data_splitter") == None
        ), "For UNIBO21, multisource_data_splitter is not supported. UNIBO21 are given with splits."
        self.multisource_data_splitter = "For UNIBO21, multisource_data_splitter is not supported. UNIBO21 are given with splits."

    @override
    def load_data(self):
        """Load data from the data source."""
        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_isplitted = True

    @override
    def get_data(self) -> dict | list[dict]:
        """Return data sources.

        Returns:
            dict | list[dict] : either dict in case we use multisource_data_splitter or list of dicts in case we use data_splitter

        """
        assert self._is_loaded is True, "Data must be loaded before getting data."
        assert self._is_isplitted is True, "Data must be splitted before getting data."

        return SplitDatasetContainer(**self.data_dict)

    def _load_data(self) -> dict:
        # Read raw CSV
        data = self.read_data()

        # Convert to desired format
        train = convert_outer_list_to_inner(data["train"])
        val = convert_outer_list_to_inner(data["val"])
        test = convert_outer_list_to_inner(data["test"])

        out_dict = {}
        for key in train.keys():
            out_dict[key] = {
                "train": train[key],
                "val": val[key],
                "test": test[key],
            }

        logger.info(f"Number of train units: {len(out_dict['features']['train'])}")
        logger.info(f"Number of val units: {len(out_dict['features']['val'])}")
        logger.info(f"Number of test units: {len(out_dict['features']['test'])}")
        return out_dict

    def read_data(
        self,
    ):
        data = defaultdict(list)
        dataset = UniboPowertoolsData(
            test_types=[],
            chunk_size=1000000,
            lines=[37, 40],
            charge_line=37,
            discharge_line=40,
            base_path=self.data_path,
            force_download=False,
        )

        rul_handler = RulHandler()

        dataset.prepare_data(train_names, test_names)
        dataset_handler = ModelDataHandler(
            dataset, [CycleCols.VOLTAGE, CycleCols.CURRENT, CycleCols.TEMPERATURE]
        )

        CAPACITY_THRESHOLDS = {
            3.0: 2.7,  # th 90% - min 2.1, 70%
            2.85: 2.7,  # th 94.7% - min 2.622, 92%
            2.0: 1.93,  # th 96.5% - min 1.93, 96.5%
            4.0: 3.77,  # th 94.2% - min 3.77 94.2%
            4.9: 4.7,  # th 95.9% - min 4.3, 87.7%
            5.0: 4.5,  # th 90% - min 3.63, 72.6%
        }

        (
            train_x,
            train_y_soh,
            test_x,
            test_y_soh,
            train_battery_range,
            test_battery_range,
            time_train,
            time_test,
            current_train,
            current_test,
            initial_cycle_lenghts_train,
            initial_cycle_lenghts_test,
        ) = dataset_handler.get_discharge_whole_cycle_future(train_names, test_names)

        train_y = rul_handler.prepare_y_future(
            train_names,
            train_battery_range,
            train_y_soh,
            current_train,
            time_train,
            CAPACITY_THRESHOLDS,
        )
        del current_train
        del time_train

        test_y = rul_handler.prepare_y_future(
            test_names,
            test_battery_range,
            test_y_soh,
            current_test,
            time_test,
            CAPACITY_THRESHOLDS,
        )
        del current_test
        del time_test

        # first one is SOH, we keep only RUL
        train_y = train_y[:, 1]
        test_y = test_y[:, 1]

        # Get train
        data = defaultdict(list)
        # Get train
        data = defaultdict(list)

        data["train"] = self._process_split(
            x=train_x,
            y=train_y,
            battery_range=train_battery_range,
            names=train_names,
            valid_lengths=initial_cycle_lenghts_train,
            flatten=True,
        )

        # Get test
        test_entries = self._process_split(
            x=test_x,
            y=test_y,
            battery_range=test_battery_range,
            names=test_names,
            valid_lengths=initial_cycle_lenghts_test,
            flatten=True,
        )
        data["test"] = test_entries
        data["val"] = copy.deepcopy(test_entries)

        if self.debug_subset_range is not None:
            data["train"] = data["train"][:2]
            data["val"] = data["val"][:1]
            data["test"] = data["test"][:1]
            logger.info(
                f"Debug subset range applied. New train size: {len(data['train'])}, val size: {len(data['val'])}, test size: {len(data['test'])}"
            )

        return data

    def _process_split(
        self, x, y, battery_range, names, valid_lengths=None, flatten=False
    ):
        """
        Process one split (train/test/val).

        Args:
            x: np.ndarray of features (n_samples, ...)
            y: np.ndarray of labels
            battery_range: iterable of right indices per unit
            names: iterable of unit names
            valid_lengths: list of lists with valid cycle lengths (only for train)
            flatten: whether to unpad and flatten cycles

        Returns:
            list of dicts with keys: features, target, metadata
        """
        data_split = []
        left_idx = 0

        for idx, (right_idx, unit_name) in enumerate(zip(battery_range, names)):
            curr_x = x[left_idx:right_idx]
            curr_y = y[left_idx:right_idx]

            if flatten:
                # Use valid_lengths to unpad and flatten
                lengths = valid_lengths[left_idx:right_idx]
                flattened_x = table_to_ak_array(curr_x, lengths)
                # flattened_x, flattened_y, cycle_indices = flatten_cycles(curr_x, curr_y, lengths)
                entry = {
                    "features": flattened_x,
                    # "time_features": ak.from_numpy(np.array(curr_y.reshape(-1, 1))),
                    "target": ak.from_numpy(curr_y.reshape(-1, 1)),  # flattened_y
                    "metadata": {
                        "unit_name": unit_name,
                        # "cycle_indices": cycle_indices,
                        "n_cycles": len(lengths),
                    },
                }
            else:
                # Keep as is (no flattening)
                entry = {
                    "features": curr_x,
                    "target": curr_y,
                    "metadata": {"unit_name": unit_name},
                }

            if self.debug_subsample_rate is not None:
                # subsample the data
                entry["features"] = entry["features"][:: self.debug_subsample_rate]
                entry["target"] = entry["target"][:: self.debug_subsample_rate]

            data_split.append(entry)
            left_idx = right_idx

        return data_split

    def get_meta_data(self) -> Dict[str, Any]:
        """
        Return metadata about the data source.
        This can include information like data source name, version, etc.
        """
        assert self._is_loaded, "Data must be loaded before getting metadata."
        return self.meta_data

    def get_data_name(self) -> list[str]:
        """Return source data names."""
        return self.data_dict["metadata"]

    def get_multisource_data_splitter(self):
        """Return multisource data splitter."""
        return self.multisource_data_splitter

    @override
    def split_data(self):
        logger.warning(
            f"{self.data_name} does not support data splitting. The project relies on predefined splits."
        )
