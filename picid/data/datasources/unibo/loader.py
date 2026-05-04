"""
UNIBO21 datasource loader with predefined battery splits.

The loader follows the same high-level pattern as NB14, but it operates on the
UNIBO Powertools battery data. The split membership is encoded through the
module-level ``train_names``, ``val_names``, and ``test_names`` lists so the
loader reproduces the experiment setup already used by the project.
"""

import logging
import warnings
from collections import defaultdict
from typing import Optional

import awkward as ak
import numpy as np

from picid.data.datasources.base.predefined_split_loader import (
    PredefinedSplitLoaderBase,
)
from picid.data.datasources.nb14.prepare_rul_data import RulHandler
from picid.data.datasources.nb14.utils import table_to_ak_array
from picid.data.datasources.unibo.model_data_handler import ModelDataHandler
from picid.data.datasources.unibo.unibo_powertools_data import (
    CycleCols,
    UniboPowertoolsData,
)
from picid.data.datasources.utils import convert_outer_list_to_inner

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# Ideally we should adapt the processing code to consume train
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

UNIT_NAMES_TO_ID = train_names + test_names
UNIT_NAMES_TO_ID = {name: int(name.split("-")[0]) for name in UNIT_NAMES_TO_ID}


train_names = [
    # Group 1
    "000-DM-3.0-4019-S",  # minimum capacity 1.48
    "001-DM-3.0-4019-S",  # minimum capacity 1.81
    # Group 2
    "009-DM-3.0-4019-H",  # minimum capacity 1.41
    # Group 3
    "014-DM-3.0-4019-P",  # minimum capacity 1.7
    "015-DM-3.0-4019-P",  # minimum capacity 1.76
    "017-DM-3.0-4019-P",  # minimum capacity 1.29
    #'047-DM-3.0-4019-P',#new 1.98 Excluded by Bosello et al. 2023
    #'049-DM-3.0-4019-P',#new 2.19 Excluded by Bosello et al. 2023
    # Group 4
    "007-EE-2.85-0820-S",  # 2.5
    "008-EE-2.85-0820-S",  # 2.49
    # Group 5
    "043-EE-2.85-0820-H",  # 2.31
    # Group 6
    "018-DP-2.00-1320-S",  # minimum capacity 1.82
    # '019-DP-2.00-1320-S',# minimum capacity 1.61; Excluded by Bosello et al. 2023
    "036-DP-2.00-1720-S",  # minimum capacity 1.91
    "037-DP-2.00-1720-S",  # minimum capacity 1.84
    "038-DP-2.00-2420-S",  # minimum capacity 1.854 (to 0)
    "050-DP-2.00-4020-S",  # new 1.81
    # Group 7
    "040-DM-4.00-2320-S",  # minimum capacity 3.75, cycles 188
]

val_names = [
    # Group 1
    "002-DM-3.0-4019-S",  # minimum capacity 2.06
    # Group 2
    "010-DM-3.0-4019-H",  # minimum capacity 1.44
    # Group 3
    "016-DM-3.0-4019-P",  # minimum capacity 1.56
    # Group 4
    "042-EE-2.85-0820-S",  # 2.51
    # Group 5 is not present in val as there are only 2 units, one for train one for test
    # Group 6
    "051-DP-2.00-4020-S",  # new 1.866
    # Group 7 is not present in val as there are only 2 units, one for train one for test
]


test_names = [
    # Group 1
    "003-DM-3.0-4019-S",  # minimum capacity 1.84
    # Group 2
    "011-DM-3.0-4019-H",  # minimum capacity 1.36
    # Group 3
    "013-DM-3.0-4019-P",  # minimum capacity 1.6
    # Group 4
    "006-EE-2.85-0820-S",  # 2.621
    # Group 5
    "044-EE-2.85-0820-H",  # 2.43
    # Group 6
    "039-DP-2.00-2420-S",  # minimum capacity 1.93
    # Group 7
    "041-DM-4.00-2320-S",  # minimum capacity 3.76, cycles 190
]


class UNIBO21Loader(PredefinedSplitLoaderBase):
    """
    Load the UNIBO21 battery dataset and expose predefined split payloads.

    Parameters
    ----------
    rul_anomaly_threshold : float | None, default=None
        Absolute RUL threshold used for anomaly detection targets.
    rul_anomaly_fraction : float | None, default=None
        Relative fraction of the battery lifetime used as the anomaly
        threshold.
    **kwargs
        Additional loader configuration forwarded to the predefined-split base
        class.

    Notes
    -----
    For anomaly detection, the loader keeps the existing convention that the
    target is derived from an RUL threshold. The threshold can be specified
    either as an absolute value or as a fraction of the battery lifetime.
    """

    def __init__(
        self,
        *,
        rul_anomaly_threshold: Optional[float] = None,
        rul_anomaly_fraction: Optional[float] = None,
        **kwargs,
    ):
        self.data_path = kwargs["data_dir"]
        effective_threshold = kwargs.get(
            "rul_anomaly_threshold", rul_anomaly_threshold
        )
        effective_fraction = kwargs.get(
            "rul_anomaly_fraction", rul_anomaly_fraction
        )

        if kwargs["task_mode"] == "anomaly_detection":
            if effective_threshold is not None and effective_fraction is not None:
                raise ValueError(
                    "For task_mode='anomaly_detection', set exactly one of "
                    "rul_anomaly_threshold (absolute) or rul_anomaly_fraction (relative)."
                )
            if effective_threshold is None and effective_fraction is None:
                effective_fraction = 0.3  # default: last 30% of life

        self.rul_anomaly_threshold = effective_threshold
        self.rul_anomaly_fraction = effective_fraction
        kwargs["rul_anomaly_threshold"] = self.rul_anomaly_threshold
        kwargs["rul_anomaly_fraction"] = self.rul_anomaly_fraction

        super().__init__(**kwargs)

    def get_multisource_data_splitter(self):
        return self.multisource_data_splitter

    def _load_data(self) -> dict:
        """
        Read the dataset and assemble the predefined split container.

        Returns
        -------
        dict
            Split-aware container with one list of unit payloads per field and
            per split.
        """
        data = self.read_data()

        # Convert the split-wise list-of-dicts payload into columnar containers.
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

        logger.info(
            f"Number of train units: {len(out_dict['features']['train'])}, units: {[entry['unit_name'][:3] for entry in out_dict['metadata']['train']]}, n_cycles: {[entry['n_cycles'] for entry in out_dict['metadata']['train']]}"
        )
        logger.info(
            f"Number of val units: {len(out_dict['features']['val'])}, units: {[entry['unit_name'][:3] for entry in out_dict['metadata']['val']]}, n_cycles: {[entry['n_cycles'] for entry in out_dict['metadata']['val']]}"
        )
        logger.info(
            f"Number of test units: {len(out_dict['features']['test'])}, units: {[entry['unit_name'][:3] for entry in out_dict['metadata']['test']]}, n_cycles: {[entry['n_cycles'] for entry in out_dict['metadata']['test']]}"
        )

        # Extract metadata from each split so downstream code can resolve unit
        # ids and names without re-walking the raw payload.
        unit_ids_meta_data = {
            "unit_ids": {
                "train": [
                    UNIT_NAMES_TO_ID[entry["metadata"]["unit_name"]]
                    for entry in data["train"]
                ],
                "val": [
                    UNIT_NAMES_TO_ID[entry["metadata"]["unit_name"]]
                    for entry in data["val"]
                ],
                "test": [
                    UNIT_NAMES_TO_ID[entry["metadata"]["unit_name"]]
                    for entry in data["test"]
                ],
            },
            "unit_names": {
                "train": [entry["metadata"]["unit_name"] for entry in data["train"]],
                "val": [entry["metadata"]["unit_name"] for entry in data["val"]],
                "test": [entry["metadata"]["unit_name"] for entry in data["test"]],
            },
        }

        # Persist the derived metadata alongside the split payload.
        self.meta_data.update(unit_ids_meta_data)

        return out_dict

    def read_data(
        self,
    ):
        """
        Read the raw UNIBO battery files and prepare split-wise unit payloads.

        Returns
        -------
        collections.defaultdict[list]
            Mapping from split name to a list of processed unit dictionaries.

        Notes
        -----
        The raw UNIBO dataset is first prepared through
        :class:`UniboPowertoolsData`, then converted into cycle-level tensors by
        :class:`ModelDataHandler`, and finally transformed into the per-battery
        payload expected by the datasource layer.
        """
        # 1. Set up the raw dataset handler.
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

        # Pass all three split name lists to the dataset preparer.
        dataset.prepare_data(train_names, test_names, val_names)

        dataset_handler = ModelDataHandler(
            dataset, [CycleCols.VOLTAGE, CycleCols.CURRENT, CycleCols.TEMPERATURE]
        )

        CAPACITY_THRESHOLDS = {
            3.0: 2.7,
            2.85: 2.7,
            2.0: 1.93,
            4.0: 3.77,
            4.9: 4.7,
            5.0: 4.5,
        }

        # The dataset handler returns train, validation, and test tensors in one call.
        (
            train_x,
            train_y_soh,
            validation_x,
            validation_y_soh,
            test_x,
            test_y_soh,
            train_battery_range,
            validation_battery_range,
            test_battery_range,
            time_train,
            time_validation,
            time_test,
            current_train,
            current_validation,
            current_test,
            initial_cycle_lenghts_train,
            initial_cycle_lenghts_validation,
            initial_cycle_lenghts_test,
        ) = dataset_handler.get_discharge_whole_cycle_future(
            train_names,
            test_names,
            val_names,  # Pass all three
        )

        # 3. --- Organize Data for Looping ---
        split_data = {
            "train": {
                "x": train_x,
                "y_soh": train_y_soh,
                "battery_range": train_battery_range,
                "time": time_train,
                "current": current_train,
                "valid_lengths": initial_cycle_lenghts_train,
                "names": train_names,
            },
            "val": {
                "x": validation_x,
                "y_soh": validation_y_soh,
                "battery_range": validation_battery_range,
                "time": time_validation,
                "current": current_validation,
                "valid_lengths": initial_cycle_lenghts_validation,
                "names": val_names,
            },
            "test": {
                "x": test_x,
                "y_soh": test_y_soh,
                "battery_range": test_battery_range,
                "time": time_test,
                "current": current_test,
                "valid_lengths": initial_cycle_lenghts_test,
                "names": test_names,
            },
        }

        processed_data = defaultdict(list)

        # 4. --- Process in a Loop ---
        for split_name, data in split_data.items():
            # Skip if data is missing (e.g., val_names was None)
            if data["x"] is None:
                continue

            # Prepare Y data (RUL)
            y_raw = rul_handler.prepare_y_future(
                data["names"],
                data["battery_range"],
                data["y_soh"],
                data["current"],
                data["time"],
                CAPACITY_THRESHOLDS,
            )

            # Keep only RUL (second column)
            y_rul = y_raw[:, 1]

            # Process the split and store it
            processed_data[split_name] = self._process_split(
                x=data["x"],
                y=y_rul,
                battery_range=data["battery_range"],
                names=data["names"],
                valid_lengths=data["valid_lengths"],
                flatten=True,
            )

        return processed_data

    def _process_split(
        self, x, y, battery_range, names, valid_lengths=None, flatten=False
    ):
        """
        Create per-unit payloads for one split.

        Parameters
        ----------
        x : np.ndarray
            Feature tensor for the whole split.
        y : np.ndarray
            Label tensor for the whole split.
        battery_range : collections.abc.Iterable[int]
            Right indices delimiting the unit ranges inside the split tensors.
        names : collections.abc.Iterable[str]
            Unit names aligned with ``battery_range``.
        valid_lengths : list[list[int]] | None, default=None
            Valid cycle lengths used to unpad and flatten cycles.
        flatten : bool, default=False
            Whether to unpad and flatten cycles.

        Returns
        -------
        list[dict]
            List of per-unit payload dictionaries.
        """
        data_split = []
        left_idx = 0

        for idx, (right_idx, unit_name) in enumerate(zip(battery_range, names)):
            curr_x = x[left_idx:right_idx]
            curr_y = y[left_idx:right_idx].copy()

            if self.task_mode == "anomaly_detection":
                if self.rul_anomaly_fraction is not None:
                    max_rul = float(np.max(curr_y))
                    threshold = self.rul_anomaly_fraction * max_rul
                    curr_y = (curr_y <= threshold).astype(np.float32)
                else:
                    curr_y = (curr_y <= self.rul_anomaly_threshold).astype(np.float32)

            if flatten:
                # Use valid_lengths to unpad and flatten
                lengths = valid_lengths[left_idx:right_idx]
                features = table_to_ak_array(curr_x, lengths)

                lengths = ak.num(features, axis=1)
                curr_y = curr_y.reshape(-1, 1)

                output_list = [
                    np.repeat(curr_y[i : i + 1], lengths[i], axis=0)
                    for i in range(curr_y.shape[0])
                ]
                target = ak.to_regular(ak.Array(output_list), axis=-1)

                entry = {
                    "features": features,  # N cycles, cycle_len, n_features
                    "target": target,
                    "unit_id": ak.from_numpy(np.array([UNIT_NAMES_TO_ID[unit_name]])),
                    "metadata": {
                        "unit_name": unit_name,
                        "unit_id": UNIT_NAMES_TO_ID[unit_name],
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
