import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, override

import awkward as ak
import numpy as np
from picid.data.data_objects.data import SplitDatasetContainer
from picid.data.datasources.base.interfaces import AbstractDataSourceLoader
from picid.data.datasources.nb14.nasa_random_data import NasaRandomizedData
from picid.data.datasources.nb14.prepare_rul_data import RulHandler
from picid.data.datasources.nb14.utils import table_to_ak_array

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
    # Group 1
    "Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/RW1",
    "Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/RW2",
    # Group 2
    #'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/RW3',
    "Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/RW4",
    # Group 3
    #'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/RW9',
    #'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/RW10',
    #'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/RW11',
    # Group 4
    "RW_Skewed_Low_Room_Temp_DataSet_2Post/RW13",
    "RW_Skewed_Low_Room_Temp_DataSet_2Post/RW14",
    # Group 5
    "RW_Skewed_High_Room_Temp_DataSet_2Post/RW17",
    # "RW_Skewed_High_Room_Temp_DataSet_2Post/RW19",
    # Group 6
    "RW_Skewed_Low_40C_DataSet_2Post/RW21",
    "RW_Skewed_Low_40C_DataSet_2Post/RW22",
    # Group 7
    "RW_Skewed_High_40C_DataSet_2Post/RW25",
    "RW_Skewed_High_40C_DataSet_2Post/RW26",
]
val_names = [
    # Group 1
    "Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/RW7",
    # Group 2
    "Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/RW5",
    # Group 4
    "RW_Skewed_Low_Room_Temp_DataSet_2Post/RW15",
    # Group 5
    "RW_Skewed_High_Room_Temp_DataSet_2Post/RW18",
    # Group 6
    "RW_Skewed_Low_40C_DataSet_2Post/RW23",
    # Group 7
    "RW_Skewed_High_40C_DataSet_2Post/RW27",
]


test_names = [
    # Group 1
    "Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/RW8",
    # Group 2
    "Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/RW6",
    # Group 3
    # "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/RW12",
    # Group 4
    "RW_Skewed_Low_Room_Temp_DataSet_2Post/RW16",
    # Group 5
    "RW_Skewed_High_Room_Temp_DataSet_2Post/RW19",
    # "RW_Skewed_High_Room_Temp_DataSet_2Post/RW20",
    # Group 6
    "RW_Skewed_Low_40C_DataSet_2Post/RW24",
    # Group 7
    "RW_Skewed_High_40C_DataSet_2Post/RW28",
]

UNIT_NAMES_TO_ID = train_names + val_names + test_names
UNIT_NAMES_TO_ID = {
    name: int(name.split("/")[1].strip("RW")) for name in UNIT_NAMES_TO_ID
}


class NB14Loader(AbstractDataSourceLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_path = kwargs["data_dir"]

        self._is_splitted = False
        self._is_loaded = False
        self.data_dict = None
        self.meta_data = {}

        # Instantiate multisource_data_splitter if is not none
        assert (
            kwargs.get("multisource_data_splitter") == None
        ), "For RigidBodyDynanics, multisource_data_splitter is not supported. RigidBodyDynanics are given with splits."
        self.multisource_data_splitter = "For RigidBodyDynanics, multisource_data_splitter is not supported. RigidBodyDynanics are given with splits."

    @override
    def load_data(self):
        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_splitted = True

    @override
    def get_data(self) -> dict | list[dict]:
        """Return data sources.

        Returns
        -------
        dict | list[dict]
            Either dict in case we use multisource_data_splitter or list of dicts in case we use data_splitter

        """
        assert self._is_loaded is True, "Data must be loaded before getting data."
        assert self._is_splitted is True, "Data must be splitted before getting data."
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

        logger.info(
            f"Number of train units: {len(out_dict['features']['train'])}, units: {[entry['unit_name'].split('/')[-1] for entry in out_dict['metadata']['train']]}, n_cycles: {[entry['n_cycles'] for entry in out_dict['metadata']['train']]}"
        )
        logger.info(
            f"Number of val units: {len(out_dict['features']['val'])}, units: {[entry['unit_name'].split('/')[-1] for entry in out_dict['metadata']['val']]}, n_cycles: {[entry['n_cycles'] for entry in out_dict['metadata']['val']]}"
        )
        logger.info(
            f"Number of test units: {len(out_dict['features']['test'])}, units: {[entry['unit_name'].split('/')[-1] for entry in out_dict['metadata']['test']]}, n_cycles: {[entry['n_cycles'] for entry in out_dict['metadata']['test']]}"
        )

        # Extract metadata from the splits to be accessible later by return self.meta_data_dict["unit_ids"][self.meta_data_dict["current_data_split"]][self.meta_data_dict["concat_dataset_index"]]
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

        # update the meta_data with any other info you want to store
        self.meta_data.update(unit_ids_meta_data)

        return out_dict

    def read_data(self):
        nasa_data_handler = NasaRandomizedData(self.data_path)
        rul_handler = RulHandler()

        CAPACITY_THRESHOLDS = None
        NOMINAL_CAPACITY = 2.2

        # 1. Get all data: Note the CORRECTED variable names
        (
            train_x,
            train_y,
            validation_x,
            validation_y,
            test_x,
            test_y,
            battery_n_cycle_train,
            battery_n_cycle_validation,
            battery_n_cycle_test,
            time_train,
            time_validation,
            time_test,
            current_train,
            current_validation,
            current_test,
            initial_cycle_lenghts_train,
            initial_cycle_lenghts_validation,
            initial_cycle_lenghts_test,
        ) = nasa_data_handler.get_discharge_whole_cycle_future(
            train_names=train_names, test_names=test_names, validation_names=val_names
        )

        # 2. Organize the data into a structure we can loop over.
        #    (Assuming train_names, val_names, test_names are available in this scope)
        split_data = {
            "train": {
                "x": train_x,
                "y_soh": train_y,
                "battery_range": battery_n_cycle_train,
                "time": time_train,
                "current": current_train,
                "valid_lengths": initial_cycle_lenghts_train,
                "names": train_names,
            },
            "val": {
                "x": validation_x,
                "y_soh": validation_y,
                "battery_range": battery_n_cycle_validation,
                "time": time_validation,
                "current": current_validation,
                "valid_lengths": initial_cycle_lenghts_validation,
                "names": val_names,
            },
            "test": {
                "x": test_x,
                "y_soh": test_y,
                "battery_range": battery_n_cycle_test,
                "time": time_test,
                "current": current_test,
                "valid_lengths": initial_cycle_lenghts_test,
                "names": test_names,
            },
        }

        processed_data = defaultdict(list)

        # 3. Loop through the structure and apply the logic *once*
        for split_name, data in split_data.items():
            # Prepare the Y data (RUL)
            # We pass data["y_soh"] and data["battery_range"]
            # which now correctly hold train_y and battery_n_cycle_train
            y_raw = rul_handler.prepare_y_future(
                data["names"],
                data["battery_range"],
                data["y_soh"],
                data["current"],
                data["time"],
                CAPACITY_THRESHOLDS,
                capacity=NOMINAL_CAPACITY,
            )

            # Keep only RUL (the second column)
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
        Process one split (train/test/val).

        Parameters
        ----------
        x
            np.ndarray of features (n_samples, ...)
        y
            np.ndarray of labels
        battery_range
            iterable of right indices per unit
        names
            iterable of unit names
        valid_lengths
            list of lists with valid cycle lengths (only for train)
        flatten
            whether to unpad and flatten cycles

        Returns
        -------
        list
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
                features = table_to_ak_array(curr_x, lengths)

                lengths = ak.num(features, axis=1)
                curr_y = curr_y.reshape(-1, 1)

                output_list = [
                    np.repeat(curr_y[i : i + 1], lengths[i], axis=0)
                    for i in range(curr_y.shape[0])
                ]

                target = ak.to_regular(ak.Array(output_list), axis=-1)

                entry = {
                    "features": features,
                    "target": target,
                    "unit_id": ak.from_numpy(np.array([UNIT_NAMES_TO_ID[unit_name]])),
                    "metadata": {
                        "unit_name": unit_name,
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
