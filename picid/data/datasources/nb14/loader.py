"""
NB14 datasource loader with predefined train/validation/test battery splits.

The loader wraps the NASA randomized battery data helpers already used in the
project and preserves the Bosello-style split definition encoded in the module
constants:

- ``train_names`` lists the batteries assigned to training
- ``val_names`` lists the validation batteries
- ``test_names`` lists the held-out batteries

Each split is converted into the standard per-unit payload expected by the rest
of the preprocessing pipeline.
"""

import logging
import warnings
from collections import defaultdict

import awkward as ak
import numpy as np
from picid.data.datasources.base.predefined_split_loader import PredefinedSplitLoaderBase
from picid.data.datasources.nb14.nasa_random_data import NasaRandomizedData
from picid.data.datasources.nb14.prepare_rul_data import RulHandler
from picid.data.datasources.nb14.utils import table_to_ak_array

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def convert_outer_list_to_inner(data_list):
    """
    Convert a list of per-unit dictionaries into a dict of per-field lists.

    Parameters
    ----------
    data_list : list[dict]
        Per-unit payload represented as one dictionary per unit.

    Returns
    -------
    dict
        Columnar payload where each field maps to the list of unit values.

    Examples
    --------
    A split stored as::

        [{"features": x1, "target": y1}, {"features": x2, "target": y2}]

    becomes::

        {"features": [x1, x2], "target": [y1, y2]}
    """
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


class NB14Loader(PredefinedSplitLoaderBase):
    """
    Load the NB14 battery dataset and expose predefined battery splits.

    Parameters
    ----------
    **kwargs
        Loader configuration forwarded to
        :class:`PredefinedSplitLoaderBase`. The ``data_dir`` entry is reused as
        the raw NB14 data location.

    Notes
    -----
    The loader keeps the split definitions in Python constants instead of
    learning them from the raw files. This mirrors the existing experimental
    setup, where battery ids are explicitly assigned to one split.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_path = kwargs["data_dir"]
        self.data_dict = None
        self.meta_data = {}

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
            f"Number of train units: {len(out_dict['features']['train'])}, units: {[entry['unit_name'].split('/')[-1] for entry in out_dict['metadata']['train']]}, n_cycles: {[entry['n_cycles'] for entry in out_dict['metadata']['train']]}"
        )
        logger.info(
            f"Number of val units: {len(out_dict['features']['val'])}, units: {[entry['unit_name'].split('/')[-1] for entry in out_dict['metadata']['val']]}, n_cycles: {[entry['n_cycles'] for entry in out_dict['metadata']['val']]}"
        )
        logger.info(
            f"Number of test units: {len(out_dict['features']['test'])}, units: {[entry['unit_name'].split('/')[-1] for entry in out_dict['metadata']['test']]}, n_cycles: {[entry['n_cycles'] for entry in out_dict['metadata']['test']]}"
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

    def read_data(self):
        """
        Read the raw NB14 battery files and prepare split-wise unit payloads.

        Returns
        -------
        collections.defaultdict[list]
            Mapping from split name to a list of processed unit dictionaries.

        Notes
        -----
        The raw handler first returns cycle-aligned signals. The loader then
        uses :class:`RulHandler` to derive the future RUL target and keeps only
        the RUL column before converting each battery into the final payload.
        """
        nasa_data_handler = NasaRandomizedData(self.data_path)
        rul_handler = RulHandler()

        CAPACITY_THRESHOLDS = None
        NOMINAL_CAPACITY = 2.2

        # 1. Retrieve all split tensors from the raw handler.
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

        # 2. Organize the split tensors into one structure we can iterate over.
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

        # 3. Process each split with the shared RUL preparation logic.
        for split_name, data in split_data.items():
            # Prepare the RUL target from the SOH signal and cycle ranges.
            y_raw = rul_handler.prepare_y_future(
                data["names"],
                data["battery_range"],
                data["y_soh"],
                data["current"],
                data["time"],
                CAPACITY_THRESHOLDS,
                capacity=NOMINAL_CAPACITY,
            )

            # Keep only the RUL target column.
            y_rul = y_raw[:, 1]

            # Convert the split into the loader's per-unit payload format.
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

    def get_multisource_data_splitter(self):
        """
        Return the multisource data splitter configured on the loader.

        Returns
        -------
        object | None
            Multisource data splitter instance, if configured.
        """
        return self.multisource_data_splitter
