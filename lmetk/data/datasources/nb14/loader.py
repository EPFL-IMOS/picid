import warnings
import logging
from lmetk.data.datasources import SingleSourceLoader
from lmetk.data.datasources.kubric_movie.movi_dataset import MoviDataset
from collections import defaultdict
from lmetk.data.datasources.base.interfaces import AbstractDataSourceLoader
from lmetk.data.data_objects.data import SplitDatasetContainer
from typing import Any, Dict, override
import torch
from lmetk.data.datasources.nb14.nasa_random_data import NasaRandomizedData
from lmetk.data.datasources.nb14.prepare_rul_data import RulHandler

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
        'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/RW1',
        
        'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/RW2',
        'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/RW7',

        #'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/RW3',
        
        'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/RW4',
        'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/RW5',

        #'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/RW9',
        #'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/RW10',
        #'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/RW11',

       'RW_Skewed_Low_Room_Temp_DataSet_2Post/RW13',
        'RW_Skewed_Low_Room_Temp_DataSet_2Post/RW14',
        'RW_Skewed_Low_Room_Temp_DataSet_2Post/RW15',

       'RW_Skewed_High_Room_Temp_DataSet_2Post/RW17',
        'RW_Skewed_High_Room_Temp_DataSet_2Post/RW18',
        'RW_Skewed_High_Room_Temp_DataSet_2Post/RW19',

       'RW_Skewed_Low_40C_DataSet_2Post/RW21',
        'RW_Skewed_Low_40C_DataSet_2Post/RW22',
        'RW_Skewed_Low_40C_DataSet_2Post/RW23',

       'RW_Skewed_High_40C_DataSet_2Post/RW25',
        'RW_Skewed_High_40C_DataSet_2Post/RW26',
        'RW_Skewed_High_40C_DataSet_2Post/RW27',
]

test_names = [
        'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/RW8',
        #'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/RW6',
        #'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/RW12',
        'RW_Skewed_Low_Room_Temp_DataSet_2Post/RW16',
        'RW_Skewed_High_Room_Temp_DataSet_2Post/RW20',
        'RW_Skewed_Low_40C_DataSet_2Post/RW24',
        'RW_Skewed_High_40C_DataSet_2Post/RW28',
]

class NB14Loader(AbstractDataSourceLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_path = kwargs['cache_dir']
        
        
        self._is_isplitted = False
        self._is_loaded = False
        self.data_dict = None
        self.meta_data = None

        # Instantiate multisource_data_splitter if is not none
        assert (
            kwargs.get("multisource_data_splitter") == None
        ), "For RigidBodyDynanics, multisource_data_splitter is not supported. RigidBodyDynanics are given with splits."
        self.multisource_data_splitter = "For RigidBodyDynanics, multisource_data_splitter is not supported. RigidBodyDynanics are given with splits."
    
    @override
    def load_data(self):
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

        nasa_data_handler = NasaRandomizedData(self.data_path)
        rul_handler = RulHandler()
        
        # These parameters are taken from Bosello
        CAPACITY_THRESHOLDS = None
        NOMINAL_CAPACITY = 2.2

        (train_x, train_y_soh, test_x, test_y_soh,
        train_battery_range, test_battery_range,
        time_train, time_test, current_train, current_test) = nasa_data_handler.get_discharge_whole_cycle_future(train_names, test_names)

        train_y = rul_handler.prepare_y_future(train_names, train_battery_range, train_y_soh, current_train, time_train, CAPACITY_THRESHOLDS, capacity=NOMINAL_CAPACITY)
        del current_train
        del time_train

        test_y = rul_handler.prepare_y_future(test_names, test_battery_range, test_y_soh, current_test, time_test, CAPACITY_THRESHOLDS, capacity=NOMINAL_CAPACITY)
        del current_test
        del time_test
            
        # first one is SOH, we keep only RUL
        train_y = train_y[:,1]
        test_y = test_y[:,1]

        # Get train 
        data = defaultdict(list)
        left_idx = 0
        for right_idx, unit_name in zip(train_battery_range, train_names):
            data['train'].append(
                {
                "features": train_x[left_idx:right_idx],
                "target": train_y[left_idx:right_idx],
                "metadata": {"unit_name": unit_name}
                }
            )
            left_idx = right_idx


        # Get test 
        left_idx = 0
        for right_idx, unit_name in zip(test_battery_range, test_names):
            data['test'].append(
                {
                "features": test_x[left_idx:right_idx],
                "target": test_y[left_idx:right_idx],
                "metadata": {"unit_name": unit_name}
                }
            )
            # Add also test as val
            data['val'].append(
                {
                "features": test_x[left_idx:right_idx],
                "target": test_y[left_idx:right_idx],
                "metadata": {"unit_name": unit_name}
                }
            )
            left_idx = right_idx
        

        return data

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
            "RigidBodyDynamics does not support data splitting. The project relies on predefined splits."
        )