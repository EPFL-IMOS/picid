
from typing import Dict
from box import Box
import numpy as np
from omegaconf import DictConfig
from sklearn.base import TransformerMixin
from lmetk.data.pipeline.base import TimeSeriesDatasetInterface

class OneSourcePipeline(TimeSeriesDatasetInterface):
    def __init__(self, data_source, data_splitter, scalers, **kwargs):
        # self.load_arguments: DictConfig = load_arguments
        self.data_source = data_source
        self.data_splitter = data_splitter
        self.scalers: Dict[str, TransformerMixin] = scalers

        self._init_instance_variables()
    
    def _init_instance_variables(self,):
        # Init the flags
        self._is_loaded = False  # Flag to check if data is loaded
        self._is_split = False  # Flag to check if data is split    
        self._is_transformed = False  # Flag to check if data is transformed

        self.data_dict: (
            Dict[str, Dict[str, np.ndarray]] | Box
        ) = {}  # centralized storage

    def _load_data(self,):
        if getattr(self.data_source, '_is_loaded', False):
            self.data_dict.update(self.data_source)
        self._is_loaded = True
    
    def _pipeline(self):
        self._load_data()
        self._generate_splits()
        self._apply_transformations()
    
    def get_data_dict(self) -> Dict[str, Dict[str, np.ndarray]] | Dict [str, np.ndarray]:
        """
        Return the data dictionary containing features and timestamps.
        """
        assert self._is_loaded, "Data must be loaded before accessing the data dictionary."
        return self.data_dict

    def _generate_splits(self):
        # Split the data (e.g., train/val/test sequences)

        assert self._is_loaded, "Data must be loaded before splitting."
        assert not self._is_split, (
            "Data has already been split. "
            "If you want to re-split, please reset the data source."
        )

        splitted_data, split_masks = self.data_splitter.split_data(
            self.data_dict["features"]
        )
        
        splitted_timestamps, _ = (
            self.data_splitter.split_data(self.data_dict["timestamps"])
        )
        
        # Check 
        for key in splitted_data.keys():
            # Check that splits are consistent
            assert splitted_data[key].shape[0] == splitted_timestamps[key].shape[0], (
                f"Mismatch in {key} data and timestamps length"
            )
           

        # Update the self.data_dict
        self.data_dict.update(
            {"features": {"train": splitted_data["train"], "val": splitted_data["val"], "test": splitted_data["test"]}}
        )

        self.data_dict.update(
            {
                "timestamps": {
                    "train": splitted_timestamps["train"],
                    "val": splitted_timestamps['val'],
                    "test": splitted_timestamps['test'],
                }
            }
        )

        self.data_dict.update({
                "split_masks":{
                    "train": split_masks["train"],
                    "val": split_masks['val'],
                    "test": split_masks['test'],
                }
            }
        )

        self._is_split = True  # Mark data as split

    def _apply_transformations(self):

        assert self._is_split, "Data must be split before applying transformations."
        assert self._is_loaded, "Data must be loaded before applying transformations."
        assert not self._is_transformed, (
            "Data has already been transformed. "
            "If you want to re-transform, please reset the data source."
        )

        # Optionally scale the data
        if self.load_arguments.get("scale", True):
            for key, scaler in self.scalers.items():
                scaler.fit(self.data_dict[key]["train"])

            for split_name in ["train", "val", "test"]:
                if split_name in self.data_dict["features"]:
                    for key, scaler in self.scalers.items():
                        self.data_dict[key][split_name] = scaler.transform(
                            self.data_dict[key][split_name]
                        )


        self.data_dict = Box(
            self.data_dict
        )  # Convert to a Box object for easier access

        self._is_transformed = True  # Mark data as transformed

    def __repr__(self) -> str:
        return (
            f"data source: {self.data_source}, "
            f"data_splitter: {self.data_splitter}, "
            f"scaler: {self.scalers}"
        )


    # def get_data(self):
    #     return self.data_dict["features"], self.data_dict["timestamps"]

    # def get_scaler(self):
    #     return self.scalers

   

    
    
    # def update_data_dict(self, data_dict: Dict [str, np.ndarray]):
    #     """
    #     Set the data dictionary.
    #     This is useful for updating the data after loading.
    #     """
    #     assert isinstance(data_dict, dict), "data_dict must be a dictionary."
    #     # This feature is mostly used by MultiSourcePipeline to scale the data.
    #     # So it does not work with by source splitting.
    #     assert not self._is_split, "Cannot update data_dict after splitting."
    #     self.data_dict.update(data_dict)
    #     self._is_loaded = True

    # @abstractmethod
    # def _load_data(self) -> dict:
    #     """
    #     Expected to return a dict with at least:
    #         - "features"
    #         - "timestamps"
    #     Optional keys like:
    #         - "mask", "labels", etc.
    #     """
    #     self._is_loaded = True  # Mark data as loaded


    
