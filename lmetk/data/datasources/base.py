from abc import ABC, abstractmethod
from typing import Dict
from box import Box
import numpy as np
from omegaconf import DictConfig
from sklearn.base import TransformerMixin
from box import Box
from collections import OrderedDict

class DataSourceLoader(ABC):
    def __init__(self, **kwargs):
        
        # Store kwargs for repr
        self._init_kwargs = kwargs
        

        # Auto-assign keyword arguments to instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Define data source loader storage dict
        self.data_dict: (
            Dict[str, Dict[str, np.ndarray]] | Box
        ) = {}  # centralized storage

        # Split flag, create before load, as some datasets might have predefined splits, hence then _load_data method has to upadte self._is_splitted flag
        self._is_splitted = False
        
        # Load the data
        self.data_dict.update(self._load_data())
        self.data_name = kwargs['data_name']
        self._is_loaded = True    
        
        # Check if splits already has been predefined and done during the loading
        if self._is_splitted==False:
            # Get already instantiated data splitter
            self.data_splitter = kwargs['data_splitter']
            # By default create splits for features and timestamps
            for key in getattr(self.data_splitter, 'create_splits_for', ['features', 'timestamps']):
                self._split_data(key)

            self._is_splitted = True

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}({args})"
    
    def get_data(self) -> Dict[str, np.ndarray]:
        """Return splitted data dict."""
        return self.data_dict
    
    def get_data_name(self):
        """Return data name."""
        return self.data_name
    
    def get_data_default_transforms(self):
        """Return data default transforms."""
        return self.data_default_transforms
    
    def get_transforms(self) -> None:
        """Append the instantiated transforms to the instance data_transforms_lst."""
        source_transforms = OrderedDict()
        if getattr(self, 'data_default_transforms', None) is not None:
            for key, item in self.data_default_transforms.items():
                source_transforms[key] = item
            
        
        if getattr(self, 'transforms', None) is not None: 
            for key, item in self.transforms.items():
                # Check if such key already exist
                if key in source_transforms.keys():
                    raise KeyError('The key is already exist, make sure to give unique name for each transform applied')
                
                source_transforms[key] = item

        return source_transforms

    def _split_data(self, variable) ->  Dict[str, np.ndarray]:
        assert self._is_loaded, "Data must be loaded before splitting."
        
        splitted_data, split_masks = self.data_splitter.split_data(
            self.data_dict[variable]
        )
        
        # splitted_timestamps, _ = (
        #     self.data_splitter.split_data(self.data_dict["timestamps"])
        # )
        
        # Check 
        if not self.data_dict.get("split_masks", None):
            self.data_dict.update({
                "split_masks":{
                    "train": split_masks["train"],
                    "val": split_masks['val'],
                    "test": split_masks['test'],
                }
            }
            )
        else: 
            # Check that splits masks are consistent
            for key in split_masks.keys():
                assert splitted_data[key].shape[0] == self.data_dict['split_masks'][key].shape[0], (
                    f"Mismatch in {key} data and timestamps length"
                )
        
        # Update the self.data_dict
        self.data_dict.update(
            {variable: {"train": splitted_data["train"], "val": splitted_data["val"], "test": splitted_data["test"]}}
        )

        

    # Mark data as split
    # def _split_data(self) ->  Dict[str, np.ndarray]:
    #     assert self._is_loaded, "Data must be loaded before splitting."
    #     assert not self._is_splitted, (
    #         "Data has already been split. "
    #         "If you want to re-split, please reset the data source."
    #     )

    #     splitted_data, split_masks = self.data_splitter.split_data(
    #         self.data_dict["features"]
    #     )
        
    #     splitted_timestamps, _ = (
    #         self.data_splitter.split_data(self.data_dict["timestamps"])
    #     )
        
    #     # Check 
    #     for key in splitted_data.keys():
    #         # Check that splits are consistent
    #         assert splitted_data[key].shape[0] == splitted_timestamps[key].shape[0], (
    #             f"Mismatch in {key} data and timestamps length"
    #         )
           

    #     # Update the self.data_dict
    #     self.data_dict.update(
    #         {"features": {"train": splitted_data["train"], "val": splitted_data["val"], "test": splitted_data["test"]}}
    #     )

    #     self.data_dict.update(
    #         {
    #             "timestamps": {
    #                 "train": splitted_timestamps["train"],
    #                 "val": splitted_timestamps['val'],
    #                 "test": splitted_timestamps['test'],
    #             }
    #         }
    #     )

    #     self.data_dict.update({
    #             "split_masks":{
    #                 "train": split_masks["train"],
    #                 "val": split_masks['val'],
    #                 "test": split_masks['test'],
    #             }
    #         }
    #     )

    #     self._is_splitted = True  # Mark data as split

    
    @abstractmethod
    def _load_data(self) -> dict:
        """Load the raw dataset and timestamp information."""
        pass
    

    # @abstractmethod
    # def get_data_dict(self) -> Dict[str, Dict[str, np.ndarray]] | Dict [str, np.ndarray]:
    #     """Return the data dictionary containing features and timestamps."""
    #     pass
    
    # @abstractmethod
    # def update_data_dict(self, data_dict: Dict [str, np.ndarray]):
    #     """Set the data dictionary."""
    #     pass




# class AbstractTimeSeriesDataSource(TimeSeriesDatasetInterface):
#     def __init__(self, load_arguments, data_splitter, scalers, **kwargs):
#         self.load_arguments: DictConfig = load_arguments
#         self.data_splitter = data_splitter
#         self.scalers: Dict[str, TransformerMixin] = scalers

#         self._is_loaded = False  # Flag to check if data is loaded
#         self._is_split = False  # Flag to check if data is split    
#         self._is_transformed = False  # Flag to check if data is transformed

#         self.data_dict: (
#             Dict[str, Dict[str, np.ndarray]] | Box
#         ) = {}  # centralized storage

#     def get_data(self):
#         return self.data_dict["features"], self.data_dict["timestamps"]

#     def get_scaler(self):
#         return self.scalers

#     def _pipeline(self):
#         self.data_dict.update(self._load_data())
#         self._generate_splits()
#         self._apply_transformations()

#     def get_data_dict(self) -> Dict[str, Dict[str, np.ndarray]] | Dict [str, np.ndarray]:
#         """
#         Return the data dictionary containing features and timestamps.
#         """
#         assert self._is_loaded, "Data must be loaded before accessing the data dictionary."
#         return self.data_dict
    
#     def update_data_dict(self, data_dict: Dict [str, np.ndarray]):
#         """
#         Set the data dictionary.
#         This is useful for updating the data after loading.
#         """
#         assert isinstance(data_dict, dict), "data_dict must be a dictionary."
#         # This feature is mostly used by MultiSourcePipeline to scale the data.
#         # So it does not work with by source splitting.
#         assert not self._is_split, "Cannot update data_dict after splitting."
#         self.data_dict.update(data_dict)
#         self._is_loaded = True

#     @abstractmethod
#     def _load_data(self) -> dict:
#         """
#         Expected to return a dict with at least:
#             - "features"
#             - "timestamps"
#         Optional keys like:
#             - "mask", "labels", etc.
#         """
#         self._is_loaded = True  # Mark data as loaded


#     def _generate_splits(self):
#         # Split the data (e.g., train/val/test sequences)

#         assert self._is_loaded, "Data must be loaded before splitting."
#         assert not self._is_split, (
#             "Data has already been split. "
#             "If you want to re-split, please reset the data source."
#         )

#         train_data, val_data, test_data = self.data_splitter.split_data(
#             self.data_dict["features"]
#         )
#         train_timestamps, val_timestamps, test_timestamps = (
#             self.data_splitter.split_data(self.data_dict["timestamps"])
#         )

#         # Check that splits are consistent
#         assert train_data.shape[0] == train_timestamps.shape[0], (
#             "Mismatch in train data and timestamps length"
#         )
#         assert val_data.shape[0] == val_timestamps.shape[0], (
#             "Mismatch in val data and timestamps length"
#         )
#         assert test_data.shape[0] == test_timestamps.shape[0], (
#             "Mismatch in test data and timestamps length"
#         )

#         # Update the self.data_dict
#         self.data_dict.update(
#             {"features": {"train": train_data, "val": val_data, "test": test_data}}
#         )
#         self.data_dict.update(
#             {
#                 "timestamps": {
#                     "train": train_timestamps,
#                     "val": val_timestamps,
#                     "test": test_timestamps,
#                 }
#             }
#         )

#         self._is_split = True  # Mark data as split

#     def _apply_transformations(self):

#         assert self._is_split, "Data must be split before applying transformations."
#         assert self._is_loaded, "Data must be loaded before applying transformations."
#         assert not self._is_transformed, (
#             "Data has already been transformed. "
#             "If you want to re-transform, please reset the data source."
#         )

#         # Optionally scale the data
#         if self.load_arguments.get("scale", True):
#             for key, scaler in self.scalers.items():
#                 scaler.fit(self.data_dict[key]["train"])

#             for split_name in ["train", "val", "test"]:
#                 if split_name in self.data_dict["features"]:
#                     for key, scaler in self.scalers.items():
#                         self.data_dict[key][split_name] = scaler.transform(
#                             self.data_dict[key][split_name]
#                         )


#         self.data_dict = Box(
#             self.data_dict
#         )  # Convert to a Box object for easier access

#         self._is_transformed = True  # Mark data as transformed

#     def __repr__(self) -> str:
#         return (
#             f"loader_arguments: {self.load_arguments}, "
#             f"data_splitter: {self.data_splitter}, "
#             f"scaler: {self.scalers}"
#         )
