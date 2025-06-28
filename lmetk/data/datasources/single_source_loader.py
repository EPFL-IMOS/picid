from typing import Dict
from box import Box
import numpy as np
from omegaconf import DictConfig
from lmetk.data.datasources.base import DataSourceLoaderInterface
from abc import abstractmethod

class DataSourceLoader(DataSourceLoaderInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
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
        
        # Get already instantiated data splitter
        self.multisource_data_splitter = kwargs.get('multisource_data_splitter', None)
        if not self.multisource_data_splitter:
            self.data_splitter = kwargs['data_splitter']
            self.split_data(kwargs)
    
    @abstractmethod
    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load data from the data source."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def split_data(self, kwargs):
        # Check if splits already has been predefined and done during the loading
        if not self._is_splitted:
            # By default create splits for features and timestamps
            for key in getattr(self.data_splitter, 'create_splits_for', ['features', 'timestamps']):
                self._split_data(key)

            self._is_splitted = True
    
    def get_data(self) -> Dict[str, np.ndarray]:
        """Return splitted data dict."""
        return self.data_dict
    
    def get_data_name(self):
        """Return data name."""
        return self.data_name

    def _split_data(self, variable) ->  Dict[str, np.ndarray]:
        assert self._is_loaded, "Data must be loaded before splitting."
        
        splitted_data, split_masks = self.data_splitter.split_data(
            data_dict = self.data_dict.copy(), split_variable = variable,
        )
        
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
    
    