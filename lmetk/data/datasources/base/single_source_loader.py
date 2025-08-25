from abc import abstractmethod
from typing import Any, Dict

import numpy as np
from box import Box

from lmetk.data.datasources.base.interfaces import DataSourceLoaderInterface
from lmetk.data.data_objects.data import DatasetContainer

class SingleSourceLoader(DataSourceLoaderInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Auto-assign keyword arguments to instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Define data source loader storage dict
        self.data_dict: Dict[str, Dict[str, np.ndarray]] | Box = (
            {}
        )  # centralized storage

        # Split flag, create before load, as some datasets might have predefined splits,
        # hence then _load_data method has to update self._is_splitted flag
        self._is_splitted = False
        self._is_loaded = False
        

    @abstractmethod
    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load data from the data source."""
        raise NotImplementedError("Subclasses must implement this method.")

    def split_data(self):
        # Check if splits already has been predefined and done during the loading
        if not self._is_splitted:
            # By default create splits for features and timestamps
            for key in getattr(
                self.data_splitter, "create_splits_for", ["features", "timestamps"]
            ):
                self._split_data(key)

            self._is_splitted = True

    def get_data(self) -> Dict[str, np.ndarray]:
        """Return splitted data dict."""
        if self._is_loaded == False:
            # Load the data
            self.data_dict.update(self._load_data())
            # self.data_name = kwargs["data_name"]
            self._is_loaded = True

            # Get already instantiated data splitter
            self.multisource_data_splitter = getattr(self, "multisource_data_splitter", None)
            if not self.multisource_data_splitter:
                self.data_splitter = getattr(self,"data_splitter", None)
                assert self.data_splitter is not None, "Data splitter must be set."
                self.split_data()

        assert self._is_loaded==True, "Data must be loaded before getting data." 
        
        # We use MultiDataChunk as it will simply wrap values of the self.data_dict into a list
        return DatasetContainer(**self.data_dict)

    def get_meta_data(self) -> Dict[str, Any]:
        """
        Return metadata about the data source.
        This can include information like data source name, version, etc.
        """
        assert self._is_loaded, "Data must be loaded before getting metadata."
        return {}

    def get_data_name(self):
        """Return data name."""
        return self.data_name

    def _split_data(self, variable) -> Dict[str, np.ndarray]:
        assert self._is_loaded, "Data must be loaded before splitting."

        splitted_data, split_masks = self.data_splitter.split_data(
            data_dict=self.data_dict.copy(),
            split_variable=variable,
        )

        # Check
        if not self.data_dict.get("split_masks", None):
            self.data_dict.update(
                {
                    "split_masks": {
                        "train": split_masks["train"],
                        "val": split_masks["val"],
                        "test": split_masks["test"],
                    }
                }
            )
        else:
            # Check that splits masks are consistent
            for key in split_masks.keys():
                assert (
                    splitted_data[key].shape[0]
                    == self.data_dict["split_masks"][key].shape[0]
                ), f"Mismatch in {key} data and timestamps length"

        # Update the self.data_dict
        self.data_dict.update(
            {
                variable: {
                    "train": splitted_data["train"],
                    "val": splitted_data["val"],
                    "test": splitted_data["test"],
                }
            }
        )
