from abc import ABC, abstractmethod
from typing import Dict
from box import Box
import numpy as np
from omegaconf import DictConfig
from sklearn.base import TransformerMixin


class TimeSeriesDatasetInterface(ABC):
    @abstractmethod
    def __init__(self, load_arguments, data_splitter, scaler):
        pass

    @abstractmethod
    def get_data(self):
        """Return splitted data and timestamps."""
        pass

    @abstractmethod
    def get_scaler(self):
        """Return the fitted scaler."""
        pass

    @abstractmethod
    def _pipeline(self):
        """Load, split, and transform the data."""
        pass

    @abstractmethod
    def _load_data(self):
        """Load the raw dataset and timestamp information."""
        pass

    @abstractmethod
    def _generate_splits(self):
        """Split the data and timestamps into train/val/test sets."""
        pass

    @abstractmethod
    def _apply_transformations(self):
        """Apply any preprocessing, such as scaling."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class AbstractTimeSeriesDataSource(TimeSeriesDatasetInterface):

    def __init__(self, load_arguments, data_splitter, scalers, **kwargs):

        self.load_arguments: DictConfig = load_arguments
        self.data_splitter = data_splitter
        self.scalers: Dict[str, TransformerMixin] = scalers  
        self.data_dict: Dict[str, Dict[str, np.ndarray]] | Box = {} # centralized storage
        # self._pipeline() # Why do we call pipeline here?

    def get_data(self):
        return self.data_dict["features"], self.data_dict["timestamps"]

    def get_scaler(self):
        return self.scaler

    def _pipeline(self):
        self.data_dict.update(self._load_data())
        self._generate_splits()
        self._apply_transformations()

    @abstractmethod
    def _load_data(self)-> dict:
        """
        Expected to return a dict with at least:
            - "features"
            - "timestamps"
        Optional keys like:
            - "mask", "labels", etc.
        """
        pass

    def _generate_splits(self):
        # Split the data (e.g., train/val/test sequences)
        
        train_data, val_data, test_data = self.data_splitter.split_data(self.data_dict['features'])
        train_timestamps, val_timestamps, test_timestamps = self.data_splitter.split_data(self.data_dict['timestamps'])

        # Check that splits are consistent
        assert train_data.shape[0] == train_timestamps.shape[0], "Mismatch in train data and timestamps length"
        assert val_data.shape[0] == val_timestamps.shape[0], "Mismatch in val data and timestamps length"
        assert test_data.shape[0] == test_timestamps.shape[0], "Mismatch in test data and timestamps length"

        # Update the self.data_dict
        self.data_dict.update({"features": {"train": train_data, "val": val_data, "test": test_data}})
        self.data_dict.update({"timestamps": {"train": train_timestamps, "val": val_timestamps, "test": test_timestamps}})

    def _apply_transformations(self):
        # Optionally scale the data
        if self.load_arguments.get("scale", True):

            for key, scaler in self.scalers.items():
                scaler.fit(self.data_dict[key]["train"])
            
            for split_name in ["train", "val", "test"]:
                if split_name in self.data_dict["features"]:
                    for key, scaler in self.scalers.items():
                        print(f"Scaling {key} for {split_name} split using {scaler.__class__.__name__}")  
                        self.data_dict[key][split_name] = \
                              scaler.transform(self.data_dict[key][split_name])
        
        self.data_dict = Box(self.data_dict)  # Convert to a Box object for easier access

    def __repr__(self) -> str:
        return f"loader_arguments: {self.load_arguments}, " \
                f"data_splitter: {self.data_splitter}, " \
                f"scaler: {self.scaler}"