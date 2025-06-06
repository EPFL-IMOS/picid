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
    def get_data(self) -> Dict[str, np.ndarray]:
        """Return splitted data and timestamps."""
        pass

    @abstractmethod
    def get_data_dict(self) -> Dict[str, Dict[str, np.ndarray]] | Dict [str, np.ndarray]:
        """Return the data dictionary containing features and timestamps."""
        pass

    
    @abstractmethod
    def update_data_dict(self, data_dict: Dict [str, np.ndarray]):
        """Set the data dictionary."""
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

        self._is_loaded = False  # Flag to check if data is loaded
        self._is_split = False  # Flag to check if data is split    
        self._is_transformed = False  # Flag to check if data is transformed

        self.data_dict: (
            Dict[str, Dict[str, np.ndarray]] | Box
        ) = {}  # centralized storage

    def get_data(self):
        return self.data_dict["features"], self.data_dict["timestamps"]

    def get_scaler(self):
        return self.scalers

    def _pipeline(self):
        self.data_dict.update(self._load_data())
        self._generate_splits()
        self._apply_transformations()

    def get_data_dict(self) -> Dict[str, Dict[str, np.ndarray]] | Dict [str, np.ndarray]:
        """
        Return the data dictionary containing features and timestamps.
        """
        assert self._is_loaded, "Data must be loaded before accessing the data dictionary."
        return self.data_dict
    
    def update_data_dict(self, data_dict: Dict [str, np.ndarray]):
        """
        Set the data dictionary.
        This is useful for updating the data after loading.
        """
        assert isinstance(data_dict, dict), "data_dict must be a dictionary."
        # This feature is mostly used by MultiSourcePipeline to scale the data.
        # So it does not work with by source splitting.
        assert not self._is_split, "Cannot update data_dict after splitting."
        self.data_dict.update(data_dict)
        self._is_loaded = True

    @abstractmethod
    def _load_data(self) -> dict:
        """
        Expected to return a dict with at least:
            - "features"
            - "timestamps"
        Optional keys like:
            - "mask", "labels", etc.
        """
        self._is_loaded = True  # Mark data as loaded


    def _generate_splits(self):
        # Split the data (e.g., train/val/test sequences)

        assert self._is_loaded, "Data must be loaded before splitting."
        assert not self._is_split, (
            "Data has already been split. "
            "If you want to re-split, please reset the data source."
        )

        train_data, val_data, test_data = self.data_splitter.split_data(
            self.data_dict["features"]
        )
        train_timestamps, val_timestamps, test_timestamps = (
            self.data_splitter.split_data(self.data_dict["timestamps"])
        )

        # Check that splits are consistent
        assert train_data.shape[0] == train_timestamps.shape[0], (
            "Mismatch in train data and timestamps length"
        )
        assert val_data.shape[0] == val_timestamps.shape[0], (
            "Mismatch in val data and timestamps length"
        )
        assert test_data.shape[0] == test_timestamps.shape[0], (
            "Mismatch in test data and timestamps length"
        )

        # Update the self.data_dict
        self.data_dict.update(
            {"features": {"train": train_data, "val": val_data, "test": test_data}}
        )
        self.data_dict.update(
            {
                "timestamps": {
                    "train": train_timestamps,
                    "val": val_timestamps,
                    "test": test_timestamps,
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
            f"loader_arguments: {self.load_arguments}, "
            f"data_splitter: {self.data_splitter}, "
            f"scaler: {self.scalers}"
        )
