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

