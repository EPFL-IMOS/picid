import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from typing import Any, List, Union, Tuple, Type, Optional, Dict
from lmetk.transforms.utils import _convert_to_numpy, _convert_from_numpy
from numpy import ndarray


class BaseTransform(ABC):
    """
    Simplified base transform interface. Transforms only need to implement
    the core transformation logic, not the fitting strategies.
    """
    
    @abstractmethod
    def transform_data(self, data_segment: ndarray | list[ndarray] , auxiliary_data: Optional[Dict[str, Any]] = None, 
                      mode: Optional[str] = None) -> Any:
        """
        Transform a single data segment.
        
        Parameters
        ----------
        data : Any
            The data to transform
        auxiliary_data : Optional[Dict[str, Any]]
            Additional data needed for transformation
        mode : Optional[str]
            Transformation mode (e.g., 'train', 'val', 'test')
        """
        pass
    
    def fit_data(self, data_segment: ndarray | list[ndarray], auxiliary_data: Optional[Dict[str, Any]] = None) -> 'BaseTransform':
        """
        Fit the transform on data. Default implementation does nothing (for stateless transforms).
        Override this method for transforms that need fitting.
        """
        return self
    
    def supports_multi_source_fit(self) -> bool:
        """
        Whether this transform can handle multi-source fitting.
        Override to return True if the transform can handle aggregated data from multiple sources.
        """
        return True  # Most transforms can handle concatenated data

# --- Adapter Examples for External Libraries ---

# Example: Adapter for sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class SklearnStandardScalerAdapter(BaseTransform):
    """
    An adapter for sklearn.preprocessing.StandardScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]):
        """Fits the internal StandardScaler on a single NumPy array."""
        np_data, _ = _convert_to_numpy(data_segment)
        self.scaler.fit(np_data) 
        
    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Transforms a single data segment using the internal StandardScaler."""
        np_data, original_type = _convert_to_numpy(data_segment)
        transformed_np_data = self.scaler.transform(np_data)
        return transformed_np_data #_convert_from_numpy(transformed_np_data, original_type, data_segment)

# Example: Adapter for sklearn.preprocessing.MinMaxScaler


class SklearnMinMaxScalerAdapter(BaseTransform):
    """
    An adapter for sklearn.preprocessing.MinMaxScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def fit_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]):
        """Fits the internal MinMaxScaler on a single NumPy array."""
        np_data, _ = _convert_to_numpy(data_segment)
        self.scaler.fit(np_data)
        
    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Transforms a single data segment using the internal MinMaxScaler."""
        np_data, original_type = _convert_to_numpy(data_segment)
        transformed_np_data = self.scaler.transform(np_data)
        return transformed_np_data #_convert_from_numpy(transformed_np_data, original_type, data_segment)

# Example: A simple custom direct transform (no fitting)
class LogTransformAdapter(BaseTransform):
    """
    A simple adapter for a logarithmic transformation, conforming to BaseTransform.
    This is a direct transform (no fitting required). Handles various input data types
    by converting to numpy.ndarray internally and converting back.
    """
    def fit_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]):
        """No fitting needed for a direct log transform."""
        pass 

    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Applies a natural logarithm transformation."""
        np_data, original_type = _convert_to_numpy(data_segment)
        
        # Add a small constant to avoid log(0) issues for non-negative data
        # Use np.maximum to handle negative numbers gracefully if they occur
        transformed_np_data = np.log1p(np.maximum(0, np_data)) # log(1 + x) for x >= 0

        return transformed_np_data #_convert_from_numpy(transformed_np_data, original_type, data_segment)