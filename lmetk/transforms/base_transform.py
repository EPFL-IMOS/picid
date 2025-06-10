import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from typing import Any, List, Union, Tuple, Type


# --- Utility for Type Conversion ---
def _convert_to_numpy(data: Any) -> Tuple[np.ndarray, Type]:
    """Converts input data to numpy.ndarray and returns its original type."""
    original_type = type(data)
    
    if isinstance(data, np.ndarray):
        return data, original_type
    elif isinstance(data, pd.DataFrame):
        return data.values, original_type
    elif isinstance(data, pd.Series):
        return data.values, original_type
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy(), original_type # Move to CPU if on GPU, then convert
    else:
        # For other types, try direct conversion or raise error if unsupported
        try:
            return np.asarray(data), original_type
        except Exception as e:
            raise TypeError(f"Unsupported data type for conversion to numpy: {original_type}. Error: {e}")

def _convert_from_numpy(numpy_data: np.ndarray, original_type: Type, original_data: Any) -> Any:
    """Converts numpy.ndarray back to the original data type."""
    if original_type is np.ndarray:
        return numpy_data
    elif original_type is pd.DataFrame:
        # Attempt to reconstruct DataFrame, preserving index and columns if possible
        if isinstance(original_data, pd.DataFrame):
            return pd.DataFrame(numpy_data, index=original_data.index, columns=original_data.columns)
        return pd.DataFrame(numpy_data) # Fallback
    elif original_type is pd.Series:
        # Attempt to reconstruct Series, preserving index and name if possible
        if isinstance(original_data, pd.Series):
            return pd.Series(numpy_data.flatten(), index=original_data.index, name=original_data.name)
        return pd.Series(numpy_data.flatten()) # Fallback
    elif original_type is torch.Tensor:
        # Convert back to Tensor, preserving original device if possible
        if isinstance(original_data, torch.Tensor):
            return torch.from_numpy(numpy_data).to(original_data.device)
        return torch.from_numpy(numpy_data) # Fallback to CPU
    else:
        # If the original type was not explicitly handled, return numpy array
        # as it was the intermediate format. This might lead to unexpected types
        # downstream if not careful.
        return numpy_data

# --- BaseTransform ABC ---

class BaseTransform(ABC):
    """
    Abstract base class for all custom and wrapped external transformation objects.
    This defines the expected interface for fitting and transforming data within
    the lmetk preprocessing pipeline, especially considering multi-source scenarios.

    All concrete transformation implementations (including adapters for external libraries)
    must inherit from this class and implement its abstract methods.
    """
    
    @abstractmethod
    def fit_single_source(self, data_segment: Any):
        """
        Fits the transform on a single data segment (e.g., a 'train' split from one source).
        
        Parameters
        ----------
        data_segment : Any
            The data segment to fit on. Its type can be numpy.ndarray, pandas.DataFrame,
            pandas.Series, torch.Tensor, or other array-like structures.
        """
        pass
        
    @abstractmethod
    def fit_multi_source(self, list_of_data_segments: List[Any]):
        """
        Fits the transform on a list of data segments, aggregated across multiple sources.
        
        This method is responsible for:
        1. Aggregating/stacking the `list_of_data_segments` into a single structure
           suitable for the underlying transformation logic (e.g., vstacking NumPy arrays).
        2. Performing the fitting operation on this aggregated data.

        Parameters
        ----------
        list_of_data_segments : List[Any]
            A list where each element is a data segment (e.g., 'train' split)
            from a different source.
        """
        pass
        
    @abstractmethod
    def transform_data(self, data_segment: Any) -> Any:
        """
        Applies the transformation to a single data segment.

        Parameters
        ----------
        data_segment : Any
            The data segment to transform. Its type can be numpy.ndarray, pandas.DataFrame,
            pandas.Series, torch.Tensor, or other array-like structures.

        Returns
        -------
        Any
            The transformed data segment. Its type should ideally match the input type.
        """
        pass

# --- Adapter Examples for External Libraries ---

# Example: Adapter for sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import StandardScaler

class SklearnStandardScalerAdapter(BaseTransform):
    """
    An adapter for sklearn.preprocessing.StandardScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit_single_source(self, data_segment: Any):
        """Fits the internal StandardScaler on a single NumPy array."""
        np_data, _ = _convert_to_numpy(data_segment)
        self.scaler.fit(np_data)
        
    def fit_multi_source(self, list_of_data_segments: List[Any]):
        """
        Aggregates multiple data segments by vertical stacking (after converting to NumPy) and
        fits the internal StandardScaler.
        """
        if not list_of_data_segments:
            raise ValueError("No data segments provided for multi-source fitting.")
        
        # Convert all segments to NumPy and then stack
        numpy_segments = [_convert_to_numpy(ds)[0] for ds in list_of_data_segments]
        
        try:
            aggregated_data = np.vstack(numpy_segments) 
        except ValueError as e:
            raise ValueError(f"Could not stack data segments for multi-source fitting. Ensure all segments have compatible shapes. Error: {e}")
        
        self.scaler.fit(aggregated_data)
        
    def transform_data(self, data_segment: Any) -> Any:
        """Transforms a single data segment using the internal StandardScaler."""
        np_data, original_type = _convert_to_numpy(data_segment)
        transformed_np_data = self.scaler.transform(np_data)
        return _convert_from_numpy(transformed_np_data, original_type, data_segment)

# Example: Adapter for sklearn.preprocessing.MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

class SklearnMinMaxScalerAdapter(BaseTransform):
    """
    An adapter for sklearn.preprocessing.MinMaxScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def fit_single_source(self, data_segment: Any):
        """Fits the internal MinMaxScaler on a single NumPy array."""
        np_data, _ = _convert_to_numpy(data_segment)
        self.scaler.fit(np_data)
        
    def fit_multi_source(self, list_of_data_segments: List[Any]):
        """
        Aggregates multiple data segments by vertical stacking (after converting to NumPy) and
        fits the internal MinMaxScaler.
        """
        if not list_of_data_segments:
            raise ValueError("No data segments provided for multi-source fitting.")
        
        numpy_segments = [_convert_to_numpy(ds)[0] for ds in list_of_data_segments]
        
        try:
            aggregated_data = np.vstack(numpy_segments) 
        except ValueError as e:
            raise ValueError(f"Could not stack data segments for multi-source fitting. Ensure all segments have compatible shapes. Error: {e}")
        
        self.scaler.fit(aggregated_data)
        
    def transform_data(self, data_segment: Any) -> Any:
        """Transforms a single data segment using the internal MinMaxScaler."""
        np_data, original_type = _convert_to_numpy(data_segment)
        transformed_np_data = self.scaler.transform(np_data)
        return _convert_from_numpy(transformed_np_data, original_type, data_segment)

# Example: A simple custom direct transform (no fitting)
class LogTransformAdapter(BaseTransform):
    """
    A simple adapter for a logarithmic transformation, conforming to BaseTransform.
    This is a direct transform (no fitting required). Handles various input data types
    by converting to numpy.ndarray internally and converting back.
    """
    def fit_single_source(self, data_segment: Any):
        """No fitting needed for a direct log transform."""
        pass 

    def fit_multi_source(self, list_of_data_segments: List[Any]):
        """No fitting needed for a direct log transform."""
        pass 

    def transform_data(self, data_segment: Any) -> Any:
        """Applies a natural logarithm transformation."""
        np_data, original_type = _convert_to_numpy(data_segment)
        
        # Add a small constant to avoid log(0) issues for non-negative data
        # Use np.maximum to handle negative numbers gracefully if they occur
        transformed_np_data = np.log1p(np.maximum(0, np_data)) # log(1 + x) for x >= 0

        return _convert_from_numpy(transformed_np_data, original_type, data_segment)