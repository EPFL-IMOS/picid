import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional, Union

from lmetk.transforms.base_transform import BaseTransform 

import logging
logger = logging.getLogger(__name__) 

class ConcatenateTransform(BaseTransform):

    def __init__(self, exclude_keys: Optional[List[str]] = None, **kwargs):
        """
        Initialize the transform.
        
        Args:
            exclude_keys: List of auxiliary_data keys to exclude from concatenation
            **kwargs: Additional keyword arguments
        """
        self._init_kwargs = kwargs 
        self.exclude_keys = exclude_keys or []

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        exclude_str = f"exclude_keys={self.exclude_keys!r}"
        if args:
            return f"{self.__class__.__name__}({exclude_str}, {args})"
        return f"{self.__class__.__name__}({exclude_str})"

    def fit_single_source(self, data_segment: Any):
        """
        Concatenation is a stateless operation, so no fitting is required.
        This method does nothing.
        """
        pass

    def fit_multi_source(self, list_of_data_segments: List[Any]):
        """
        Concatenation is a stateless operation, so no fitting is required
        across multiple sources. This method does nothing.
        """
        pass

    def transform_data(self, data_segment: Any, auxiliary_data: Dict,  mode: str) -> Any:
        """
        Concatenate all auxiliary_data values to the data_segment.
        
        Args:
            data_segment: The main data segment (assumed to be numpy array)
            auxiliary_data: Dictionary containing auxiliary data to concatenate
            
        Returns:
            Concatenated data segment with auxiliary data
        """
        # Convert data_segment to numpy array if it isn't already
        if not isinstance(data_segment, np.ndarray):
            data_segment = np.array(data_segment)
        
        # Ensure data_segment is 2D
        if data_segment.ndim == 1:
            data_segment = data_segment.reshape(-1, 1)
        
        # Collect all auxiliary data arrays to concatenate
        aux_arrays = []
        
        for key, value in auxiliary_data.items():
            # Skip excluded keys
            if key in self.exclude_keys:
                continue
                
            # Convert value to numpy array
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            
            # Ensure the auxiliary data has the same number of rows as data_segment
            if value.ndim == 1:
                if len(value) == data_segment.shape[0]:
                    value = value.reshape(-1, 1)
                else:
                    raise ValueError(f"Auxiliary data key '{key}' length mismatch. "
                                   f"Expected {data_segment.shape[0]}, got {len(value)}")
            elif value.ndim == 2:
                if value.shape[0] != data_segment.shape[0]:
                    raise ValueError(f"Auxiliary data key '{key}' row count mismatch. "
                                   f"Expected {data_segment.shape[0]}, got {value.shape[0]}")
            else:
                raise ValueError(f"Auxiliary data key '{key}' has unsupported dimensions: {value.ndim}. "
                               f"Only 1D and 2D arrays are supported.")
            
            aux_arrays.append(value)
            logging.info(f"Adding auxiliary data '{key}' with shape {value.shape}")
        
        # Concatenate all arrays horizontally
        if aux_arrays:
            all_arrays = [data_segment] + aux_arrays
            result = np.concatenate(all_arrays, axis=1)
            logging.info(f"Final concatenated shape: {result.shape}")
        else:
            logging.info("No auxiliary data to concatenate")
            result = data_segment
        
        return result

    def __call__(self, data_segment: Any, auxiliary_data: Dict,  mode: str) -> Any:
        return self.transform_data(data_segment, auxiliary_data,  mode=mode)