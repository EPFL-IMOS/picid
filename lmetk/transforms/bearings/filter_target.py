import logging
from typing import Any, Dict, List

import numpy as np

from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.multisource import BaseMultiSourceTransformMixin

logger = logging.getLogger(__name__) 

class FilterTarget(BaseTransform, BaseMultiSourceTransformMixin):
    def __init__(self, win_len: int, hop: int, **kwargs):
        """
        Initialize the transform.
        
        Args:
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.win_len = win_len
        self.hop = hop
        
    def _convert_to_numpy(self, value: Any, key: str) -> np.ndarray:
        """Convert value to numpy array if it isn't already."""
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value)
                logging.info(f"Converted key '{key}' to numpy array")
            except Exception as e:
                raise ValueError(f"Could not convert key '{key}' to numpy array: {e}")
        return value
    
    def _ensure_2d(self, value: np.ndarray, key: str) -> np.ndarray:
        """Ensure array is 2D for concatenation."""
        if value.ndim == 1:
            value = value.reshape(-1, 1)
            logging.info(f"Reshaped 1D array '{key}' to 2D: {value.shape}")
        elif value.ndim > 2:
            raise ValueError(f"Key '{key}' has unsupported dimensions: {value.ndim}. "
                           f"Only 1D and 2D arrays are supported.")
        return value
    
    def _check_for_nans(self, arrays: List[np.ndarray], keys: List[str]) -> None:
        """Check for NaN values and raise error if found."""
        for arr, key in zip(arrays, keys):
            if np.any(np.isnan(arr)):
                raise ValueError(f"NaN values found in array '{key}'")
    
    def transform_data(self, data: Dict, metadata: Dict) -> np.ndarray:
        """
        Filter targets correponding to the stftt
        
        Args:
            data: Dictionary containing numpy arrays to concatenate
            metadata: Dictionary containing auxiliary metadata (unused in this transform)
            
        Returns:
            Concatenated numpy array
        """
        features = data['features']
        assert 'target' in data, "Missing 'target' key in data"
        
        if not data:
            raise ValueError("No data provided for concatenation")
        
        
        target, key= data['target'], 'target'
        # Convert to numpy array
        target = self._convert_to_numpy(target, key)
        
        # Ensure 2D shape
        target = self._ensure_2d(target, key)
        logging.info(f"Added data '{key}' with shape {target.shape}")
        
        # Check for NaN values (will raise error if found)
        self._check_for_nans([target], ['target'])
        
        # Allign targets
        n_frames = features.shape[0]               # 103 here
        centers = np.arange(n_frames) * self.hop   # in samples
        end_idx = np.clip(centers + self.win_len//2 - 1, 0, len(target)-1)
        frame_targets = target[end_idx]       # aligned targets per time frame
        
        # Concatenate and reshape (-1,1)
        target = np.array(frame_targets).reshape(-1, 1)
        logging.info(f"Final target shape: {target.shape}")

        assert target.shape[0] == features.shape[0], "Target and result shapes do not match"
        assert target[-1] == 0, "For the RUL the last value has to be 0"

        return target
    
    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)