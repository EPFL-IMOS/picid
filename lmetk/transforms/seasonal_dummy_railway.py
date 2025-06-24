import pandas as pd
import numpy as np
import logging
from typing import Any, List, Dict

from lmetk.transforms.base_transform import BaseTransform 
logger = logging.getLogger(__name__) 

class SeasonalDummyTransform(BaseTransform):

    def __init__(self, shift: int, seq_len: int=168, **kwargs):
        """
        Initialize the seasonal dummy transform.
        
        Args:
            shift: Number of time steps to shift the target (e.g., 168 for weekly)
            seq_len: Sequence length that defines the seasonal period
            **kwargs: Additional keyword arguments
        """
        self._init_kwargs = kwargs 
        self.shift = shift
        self.seq_len = seq_len

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}(shift={self.shift!r}, seq_len={self.seq_len!r}, {args})"

    def fit_single_source(self, data_segment: Any):
        """
        This is a stateless operation, so no fitting is required.
        """
        pass

    def fit_multi_source(self, list_of_data_segments: List[Any]):
        """
        This is a stateless operation, so no fitting is required.
        """
        pass

    def transform_data(self, data_segment: Any, auxiliary_data: Dict) -> Any:
        """
        Transform data by adding shifted target values and seasonal statistics.
        
        Args:
            data_segment: The main data segment (numpy array)
            auxiliary_data: Dictionary containing 'target' and other auxiliary data
            
        Returns:
            Concatenated data with shifted targets and seasonal statistics
        """
        # Validate inputs
        if 'target' not in auxiliary_data:
            raise ValueError("auxiliary_data must contain 'target' key")
        
        target = auxiliary_data['target']
        if not isinstance(target, np.ndarray):
            target = np.array(target)
        
        # Ensure target is 2D
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        
        # Ensure data_segment is 2D
        if not isinstance(data_segment, np.ndarray):
            data_segment = np.array(data_segment)
        if data_segment.ndim == 1:
            data_segment = data_segment.reshape(-1, 1)
        
        # Validate dimensions
        if target.shape[0] != data_segment.shape[0]:
            raise ValueError(f"Target length {target.shape[0]} doesn't match data_segment length {data_segment.shape[0]}")
        
        # Create shifted target (dummy leak)
        dummy_leak = np.concatenate([np.zeros((self.shift, target.shape[1])), target[:-self.shift]])
        
        # Calculate seasonal statistics
        seasonal_stats = self._calculate_seasonal_stats(dummy_leak, target.shape[0])
        
        # Concatenate all features
        result = np.concatenate([
            data_segment,           # Original features
            dummy_leak,            # Shifted target values
            *seasonal_stats        # All seasonal statistics
        ], axis=1)
        
        logger.info(f"Original data shape: {data_segment.shape}")
        logger.info(f"Shifted target shape: {dummy_leak.shape}")
        logger.info(f"Seasonal stats shapes: {[stat.shape for stat in seasonal_stats]}")
        logger.info(f"Final concatenated shape: {result.shape}")
        
        return result

    def _calculate_seasonal_stats(self, shifted_target: np.ndarray, total_length: int) -> List[np.ndarray]:
        """
        Calculate comprehensive seasonal statistics based on the seasonal pattern.
        
        Args:
            shifted_target: The shifted target values
            total_length: Total length of the sequence
            
        Returns:
            List of seasonal statistics arrays: [mean, std, min, max, median, q25, q75, count, trend]
        """
        # Initialize all statistics arrays
        seasonal_mean = np.zeros_like(shifted_target)
        seasonal_std = np.zeros_like(shifted_target)
        seasonal_min = np.zeros_like(shifted_target)
        seasonal_max = np.zeros_like(shifted_target)
        seasonal_median = np.zeros_like(shifted_target)
        seasonal_q25 = np.zeros_like(shifted_target)
        seasonal_q75 = np.zeros_like(shifted_target)
        seasonal_count = np.zeros_like(shifted_target)
        seasonal_trend = np.zeros_like(shifted_target)
        
        # For each position, calculate stats from all previous values at the same seasonal position
        for i in range(total_length):
            # Find all previous seasonal positions
            seasonal_positions = []
            pos = i
            while pos >= self.seq_len:
                pos -= self.seq_len
                seasonal_positions.append(pos)
            
            if seasonal_positions:
                # Get values from all previous seasonal positions
                seasonal_values = shifted_target[seasonal_positions]
                
                # Remove zeros (which represent unavailable past values)
                non_zero_mask = np.any(seasonal_values != 0, axis=1)
                if np.any(non_zero_mask):
                    valid_seasonal_values = seasonal_values[non_zero_mask]
                    
                    # Calculate comprehensive statistics
                    seasonal_mean[i] = np.mean(valid_seasonal_values, axis=0)
                    seasonal_std[i] = np.std(valid_seasonal_values, axis=0)
                    seasonal_min[i] = np.min(valid_seasonal_values, axis=0)
                    seasonal_max[i] = np.max(valid_seasonal_values, axis=0)
                    seasonal_median[i] = np.median(valid_seasonal_values, axis=0)
                    seasonal_q25[i] = np.percentile(valid_seasonal_values, 25, axis=0)
                    seasonal_q75[i] = np.percentile(valid_seasonal_values, 75, axis=0)
                    seasonal_count[i] = len(valid_seasonal_values)
                    
                    # Calculate trend (slope of linear regression over time)
                    if len(valid_seasonal_values) >= 2:
                        # Simple linear trend calculation
                        x = np.arange(len(valid_seasonal_values))
                        trends = []
                        for col in range(valid_seasonal_values.shape[1]):
                            y = valid_seasonal_values[:, col]
                            # Calculate slope using least squares
                            slope = np.polyfit(x, y, 1)[0] if len(np.unique(y)) > 1 else 0
                            trends.append(slope)
                        seasonal_trend[i] = np.array(trends)
                    else:
                        seasonal_trend[i] = 0
                        
                else:
                    # If no valid values, keep as zeros
                    pass  # Already initialized to zeros
            else:
                # For positions without enough history, use zeros
                pass  # Already initialized to zeros
        
        return [
            seasonal_mean,
            seasonal_std, 
            # seasonal_min,
            # seasonal_max,
            # seasonal_median,
            # seasonal_q25,
            # seasonal_q75,
            # seasonal_count,
            # seasonal_trend
        ]

    def __call__(self, data_segment: Any, auxiliary_data: Dict) -> Any:
        return self.transform_data(data_segment, auxiliary_data)