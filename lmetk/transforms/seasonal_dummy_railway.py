import pandas as pd
import numpy as np
import logging
from typing import Any, List, Dict, Optional

from lmetk.transforms.base_transform import BaseTransform 
logger = logging.getLogger(__name__) 

class SeasonalDummyTransform(BaseTransform):

    def __init__(self, shift: int, seq_len: int = 168, **kwargs):
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
        
        # Storage for seasonal history - maps seasonal_pos to list of all historical values
        self.seasonal_history = {i: [] for i in range(self.seq_len)}

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}(shift={self.shift!r}, seq_len={self.seq_len!r}, {args})"

    def fit_single_source(self, data_segment: Any, auxiliary_data: Optional[Dict] = None):
        """
        No-op for this transform since all logic is in transform_data.
        Just initialize the seasonal history.
        """
        self.seasonal_history = {i: [] for i in range(self.seq_len)}
        return data_segment

    def fit_multi_source(self, list_of_data_segments: List[Any]):
        """
        This transform is designed for single source fitting.
        """
        raise NotImplementedError("SeasonalDummyTransform should use fit_single_source")

    def transform_data(self, data_segment: Any, auxiliary_data: Dict, mode: str) -> Any:
        """
        Transform data by adding shifted target values and seasonal statistics.
        All seasonal history management happens here.
        
        Args:
            data_segment: The main data segment (numpy array)
            auxiliary_data: Dictionary containing 'target' and other auxiliary data
            mode: The mode of transformation ('train', 'val', 'test')
            
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
        
        # Calculate seasonal statistics and update history
        seasonal_stats = self._calculate_seasonal_stats_vectorized(dummy_leak, mode)
        
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

    def _calculate_seasonal_stats_vectorized(self, shifted_target: np.ndarray, mode: str) -> List[np.ndarray]:
        """
        Highly optimized vectorized version of seasonal statistics calculation.
        """
        n_samples, n_features = shifted_target.shape
        seasonal_mean = np.zeros_like(shifted_target)
        seasonal_std = np.zeros_like(shifted_target)

        desired_polynom = 0
        seasonal_trend = np.zeros((shifted_target.shape[0], desired_polynom+2))
        
        # Pre-compute seasonal positions
        seasonal_positions = np.arange(n_samples) % self.seq_len
        #non_zero_mask = ~np.allclose(shifted_target, 0, axis=1)
        
        # For each unique seasonal position
        for seasonal_pos in range(self.seq_len):
            # Find all indices with this seasonal position
            pos_indices = np.where(seasonal_positions == seasonal_pos)[0]
            #pos_indices = pos_indices[non_zero_mask[pos_indices]]  # Only non-zero values
            
            # Get historical values for this seasonal position
            history = self.seasonal_history[seasonal_pos]
            
            # For each occurrence of this seasonal position
            seen_idxs = []
            for current_idx in pos_indices:
                # Get all prior indices with the same seasonal position

                # The ones that we have in split
                if current_idx >= self.shift:
                    seen_idxs.append(current_idx)
                
                observations = shifted_target[seen_idxs]
                
                # if mode != 'train':     
                    # # Values from history
                    # observations = np.concatenate([np.array(history[-168:]).reshape(-1,1), observations], axis=0)
        
                
        

                seasonal_mean[current_idx] = np.mean(observations, axis=0)
                seasonal_std[current_idx] = np.std(observations, axis=0)

                eff_observations = observations[:-1]
                # Calculate trend (slope of linear regression over time)
                if len(eff_observations) > desired_polynom+1:
                    trends = np.zeros(desired_polynom+1)
                    # Calculate slope using least squares
                    trends = np.polyfit(np.arange(len(eff_observations)), eff_observations.flatten(), desired_polynom)
                    y_pred = np.poly1d(trends)(len(observations))
                    error = y_pred - observations[-1]
                    seasonal_trend[current_idx] = np.concatenate([trends, error])

        
        # Update seasonal history
        self._update_seasonal_history(shifted_target, mode)
        # seasonal_trend
        return [seasonal_mean, seasonal_std]

    def _update_seasonal_history(self, shifted_target: np.ndarray, mode: str):
        """
        Update the seasonal history with new values from the current data segment.
        
        Args:
            shifted_target: The shifted target values
            mode: The transformation mode ('train', 'val', 'test')
        """
        # Determine which values to add to history
        if mode == 'train':
            # In training, add all non-zero values after the shift period
            start_idx = 0  # Skip the initial zeros from shifting

            # # In val/test, we might have lookback data that shouldn't be double-counted
            # # This depends on your data pipeline structure
            # start_idx = self.shift
        
            # Pre-compute seasonal positions
            n_samples, n_features = shifted_target[start_idx:].shape
            seasonal_positions = np.arange(n_samples) % self.seq_len
            for seasonal_pos in range(self.seq_len):
                #if not np.allclose(shifted_target[i], 0):  # Skip zero/missing values
                pos_indices = np.where(seasonal_positions == seasonal_pos)[0]
                self.seasonal_history[seasonal_pos].extend(shifted_target[pos_indices].copy().squeeze().tolist())

    def get_seasonal_history_summary(self):
        """
        Get a summary of the seasonal history for debugging/validation.
        """
        summary = {}
        for pos in range(self.seq_len):
            summary[pos] = {
                'count': len(self.seasonal_history[pos]),
                'last_values': self.seasonal_history[pos][-3:] if self.seasonal_history[pos] else []
            }
        return summary

    def reset_history(self):
        """
        Reset the seasonal history. Useful for starting fresh with new data.
        """
        self.seasonal_history = {i: [] for i in range(self.seq_len)}

    def __call__(self, data_segment: Any, auxiliary_data: Dict, mode: str) -> Any:
        return self.transform_data(data_segment, auxiliary_data, mode)










# import pandas as pd
# import numpy as np
# from typing import Any, List, Dict

# from lmetk.transforms.base_transform import BaseTransform 


# class SeasonalDummyTransform(BaseTransform):

#     def __init__(self, shift: int, seq_len: int=168, **kwargs):
#         """
#         Initialize the seasonal dummy transform.
        
#         Args:
#             shift: Number of time steps to shift the target (e.g., 168 for weekly)
#             seq_len: Sequence length that defines the seasonal period
#             **kwargs: Additional keyword arguments
#         """
#         self._init_kwargs = kwargs 
#         self.shift = shift
#         self.seq_len = seq_len

#     def __repr__(self):
#         args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
#         return f"{self.__class__.__name__}(shift={self.shift!r}, seq_len={self.seq_len!r}, {args})"

#     def fit_single_source(self, data_segment: Any):
#         """
#         This is a stateless operation, so no fitting is required.
#         """
#         pass

#     def fit_multi_source(self, list_of_data_segments: List[Any]):
#         """
#         This is a stateless operation, so no fitting is required.
#         """
#         pass

#     def transform_data(self, data_segment: Any, auxiliary_data: Dict) -> Any:
#         """
#         Transform data by adding shifted target values and seasonal statistics.
        
#         Args:
#             data_segment: The main data segment (numpy array)
#             auxiliary_data: Dictionary containing 'target' and other auxiliary data
            
#         Returns:
#             Concatenated data with shifted targets and seasonal statistics
#         """
#         # Validate inputs
#         if 'target' not in auxiliary_data:
#             raise ValueError("auxiliary_data must contain 'target' key")
        
#         target = auxiliary_data['target']
#         if not isinstance(target, np.ndarray):
#             target = np.array(target)
        
#         # Ensure target is 2D
#         if target.ndim == 1:
#             target = target.reshape(-1, 1)
        
#         # Ensure data_segment is 2D
#         if not isinstance(data_segment, np.ndarray):
#             data_segment = np.array(data_segment)
#         if data_segment.ndim == 1:
#             data_segment = data_segment.reshape(-1, 1)
        
#         # Validate dimensions
#         if target.shape[0] != data_segment.shape[0]:
#             raise ValueError(f"Target length {target.shape[0]} doesn't match data_segment length {data_segment.shape[0]}")
        
#         # Create shifted target (dummy leak)
#         dummy_leak = np.concatenate([np.zeros((self.shift, target.shape[1])), target[:-self.shift]])
        
#         # Calculate seasonal statistics
#         seasonal_stats = self._calculate_seasonal_stats(dummy_leak, target.shape[0])
        
#         # Concatenate all features
#         result = np.concatenate([
#             data_segment,           # Original features
#             dummy_leak,            # Shifted target values
#             *seasonal_stats        # All seasonal statistics
#         ], axis=1)
        
#         print(f"Original data shape: {data_segment.shape}")
#         print(f"Shifted target shape: {dummy_leak.shape}")
#         print(f"Seasonal stats shapes: {[stat.shape for stat in seasonal_stats]}")
#         print(f"Final concatenated shape: {result.shape}")
        
#         return result
    
#     def _calculate_seasonal_stats(self, shifted_target: np.ndarray, total_length: int) -> List[np.ndarray]:
#         """
#         Calculate comprehensive seasonal statistics based on the seasonal pattern.
#         Now considers daily seasonality within the weekly pattern.
        
#         Args:
#             shifted_target: The shifted target values
#             total_length: Total length of the sequence
            
#         Returns:
#             List of seasonal statistics arrays: [mean, std, min, max, median, q25, q75, count, trend]
#         """
#         # Initialize all statistics arrays
#         seasonal_mean = np.zeros_like(shifted_target)
#         seasonal_std = np.zeros_like(shifted_target)
#         seasonal_min = np.zeros_like(shifted_target)
#         seasonal_max = np.zeros_like(shifted_target)
#         seasonal_median = np.zeros_like(shifted_target)
#         seasonal_q25 = np.zeros_like(shifted_target)
#         seasonal_q75 = np.zeros_like(shifted_target)
#         seasonal_count = np.zeros_like(shifted_target)
#         seasonal_trend = np.zeros_like(shifted_target)
        
#         # For each position, calculate stats from all previous values at the same seasonal position
#         for i in range(total_length):
#             # Find all previous positions with the same hour of the same day of week
#             seasonal_positions = []
#             # Start from the current position and go backwards by seq_len steps
#             pos = i - self.seq_len
#             while pos >= 0:
#                 seasonal_positions.append(pos)
#                 pos -= self.seq_len  # Move to previous week, same day/hour
            
#             if seasonal_positions:
#                 # Get values from all previous seasonal positions
#                 seasonal_values = shifted_target[seasonal_positions]
                
#                 # Remove zeros (which represent unavailable past values)
#                 non_zero_mask = np.any(seasonal_values != 0, axis=1)
#                 if np.any(non_zero_mask):
#                     valid_seasonal_values = seasonal_values[non_zero_mask]
                    
#                     # Calculate comprehensive statistics
#                     seasonal_mean[i] = np.mean(valid_seasonal_values, axis=0)
#                     seasonal_std[i] = np.std(valid_seasonal_values, axis=0)
#                     seasonal_min[i] = np.min(valid_seasonal_values, axis=0)
#                     seasonal_max[i] = np.max(valid_seasonal_values, axis=0)
#                     seasonal_median[i] = np.median(valid_seasonal_values, axis=0)
#                     seasonal_q25[i] = np.percentile(valid_seasonal_values, 25, axis=0)
#                     seasonal_q75[i] = np.percentile(valid_seasonal_values, 75, axis=0)
#                     seasonal_count[i] = len(valid_seasonal_values)
                    
#                     # Calculate trend (slope of linear regression over time)
#                     if len(valid_seasonal_values) >= 2:
#                         # Simple linear trend calculation
#                         x = np.arange(len(valid_seasonal_values))
#                         trends = []
#                         for col in range(valid_seasonal_values.shape[1]):
#                             y = valid_seasonal_values[:, col]
#                             # Calculate slope using least squares
#                             slope = np.polyfit(x, y, 1)[0] if len(np.unique(y)) > 1 else 0
#                             trends.append(slope)
#                         seasonal_trend[i] = np.array(trends)
#                     else:
#                         seasonal_trend[i] = 0
                        
#                 else:
#                     # If no valid values, keep as zeros
#                     pass  # Already initialized to zeros
#             else:
#                 # For positions without enough history, use zeros
#                 pass  # Already initialized to zeros
        
#         return [
#             seasonal_mean,
#             seasonal_std, 
#             # seasonal_min,
#             # seasonal_max,
#             # seasonal_median,
#             # seasonal_q25,
#             # seasonal_q75,
#             # seasonal_count,
#             seasonal_trend
#         ]

#     # def _calculate_seasonal_stats(self, shifted_target: np.ndarray, total_length: int) -> List[np.ndarray]:
#     #     """
#     #     Calculate comprehensive seasonal statistics based on the seasonal pattern.
        
#     #     Args:
#     #         shifted_target: The shifted target values
#     #         total_length: Total length of the sequence
            
#     #     Returns:
#     #         List of seasonal statistics arrays: [mean, std, min, max, median, q25, q75, count, trend]
#     #     """
#     #     # Initialize all statistics arrays
#     #     seasonal_mean = np.zeros_like(shifted_target)
#     #     seasonal_std = np.zeros_like(shifted_target)
#     #     seasonal_min = np.zeros_like(shifted_target)
#     #     seasonal_max = np.zeros_like(shifted_target)
#     #     seasonal_median = np.zeros_like(shifted_target)
#     #     seasonal_q25 = np.zeros_like(shifted_target)
#     #     seasonal_q75 = np.zeros_like(shifted_target)
#     #     seasonal_count = np.zeros_like(shifted_target)
#     #     seasonal_trend = np.zeros_like(shifted_target)
        
#     #     # For each position, calculate stats from all previous values at the same seasonal position
#     #     for i in range(total_length):
#     #         # Find all previous seasonal positions
#     #         seasonal_positions = []
#     #         pos = i
#     #         while pos >= self.seq_len:
#     #             pos -= self.seq_len
#     #             seasonal_positions.append(pos)
            
#     #         if seasonal_positions:
#     #             # Get values from all previous seasonal positions
#     #             seasonal_values = shifted_target[seasonal_positions]
                
#     #             # Remove zeros (which represent unavailable past values)
#     #             non_zero_mask = np.any(seasonal_values != 0, axis=1)
#     #             if np.any(non_zero_mask):
#     #                 valid_seasonal_values = seasonal_values[non_zero_mask]
                    
#     #                 # Calculate comprehensive statistics
#     #                 seasonal_mean[i] = np.mean(valid_seasonal_values, axis=0)
#     #                 seasonal_std[i] = np.std(valid_seasonal_values, axis=0)
#     #                 seasonal_min[i] = np.min(valid_seasonal_values, axis=0)
#     #                 seasonal_max[i] = np.max(valid_seasonal_values, axis=0)
#     #                 seasonal_median[i] = np.median(valid_seasonal_values, axis=0)
#     #                 seasonal_q25[i] = np.percentile(valid_seasonal_values, 25, axis=0)
#     #                 seasonal_q75[i] = np.percentile(valid_seasonal_values, 75, axis=0)
#     #                 seasonal_count[i] = len(valid_seasonal_values)
                    
#     #                 # Calculate trend (slope of linear regression over time)
#     #                 if len(valid_seasonal_values) >= 2:
#     #                     # Simple linear trend calculation
#     #                     x = np.arange(len(valid_seasonal_values))
#     #                     trends = []
#     #                     for col in range(valid_seasonal_values.shape[1]):
#     #                         y = valid_seasonal_values[:, col]
#     #                         # Calculate slope using least squares
#     #                         slope = np.polyfit(x, y, 1)[0] if len(np.unique(y)) > 1 else 0
#     #                         trends.append(slope)
#     #                     seasonal_trend[i] = np.array(trends)
#     #                 else:
#     #                     seasonal_trend[i] = 0
                        
#     #             else:
#     #                 # If no valid values, keep as zeros
#     #                 pass  # Already initialized to zeros
#     #         else:
#     #             # For positions without enough history, use zeros
#     #             pass  # Already initialized to zeros
        
#     #     return [
#     #         seasonal_mean,
#     #         seasonal_std, 
#     #         # seasonal_min,
#     #         # seasonal_max,
#     #         # seasonal_median,
#     #         # seasonal_q25,
#     #         # seasonal_q75,
#     #         # seasonal_count,
#     #         # seasonal_trend
#     #     ]

#     def __call__(self, data_segment: Any, auxiliary_data: Dict) -> Any:
#         return self.transform_data(data_segment, auxiliary_data)