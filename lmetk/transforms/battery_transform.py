


import pandas as pd
import numpy as np
from typing import Optional, Union


import pandas as pd
import numpy as np
from typing import Any, List, Dict

# Assuming this import path is correct for your project

from lmetk.transforms.base_transform import BaseTransform 

class BatteryTransform(BaseTransform):

    def __init__(self, **kwargs):
        self._init_kwargs = kwargs 
        self.shift = kwargs['shift']
        self.polynom_degree = kwargs.get('polynom_degree', 4)  # Degree of polynomial for trend fitting
        self.lookback_window = kwargs.get('lookback_window', 100)  # How far back to look for patterns
        self.rolling_window = kwargs.get('rolling_window', 1000000)  # Rolling window for local statistics
        
        # History for polynomial trend fitting
        self.value_history = []
        self.time_history = []

    def __repr__(self):
        # Ensure 'freq' is included in the representation, along with any other init args
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}(freq={self.freq!r}, {args})"

    def transform_data(self, data_segment: Any, auxiliary_data: Dict, mode: str) -> Any:
        
        #assert self.shift in [24, 168, 100], "dummy shift must be 24 for the timepfn, and 168 for the tabpfn for railway and 100 for the battery!"

        # Features of the target shifted
        dummy_leak = np.concatenate([np.zeros((self.shift, 1)), auxiliary_data['target'][:-self.shift]])
        assert dummy_leak.shape[0] == data_segment.shape[0]

        initial_dates = list(auxiliary_data['initial_dates'])
        initial_dates_shifted = initial_dates[self.shift:]
        print(initial_dates[:5])
        print(initial_dates_shifted[:5])
        
        # Concatenate original features with shifted target
        data_segment = np.concatenate([data_segment, dummy_leak], axis=1)

        # Calculate polynomial-based seasonal statistics and update history
        #seasonal_stats = self._calculate_polynomial_stats_vectorized(dummy_leak, mode)
        
        # Concatenate all features
        result = np.concatenate([
            data_segment,           # Original features + shifted target
            #*seasonal_stats         # All polynomial-based statistics
        ], axis=1)
        
        print(f"Mode: {mode}, Original data shape: {data_segment.shape}")
        print(f"Shifted target shape: {dummy_leak.shape}")
        # print(f"Polynomial stats shapes: {[stat.shape for stat in seasonal_stats]}")
        print(f"Final concatenated shape: {result.shape}")
        
        return result

    def _calculate_polynomial_stats_vectorized(self, shifted_target: np.ndarray, mode: str) -> List[np.ndarray]:
        """
        Calculate polynomial-based statistics to capture seasonal patterns without explicit seasonality.
        Uses polynomial fitting on historical data to derive trend and pattern features.
        """
        n_samples, n_features = shifted_target.shape
        
        # Initialize output arrays
        polynomial_trend = np.zeros((n_samples, self.polynom_degree + 2))  # coefficients + prediction + error
        rolling_mean = np.zeros_like(shifted_target)
        rolling_std = np.zeros_like(shifted_target)
        local_trend = np.zeros_like(shifted_target)
        
        # Process each time step
        for i in range(n_samples):
            current_value = shifted_target[i]
            
            # Skip if we're in the initial shift period (zeros)
            if i < self.shift:
                # For initial period, use simple defaults
                rolling_mean[i] = current_value
                rolling_std[i] = 0
                local_trend[i] = 0
                polynomial_trend[i] = 0
                continue
            
            # Get available history up to current point
            available_history = shifted_target[self.shift:i+1]  # Exclude the zero-padded portion
            
            if len(available_history) >= 2:
                # Calculate rolling statistics with available data
                window_start = max(0, len(available_history) - self.rolling_window)
                window_data = available_history[window_start:]
                
                rolling_mean[i] = np.mean(window_data, axis=0)
                rolling_std[i] = np.std(window_data, axis=0) if len(window_data) > 1 else 0
                
                # Calculate local trend (slope over recent window)
                if len(window_data) > 1:
                    x_local = np.arange(len(window_data))
                    local_slope = np.polyfit(x_local, window_data.flatten(), 1)[0]
                    local_trend[i] = local_slope
                
                # Polynomial trend fitting on longer history
                if len(available_history) > self.polynom_degree + 1:
                    # Use lookback window or all available data, whichever is smaller
                    lookback_start = max(0, len(available_history) - self.lookback_window)
                    trend_data = available_history[lookback_start:]
                    
                    try:
                        # Fit polynomial to historical data
                        x_trend = np.arange(len(trend_data))
                        poly_coeffs = np.polyfit(x_trend, trend_data.flatten(), self.polynom_degree)
                        
                        # Predict next value using the polynomial
                        poly_func = np.poly1d(poly_coeffs)
                        predicted_value = poly_func(len(trend_data))
                        
                        # Calculate prediction error
                        actual_value = current_value[0] if current_value.ndim > 0 else current_value
                        prediction_error = predicted_value - actual_value
                        
                        # Store polynomial coefficients, prediction, and error
                        polynomial_trend[i, :len(poly_coeffs)] = poly_coeffs
                        polynomial_trend[i, -2] = predicted_value
                        polynomial_trend[i, -1] = prediction_error
                        
                    except np.linalg.LinAlgError:
                        # Handle numerical issues with polynomial fitting
                        polynomial_trend[i] = 0
            else:
                # Not enough data for meaningful statistics
                rolling_mean[i] = current_value
                rolling_std[i] = 0
                local_trend[i] = 0
                polynomial_trend[i] = 0
        
        # Update history for future use
        self._update_value_history(shifted_target, mode)
        
        return [
            rolling_mean, 
            rolling_std, 
            #local_trend, 
            #polynomial_trend
            ]

    def _update_value_history(self, shifted_target: np.ndarray, mode: str):
        """
        Update the value history with new observations for future polynomial fitting.
        """
        
        # Add non-zero values to history (skip the initial shift period)
        valid_values = shifted_target[self.shift:]
        self.value_history.extend(valid_values.flatten().tolist())
        
        # Keep history manageable by limiting size
        max_history_size = self.lookback_window * 3
        if len(self.value_history) > max_history_size:
            self.value_history = self.value_history[-max_history_size:]

    def get_polynomial_history_summary(self):
        """
        Get a summary of the polynomial fitting history for debugging/validation.
        """
        return {
            'history_length': len(self.value_history),
            'recent_values': self.value_history[-10:] if self.value_history else [],
            'polynom_degree': self.polynom_degree,
            'lookback_window': self.lookback_window,
            'rolling_window': self.rolling_window
        }

    def reset_history(self):
        """
        Reset the value history. Useful for starting fresh with new data.
        """
        self.value_history = []
        self.time_history = []

    def __call__(self, data_segment: Any, auxiliary_data: Dict, mode: str) -> Any:
        return self.transform_data(data_segment, auxiliary_data, mode)


# class BatteryTransform(BaseTransform):

#     def __init__(self, **kwargs):
#         self._init_kwargs = kwargs 
#         self.shift = kwargs['shift']

#     def __repr__(self):
#         # Ensure 'freq' is included in the representation, along with any other init args
#         args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
#         return f"{self.__class__.__name__}(freq={self.freq!r}, {args})"

#     def fit_single_source(self, data_segment: Any):
#         """
#         Time feature extraction is a stateless operation, so no fitting is required.
#         This method does nothing.
#         """
#         pass

#     def fit_multi_source(self, list_of_data_segments: List[Any]):
#         """
#         Time feature extraction is a stateless operation, so no fitting is required
#         across multiple sources. This method does nothing.
#         """
#         pass

#     def transform_data(self, data_segment: Any, auxiliary_data: Dict,  mode: str) -> Any:
        
#         # Your original working logic: Convert input to DataFrame and extract 'date' values.
#         # This assumes the 'data_segment' can be directly converted to a DataFrame
#         # and has a 'date' column, or can be implicitly handled by pd.to_datetime.
        
#         assert self.shift in [24, 168, 100], "dummy shift must be 24 for the timepfn, and 168 for the tabpfn for railway and 100 for the battery!"

#         # Features of the target shifted
#         dummy_leak = np.concatenate([np.zeros((self.shift,1)), auxiliary_data['target'][:-self.shift]])
#         assert dummy_leak.shape[0] == data_segment.shape[0]

#         # Features shift as well, random because 
#         # data_segment = np.concatenate([np.random.random((self.shift, data_segment.shape[1])), data_segment[:-self.shift, :]], axis=0)

#         initial_dates = list(auxiliary_data['initial_dates'])
#         initial_dates_shifted = initial_dates[self.shift:]
#         print(initial_dates[:5])
#         print(initial_dates_shifted[:5])
        
#         data_segment = np.concatenate([data_segment, dummy_leak], axis=1)

#         # Calculate seasonal statistics and update history
#         seasonal_stats = self._calculate_seasonal_stats_vectorized(dummy_leak, mode)
        
#         # Concatenate all features
#         result = np.concatenate([
#             data_segment,           # Original features
#             dummy_leak,            # Shifted target values
#             *seasonal_stats        # All seasonal statistics
#         ], axis=1)
        
#         print(f"Mode: {mode}, Original data shape: {data_segment.shape}")
#         print(f"Shifted target shape: {dummy_leak.shape}")
#         print(f"Seasonal stats shapes: {[stat.shape for stat in seasonal_stats]}")
#         print(f"Final concatenated shape: {result.shape}")
        
#         return result

        
       

#     def _calculate_seasonal_stats_vectorized(self, shifted_target: np.ndarray, mode: str) -> List[np.ndarray]:
#         """
#         Highly optimized vectorized version of seasonal statistics calculation.
#         """
#         n_samples, n_features = shifted_target.shape
#         seasonal_mean = np.zeros_like(shifted_target)
#         seasonal_std = np.zeros_like(shifted_target)

#         desired_polynom = 0
#         seasonal_trend = np.zeros((shifted_target.shape[0], desired_polynom+2))
        
#         # Pre-compute seasonal positions
#         seasonal_positions = np.arange(n_samples) % self.seq_len
#         #non_zero_mask = ~np.allclose(shifted_target, 0, axis=1)
        
#         # For each unique seasonal position
#         for seasonal_pos in range(self.seq_len):
#             # Find all indices with this seasonal position
#             pos_indices = np.where(seasonal_positions == seasonal_pos)[0]
#             #pos_indices = pos_indices[non_zero_mask[pos_indices]]  # Only non-zero values
            
#             # Get historical values for this seasonal position
#             history = self.seasonal_history[seasonal_pos]
            
#             # For each occurrence of this seasonal position
#             seen_idxs = []
#             for current_idx in pos_indices:
#                 # Get all prior indices with the same seasonal position

#                 # The ones that we have in split
#                 if current_idx >= self.shift:
#                     seen_idxs.append(current_idx)
                
#                 observations = shifted_target[seen_idxs]
                
#                 # if mode != 'train':     
#                     # # Values from history
#                     # observations = np.concatenate([np.array(history[-168:]).reshape(-1,1), observations], axis=0)
        
                
        

#                 seasonal_mean[current_idx] = np.mean(observations, axis=0)
#                 seasonal_std[current_idx] = np.std(observations, axis=0)

#                 eff_observations = observations[:-1]
#                 # Calculate trend (slope of linear regression over time)
#                 if len(eff_observations) > desired_polynom+1:
#                     trends = np.zeros(desired_polynom+1)
#                     # Calculate slope using least squares
#                     trends = np.polyfit(np.arange(len(eff_observations)), eff_observations.flatten(), desired_polynom)
#                     y_pred = np.poly1d(trends)(len(observations))
#                     error = y_pred - observations[-1]
#                     seasonal_trend[current_idx] = np.concatenate([trends, error])

        
#         # Update seasonal history
#         self._update_seasonal_history(shifted_target, mode)
#         # seasonal_trend
#         return [seasonal_mean, seasonal_std]

#     def _update_seasonal_history(self, shifted_target: np.ndarray, mode: str):
#         """
#         Update the seasonal history with new values from the current data segment.
        
#         Args:
#             shifted_target: The shifted target values
#             mode: The transformation mode ('train', 'val', 'test')
#         """
#         # Determine which values to add to history
#         if mode == 'train':
#             # In training, add all non-zero values after the shift period
#             start_idx = 0  # Skip the initial zeros from shifting

#             # # In val/test, we might have lookback data that shouldn't be double-counted
#             # # This depends on your data pipeline structure
#             # start_idx = self.shift
        
#             # Pre-compute seasonal positions
#             n_samples, n_features = shifted_target[start_idx:].shape
#             seasonal_positions = np.arange(n_samples) % self.seq_len
#             for seasonal_pos in range(self.seq_len):
#                 #if not np.allclose(shifted_target[i], 0):  # Skip zero/missing values
#                 pos_indices = np.where(seasonal_positions == seasonal_pos)[0]
#                 self.seasonal_history[seasonal_pos].extend(shifted_target[pos_indices].copy().squeeze().tolist())

#     def get_seasonal_history_summary(self):
#         """
#         Get a summary of the seasonal history for debugging/validation.
#         """
#         summary = {}
#         for pos in range(self.seq_len):
#             summary[pos] = {
#                 'count': len(self.seasonal_history[pos]),
#                 'last_values': self.seasonal_history[pos][-3:] if self.seasonal_history[pos] else []
#             }
#         return summary

#     def reset_history(self):
#         """
#         Reset the seasonal history. Useful for starting fresh with new data.
#         """
#         self.seasonal_history = {i: [] for i in range(self.seq_len)}

#     def __call__(self, data_segment: Any, auxiliary_data: Dict, mode: str) -> Any:
#         return self.transform_data(data_segment, auxiliary_data, mode)
    