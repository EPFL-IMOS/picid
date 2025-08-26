import logging
from typing import Any, Dict, List

import numpy as np

from lmetk.transforms.base.base_transform import BaseTransform

# Assuming this import path is correct for your project


logger = logging.getLogger(__name__)


class BatteryTransform(BaseTransform):

    def __init__(self, **kwargs):
        self._init_kwargs = kwargs
        self.shift = kwargs["shift"]
        self.polynom_degree = kwargs.get(
            "polynom_degree", 4
        )  # Degree of polynomial for trend fitting
        self.lookback_window = kwargs.get(
            "lookback_window", 100
        )  # How far back to look for patterns
        self.rolling_window = kwargs.get(
            "rolling_window", 1000000
        )  # Rolling window for local statistics

        # History for polynomial trend fitting
        self.value_history = []
        self.time_history = []

    def __repr__(self):
        # Ensure 'freq' is included in the representation, along with any other init args
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}(freq={self.freq!r}, {args})"

    def transform_data(self, data_segment: Any, auxiliary_data: Dict, mode: str) -> Any:

        # assert self.shift in [24, 168, 100], "dummy shift must be 24 for the timepfn, and 168 for the tabpfn for railway and 100 for the battery!"

        # Features of the target shifted
        dummy_leak = np.concatenate(
            [np.zeros((self.shift, 1)), auxiliary_data["target"][: -self.shift]]
        )
        assert dummy_leak.shape[0] == data_segment.shape[0]

        initial_dates = list(auxiliary_data["initial_dates"])
        initial_dates_shifted = initial_dates[self.shift :]
        logger.info(initial_dates[:5])
        logger.info(initial_dates_shifted[:5])

        # Concatenate original features with shifted target
        data_segment = np.concatenate([data_segment, dummy_leak], axis=1)

        # Calculate polynomial-based seasonal statistics and update history
        # seasonal_stats = self._calculate_polynomial_stats_vectorized(dummy_leak, mode)

        # Concatenate all features
        result = np.concatenate(
            [
                data_segment,  # Original features + shifted target
                # *seasonal_stats         # All polynomial-based statistics
            ],
            axis=1,
        )

        logger.info(f"Mode: {mode}, Original data shape: {data_segment.shape}")
        logger.info(f"Shifted target shape: {dummy_leak.shape}")
        # logger.info(f"Polynomial stats shapes: {[stat.shape for stat in seasonal_stats]}")
        logger.info(f"Final concatenated shape: {result.shape}")

        return result

    def _calculate_polynomial_stats_vectorized(
        self, shifted_target: np.ndarray, mode: str
    ) -> List[np.ndarray]:
        """
        Calculate polynomial-based statistics to capture seasonal patterns without explicit seasonality.
        Uses polynomial fitting on historical data to derive trend and pattern features.
        """
        n_samples, n_features = shifted_target.shape

        # Initialize output arrays
        polynomial_trend = np.zeros(
            (n_samples, self.polynom_degree + 2)
        )  # coefficients + prediction + error
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
            available_history = shifted_target[
                self.shift : i + 1
            ]  # Exclude the zero-padded portion

            if len(available_history) >= 2:
                # Calculate rolling statistics with available data
                window_start = max(0, len(available_history) - self.rolling_window)
                window_data = available_history[window_start:]

                rolling_mean[i] = np.mean(window_data, axis=0)
                rolling_std[i] = (
                    np.std(window_data, axis=0) if len(window_data) > 1 else 0
                )

                # Calculate local trend (slope over recent window)
                if len(window_data) > 1:
                    x_local = np.arange(len(window_data))
                    local_slope = np.polyfit(x_local, window_data.flatten(), 1)[0]
                    local_trend[i] = local_slope

                # Polynomial trend fitting on longer history
                if len(available_history) > self.polynom_degree + 1:
                    # Use lookback window or all available data, whichever is smaller
                    lookback_start = max(
                        0, len(available_history) - self.lookback_window
                    )
                    trend_data = available_history[lookback_start:]

                    try:
                        # Fit polynomial to historical data
                        x_trend = np.arange(len(trend_data))
                        poly_coeffs = np.polyfit(
                            x_trend, trend_data.flatten(), self.polynom_degree
                        )

                        # Predict next value using the polynomial
                        poly_func = np.poly1d(poly_coeffs)
                        predicted_value = poly_func(len(trend_data))

                        # Calculate prediction error
                        actual_value = (
                            current_value[0]
                            if current_value.ndim > 0
                            else current_value
                        )
                        prediction_error = predicted_value - actual_value

                        # Store polynomial coefficients, prediction, and error
                        polynomial_trend[i, : len(poly_coeffs)] = poly_coeffs
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
            # local_trend,
            # polynomial_trend
        ]

    def _update_value_history(self, shifted_target: np.ndarray, mode: str):
        """
        Update the value history with new observations for future polynomial fitting.
        """

        # Add non-zero values to history (skip the initial shift period)
        valid_values = shifted_target[self.shift :]
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
            "history_length": len(self.value_history),
            "recent_values": self.value_history[-10:] if self.value_history else [],
            "polynom_degree": self.polynom_degree,
            "lookback_window": self.lookback_window,
            "rolling_window": self.rolling_window,
        }

    def reset_history(self):
        """
        Reset the value history. Useful for starting fresh with new data.
        """
        self.value_history = []
        self.time_history = []

    def __call__(self, data_segment: Any, auxiliary_data: Dict, mode: str) -> Any:
        return self.transform_data(data_segment, auxiliary_data, mode)
