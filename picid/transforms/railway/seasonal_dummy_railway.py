import logging
from typing import Any, Dict, List

import numpy as np

from picid.data.data_objects.data import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin

logger = logging.getLogger(__name__)


class SeasonalDummyTransform(NoFitPerSegmentMixin, DenseTransform):
    def __init__(self, shift: int, seq_len: int = 168, **kwargs):
        """
        Initialize the seasonal dummy transform.

        Parameters
        ----------
        shift : int
            Number of time steps to shift the target (e.g., 168 for weekly)
        seq_len : int
            Sequence length that defines the seasonal period
        **kwargs
            Additional keyword arguments
        """
        self._init_kwargs = kwargs
        self.shift = shift
        self.seq_len = seq_len

        # Storage for seasonal history - maps seasonal_pos to list of all historical values
        self.seasonal_history = {i: [] for i in range(self.seq_len)}

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}(shift={self.shift!r}, seq_len={self.seq_len!r}, {args})"

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> Any:
        """
        Transform data by adding shifted target values and seasonal statistics.
        All seasonal history management happens here.

        Parameters
        ----------
        data_segment : dict
            The main data segment (dict)

        Returns
        -------
        np.ndarray
            Concatenated data with shifted targets and seasonal statistics
        """
        data_segment = data

        # Validate inputs
        if "target" not in data_segment:
            raise ValueError(
                "data_segment must contain 'target' key to make the hist. features."
            )
        target = data_segment["target"]

        if "features" not in data_segment:
            raise ValueError(
                "data_segment must contain 'features' key to make the hist. features."
            )
        features = data_segment["features"]

        if not isinstance(target, np.ndarray):
            target = np.array(target)

        # Ensure target is 2D
        if target.ndim == 1:
            target = target.reshape(-1, 1)

        # Ensure features is 2D
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        # Validate dimensions
        if target.shape[0] != features.shape[0]:
            raise ValueError(
                f"Target length {target.shape[0]} doesn't match features length {features.shape[0]}"
            )

        # Create shifted target (dummy leak)
        dummy_leak = np.concatenate(
            [np.zeros((self.shift, target.shape[1])), target[: -self.shift]]
        )

        # Calculate seasonal statistics and update history
        seasonal_stats = self._calculate_seasonal_stats_vectorized(
            dummy_leak, metadata["mode"]
        )

        # Concatenate all features
        result = np.concatenate(
            [
                features,  # Original features
                dummy_leak,  # Shifted target values
                *seasonal_stats,  # All seasonal statistics
            ],
            axis=1,
        )

        logger.info(f"Original data shape: {features.shape}")
        logger.info(f"Shifted target shape: {dummy_leak.shape}")
        logger.info(f"Seasonal stats shapes: {[stat.shape for stat in seasonal_stats]}")
        logger.info(f"Final concatenated shape: {result.shape}")

        return result

    def _calculate_seasonal_stats_vectorized(
        self, shifted_target: np.ndarray, mode: str
    ) -> List[np.ndarray]:
        """
        Highly optimized vectorized version of seasonal statistics calculation.
        """
        n_samples, n_features = shifted_target.shape
        seasonal_mean = np.zeros_like(shifted_target)
        seasonal_std = np.zeros_like(shifted_target)

        desired_polynom = 0
        seasonal_trend = np.zeros((shifted_target.shape[0], desired_polynom + 2))

        # Pre-compute seasonal positions
        seasonal_positions = np.arange(n_samples) % self.seq_len
        # non_zero_mask = ~np.allclose(shifted_target, 0, axis=1)

        # For each unique seasonal position
        for seasonal_pos in range(self.seq_len):
            # Find all indices with this seasonal position
            pos_indices = np.where(seasonal_positions == seasonal_pos)[0]
            # pos_indices = pos_indices[non_zero_mask[pos_indices]]  # Only non-zero values

            # Get historical values for this seasonal position
            _history = self.seasonal_history[seasonal_pos]

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
                if len(eff_observations) > desired_polynom + 1:
                    trends = np.zeros(desired_polynom + 1)
                    # Calculate slope using least squares
                    trends = np.polyfit(
                        np.arange(len(eff_observations)),
                        eff_observations.flatten(),
                        desired_polynom,
                    )
                    y_pred = np.poly1d(trends)(len(observations))
                    error = y_pred - observations[-1]
                    seasonal_trend[current_idx] = np.concatenate([trends, error])

        # Update seasonal history
        self._update_seasonal_history(shifted_target, mode)
        # seasonal_trend

        # Check if there are any NaNs in seasonal_mean or seasonal_std, fill with 0 and log a warning
        if np.isnan(seasonal_mean).any() or np.isnan(seasonal_std).any():
            logger.warning(
                f"NaNs detected in seasonal statistics. Filling NaNs with 0. Num. NaNs {np.isnan(seasonal_mean).sum()} for seasonal_mean"
            )
            seasonal_mean = np.nan_to_num(seasonal_mean, nan=0.0)
            seasonal_std = np.nan_to_num(seasonal_std, nan=0.0)

        return [seasonal_mean, seasonal_std]

    def _update_seasonal_history(self, shifted_target: np.ndarray, mode: str):
        """
        Update the seasonal history with new values from the current data segment.

        Parameters
        ----------
        shifted_target : np.ndarray
            The shifted target values
        mode : str
            The transformation mode ('train', 'val', 'test')
        """
        # Determine which values to add to history
        if mode == "train":
            # In training, add all non-zero values after the shift period
            start_idx = 0  # Skip the initial zeros from shifting

            # # In val/test, we might have lookback data that shouldn't be double-counted
            # # This depends on your data pipeline structure
            # start_idx = self.shift

            # Pre-compute seasonal positions
            n_samples, n_features = shifted_target[start_idx:].shape
            seasonal_positions = np.arange(n_samples) % self.seq_len
            for seasonal_pos in range(self.seq_len):
                # if not np.allclose(shifted_target[i], 0):  # Skip zero/missing values
                pos_indices = np.where(seasonal_positions == seasonal_pos)[0]
                self.seasonal_history[seasonal_pos].extend(
                    shifted_target[pos_indices].copy().squeeze().tolist()
                )

    def get_seasonal_history_summary(self):
        """
        Get a summary of the seasonal history for debugging/validation.
        """
        summary = {}
        for pos in range(self.seq_len):
            summary[pos] = {
                "count": len(self.seasonal_history[pos]),
                "last_values": (
                    self.seasonal_history[pos][-3:]
                    if self.seasonal_history[pos]
                    else []
                ),
            }
        return summary

    def reset_history(self):
        """
        Reset the seasonal history. Useful for starting fresh with new data.
        """
        self.seasonal_history = {i: [] for i in range(self.seq_len)}

    def __call__(self, data_segment: Any, metadata: Dict) -> Any:
        return self.transform_data(data_segment, metadata=metadata)
