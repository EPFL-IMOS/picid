import logging
from typing import Any, Dict, List, Optional, Tuple, override
import numpy as np

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
from lmetk.data.data_objects.utils import convert_to_numpy

logger = logging.getLogger(__name__)


class MovingAverageTransform(DenseTransform, NoFitMultiSourceTransformMixin):
    """
    Applies a moving average (convolution) filter to smooth time-domain signals.

    This transform is a preprocessor and preserves the shape of the
    input data (n_rows, n_columns). It uses 'same' mode convolution
    to ensure the signal length is unchanged, handling edges with
    zero-padding implicitly.
    """

    def __init__(
        self,
        window_size: int,
        apply_to_columns: bool = True,
        **kwargs,
    ):
        """
        Initialize the MovingAverageTransform.

        Args:
            window_size: The number of samples to include in the
                moving average window.
            apply_to_columns: If True (default), apply to columns (each column
                is a signal). If False, apply to rows (each row is a signal).
            **kwargs: Additional arguments for BaseTransform.
        """
        super().__init__(**kwargs)

        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer.")

        self.window_size = window_size
        self.apply_to_columns = apply_to_columns

        if not self.apply_to_columns:
            raise NotImplementedError("apply_to_columns=False is not yet implemented.")

        # Create the averaging window
        self.window = np.ones(self.window_size) / self.window_size

        logger.debug(
            f"MovingAverageTransform initialized: window_size={self.window_size}, "
            f"apply_to_columns={self.apply_to_columns}"
        )

    def _validate_input(self, data: np.ndarray, key: str) -> np.ndarray:
        """Validate input data."""
        if data.ndim != 2:
            raise ValueError(
                f"Key '{key}' must be a 2D array (n_rows, n_columns), "
                f"got {data.ndim}D array with shape {data.shape}"
            )
        if np.any(np.isinf(data)):
            raise ValueError(f"Infinite values found in array '{key}'")

        signal_length = data.shape[0] if self.apply_to_columns else data.shape[1]
        if signal_length < self.window_size:
            axis_name = "rows" if self.apply_to_columns else "columns"
            logger.warning(
                f"Signal length {signal_length} along {axis_name} is shorter "
                f"than window_size {self.window_size} for key '{key}'. "
                "Edge effects will be significant."
            )
        return data

    @override
    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Apply moving average smoothing to all signals in the input.

        Args:
            data: Dictionary containing 2D numpy arrays (n_rows, n_columns)
            metadata: Dictionary containing auxiliary metadata (unused)

        Returns:
            A 2D numpy array of the same shape (n_rows, n_columns) containing
            the smoothed signals.
        """
        if not data:
            raise ValueError("No data provided for MovingAverageTransform")

        key, value = "features", data["features"]
        value = convert_to_numpy(value)
        value = self._validate_input(value, key)

        n_rows, n_columns = value.shape
        n_signals = n_columns  # Assumes apply_to_columns=True

        # Create an output array of the same shape
        smoothed_data = np.empty_like(value)

        # Iterate over each signal (column)
        for sig_idx in range(n_signals):
            signal = value[:, sig_idx]  # Extract 1D signal

            # Apply convolution
            # 'same' mode preserves the signal length (n_rows)
            # NaNs in the signal will propagate (which is correct behavior)
            smoothed_signal = np.convolve(signal, self.window, mode="same")

            smoothed_data[:, sig_idx] = smoothed_signal

        logger.debug(f"MovingAverageTransform result shape: {smoothed_data.shape}")
        return smoothed_data

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)

    def get_feature_names(
        self, input_keys: List[str], input_shapes: Dict[str, Tuple[int, int]]
    ) -> List[str]:
        """
        Generate feature names for the smoothed output.

        Since this transform preserves the number of features, it
        returns a list of names corresponding to the input columns,
        with a suffix.
        """
        feature_names = []
        if not input_keys:
            return []

        key = input_keys[0]
        if key not in input_shapes:
            raise KeyError(f"Input shape for key '{key}' not provided.")

        _n_rows, n_columns = input_shapes[key]
        n_signals = n_columns  # Assumes apply_to_columns=True
        axis_name = "col"

        # Generate one output feature name for each input feature (column)
        for sig_idx in range(n_signals):
            feature_names.append(
                f"{key}_{axis_name}{sig_idx}_mov_avg{self.window_size}"
            )

        return feature_names
