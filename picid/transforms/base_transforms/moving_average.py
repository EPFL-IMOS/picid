import logging
from typing import Any, Dict, List, Optional, Tuple, override
import numpy as np

from picid.data.data_objects.data import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin
from picid.data.data_objects.utils import convert_to_numpy

# Import the sequencer
from picid.data.optimization.sequencer import DenseArraySequencer

logger = logging.getLogger(__name__)


class MovingAverageTransform(NoFitPerSegmentMixin, DenseTransform):
    """
    Applies a moving average filter to smooth time-domain signals.

    This transform supports two main modes:

    1.  Continuous Mode (`chunk_size=None`):
        Applies a continuous filter over the entire signal.
        - `causal=False` (default): Centered (non-causal) average.
        - `causal=True`: Causal average (output `i` depends only on inputs `<= i`).

    2.  Chunked Mode (`chunk_size` is an int):
        For discontinuous data (e.g., concatenated experiments).
        The signal is split into non-overlapping chunks of `chunk_size`,
        the filter is applied to each chunk independently, and the
        signal is reassembled to its original shape.
    """

    def __init__(
        self,
        window_size: int,
        chunk_size: Optional[int] = None,  # <-- New chunk_size parameter
        apply_to_columns: bool = True,
        causal: bool = False,
        **kwargs,
    ):
        """
        Initialize the MovingAverageTransform.

        Args:
            window_size: The number of samples for the moving average
                filter window.
            chunk_size: The size of each independent, discontinuous
                chunk of data. If None (default), the signal is
                treated as one continuous block.
            apply_to_columns: If True (default), apply to columns.
            causal: If True and `chunk_size=None`, applies a causal filter.
                If `chunk_size` is set, the filter is always applied
                causally *within* the chunk (using mode='full' and slicing).
            **kwargs: Additional arguments for BaseTransform.
        """
        super().__init__(**kwargs)

        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer.")

        if chunk_size is not None and (
            not isinstance(chunk_size, int) or chunk_size < 1
        ):
            raise ValueError("If provided, chunk_size must be a positive integer.")

        self.window_size = window_size
        self.chunk_size = chunk_size
        self.apply_to_columns = apply_to_columns
        self.causal = causal

        # Create the averaging window
        self.window = np.ones(self.window_size) / self.window_size

        if not self.apply_to_columns:
            raise NotImplementedError("apply_to_columns=False is not yet implemented.")

        if self.chunk_size is None:
            logger.debug(
                f"MovingAverageTransform initialized in CONTINUOUS mode: "
                f"window_size={self.window_size}, causal={self.causal}"
            )
        else:
            logger.debug(
                f"MovingAverageTransform initialized in CHUNKED mode: "
                f"window_size={self.window_size}, chunk_size={self.chunk_size}"
            )
            if self.causal:
                logger.info(
                    "`causal=True` is noted, but in CHUNKED mode, the "
                    "filter is always applied *within* the chunk boundaries."
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

        signal_length = data.shape[0]

        if self.chunk_size is not None and signal_length % self.chunk_size != 0:
            raise ValueError(
                f"Signal length ({signal_length}) is not perfectly divisible "
                f"by chunk_size ({self.chunk_size}). This is required "
                "for chunked filtering."
            )

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
        """
        if not data:
            raise ValueError("No data provided for MovingAverageTransform")

        assert len(metadata["apply_to_keys"]) == 1, (
            "MovingAverageTransform expects exactly one key in metadata['apply_to_keys'], "
            f"but got {len(metadata['apply_to_keys'])} keys: {metadata['apply_to_keys']}"
        )
        apply_to_key = metadata["apply_to_keys"][0]
        key, value = apply_to_key, data[apply_to_key]

        value = convert_to_numpy(value)
        value = self._validate_input(value, key)

        n_rows, n_columns = value.shape

        if self.chunk_size is not None:
            # --- CHUNKED MODE (for discontinuous data) ---

            # 1. Use Sequencer to slice into non-overlapping chunks
            seq = DenseArraySequencer(
                array=value,
                seq_len=self.chunk_size,
                stride=self.chunk_size,  # Non-overlapping
                label_len=0,
                pred_len=0,
            )

            if len(seq) == 0:
                raise ValueError(
                    f"DenseArraySequencer produced 0 chunks. "
                    f"Input shape={value.shape}, chunk_size={self.chunk_size}"
                )

            # Get all windows: shape (n_windows, chunk_size, n_columns)
            batch_idx = np.arange(len(seq))
            seq_x, _ = seq.sequences_batch(batch_idx)

            # 2. Apply convolve to each chunk independently
            smoothed_seq_x = np.empty_like(seq_x)
            n_windows = seq_x.shape[0]

            for i in range(n_windows):
                for j in range(n_columns):
                    chunk_signal = seq_x[i, :, j]

                    if self.causal:
                        full_conv = np.convolve(chunk_signal, self.window, mode="full")
                        smoothed_chunk = full_conv[: self.chunk_size]
                    else:
                        smoothed_chunk = np.convolve(
                            chunk_signal, self.window, mode="same"
                        )

                    smoothed_seq_x[i, :, j] = smoothed_chunk

            # 3. Reshape back to original signal shape
            smoothed_data = smoothed_seq_x.reshape(n_rows, n_columns)

        else:
            # --- CONTINUOUS MODE (for continuous data) ---
            smoothed_data = np.empty_like(value)

            for sig_idx in range(n_columns):
                signal = value[:, sig_idx]  # Extract 1D signal

                if self.causal:
                    # Causal: output[i] depends on input[<=i]
                    full_conv = np.convolve(signal, self.window, mode="full")
                    smoothed_data[:, sig_idx] = full_conv[:n_rows]
                else:
                    # Non-Causal: output[i] is centered on input[i]
                    smoothed_data[:, sig_idx] = np.convolve(
                        signal, self.window, mode="same"
                    )

        logger.debug(f"MovingAverageTransform result shape: {smoothed_data.shape}")
        return smoothed_data

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)

    def get_feature_names(
        self, input_keys: List[str], input_shapes: Dict[str, Tuple[int, int]]
    ) -> List[str]:
        """
        Generate feature names for the smoothed output.
        """
        feature_names = []
        if not input_keys:
            return []

        key = input_keys[0]
        if key not in input_shapes:
            raise KeyError(f"Input shape for key '{key}' not provided.")

        _n_rows, n_columns = input_shapes[key]
        n_signals = n_columns
        axis_name = "col"

        # Suffix changes based on mode
        suffix = f"_mov_avg{self.window_size}"
        if self.chunk_size is not None:
            suffix += f"_chunked{self.chunk_size}"
        elif self.causal:
            suffix += "_causal"

        for sig_idx in range(n_signals):
            feature_names.append(f"{key}_{axis_name}{sig_idx}{suffix}")

        return feature_names
