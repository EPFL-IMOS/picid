import logging
from typing import Any, Dict, List, Optional, Tuple, override
import numpy as np
import os
import matplotlib.pyplot as plt

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import NoFitPerSegmentMixin
from lmetk.data.data_objects.utils import convert_to_numpy

# Import the sequencer
from lmetk.data.optimization.sequencer import DenseArraySequencer

logger = logging.getLogger(__name__)


class MovingRMSTransform(DenseTransform, NoFitPerSegmentMixin):
    """
    Calculates the Root Mean Square (RMS) of time-domain signals
    using either an expanding or a strided-sliding window.

    This transform supports three modes:
    1.  Expanding Window (if `window_size` is None):
        Calculates the RMS from the start of the signal up to the
        current point. Preserves signal length.
    2.  Tumbling Window (if `window_size` is N and `stride` is None):
        Calculates RMS over non-overlapping windows of size N.
        Reduces signal length.
    3.  Sliding Window (if `window_size` is N and `stride` is S):
        Calculates RMS over windows of size N, sliding by S.
        Reduces signal length.

    Can optionally save a plot of the resulting RMS signals.
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
        apply_to_columns: bool = True,
        save_images: bool = False,
        saving_path: Optional[str] = None,
        unit_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the MovingRMSTransform.

        Args:
            window_size: The number of samples to include in the
                RMS window (N). If None, an expanding (cumulative)
                window is used.
            stride: The hop size to slide the window. If None and
                `window_size` is set, `stride` defaults to `window_size`
                (tumbling, non-overlapping windows).
            apply_to_columns: If True (default), apply to columns.
            save_images: If True, a plot of the RMS signal(s) will be
                saved.
            saving_path: The directory path to save images. Required if
                `save_images` is True.
            unit_key: The key in the `metadata` dict to find the
                unit identifier (e.g., 'unit_id'). Used for the
                plot title and filename.
            **kwargs: Additional arguments for BaseTransform
        """
        super().__init__(**kwargs)

        self.window_size = window_size
        self.stride = stride
        self.apply_to_columns = apply_to_columns
        self.save_images = save_images
        self.saving_path = saving_path
        self.unit_key = unit_key
        self.effective_stride_ = None

        if self.save_images and self.saving_path is None:
            raise ValueError(
                "If 'save_images' is True, 'saving_path' must be provided."
            )

        if self.window_size is not None:
            # Sliding/Tumbling window mode
            if not isinstance(self.window_size, int) or self.window_size < 1:
                raise ValueError("If provided, window_size must be a positive integer.")

            if self.stride is not None and (
                not isinstance(self.stride, int) or self.stride < 1
            ):
                raise ValueError("If provided, stride must be a positive integer.")

            # Default stride to window_size if not given
            self.effective_stride_ = (
                self.stride if self.stride is not None else self.window_size
            )

            logger.debug(
                f"MovingRMSTransform initialized in SLIDING/TUMBLING window mode: "
                f"window_size={self.window_size}, stride={self.effective_stride_}"
            )
        else:
            # Expanding window mode
            if self.stride is not None:
                logger.warning(
                    f"stride ({self.stride}) is ignored when window_size is None."
                )
            logger.debug("MovingRMSTransform initialized in EXPANDING window mode.")

        if not self.apply_to_columns:
            raise NotImplementedError("apply_to_columns=False is not yet implemented.")

    def _validate_input(self, data: np.ndarray, key: str) -> np.ndarray:
        """Validate input data."""
        if data.ndim != 2:
            raise ValueError(
                f"Key '{key}' must be a 2D array (n_rows, n_columns), "
                f"got {data.ndim}D array with shape {data.shape}"
            )
        if np.any(np.isinf(data)):
            raise ValueError(f"Infinite values found in array '{key}'")

        if self.window_size is not None:
            signal_length = data.shape[0] if self.apply_to_columns else data.shape[1]
            if signal_length < self.window_size:
                axis_name = "rows" if self.apply_to_columns else "columns"
                raise ValueError(
                    f"Signal length {signal_length} along {axis_name} is shorter "
                    f"than window_size {self.window_size} for key '{key}'."
                )
        return data

    def _apply_expanding_rms(self, signal: np.ndarray) -> np.ndarray:
        """Calculates the cumulative/expanding RMS."""
        signal_squared = np.square(signal)
        cumulative_sum_sq = np.cumsum(signal_squared)
        counts = np.arange(1, len(signal) + 1)
        cumulative_mean_sq = cumulative_sum_sq / counts
        return np.sqrt(cumulative_mean_sq)

    def _save_plot(self, rms_data: np.ndarray, metadata: Dict):
        """Saves a plot of the RMS data."""
        os.makedirs(self.saving_path, exist_ok=True)

        title = ""
        unit_id_str = "unknown_unit"

        if self.unit_key and metadata and self.unit_key in metadata:
            unit_id = metadata[self.unit_key]
            if isinstance(unit_id, (tuple, list)):
                unit_id_str = "_".join(map(str, unit_id))
            else:
                unit_id_str = str(unit_id).replace(" ", "_")
            title = unit_id_str

        filename = f"{unit_id_str}.png"
        file_path = os.path.join(self.saving_path, filename)

        # Set x-axis label based on mode
        xlabel = "Window Index" if self.window_size is not None else "Sample (n)"

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(rms_data)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("RMS")
        ax.set_title(title)

        if 1 < rms_data.shape[1] <= 10:
            ax.legend([f"Col {i}" for i in range(rms_data.shape[1])])

        try:
            plt.savefig(file_path)
            logger.debug(f"Saved RMS plot to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save RMS plot to {file_path}: {e}")
        finally:
            plt.close(fig)

    @override
    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Apply moving RMS to all signals in the input.
        """
        if not data:
            raise ValueError("No data provided for MovingRMSTransform")

        key = "features" if "features" in data else list(data.keys())[0]
        value = convert_to_numpy(data[key])
        value = self._validate_input(value, key)

        n_rows, n_columns = value.shape

        if self.window_size is None:
            # --- Expanding Window Mode (Preserves shape) ---
            rms_data = np.empty_like(value)
            for sig_idx in range(n_columns):
                signal = value[:, sig_idx]
                rms_data[:, sig_idx] = self._apply_expanding_rms(signal)

        else:
            # --- Sliding/Tumbling Window Mode (Uses Sequencer) ---

            # Initialize the sequencer on the (n_rows, n_columns) data
            seq = DenseArraySequencer(
                array=value,
                seq_len=self.window_size,
                stride=self.effective_stride_,
                label_len=0,  # We don't need a label
                pred_len=0,  # We don't need a prediction
            )

            if len(seq) == 0:
                raise ValueError(
                    f"DenseArraySequencer produced 0 windows. "
                    f"Input shape={value.shape}, window_size={self.window_size}, "
                    f"stride={self.effective_stride_}"
                )

            # Get all windows at once
            batch_idx = np.arange(len(seq))
            # seq_x shape: (n_windows, window_size, n_columns)
            seq_x, _ = seq.sequences_batch(batch_idx)
            seq_x = np.array(seq_x)

            # 1. Square the windows
            squared_windows = np.square(seq_x)

            # 2. Calculate mean along the window_size dimension (axis=1)
            mean_squared = np.mean(squared_windows, axis=1)

            # 3. Take the square root
            rms_data = np.sqrt(mean_squared)
            # Final shape: (n_windows, n_columns)

        # Save image if requested
        if self.save_images:
            self._save_plot(
                rms_data,
                (
                    {self.unit_key: data.get(self.unit_key, "unknown_unit")}
                    if self.unit_key
                    else {}
                ),
            )

        logger.debug(f"MovingRMSTransform result shape: {rms_data.shape}")
        return rms_data

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)

    def get_feature_names(
        self, input_keys: List[str], input_shapes: Dict[str, Tuple[int, int]]
    ) -> List[str]:
        """
        Generate feature names for the moving RMS output.
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

        if self.window_size is None:
            suffix = "_exp_rms"
        else:
            # Use the effective_stride_ for the name
            suffix = f"_rms_w{self.window_size}_s{self.effective_stride_}"

        for sig_idx in range(n_signals):
            feature_names.append(f"{key}_{axis_name}{sig_idx}{suffix}")

        return feature_names
