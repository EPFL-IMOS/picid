import logging
from typing import Any, Dict, List, Optional, Tuple, override
import numpy as np
from scipy.signal.windows import get_window
from scipy.signal import ShortTimeFFT
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin
from lmetk.data.data_objects.utils import convert_to_numpy

logger = logging.getLogger(__name__)


class STFTTransform(DenseTransform, ConcatinatedFitMultiSourceTransformMixin):
    def __init__(
        self,
        win_len: int = 256,
        hop: int = None,
        fs: float = 1.0,
        fft_mode: str = "onesided",
        mfft: Optional[int] = None,
        dual_win: Optional[np.ndarray] = None,
        scale_to: Optional[str] = None,
        phase_shift: int = 0,
        window_type: str = "hann",
        output_format: str = "magnitude",
        apply_to_columns: bool = True,
        **kwargs,
    ):
        """Initialize STFT transform using scipy's ShortTimeFFT.

        Args:
            win_len: Window length in samples
            hop: Window shift increment. Defaults to win_len // 4 if None
            fs: Sampling frequency in Hz
            fft_mode: FFT mode ('onesided', 'twosided', 'centered', 'onesided2X')
            mfft: FFT length for zero-padding. Uses win_len if None
            dual_win: Dual window array. Auto-calculated if None
            scale_to: Scaling mode (None, 'magnitude', 'psd')
            phase_shift: Linear phase shift per frequency bin
            window_type: Window function type ('hann', 'hamming', etc.)
            output_format: Output format ('magnitude', 'phase', 'complex', 'power', 'log_power')
            apply_to_columns: If True, apply STFT to columns (each column is a signal).
                            If False, apply to rows (each row is a signal)
            **kwargs: Additional arguments for BaseTransform
        """
        super().__init__(**kwargs)

        self.win_len = win_len
        self.hop = hop if hop is not None else win_len // 4
        self.fs = fs
        self.fft_mode = fft_mode
        self.mfft = mfft
        self.dual_win = dual_win
        self.scale_to = scale_to
        self.phase_shift = phase_shift
        self.window_type = window_type
        self.output_format = output_format
        self.apply_to_columns = apply_to_columns

        # Validate parameters
        valid_fft_modes = ["onesided", "twosided", "centered", "onesided2X"]
        if self.fft_mode not in valid_fft_modes:
            raise ValueError(
                f"fft_mode must be one of {valid_fft_modes}, got {self.fft_mode}"
            )

        valid_scale_modes = [None, "magnitude", "psd"]
        if self.scale_to not in valid_scale_modes:
            raise ValueError(
                f"scale_to must be one of {valid_scale_modes}, got {self.scale_to}"
            )

        valid_formats = ["magnitude", "phase", "complex", "power", "log_power"]
        if self.output_format not in valid_formats:
            raise ValueError(
                f"output_format must be one of {valid_formats}, got {self.output_format}"
            )

        # Create window
        self.window = get_window(self.window_type, self.win_len)

        # Initialize ShortTimeFFT object
        self.stft = ShortTimeFFT(
            win=self.window,
            hop=self.hop,
            fs=self.fs,
            fft_mode=self.fft_mode,
            mfft=self.mfft,
            dual_win=self.dual_win,
            scale_to=self.scale_to,
            phase_shift=self.phase_shift,
        )

        logger.debug(
            f"STFT initialized: win_len={self.win_len}, hop={self.hop}, "
            f"fs={self.fs}, fft_mode={self.fft_mode}, output_format={self.output_format}, "
            f"apply_to_columns={self.apply_to_columns}"
        )

    def _validate_input(self, data: np.ndarray, key: str) -> np.ndarray:
        """Validate and prepare input data for STFT."""
        # Check dimensions - expect 2D array (n_rows, n_columns)
        if data.ndim != 2:
            raise ValueError(
                f"Key '{key}' must be a 2D array with shape (n_rows, n_columns), "
                f"got {data.ndim}D array with shape {data.shape}"
            )

        n_rows, n_columns = data.shape
        logger.debug(f"Input '{key}' shape: ({n_rows}, {n_columns})")

        # Check for NaN values
        if np.any(np.isnan(data)):
            raise ValueError(f"NaN values found in array '{key}'")

        # Check for infinite values
        if np.any(np.isinf(data)):
            raise ValueError(f"Infinite values found in array '{key}'")

        # Check minimum length against window size
        signal_length = n_rows if self.apply_to_columns else n_columns
        if signal_length < self.win_len:
            axis_name = "rows" if self.apply_to_columns else "columns"
            raise ValueError(
                f"Signal length {signal_length} along {axis_name} is shorter than "
                f"window length {self.win_len} for key '{key}'"
            )

        return data

    def _apply_stft_to_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply STFT to a 1D signal."""
        # Compute STFT - returns complex-valued matrix
        stft_result = self.stft.stft(signal)

        # Apply output format transformation
        if self.output_format == "magnitude":
            result = np.abs(stft_result)
        elif self.output_format == "phase":
            result = np.angle(stft_result)
        elif self.output_format == "complex":
            # For complex output, flatten real and imaginary parts
            result = np.stack([stft_result.real, stft_result.imag], axis=0)
        elif self.output_format == "power":
            result = np.abs(stft_result) ** 2
        elif self.output_format == "log_power":
            power = np.abs(stft_result) ** 2
            result = np.log10(power + 1e-10)  # Add small epsilon to avoid log(0)

        return result

    def _reshape_for_concatenation(
        self, stft_result: np.ndarray, key: str, signal_idx: int
    ) -> np.ndarray:
        """Reshape STFT result for concatenation."""
        if self.output_format == "complex":
            # Handle complex output: (2, n_freq_bins, n_time_frames) -> (n_time_frames, 2 * n_freq_bins)
            real_part, imag_part = stft_result[0], stft_result[1]
            reshaped = np.stack([real_part, imag_part], axis=0)  # (2, freq, time)
            reshaped = reshaped.transpose(2, 0, 1)  # (time, 2, freq)
            reshaped = reshaped.reshape(reshaped.shape[0], -1)  # (time, 2*freq)
        else:
            # Standard case: (n_freq_bins, n_time_frames) -> (n_time_frames, n_freq_bins)
            reshaped = stft_result.T

        axis_name = "column" if self.apply_to_columns else "row"
        logger.debug(
            f"STFT for '{key}' {axis_name} {signal_idx}: {stft_result.shape} -> {reshaped.shape}"
        )
        return reshaped

    @override
    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """Apply STFT to signals and concatenate results.

        Args:
            data: Dictionary containing 2D numpy arrays (n_rows, n_columns)
            metadata: Dictionary containing auxiliary metadata (unused in this transform)

        Returns:
            Concatenated STFT results as numpy array with shape (n_time_frames, total_features)
        """

        if not data:
            raise ValueError("No data provided for STFT transform")

        key, value = "features", data["features"]
        # Convert to numpy array
        value = convert_to_numpy(value)

        # Validate additional STFFT
        value = self._validate_input(value, key)

        _, n_columns = value.shape

        # Apply STFT based on chosen axis
        stft_results_for_key = []

        # Apply STFT to each column (each column is a signal)
        for col_idx in range(n_columns):
            signal = value[:, col_idx]  # Extract 1D signal from column

            # Apply STFT to this column
            stft_result = self._apply_stft_to_signal(signal)

            # Reshape for concatenation
            reshaped_result = self._reshape_for_concatenation(stft_result, key, col_idx)
            stft_results_for_key.append(reshaped_result)

        # Concatenate all signals for this key along feature dimension
        if len(stft_results_for_key) > 1:
            final_result = np.concatenate(stft_results_for_key, axis=1)
        else:
            final_result = stft_results_for_key[0]

        axis_name = "columns"
        logger.debug(
            f"Combined STFT result for key '{key}' ({axis_name}): {final_result.shape}"
        )
        logger.debug(f"Final concatenated STFT shape: {final_result.shape}")

        if "target" not in data.keys():
            return final_result
        # TODO: DO WE STILL NEED THIS PART? It is responsigble for handling the targets alignment
        # metadata
        out_dict = {"features": final_result}

        target = data["target"]
        # Convert to numpy array
        target = convert_to_numpy(target, key)

        logger.debug(f"Added data '{key}' with shape {target.shape}")

        # Allign targets
        n_frames = final_result.shape[0]
        centers = np.arange(n_frames) * self.hop  # in samples
        end_idx = np.clip(centers + self.win_len // 2 - 1, 0, len(target) - 1)
        frame_targets = target[end_idx]  # aligned targets per time frame

        # Concatenate and reshape (-1,1)
        target = np.array(frame_targets).reshape(-1, 1)
        logger.debug(f"Final target shape: {target.shape}")

        assert target.shape[0] == n_frames, "Target and result shapes do not match"

        out_dict["target"] = target
        return out_dict

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)

    def get_feature_names(
        self, input_keys: List[str], input_shapes: Dict[str, Tuple[int, int]]
    ) -> List[str]:
        """
        Generate feature names for the STFT output.

        Args:
            input_keys: List of input data keys
            input_shapes: Dictionary mapping keys to their (n_rows, n_columns) shapes

        Returns:
            List of feature names
        """
        feature_names = []

        # Get number of frequency bins from STFT object
        n_freq_bins = self.stft.f_pts

        for key in input_keys:
            n_rows = input_shapes[key][0]

            for row_idx in range(n_rows):
                if self.output_format == "complex":
                    # Complex output has both real and imaginary parts
                    for freq_idx in range(n_freq_bins):
                        feature_names.append(f"{key}_row{row_idx}_freq{freq_idx}_real")
                        feature_names.append(f"{key}_row{row_idx}_freq{freq_idx}_imag")
                else:
                    for freq_idx in range(n_freq_bins):
                        feature_names.append(
                            f"{key}_row{row_idx}_{self.output_format}_freq{freq_idx}"
                        )

        return feature_names

    def get_time_frequency_info(self) -> Dict[str, Any]:
        """
        Get information about the time-frequency grid of the STFT.

        Returns:
            Dictionary containing STFT grid information
        """
        return {
            "n_freq_bins": self.stft.f_pts,
            "freq_bins": self.stft.f,
            "delta_f": self.stft.delta_f,
            "delta_t": self.stft.delta_t,
            "hop_samples": self.hop,
            "window_length": self.win_len,
            "fft_mode": self.fft_mode,
            "sampling_rate": self.fs,
        }
