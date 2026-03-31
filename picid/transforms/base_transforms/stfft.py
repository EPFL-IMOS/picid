import logging
from typing import Any, Dict, List, Optional, Tuple, override
import numpy as np
from scipy.signal.windows import get_window
from scipy.signal import ShortTimeFFT
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin
from picid.data.data_objects.utils import convert_to_numpy

logger = logging.getLogger(__name__)


class STFTTransform(NoFitPerSegmentMixin, DenseTransform):
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
        subbands: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ):
        """
        Initialize the STFT transform.

        Parameters
        ----------
        win_len : int
            Window length in samples.
        hop : int, optional
            Window shift increment. Defaults to ``win_len // 4`` when ``None``.
        fs : float
            Sampling frequency in Hz.
        fft_mode : str
            FFT mode. Supported values are ``'onesided'``, ``'twosided'``,
            ``'centered'``, and ``'onesided2X'``.
        mfft : int, optional
            FFT length for zero-padding. Uses ``win_len`` when ``None``.
        dual_win : np.ndarray, optional
            Dual window array. Auto-calculated when ``None``.
        scale_to : str, optional
            Scaling mode. Supported values are ``None``, ``'magnitude'``, and
            ``'psd'``.
        phase_shift : int
            Linear phase shift per frequency bin.
        window_type : str
            Window function type such as ``'hann'`` or ``'hamming'``.
        output_format : str
            Output format. Supported values are ``'magnitude'``, ``'phase'``,
            ``'complex'``, ``'power'``, and ``'log_power'``.
        apply_to_columns : bool
            If True, apply STFT to columns (each column is a signal).
            If False, apply to rows (each row is a signal).
        subbands : list of tuple of float, optional
            Frequency bands to extract from the spectrum, expressed as
            ``(low_freq, high_freq)`` pairs in Hz.
        **kwargs
            Additional arguments for :class:`BaseTransform`.
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
        self.subbands = subbands  # list of (low_freq, high_freq) in Hz

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
        """
        Validate and prepare input data for STFT.

        Parameters
        ----------
        data : np.ndarray
            Dense input array to transform.
        key : str
            Logical data key used for error messages.

        Returns
        -------
        np.ndarray
            The validated input array.
        """
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
        """
        Apply STFT to a 1D signal.

        Parameters
        ----------
        signal : np.ndarray
            One-dimensional signal to transform.

        Returns
        -------
        np.ndarray
            STFT representation of the signal.
        """
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

        # result at this point is frequencies x time_points
        # Apply subband filtering if specified
        if self.subbands is not None:
            freqs = self.stft.f
            band_results = []
            for low, high in self.subbands:
                mask = (freqs >= low) & (freqs <= high)
                if np.any(mask):
                    band_data = result[mask, :]
                    freq_band = freqs[mask]
                    band_features = self.frequency_features_np(
                        band_data, freq_band, as_list=True
                    )
                    band_results.append(np.array(band_features))
                else:
                    logger.warning(f"No frequencies found in range {low}-{high} Hz")
            if band_results:
                result = np.concatenate(band_results, axis=0)
            else:
                logger.warning("No valid subbands selected, returning empty array.")
                result = np.zeros_like(result)

        return result

    @staticmethod
    def frequency_features_np(S, freqs, as_list: bool = False) -> Dict[str, np.ndarray]:
        """
        Compute frequency-domain statistical features F12–F24 using NumPy.

        Parameters
        ----------
        S : np.ndarray
            Magnitude spectrum with shape ``(K, T)``.
        freqs : np.ndarray
            Frequency values with shape ``(K,)``.
        as_list : bool, default=False
            Return the features as a list instead of a dictionary.

        Returns
        -------
        dict or list of np.ndarray
            Feature arrays with shape ``(T,)``.
        """
        K, T = S.shape
        eps = 1e-12
        s_sum = np.sum(S, axis=0) + eps

        # F12: spectral mean
        F12 = np.mean(S, axis=0)

        # F13: RMS of spectrum
        F13 = np.sqrt(np.mean((S - F12) ** 2, axis=0))

        # F14: frequency statistic value 3
        F14 = np.sum((S - F12) ** 3, axis=0) / ((K - 1) * (F13**3 + eps))

        # F15: frequency statistic value 4
        F15 = np.sum((S - F12) ** 4, axis=0) / ((K - 1) * (F13**4 + eps))

        # F16: frequency centre of gravity
        F16 = np.sum(freqs[:, None] * S, axis=0) / s_sum

        # F17: frequency statistic value 6 (std around F16)
        F17 = np.sqrt(np.sum(((freqs[:, None] - F16) ** 2) * S, axis=0) / (K - 1))

        # F18: root mean square frequency
        F18 = np.sqrt(np.sum((freqs[:, None] ** 2) * S, axis=0) / s_sum)

        # F19: frequency statistic value 8
        num = np.sum((freqs[:, None] ** 4) * S, axis=0)
        den = np.sum((freqs[:, None] ** 2) * S, axis=0)
        F19 = np.sqrt(num / (den + eps))

        # F20: frequency statistic value 9
        num = np.sum((freqs[:, None] ** 2) * S, axis=0)
        den = np.sqrt(np.sum(S, axis=0) * np.sum((freqs[:, None] ** 4) * S, axis=0))
        F20 = num / (den + eps)

        # F21: frequency statistic value 10
        F21 = F17 / (F16 + eps)

        # F22: frequency statistic value 11
        F22 = np.sum(((freqs[:, None] - F16) ** 3) * S, axis=0) / (
            (K - 1) * (F17**3 + eps)
        )

        # F23: frequency statistic value 12
        F23 = np.sum(((freqs[:, None] - F16) ** 4) * S, axis=0) / (
            (K - 1) * (F17**4 + eps)
        )

        # There seems to be an error in the original definition of F24.
        # F24: standard deviation frequency (corrected exponent 1/2)
        # F24 = np.sum(((freqs[:, None] - F16) ** 0.5) * S, axis=0) / (
        #     (K - 1) * (F17**0.5 + eps)
        # )

        if as_list:
            return [
                F12,
                F13,
                F14,
                F15,
                F16,
                F17,
                F18,
                F19,
                F20,
                F21,
                F22,
                F23,
                # F24,
            ]

        else:
            return {
                "F12": F12,
                "F13": F13,
                "F14": F14,
                "F15": F15,
                "F16": F16,
                "F17": F17,
                "F18": F18,
                "F19": F19,
                "F20": F20,
                "F21": F21,
                "F22": F22,
                "F23": F23,
                # "F24": F24,
            }

    def _reshape_for_concatenation(
        self, stft_result: np.ndarray, key: str, signal_idx: int
    ) -> np.ndarray:
        """
        Reshape STFT result for concatenation.

        Parameters
        ----------
        stft_result : np.ndarray
            Raw STFT output for a single signal.
        key : str
            Logical data key used for logging.
        signal_idx : int
            Index of the signal within the input segment.

        Returns
        -------
        np.ndarray
            STFT output reshaped to a concatenation-friendly layout.
        """
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
        """
        Apply STFT to signals and concatenate results.

        Parameters
        ----------
        data : dict
            Dictionary containing 2D NumPy arrays with shape ``(n_rows, n_columns)``.
        metadata : dict
            Dictionary containing auxiliary metadata (unused in this transform).

        Returns
        -------
        np.ndarray
            Concatenated STFT results with shape ``(n_time_frames, total_features)``.
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

        return final_result

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)

    def get_feature_names(
        self, input_keys: List[str], input_shapes: Dict[str, Tuple[int, int]]
    ) -> List[str]:
        """
        Generate feature names for the STFT output.

        Parameters
        ----------
        input_keys : list
            List of input data keys.
        input_shapes : dict
            Dictionary mapping keys to their ``(n_rows, n_columns)`` shapes.

        Returns
        -------
        list
            List of feature names.
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

        Returns
        -------
        dict
            Dictionary containing STFT grid information.
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
