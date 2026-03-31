import logging
from typing import Dict, List, Tuple, override
import numpy as np
from numpy.fft import fft
from scipy import stats
from scipy.stats import entropy

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin
from picid.data.data_objects.utils import convert_to_numpy

logger = logging.getLogger(__name__)


class SpectralStatsTransform(NoFitPerSegmentMixin, DenseTransform):
    """
    Compute global statistical features from the frequency spectrum of signals.

    The transform applies an FFT to each signal and then computes scalar
    statistics from the resulting amplitude spectrum.

    Parameters
    ----------
    stats_to_compute : list[str]
        Statistic names to compute. Each value must be present in
        :attr:`VALID_STATS`.
    fs : float, default=1.0
        Sampling frequency in Hz used for the spectrum calculation.
    apply_to_columns : bool, default=True
        Whether to treat columns as signals. The row-wise mode is not
        implemented.
    pe_dim : int, default=5
        Embedding dimension used for permutation entropy.
    pe_tau : int, default=10
        Delay used for permutation entropy.
    **kwargs
        Extra compatibility keywords passed to the parent transform.
    """

    # Define the set of all valid statistical function names
    VALID_STATS = {
        # Basic stats
        "mean",
        "maximum",
        "minimum",
        "root_mean_square",
        "peak_to_peak_value",
        "variance",
        "skewness",
        "kurtosis",
        "abs_energy",
        # rul_features inspired
        "peak_factor",
        "change_coefficient",
        "clearance_factor",
        # Entropy stats
        "spectral_entropy",
        "shannon_entropy",  # Will be treated as spectral_entropy
        "permutation_entropy",
    }

    def __init__(
        self,
        stats_to_compute: List[str],
        fs: float = 1.0,
        apply_to_columns: bool = True,
        pe_dim: int = 5,
        pe_tau: int = 10,
        **kwargs,
    ):
        """
        Initialize the spectral-statistics transform.

        Parameters
        ----------
        stats_to_compute : list[str]
            Statistic names to compute.
        fs : float, default=1.0
            Sampling frequency in Hz used for the spectrum calculation.
        apply_to_columns : bool, default=True
            Whether to treat columns as signals.
        pe_dim : int, default=5
            Embedding dimension used for permutation entropy.
        pe_tau : int, default=10
            Delay used for permutation entropy.
        **kwargs
            Extra compatibility keywords passed to the parent transform.
        """
        super().__init__(**kwargs)
        self.fs = fs
        self.apply_to_columns = apply_to_columns
        self.pe_dim = pe_dim
        self.pe_tau = pe_tau

        # Validate the requested statistics
        self.stats_to_compute = stats_to_compute
        for stat_name in self.stats_to_compute:
            if stat_name not in self.VALID_STATS:
                raise ValueError(
                    f"Unknown statistic: '{stat_name}'. "
                    f"Valid stats are: {self.VALID_STATS}"
                )

        if not self.apply_to_columns:
            raise NotImplementedError("apply_to_columns=False is not yet implemented.")

        logger.debug(
            f"SpectralStatsTransform initialized: stats={self.stats_to_compute}, "
            f"fs={self.fs}, apply_to_columns={self.apply_to_columns}"
        )

    def _compute_fft_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the single-sided amplitude spectrum of a 1D signal.

        Parameters
        ----------
        signal : numpy.ndarray
            One-dimensional input signal.

        Returns
        -------
        numpy.ndarray
            Single-sided amplitude spectrum.
        """
        n = len(signal)
        if n == 0:
            return np.array([])

        frequency_values = fft(signal) / n
        frequency_values = frequency_values[range(int(n / 2))]
        return np.abs(frequency_values)

    def _compute_spectral_entropy(self, spectrum: np.ndarray) -> float:
        """
        Compute the Shannon entropy of the normalized power spectrum.

        Parameters
        ----------
        spectrum : numpy.ndarray
            Amplitude spectrum.

        Returns
        -------
        float
            Spectral entropy in bits.
        """
        # 1. Calculate Power Spectrum (Power = Amplitude^2)
        power_spectrum = spectrum**2

        # 2. Normalize to create a probability distribution
        total_power = np.sum(power_spectrum)
        if total_power < 1e-10:
            return 0.0  # Avoid division by zero for silent signals

        normalized_psd = power_spectrum / total_power

        # 3. Compute Shannon entropy
        # scipy.stats.entropy computes -sum(p * log(p))
        return entropy(normalized_psd, base=2)  # Use base 2 for bits

    def _compute_permutation_entropy(self, x: np.ndarray, dim: int, tau: int) -> float:
        """
        Compute permutation entropy on a 1D NumPy array.

        Parameters
        ----------
        x : numpy.ndarray
            One-dimensional signal.
        dim : int
            Embedding dimension.
        tau : int
            Delay between embedded samples.

        Returns
        -------
        float
            Permutation entropy, or NaN when the signal is too short.
        """
        x = np.array(x)
        n = len(x)

        # Check if signal is long enough
        if n < (dim - 1) * tau + 1 or n < dim:
            logger.warning(
                f"Signal length ({n}) is too short for "
                f"permutation entropy with dim={dim}, tau={tau}. "
                "Returning NaN."
            )
            return np.nan

        # 1. Create the matrix of overlapping windows
        # Create an array of starting indices
        idx = np.arange(n - (dim - 1) * tau)
        # Create the 2D array of indices for all windows
        indices = idx[:, None] + np.arange(dim) * tau
        # Get the windows
        windows = x[indices]

        # 2. Get the order patterns (indices that would sort each window)
        patterns_array = np.argsort(windows, axis=1)

        # 3. Count occurrences of each unique pattern
        # A robust way to count unique rows
        # Convert each row (pattern) to a tuple
        patterns_list = [tuple(p) for p in patterns_array]

        unique_patterns, counts = np.unique(patterns_list, axis=0, return_counts=True)

        # 4. Calculate probabilities
        probabilities = counts / len(patterns_list)

        # 5. Calculate permutation entropy (base 2)
        # pe = -sum(p * log2(p))
        pe = -np.sum(probabilities * np.log2(probabilities))

        return pe

    def _compute_stat(self, spectrum: np.ndarray, stat_name: str) -> float:
        """
        Compute a single named statistic on a given spectrum.

        Parameters
        ----------
        spectrum : numpy.ndarray
            Amplitude spectrum.
        stat_name : str
            Name of the statistic to compute.

        Returns
        -------
        float
            Computed statistic value.
        """
        if spectrum.size == 0:
            return np.nan

        # --- Basic Stats ---
        if stat_name == "mean":
            return np.mean(spectrum)
        elif stat_name == "maximum":
            return np.max(spectrum)
        elif stat_name == "minimum":
            return np.min(spectrum)
        elif stat_name == "root_mean_square":
            return np.sqrt(np.mean(spectrum**2))
        elif stat_name == "peak_to_peak_value":
            return np.max(spectrum) - np.min(spectrum)
        elif stat_name == "variance":
            return np.var(spectrum)
        elif stat_name == "skewness":
            return stats.skew(spectrum)
        elif stat_name == "kurtosis":
            return stats.kurtosis(spectrum)
        elif stat_name == "abs_energy":
            return np.sum(spectrum**2)

        # --- rul_features inspired Stats ---
        elif stat_name == "peak_factor":
            rms = np.sqrt(np.mean(spectrum**2))
            return np.max(spectrum) / (rms + 1e-10)
        elif stat_name == "change_coefficient":
            if spectrum.size < 2:
                return 0.0
            return np.mean(np.abs(np.diff(spectrum)))
        elif stat_name == "clearance_factor":
            mean_sqrt_amp = np.mean(np.sqrt(spectrum))
            return np.max(spectrum) / (mean_sqrt_amp**2 + 1e-10)

        # --- Entropy Stats (Applied to the SPECTRUM) ---
        elif stat_name == "spectral_entropy":
            return self._compute_spectral_entropy(spectrum)

        elif stat_name == "shannon_entropy":
            # Treat shannon_entropy as identical to spectral_entropy
            return self._compute_spectral_entropy(spectrum)

        elif stat_name == "permutation_entropy":
            # Apply permutation entropy to the spectrum itself
            return self._compute_permutation_entropy(
                spectrum, dim=self.pe_dim, tau=self.pe_tau
            )

        # This should not be reachable due to __init__ check
        else:
            raise ValueError(f"Unknown statistic handler: {stat_name}")

    def _validate_input(self, data: np.ndarray, key: str) -> np.ndarray:
        """
        Validate input data.

        Parameters
        ----------
        data : numpy.ndarray
            Input array to validate.
        key : str
            Payload key used in error messages.

        Returns
        -------
        numpy.ndarray
            Validated input array.
        """
        if data.ndim != 2:
            raise ValueError(
                f"Key '{key}' must be a 2D array (n_rows, n_columns), "
                f"got {data.ndim}D array with shape {data.shape}"
            )
        if np.any(np.isinf(data)):
            raise ValueError(f"Infinite values found in array '{key}'")
        signal_length = data.shape[0] if self.apply_to_columns else data.shape[1]
        if signal_length < 2:
            axis_name = "rows" if self.apply_to_columns else "columns"
            raise ValueError(
                f"Signal length {signal_length} along {axis_name} is too short "
                f"for FFT (min 2) for key '{key}'"
            )
        return data

    @override
    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Apply spectral statistics computation to all signals in the input.

        Parameters
        ----------
        data : dict[str, Any]
            Input payload dictionary.
        metadata : dict
            Transform metadata.

        Returns
        -------
        numpy.ndarray
            Concatenated spectral-statistics feature vector.
        """
        if not data:
            raise ValueError("No data provided for SpectralStatsTransform")

        key, value = "features", data["features"]
        value = convert_to_numpy(value)
        value = self._validate_input(value, key)

        n_rows, n_columns = value.shape
        n_signals = n_columns  # Assumes apply_to_columns=True

        all_features = []

        for sig_idx in range(n_signals):
            signal = value[:, sig_idx]  # Extract 1D signal

            # Handle potential NaNs in input signal
            if np.any(np.isnan(signal)):
                logger.warning(f"Signal {sig_idx} contains NaNs. Stats will be NaN.")
                spectrum = np.array([])  # Will cause stats to be np.nan
            else:
                spectrum = self._compute_fft_spectrum(signal)

            # Compute all requested stats for this signal's spectrum
            for stat_name in self.stats_to_compute:
                stat_value = self._compute_stat(spectrum, stat_name)
                # Ensure stat_value is a 1D numpy array. Convert numpy scalars to 1-element arrays.
                if isinstance(stat_value, np.ndarray):
                    stat_arr = stat_value.reshape(-1)
                elif isinstance(stat_value, np.generic) or np.isscalar(stat_value):
                    stat_arr = np.array([stat_value])
                else:
                    stat_arr = np.asarray(stat_value).reshape(-1)

                if np.isnan(stat_arr).any():
                    raise ValueError(f"Stat '{stat_name}' for signal {sig_idx} is NaN.")
                all_features.append(stat_arr)

        # Reshape all features into a single row vector
        final_result = np.concatenate(all_features)

        logger.debug(f"SpectralStatsTransform result shape: {final_result.shape}")
        return final_result

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)

    def get_feature_names(
        self, input_keys: List[str], input_shapes: Dict[str, Tuple[int, int]]
    ) -> List[str]:
        """
        Generate feature names for the spectral statistics output.

        Parameters
        ----------
        input_keys : list[str]
            Ordered list of input keys.
        input_shapes : dict[str, tuple[int, int]]
            Mapping from input key to 2D shape.

        Returns
        -------
        list[str]
            Ordered feature names matching the transform output.
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

        # The loop order MUST match the concatenation order in transform_data
        for sig_idx in range(n_signals):
            for stat_name in self.stats_to_compute:
                feature_names.append(f"{key}_{axis_name}{sig_idx}_spec_{stat_name}")

        return feature_names
