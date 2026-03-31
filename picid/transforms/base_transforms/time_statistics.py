import logging
from typing import Dict, List, Optional, Tuple, override
import numpy as np
from scipy import stats  # For skewness and kurtosis
import scipy

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin
from picid.data.data_objects.utils import convert_to_numpy

logger = logging.getLogger(__name__)


def hankel_svdvals(data, hankel_window_size, slice_window_size):
    """
    Compute mean Hankel singular values over fixed-size signal slices.

    Parameters
    ----------
    data : np.ndarray
        One-dimensional input signal.
    hankel_window_size : int
        Window size used to construct the Hankel matrix.
    slice_window_size : int
        Slice length used before averaging singular values.

    Returns
    -------
    np.ndarray
        Mean singular values across all slices.
    """

    n_slices = len(data) // slice_window_size
    hankel_svd = []

    for i in range(n_slices):
        sample_data = data[slice_window_size * i : slice_window_size * (i + 1)]
        c = sample_data[0:hankel_window_size]
        r = sample_data[hankel_window_size - 1 :]
        h = scipy.linalg.hankel(c, r)

        hankel_svd.append(scipy.linalg.svdvals(h))

    return np.array(hankel_svd).mean(axis=0)


class TimeStatsTransform(NoFitPerSegmentMixin, DenseTransform):
    """
    Computes global statistical features directly from time-domain signals.

    This transform applies a set of statistical functions (e.g., mean,
    kurtosis, peak_factor) to each signal (column or row).

    All statistics from all signals are concatenated into a single
    2D array of shape (1, n_signals * n_stats).

    Parameters
    ----------
    stats_to_compute : list of str
        Statistic names to compute.
    apply_to_columns : bool, default=True
        Whether each column should be treated as one signal.
    hankel_window_size : int, optional
        Hankel window size used by the ``hankel_svd`` statistic.
    slice_window_size : int, optional
        Slice size used by the ``hankel_svd`` statistic.
    **kwargs
        Additional keyword arguments passed to the parent transform classes.
    """

    # Define the set of all valid statistical function names
    # based on the provided file
    VALID_STATS = {
        "mean",
        "maximum",
        "minimum",
        "root_mean_square",
        "abs_avg",  # Defined as a wrapper for root_mean_square in the source
        "peak_to_peak_value",
        "standard_deviation",
        "skewness",
        "kurtosis",
        "variance",
        "peak_factor",
        "change_coefficient",  # Note: This is mean / std
        "clearance_factor",
        "abs_energy",
        "hankel_svd",
    }

    def __init__(
        self,
        stats_to_compute: List[str],
        apply_to_columns: bool = True,
        hankel_window_size: Optional[int] = 200,
        slice_window_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the TimeStatsTransform.

        Parameters
        ----------
        stats_to_compute : list of str
            Statistic names to compute. Each name must be in
            ``TimeStatsTransform.VALID_STATS``.
        apply_to_columns : bool
            If True (default), apply to columns. If False, apply to rows.
        hankel_window_size : int, optional
            Hankel window size used by the ``hankel_svd`` statistic.
        slice_window_size : int, optional
            Slice size used by the ``hankel_svd`` statistic.
        **kwargs
            Additional arguments for the parent transform classes.
        """
        super().__init__(**kwargs)
        self.apply_to_columns = apply_to_columns
        self.hankel_window_size = hankel_window_size
        self.slice_window_size = slice_window_size

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
            f"TimeStatsTransform initialized: stats={self.stats_to_compute}, "
            f"apply_to_columns={self.apply_to_columns}"
        )

    def _compute_stat(self, signal: np.ndarray, stat_name: str) -> float:
        """
        Compute one named statistic on a one-dimensional signal.

        Parameters
        ----------
        signal : np.ndarray
            1D time-domain signal.
        stat_name : str
            The name of the statistic to compute.

        Returns
        -------
        float
            The calculated scalar statistic.
        """
        if signal.size == 0:
            return np.nan

        # Add a small epsilon to prevent division by zero
        epsilon = 1e-10

        # --- Basic Stats ---
        if stat_name == "mean":
            return np.mean(signal)
        elif stat_name == "maximum":
            return np.max(signal)
        elif stat_name == "minimum":
            return np.min(signal)

        elif stat_name == "root_mean_square" or stat_name == "abs_avg":
            # Per source file, abs_avg is treated as identical to rms
            return np.sqrt(np.mean(signal**2))

        elif stat_name == "peak_to_peak_value":
            return np.max(signal) - np.min(signal)
        elif stat_name == "standard_deviation":
            return np.std(signal)

        elif stat_name == "skewness":
            # Using scipy.stats.skew, which is what tsfresh uses
            return stats.skew(signal)

        elif stat_name == "kurtosis":
            # Using scipy.stats.kurtosis (Fisher's definition),
            # which is what tsfresh uses
            return stats.kurtosis(signal)

        elif stat_name == "variance":
            # Using np.var, which is what tsfresh uses
            return np.var(signal)

        elif stat_name == "abs_energy":
            # Using np.sum(signal**2), which is what tsfresh uses
            return np.sum(signal**2)

        # --- Factor Stats (from Mao et al. 2020) ---
        elif stat_name == "peak_factor":
            rms_val = np.sqrt(np.mean(signal**2))
            return np.max(signal) / (rms_val + epsilon)

        elif stat_name == "change_coefficient":
            # Defined in source as mean / std
            std_val = np.std(signal)
            return np.mean(signal) / (std_val + epsilon)

        elif stat_name == "clearance_factor":
            # Defined in source as max / mean(signal**2)
            mean_sq_val = np.mean(signal**2)
            return np.max(signal) / (mean_sq_val + epsilon)

        elif stat_name == "hankel_svd":
            assert (
                len(signal) >= self.hankel_window_size
            ), f"Signal length {len(signal)} is shorter than hankel_window_size."

            return hankel_svdvals(
                signal,
                hankel_window_size=self.hankel_window_size,
                slice_window_size=(
                    self.slice_window_size
                    if self.slice_window_size is not None
                    else len(signal)
                ),
            )

        # This should not be reachable due to __init__ check
        else:
            raise ValueError(f"Unknown statistic handler: {stat_name}")

    def _validate_input(self, data: np.ndarray, key: str) -> np.ndarray:
        """
        Validate one dense input array before feature extraction.

        Parameters
        ----------
        data : np.ndarray
            Dense input array to validate.
        key : str
            Input key associated with the array.

        Returns
        -------
        np.ndarray
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
        if signal_length == 0:
            axis_name = "rows" if self.apply_to_columns else "columns"
            raise ValueError(f"Signal length is 0 along {axis_name} for key '{key}'")
        return data

    @override
    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Apply time-domain statistics computation to all signals in the input.

        Parameters
        ----------
        data : dict
            Dictionary containing 2D numpy arrays ``(n_rows, n_columns)``.
        metadata : dict
            Dictionary containing auxiliary metadata (unused).

        Returns
        -------
        np.ndarray
            A 2D numpy array of shape (1, n_signals * n_stats) containing
            the concatenated scalar features.
        """
        if not data:
            raise ValueError("No data provided for TimeStatsTransform")

        key, value = "features", data["features"]
        value = convert_to_numpy(value)
        value = self._validate_input(value, key)

        n_rows, n_columns = value.shape
        n_signals = n_columns  # Assumes apply_to_columns=True

        all_features = []

        # Iterate over each signal (column)
        for sig_idx in range(n_signals):
            signal = value[:, sig_idx]  # Extract 1D signal

            # Check for NaNs in the signal
            if np.any(np.isnan(signal)):
                logger.warning(f"Signal {sig_idx} contains NaNs. Stats will be NaN.")
                # Append NaN for all stats for this signal
                all_features.extend([np.nan] * len(self.stats_to_compute))
                continue

            # Compute all requested stats for this signal
            for stat_name in self.stats_to_compute:
                stat_value = self._compute_stat(signal, stat_name)
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

        logger.debug(f"TimeStatsTransform result shape: {final_result.shape}")
        return final_result

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)

    def get_feature_names(
        self, input_keys: List[str], input_shapes: Dict[str, Tuple[int, int]]
    ) -> List[str]:
        """
        Generate output feature names for the statistics vector.

        Parameters
        ----------
        input_keys : list of str
            Input keys seen by the transform.
        input_shapes : dict
            Mapping from key to ``(rows, columns)`` input shape tuples.

        Returns
        -------
        list of str
            Generated feature names in output order.
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
                feature_names.append(f"{key}_{axis_name}{sig_idx}_time_{stat_name}")

        return feature_names
