import logging
import warnings
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

import numpy as np
import pandas as pd

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.data.data_objects.utils import convert_to_numpy
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin

logger = logging.getLogger(__name__)


class SpectralExtrapolationImputation:
    """
    Helper class for Causal Spectral Extrapolation.
    Models the signal as a sum of sinusoids derived from the immediate past.
    """

    def __init__(self, window_len: int = 64, top_k: int = 3):
        self.window_len = window_len
        self.top_k = top_k

    def impute(self, X: np.ndarray) -> np.ndarray:
        """
        Performs spectral imputation on a 3D array (Batch, Time, Channels).
        Returns a copy of the array with gaps filled.
        """
        X_imp = X.copy()
        B, T, C = X_imp.shape

        for b in range(B):
            for c in range(C):
                signal = X_imp[b, :, c]

                # Identify NaN regions
                nan_mask = np.isnan(signal)
                if not np.any(nan_mask):
                    continue

                # Find starts and ends of gaps
                # Concatenate with False to detect edges at 0 and T
                bounded_mask = np.concatenate(([False], nan_mask, [False]))
                diffs = np.diff(bounded_mask.astype(int))
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]

                for start, end in zip(starts, ends):
                    gap_len = end - start

                    # --- DYNAMIC WINDOW LOGIC ---
                    # Use whatever history is available, up to the max window_len
                    curr_window_len = min(start, self.window_len)

                    # Safety check: If history is extremely short (e.g. < 4 points),
                    # FFT results will be meaningless/unstable. Fallback to 0.0.
                    if curr_window_len < 4:
                        signal[start:end] = 0.0
                        continue

                    hist_start = start - curr_window_len
                    history = signal[hist_start:start]

                    # Handle NaNs in history (e.g. from previous nearby gaps)
                    if np.isnan(history).any():
                        history = np.nan_to_num(history, nan=np.nanmean(history))

                    # 1. Compute FFT (Real input -> Complex output)
                    fft_vals = np.fft.rfft(history)
                    freqs = np.fft.rfftfreq(len(history))

                    # 2. Peak Picking
                    magnitudes = np.abs(fft_vals)
                    magnitudes[0] = 0  # Remove DC offset

                    # Select top K frequencies
                    if len(magnitudes) <= self.top_k:
                        peak_indices = np.arange(len(magnitudes))
                    else:
                        peak_indices = np.argsort(magnitudes)[-self.top_k :]

                    # 3. Synthesize the gap
                    # t_gap is time relative to the start of the window
                    # IMPORTANT: Use curr_window_len for the offset
                    t_gap = np.arange(curr_window_len, curr_window_len + gap_len)
                    filled_gap = np.zeros(gap_len)

                    for idx in peak_indices:
                        # Normalize amplitude based on CURRENT window length
                        amp = magnitudes[idx] / curr_window_len
                        if idx > 0 and idx < len(magnitudes) - 1:
                            amp *= 2

                        phase = np.angle(fft_vals[idx])
                        freq_cycles = freqs[idx]

                        # Generate cosine wave
                        wave = amp * np.cos(2 * np.pi * freq_cycles * t_gap + phase)
                        filled_gap += wave

                    # 4. Fill
                    signal[start:end] = filled_gap

                X_imp[b, :, c] = signal

        return X_imp


class ImputationTransform(DenseTransform, ConcatinatedFitMultiSourceTransformMixin):
    """
    Imputes missing values (NaNs) in time-series data using configurable strategies.

    This transform processes input arrays to fill in `NaN` values, ensuring data completeness
    for downstream models. It supports both statistical imputation (e.g., mean) and
    time-series specific methods (e.g., Last Observation Carried Forward, Linear Interpolation).

    The transform handles 3D arrays of shape ``(Batch, Time, Channels)`` or 2D arrays
    of shape ``(Time, Channels)``. Imputation is applied independently per channel and,
    for time-series methods, independently per batch to prevent data leakage.

    **Strategy Usage Guide:**

    The choice of strategy depends heavily on the nature of the signal (stationary vs. trending,
    smooth vs. oscillating) and whether the application allows "peeking" into the future (causality).

    | Strategy | Causality | Best Use Case | Pros/Cons |
    | :--- | :--- | :--- | :--- |
    | ``"zero"`` | N/A | Sparse signals or when 0 represents "off/missing". | **+** Simple.<br>**-** Biases statistics if 0 is not natural. |
    | ``"mean"`` | Global* | Stationary noise with no trend. | **+** Preserves global mean.<br>**-** Reduces variance; creates unnatural flat lines. |
    | ``"locf"`` | **Causal** | Step functions, digital states, or slow-moving trends. | **+** Causal; stable.<br>**-** "Freezes" the signal; bad for noise/jitter. |
    | ``"linear"``| **Non-Causal** | Smooth, continuous signals (e.g., temperature). | **+** Smooth transitions.<br>**-** Peeks at future (end of gap); smooths out high-freq variance. |
    | ``"stochastic"`` | **Causal** | **Oscillating/Jittery signals** (e.g., noisy sensors). | **+** Preserves signal texture/variance; strictly causal.<br>**-** Non-deterministic output. |
    | ``"spectral"`` | **Causal** | **Periodic/Rhythmic signals** (e.g., vibration). | **+** Restores underlying waveform.<br>**-** Computationally heavier; assumes periodicity. |

    *\* Note: "mean" imputation relies on global statistics computed during ``fit``. If these statistics are computed on the test set itself, it technically uses future information.*

    Attributes:
        strategy (Union[str, List[str]]): The imputation strategy to use.
            Can be a single string (applied to all channels) or a list of strings
            (one per channel). Must be one of:
            - ``"mean"``: Replaces NaNs with the global mean.
            - ``"locf"``: Last Observation Carried Forward.
            - ``"linear"``: Linear interpolation.
            - ``"zero"``: Replaces NaNs with 0.0.
            - ``"stochastic"``: Stochastic LOCF. Adds random Gaussian noise scaled by the channel's std dev.
            - ``"spectral"``: Causal Spectral Extrapolation. Models signal as sum of sinusoids based on history.
        apply_to (Optional[List[str]]): A list of keys in the input dictionary to transform.
        spectral_window_len (int): Window size (history) used for spectral analysis (default: 64).
        spectral_top_k (int): Number of dominant frequencies to use for extrapolation (default: 3).
        fitted_means (Dict[str, np.ndarray]): Stores computed means (for ``"mean"``).
        fitted_stds (Dict[str, np.ndarray]): Stores computed std devs (for ``"stochastic"``).
    """

    VALID_STRATEGIES = {"mean", "locf", "linear", "zero", "stochastic", "spectral"}

    def __init__(
        self,
        strategy: Union[str, List[str]] = "linear",
        apply_to: Optional[List[str]] = None,
        spectral_window_len: int = 64,
        spectral_top_k: int = 3,
        **kwargs,
    ):
        """
        Initializes the ImputationTransform with a specific strategy.

        Args:
            strategy: The name of the imputation strategy or a list of strategies per channel.
                Defaults to ``"linear"``.
            apply_to: List of dictionary keys to transform. If ``None``, all keys
                are transformed. Defaults to ``None``.
            spectral_window_len: Number of past time steps to analyze for "spectral" imputation.
            spectral_top_k: Number of frequency components to synthesize for "spectral" imputation.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If ``strategy`` is not one of the valid options.
        """
        super().__init__(**kwargs)

        # Fix for Hydra/OmegaConf: Convert ListConfig (or tuple) to standard list
        if not isinstance(strategy, str) and hasattr(strategy, "__iter__"):
            strategy = list(strategy)

        # Validate strategies
        strategies_check = [strategy] if isinstance(strategy, str) else strategy
        for s in strategies_check:
            if s not in self.VALID_STRATEGIES:
                raise ValueError(
                    f"Invalid strategy '{s}'. Valid: {self.VALID_STRATEGIES}"
                )

        self.strategy = strategy
        self.apply_to = apply_to

        # Spectral parameters
        self.spectral_window_len = spectral_window_len
        self.spectral_top_k = spectral_top_k
        self._spectral_imputer = None  # Lazy instantiation

        # Store learned parameters
        self.fitted_means: Dict[str, np.ndarray] = {}
        self.fitted_stds: Dict[str, np.ndarray] = {}

    def fit_data(self, data: NamedTransformInput, metadata: Dict):
        """
        Fits the transform to the data.

        Calculates necessary statistics based on the selected strategies:
        - ``"mean"``: Computes global mean per channel.
        - ``"stochastic"``: Computes global standard deviation per channel.

        Args:
            data: A dictionary containing input arrays.
            metadata: Metadata dictionary (unused).

        Returns:
            self: Returns the instance itself.
        """
        keys_to_process = self.apply_to if self.apply_to is not None else data.keys()

        # Determine which stats we need to compute
        if isinstance(self.strategy, list):
            uses_mean = "mean" in self.strategy
            uses_std = "stochastic" in self.strategy
        else:
            uses_mean = self.strategy == "mean"
            uses_std = self.strategy == "stochastic"

        if not (uses_mean or uses_std):
            return self

        for key, value in data.items():
            if key not in keys_to_process:
                continue

            # Ensure 3D arrays are preserved (Batch, Time, Channel)
            np_val = convert_to_numpy(value, ensure_2d=False)

            # Validate dimensions and strategy length
            n_channels = np_val.shape[-1]
            if isinstance(self.strategy, list) and len(self.strategy) != n_channels:
                raise ValueError(
                    f"Strategy list length ({len(self.strategy)}) does not match "
                    f"number of channels ({n_channels}) for key '{key}'."
                )

            # Determine axes for aggregation (collapse Batch and Time)
            if np_val.ndim == 2:
                axes = 0  # (Time, Channels) -> Mean over Time
            elif np_val.ndim == 3:
                axes = (0, 1)  # (Batch, Time, Channels) -> Mean over Batch & Time
            else:
                raise ValueError(
                    f"Data for key '{key}' has invalid dimensions: {np_val.ndim}"
                )

            # --- Compute Statistics ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                if uses_mean:
                    means = np.nanmean(np_val, axis=axes)
                    self.fitted_means[key] = np.nan_to_num(means, nan=0.0)

                if uses_std:
                    stds = np.nanstd(np_val, axis=axes)
                    # If std is 0 or NaN (e.g., constant signal), default to 0.0 variance
                    self.fitted_stds[key] = np.nan_to_num(stds, nan=0.0)

        return self

    def _get_spectral_imputer(self) -> SpectralExtrapolationImputation:
        """Lazy loader for the spectral helper class."""
        if self._spectral_imputer is None:
            self._spectral_imputer = SpectralExtrapolationImputation(
                window_len=self.spectral_window_len, top_k=self.spectral_top_k
            )
        return self._spectral_imputer

    def _impute_array(self, X: np.ndarray, key: str) -> np.ndarray:
        """
        Applies the configured imputation strategy to a single NumPy array.

        This method standardizes input to 3D, applies the strategy per channel,
        and restores original shape.
        """
        # Standardize input to (Batch, Time, Channels)
        original_shape = X.shape
        if X.ndim == 2:
            # (Time, Channels) -> (1, Time, Channels)
            X_working = X[np.newaxis, :, :]
        elif X.ndim == 3:
            X_working = X
        else:
            raise ValueError("Data must be 2D or 3D")

        # Create a copy to avoid modifying the input in-place
        X_imp = X_working.copy()
        B, T, C = X_imp.shape

        # Normalize strategies to a list for uniform processing
        if isinstance(self.strategy, list):
            if len(self.strategy) != C:
                raise ValueError(
                    f"Strategy list length ({len(self.strategy)}) does not match "
                    f"number of channels ({C})."
                )
            strategies = self.strategy
        else:
            strategies = [self.strategy] * C

        # Group channels by strategy for organized processing
        strat_map = defaultdict(list)
        for idx, s in enumerate(strategies):
            strat_map[s].append(idx)

        # --- Apply Strategies ---

        # 1. Zero Imputation
        if "zero" in strat_map:
            cols = strat_map["zero"]
            sub_arr = X_imp[:, :, cols]
            X_imp[:, :, cols] = np.nan_to_num(sub_arr, nan=0.0)

        # 2. Mean Imputation
        if "mean" in strat_map:
            cols = strat_map["mean"]
            fitted_means = self.fitted_means.get(key, np.zeros(C))

            if key not in self.fitted_means:
                logger.warning(
                    f"Key '{key}' was not seen during fit(). Using 0.0 fallback for mean."
                )

            for c_idx in cols:
                mask = np.isnan(X_imp[:, :, c_idx])
                X_imp[:, :, c_idx][mask] = fitted_means[c_idx]

        # 3. LOCF (Standard)
        if "locf" in strat_map:
            cols = strat_map["locf"]
            for b in range(B):
                for c in cols:
                    series = pd.Series(X_imp[b, :, c])
                    filled = series.ffill().bfill()
                    X_imp[b, :, c] = filled.values

        # 4. Stochastic LOCF
        if "stochastic" in strat_map:
            cols = strat_map["stochastic"]
            fitted_stds = self.fitted_stds.get(key, np.zeros(C))

            if key not in self.fitted_stds:
                logger.warning(
                    f"Key '{key}' not seen in fit(). Using 0.0 noise for stochastic."
                )

            for b in range(B):
                for c in cols:
                    # Identify where the NaNs are *before* filling
                    nan_mask = np.isnan(X_imp[b, :, c])

                    if not np.any(nan_mask):
                        continue

                    # Step A: Perform LOCF to get the baseline level
                    series = pd.Series(X_imp[b, :, c])
                    filled = series.ffill().bfill().values

                    # Step B: Generate noise ~ N(0, sigma_channel)
                    sigma = fitted_stds[c]
                    if sigma > 0:
                        noise = np.random.normal(
                            loc=0.0, scale=sigma, size=np.sum(nan_mask)
                        )
                        # Step C: Add noise ONLY to the missing regions
                        filled[nan_mask] += noise

                    X_imp[b, :, c] = filled

        # 5. Linear Interpolation
        if "linear" in strat_map:
            cols = strat_map["linear"]
            for b in range(B):
                for c in cols:
                    series = pd.Series(X_imp[b, :, c])
                    filled = series.interpolate(method="linear", limit_direction="both")
                    # Fallback for edges or all-NaN series
                    filled = filled.ffill().bfill().fillna(0.0)
                    X_imp[b, :, c] = filled.values

        # 6. Spectral Extrapolation
        if "spectral" in strat_map:
            cols = strat_map["spectral"]
            imputer = self._get_spectral_imputer()

            # Extract only the spectral columns to avoid processing others
            # We copy because slicing with a list forces a copy in NumPy anyway
            spectral_subset = X_imp[:, :, cols].copy()

            # Delegate to helper
            imputed_subset = imputer.impute(spectral_subset)

            # Assign results back
            X_imp[:, :, cols] = imputed_subset

        # Restore original dimensionality
        if len(original_shape) == 2:
            return X_imp[0]
        return X_imp

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Executes the imputation on the provided data dictionary.

        Args:
            data: Dictionary of input arrays (e.g., ``{'features': ...}``).
            metadata: Metadata dictionary (unused).

        Returns:
            Dict[str, np.ndarray]: Dictionary with imputed arrays.
        """
        output_data = {}
        keys_to_process = self.apply_to if self.apply_to is not None else data.keys()

        for key, value in data.items():
            if key in keys_to_process:
                # Ensure 3D structure is preserved for processing
                np_val = convert_to_numpy(value, ensure_2d=False)
                output_data[key] = self._impute_array(np_val, key)
            else:
                output_data[key] = value

        return output_data
