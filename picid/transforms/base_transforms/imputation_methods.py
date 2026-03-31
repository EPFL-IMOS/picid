import logging
import warnings
from typing import Dict, List, Optional, Union
from collections import defaultdict

import numpy as np
import pandas as pd

from picid.data.data_objects import NamedTransformInput
from picid.data.data_objects.utils import convert_to_numpy
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import ConcatFitAndPerSegmentTransformMixin

logger = logging.getLogger(__name__)


class SpectralExtrapolationImputation:
    """
    Impute gaps by extrapolating dominant spectral components.

    Parameters
    ----------
    window_len : int, default=64
        Number of historical samples used to estimate the local spectrum.
    top_k : int, default=3
        Number of dominant frequency components to reconstruct.
    match_variance : bool, default=False
        Whether to rescale the generated gap to match historical variance.
    """

    def __init__(
        self, window_len: int = 64, top_k: int = 3, match_variance: bool = False
    ):
        self.window_len = window_len
        self.top_k = top_k
        self.match_variance = match_variance  # <--- NEW FLAG

    def impute(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values in a 3D array using spectral extrapolation.

        Parameters
        ----------
        X : numpy.ndarray
            Input array with shape ``(batch, time, channels)``.

        Returns
        -------
        numpy.ndarray
            Copy of ``X`` with NaN gaps filled.
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
                bounded_mask = np.concatenate(([False], nan_mask, [False]))
                diffs = np.diff(bounded_mask.astype(int))
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]

                for start, end in zip(starts, ends):
                    gap_len = end - start

                    # --- DYNAMIC WINDOW LOGIC ---
                    curr_window_len = min(start, self.window_len)

                    if curr_window_len < 4:
                        signal[start:end] = 0.0
                        continue

                    hist_start = start - curr_window_len
                    history = signal[hist_start:start]

                    if np.isnan(history).any():
                        history = np.nan_to_num(history, nan=np.nanmean(history))

                    # --- 1. Prepare for Variance Matching (Optional) ---
                    hist_noise_std = 0.0
                    if self.match_variance:
                        # Use robust diff-based std to ignore steps/trends
                        if len(history) > 1:
                            hist_noise_std = np.std(np.diff(history))

                    # --- 2. Calculate and remove DC bias ---
                    dc_bias = np.mean(history)
                    history_centered = history - dc_bias

                    # --- 3. Compute FFT ---
                    fft_vals = np.fft.rfft(history_centered)
                    freqs = np.fft.rfftfreq(len(history))
                    magnitudes = np.abs(fft_vals)
                    magnitudes[0] = 0

                    if len(magnitudes) <= self.top_k:
                        peak_indices = np.arange(len(magnitudes))
                    else:
                        peak_indices = np.argsort(magnitudes)[-self.top_k :]

                    # --- 4. Synthesize the gap ---
                    t_gap = np.arange(curr_window_len, curr_window_len + gap_len)
                    filled_gap = np.zeros(gap_len)

                    for idx in peak_indices:
                        amp = magnitudes[idx] / curr_window_len
                        if idx > 0 and idx < len(magnitudes) - 1:
                            amp *= 2

                        phase = np.angle(fft_vals[idx])
                        freq_cycles = freqs[idx]

                        wave = amp * np.cos(2 * np.pi * freq_cycles * t_gap + phase)
                        filled_gap += wave

                    # --- 5. Apply Variance Matching (Optional) ---
                    if self.match_variance:
                        if len(filled_gap) > 1:
                            gap_noise_std = np.std(np.diff(filled_gap))

                            # Only scale if we have valid noise stats
                            if gap_noise_std > 1e-9 and hist_noise_std > 1e-9:
                                scale_factor = hist_noise_std / gap_noise_std
                                # Clamp to prevent explosion (e.g. max 10x boost)
                                scale_factor = min(scale_factor, 10.0)
                                filled_gap *= scale_factor

                    # --- 6. Restore DC bias ---
                    filled_gap += dc_bias

                    # --- 7. Fill ---
                    signal[start:end] = filled_gap

                X_imp[b, :, c] = signal

        return X_imp


class BlockwisePastCopyImputation:
    """
    Impute gaps by copying or mirroring past signal blocks.

    The strategy first tries to reuse a historical pattern, then falls back to
    a mirrored block, and finally uses a simple causal mean for very early gaps.
    """

    def __init__(self):
        pass

    def impute(self, X: np.ndarray) -> np.ndarray:
        X_imp = X.copy()
        B, T, C = X_imp.shape

        for b in range(B):
            for c in range(C):
                signal = X_imp[b, :, c]
                nan_mask = np.isnan(signal)
                if not np.any(nan_mask):
                    continue

                bounded = np.concatenate(([False], nan_mask, [False]))
                starts = np.where(np.diff(bounded.astype(int)) == 1)[0]
                ends = np.where(np.diff(bounded.astype(int)) == -1)[0]

                for start, end in zip(starts, ends):
                    # --- 1. CAUSAL CONTEXT GATHERING ---
                    history_so_far = signal[:start]
                    valid_hist = history_so_far[~np.isnan(history_so_far)]

                    if len(valid_hist) > 5:
                        causal_mean = np.mean(valid_hist)
                        causal_std = np.std(valid_hist) if len(valid_hist) > 1 else 0.05
                        h_range = np.ptp(valid_hist) if len(valid_hist) > 1 else 1.0
                        clamp_min, clamp_max = (
                            np.min(valid_hist) - 0.2 * h_range,
                            np.max(valid_hist) + 0.2 * h_range,
                        )

                        # Calculate local trend slope (reactive window)
                        trend_win = min(len(valid_hist), 32)
                        slope, _ = np.polyfit(
                            np.arange(trend_win), valid_hist[-trend_win:], 1
                        )
                    else:
                        # Fallback for gaps at the very beginning of a file
                        causal_mean = (
                            np.nanmean(signal) if not np.all(np.isnan(signal)) else 0.0
                        )
                        causal_std = 0.1
                        clamp_min, clamp_max = -np.inf, np.inf
                        slope = 0.0

                    current_fill_idx = start
                    while current_fill_idx < end:
                        gap_rem = end - current_fill_idx
                        found_match = False

                        # --- 2. HIERARCHICAL STRATEGY ---

                        # OPTION A: Pattern Match (Search for historical twins)
                        for q_size in [24, 12, 6]:
                            query_len = min(q_size, current_fill_idx)
                            if query_len < 6:
                                continue

                            query = signal[
                                current_fill_idx - query_len : current_fill_idx
                            ]
                            search_buf = signal[: current_fill_idx - query_len]

                            if len(search_buf) > query_len:
                                search_view = np.lib.stride_tricks.sliding_window_view(
                                    search_buf, query_len
                                )
                                dists = np.mean((search_view - query) ** 2, axis=-1)
                                best_idx = np.argmin(dists)

                                s_start = best_idx + query_len
                                p_src = signal[s_start : current_fill_idx - query_len]
                                nans = np.where(np.isnan(p_src))[0]
                                actual_max_copy = (
                                    nans[0] if len(nans) > 0 else len(p_src)
                                )

                                if actual_max_copy > 5:
                                    c_len = min(gap_rem, actual_max_copy)
                                    fill_src = signal[s_start : s_start + c_len].copy()
                                    found_match = True
                                    break

                        # OPTION B: Mirror Fallback (The 'Oscillation' Safeguard)
                        if not found_match:
                            # Use a generous window to capture a full engine cycle
                            mirror_win = min(current_fill_idx, 128)
                            if mirror_win > 4:
                                # Mirroring preserves the 'join' phase perfectly
                                fill_src = signal[
                                    current_fill_idx - mirror_win : current_fill_idx
                                ][::-1].copy()
                                c_len = min(gap_rem, len(fill_src))
                                fill_src = fill_src[:c_len]
                            else:
                                # Absolute fallback for start-of-stream
                                c_len = gap_rem
                                fill_src = np.full(
                                    c_len, causal_mean
                                ) + np.random.normal(0, causal_std * 0.1, c_len)

                        # --- 3. CLEANING & STITCHING ---
                        fill_src = np.nan_to_num(fill_src, nan=causal_mean)
                        s_idx = np.arange(len(fill_src))

                        # Detrend source -> Align DC to junction -> Re-apply trend slope
                        fill_src = fill_src - (slope * s_idx)  # Flatten
                        offset = signal[current_fill_idx - 1] - fill_src[0]
                        new_seg = (fill_src + offset) + (
                            slope * s_idx
                        )  # Tilt into future

                        # --- 4. CLAMP & PASTE ---
                        # Clamping prevents 'exploding' signals in recursive large gaps
                        signal[current_fill_idx : current_fill_idx + len(new_seg)] = (
                            np.clip(new_seg, clamp_min, clamp_max)
                        )
                        current_fill_idx += len(new_seg)

                X_imp[b, :, c] = signal
        return X_imp


class ImputationTransform(ConcatFitAndPerSegmentTransformMixin, DenseTransform):
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
    | ``"copy_past"``| **Causal** | **Repetitive/Cyclic signals** (e.g. engine cycles). | **+** Iteratively copies immediate history.<br>**-** Requires history; safe with clamping. |

    * Note: "mean" imputation relies on global statistics computed during ``fit``. If these statistics are computed on the test set itself, it technically uses future information.*

    Attributes
    ----------
    strategy : Union[str, List[str]]
        The imputation strategy to use. Can be a single string (applied to all
        channels) or a list of strings (one per channel). Must be one of:
        - ``"mean"``: Replaces NaNs with the global mean.
        - ``"locf"``: Last Observation Carried Forward.
        - ``"linear"``: Linear interpolation.
        - ``"zero"``: Replaces NaNs with 0.0.
        - ``"stochastic"``: Stochastic LOCF. Adds random Gaussian noise scaled by
          the channel's std dev.
        - ``"spectral"``: Causal Spectral Extrapolation. Models signal as sum of
          sinusoids based on history.
        - ``"copy_past"``: Blockwise Past Copy. Iteratively copies the
          immediately preceding history segment into the gap.
    apply_to : Optional[List[str]]
        A list of keys in the input dictionary to transform.
    spectral_window_len : int
        Window size (history) used for spectral analysis (default: 64).
    spectral_top_k : int
        Number of dominant frequencies to use for extrapolation (default: 3).
    spectral_match_variance : bool
        If True, the spectral imputation will attempt to match the noise
        variance of the historical signal. Defaults to False.
    fitted_means : Dict[str, np.ndarray]
        Stores computed means (for ``"mean"``).
    fitted_stds : Dict[str, np.ndarray]
        Stores computed std devs (for ``"stochastic"``).
    """

    VALID_STRATEGIES = {
        "mean",
        "locf",
        "linear",
        "zero",
        "stochastic",
        "spectral",
        "copy_past",
    }

    def __init__(
        self,
        strategy: Union[str, List[str]] = "linear",
        apply_to: Optional[List[str]] = None,
        spectral_window_len: int = 64,
        spectral_top_k: int = 3,
        spectral_match_variance: bool = False,
        **kwargs,
    ):
        """
        Initialize the imputation transform with a specific strategy.

        Parameters
        ----------
        strategy : str | list[str], default="linear"
            Imputation strategy or per-channel strategy list.
        apply_to : list[str] | None, optional
            Dictionary keys to transform. If ``None``, all keys are processed.
        spectral_window_len : int, default=64
            History length used by the spectral helper.
        spectral_top_k : int, default=3
            Number of dominant frequencies used by the spectral helper.
        spectral_match_variance : bool, default=False
            Whether the spectral helper should match historical variance.
        **kwargs
            Extra compatibility keywords passed to the parent transform.

        Raises
        ------
        ValueError
            If ``strategy`` is not one of the valid options.
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
        self.spectral_match_variance = spectral_match_variance
        self._spectral_imputer = None  # Lazy instantiation

        # Pattern Match parameters
        self._pattern_imputer = None  # Lazy instantiation

        # Store learned parameters
        self.fitted_means: Dict[str, np.ndarray] = {}
        self.fitted_stds: Dict[str, np.ndarray] = {}

    def fit_data(self, data: NamedTransformInput, metadata: Dict):
        """
        Fit the transform to the data.

        Calculates necessary statistics based on the selected strategies:
        - ``"mean"``: Computes global mean per channel.
        - ``"stochastic"``: Computes global standard deviation per channel.

        Parameters
        ----------
        data : dict
            A dictionary containing input arrays.
        metadata : dict
            Metadata dictionary (unused).

        Returns
        -------
        self
            Returns the instance itself.
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
                    self.fitted_means[key] = np.nan_to_num(
                        np.nanmean(np_val, axis=axes), nan=0.0
                    )

                if uses_std:
                    self.fitted_stds[key] = np.nan_to_num(
                        np.nanstd(np_val, axis=axes), nan=0.0
                    )

        return self

    def _get_spectral_imputer(self) -> SpectralExtrapolationImputation:
        """
        Return the spectral helper instance.

        Returns
        -------
        SpectralExtrapolationImputation
            Lazily constructed spectral helper.
        """
        if self._spectral_imputer is None:
            self._spectral_imputer = SpectralExtrapolationImputation(
                window_len=self.spectral_window_len,
                top_k=self.spectral_top_k,
                match_variance=self.spectral_match_variance,
            )
        return self._spectral_imputer

    def _get_pattern_imputer(self) -> BlockwisePastCopyImputation:
        """
        Return the pattern-matching helper instance.

        Returns
        -------
        BlockwisePastCopyImputation
            Lazily constructed blockwise copy helper.
        """
        if self._pattern_imputer is None:
            self._pattern_imputer = BlockwisePastCopyImputation()
        return self._pattern_imputer

    def _impute_array(self, X: np.ndarray, key: str) -> np.ndarray:
        """
        Impute a single NumPy array with the configured strategy.

        Parameters
        ----------
        X : numpy.ndarray
            Array to impute.
        key : str
            Payload key used to look up fitted statistics.

        Returns
        -------
        numpy.ndarray
            Imputed array with the original dimensionality restored.
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

        # 7. Pattern Matching
        if "copy_past" in strat_map:
            cols = strat_map["copy_past"]
            imputer = self._get_pattern_imputer()

            # Extract and copy subset
            pattern_subset = X_imp[:, :, cols].copy()

            # Delegate to helper
            imputed_subset = imputer.impute(pattern_subset)

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
        Transform the selected payloads by imputing missing values.

        Parameters
        ----------
        data : dict
            Dictionary of input arrays (e.g., ``{'features': ...}``).
        metadata : dict
            Metadata dictionary (unused).

        Returns
        -------
        dict
            Dictionary with imputed arrays.
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
