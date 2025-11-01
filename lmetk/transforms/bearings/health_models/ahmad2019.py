import logging
from typing import Any, Dict, List, Optional, Tuple, override
import numpy as np
from sklearn.linear_model import LinearRegression

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import FitMultiSourceTransformMixin
from lmetk.data.data_objects.utils import convert_to_numpy

logger = logging.getLogger(__name__)


class Ahmad2019FPTCutTransform(DenseTransform, FitMultiSourceTransformMixin):
    """
    Slices multiple data arrays to the First Prediction Time (FPT) based on
    the "Alarm bound technique" from Ahmad et al. 2019.
    ...
    """

    def __init__(
        self,
        signal_key: str,
        features_keys: List[str],
        normal_period_end: int = 100,
        window_size: int = 70,
        gain_threshold: float = 0.1,
        **kwargs,
    ):
        """
        Initialize the Ahmad2019FPTCutTransform.
        ...
        """
        super().__init__(**kwargs)
        self.signal_key = signal_key
        self.features_keys = features_keys
        self.normal_period_end = normal_period_end
        self.window_size = window_size
        self.gain_threshold = gain_threshold

        if not self.features_keys:
            raise ValueError("`features_keys` list cannot be empty.")

    def _linear_rectification_technique(self, signal: np.ndarray) -> np.ndarray:
        """NumPy-based implementation of linear_rectification_technique."""
        n = len(signal)
        if n < 2:
            return signal

        growth_rate = np.mean(np.diff(signal))
        smoothed = np.zeros(n)
        smoothed[0] = signal[0]

        for i in range(1, n):
            h_i = signal[i]
            h_i_min = smoothed[i - 1]

            if h_i_min <= h_i <= (1 + growth_rate) * h_i_min:
                smoothed[i] = h_i
            elif (h_i < h_i_min) or (h_i > (1 + growth_rate) * h_i_min):
                smoothed[i] = h_i_min + growth_rate

        return smoothed

    def _find_fpt(self, signal_vector: np.ndarray) -> int:
        """NumPy-based implementation of ahmad_et_al_2019."""
        if signal_vector.ndim != 1:
            raise ValueError(f"Signal from '{self.signal_key}' must be 1D.")

        if self.normal_period_end > len(signal_vector):
            logger.warning(
                f"normal_period_end ({self.normal_period_end}) is larger "
                f"than signal length ({len(signal_vector)}). Using available data."
            )

        rms_normal = np.mean(signal_vector[0 : self.normal_period_end])

        if rms_normal < 1e-10:
            logger.warning(
                "Mean of normal period is near zero. " "Normalized HI may be unstable."
            )
            rms_normal = 1e-10

        health_indicator = signal_vector / rms_normal
        lrt = self._linear_rectification_technique(health_indicator)

        if self.window_size > len(lrt):
            raise ValueError(
                f"Window size {self.window_size} is longer than the "
                f"signal length {len(lrt)}."
            )

        X_reg = np.arange(self.window_size).reshape(-1, 1)

        for i in range(len(lrt) - self.window_size):
            start_index = i
            end_index = i + self.window_size
            y_reg = lrt[start_index:end_index].reshape(-1, 1)

            try:
                lin_reg = LinearRegression().fit(X=X_reg, y=y_reg)
                slope = lin_reg.coef_[0][0]
                intercept = lin_reg.intercept_[0]
            except Exception as e:
                logger.warning(f"Linear regression failed at index {i}: {e}")
                continue

            gain = abs(slope * self.window_size)
            if intercept < 1e-10:
                continue

            if (gain / intercept) > self.gain_threshold:
                logger.info(f"Ahmad2019 FPT found at index {end_index}.")
                return end_index

        # --- MODIFIED SECTION ---
        # If the loop finishes without finding an FPT, raise an error.
        raise ValueError(
            f"Ahmad2019 FPT not found for signal '{self.signal_key}'. "
            f"The sliding window regression never met the 'gain_threshold' "
            f"(currently {self.gain_threshold}). "
            f"Consider adjusting 'normal_period_end' (currently {self.normal_period_end}), "
            f"'window_size' (currently {self.window_size}), or 'gain_threshold'."
        )
        # --- END MODIFIED SECTION ---

    @override
    def transform_data(
        self, data: NamedTransformInput, metadata: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Finds FPT, slices all feature arrays, and returns them in a dict.
        """

        if self.signal_key not in data:
            raise KeyError(
                f"signal_key '{self.signal_key}' not found in data. "
                f"Available keys: {data.keys()}"
            )

        signal_vector = convert_to_numpy(data[self.signal_key]).flatten()

        # 1. Find the First Prediction Time (FPT) index
        fpt_index = self._find_fpt(signal_vector)

        sliced_data_dict: Dict[str, np.ndarray] = {}

        for key in self.features_keys:
            if key not in data:
                raise KeyError(
                    f"features_key '{key}' not found in data. "
                    f"Available keys: {data.keys()}"
                )

            features_array = convert_to_numpy(data[key])

            if features_array.ndim < 1:  # Allow 1D or 2D
                raise ValueError(f"Features array from '{key}' must be at least 1D.")

            if len(signal_vector) != features_array.shape[0]:
                raise ValueError(
                    f"Length of signal '{self.signal_key}' ({len(signal_vector)}) does not "
                    f"match number of rows in '{key}' ({features_array.shape[0]})."
                )

            # 2. Slice the features array from that index
            sliced_features = features_array[fpt_index:, ...]  # Use ellipsis for n-dim

            if sliced_features.shape[0] == 0:
                logger.warning(
                    f"FPT cut at index {fpt_index} for key '{key}' "
                    "resulted in an empty array."
                )

            sliced_data_dict[key] = sliced_features

        # 3. Return the dictionary of sliced arrays
        return sliced_data_dict

    def __call__(self, data: Dict, metadata: Dict) -> Dict[str, np.ndarray]:
        return self.transform_data(data, metadata)
