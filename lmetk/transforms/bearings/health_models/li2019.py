import logging
from typing import Any, Dict, List, Optional, Tuple, override
import numpy as np

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.data.data_objects.utils import convert_to_numpy
from lmetk.transforms.base.multisource import FitMultiSourceTransformMixin

logger = logging.getLogger(__name__)


class Li2019FPTCutTransform(DenseTransform, FitMultiSourceTransformMixin):
    """
    Slices multiple data arrays to the First Prediction Time (FPT) based on
    the "2*sigma interval of kurtosis" method from Li et al. 2019.
    ...
    """

    def __init__(
        self,
        signal_key: str,
        features_keys: List[str],
        normal_period_start: int = 50,
        normal_period_end: int = 150,
        sigma_interval: float = 2.0,
        scan_start_index: int = 150,
        **kwargs,
    ):
        """
        Initialize the Li2019FPTCutTransform.
        ...
        """
        super().__init__(**kwargs)
        self.signal_key = signal_key
        self.features_keys = features_keys
        self.normal_period = range(normal_period_start, normal_period_end)
        self.sigma_interval = sigma_interval
        self.scan_start_index = scan_start_index

        if not self.features_keys:
            raise ValueError("`features_keys` list cannot be empty.")

    def _find_fpt(self, signal_vector: np.ndarray) -> int:
        """NumPy-based implementation of li_et_al_2019."""

        if signal_vector.ndim != 1:
            raise ValueError(f"Signal from '{self.signal_key}' must be 1D.")

        if self.normal_period.stop > len(signal_vector):
            logger.warning(
                f"normal_period_end ({self.normal_period.stop}) is larger "
                f"than signal length ({len(signal_vector)}). Using available data."
            )

        normal_data = signal_vector[self.normal_period.start : self.normal_period.stop]

        if normal_data.size == 0:
            raise ValueError(
                f"Normal period {self.normal_period} results in an empty "
                "array. Check start/end indices."
            )

        mean = np.mean(normal_data)
        std_dev = np.std(normal_data)
        threshold = self.sigma_interval * std_dev

        centered_signal_abs = np.abs(signal_vector - mean)
        n = centered_signal_abs.size

        for i in range(self.scan_start_index, n):
            if (
                centered_signal_abs[i - 1] > threshold
                and centered_signal_abs[i] > threshold
            ):
                logger.info(f"Li2019 FPT found at index {i} for {self.signal_key}.")
                return i

        # --- MODIFIED SECTION ---
        # If the loop finishes without finding an FPT, raise an error.
        raise ValueError(
            f"Li2019 FPT not found for signal '{self.signal_key}'. "
            f"The signal never crossed the threshold (mean + {self.sigma_interval}*std) "
            f"after index {self.scan_start_index}. "
            f"Consider adjusting 'normal_period_start' (currently {self.normal_period.start}), "
            f"'normal_period_end' (currently {self.normal_period.stop}), "
            f"or 'sigma_interval' (currently {self.sigma_interval})."
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
