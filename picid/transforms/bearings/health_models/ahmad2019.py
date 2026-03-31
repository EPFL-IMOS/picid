import logging
from typing import Dict, List, Optional, Tuple, override
import numpy as np
from sklearn.linear_model import LinearRegression

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import (
    NoFitPerSegmentMixin,
)
from picid.data.data_objects.utils import convert_to_numpy

logger = logging.getLogger(__name__)


class Ahmad2019FPTCutTransform(NoFitPerSegmentMixin, DenseTransform):
    """
    Cut arrays at the Ahmad et al. first prediction time.

    Slices multiple data arrays to the First Prediction Time (FPT) based on
    the "Alarm bound technique" from Ahmad et al. 2019.

    Parameters
    ----------
    signal_key : str
        Key for the 1D/2D signal array to find the FPT.
    features_keys : list[str]
        List of keys for arrays to be sliced.
    normal_period_end : int, default=100
        End index of the normal operation period.
    window_size : int, default=70
        Size of the sliding window used for regression.
    gain_threshold : float, optional
        Threshold to trigger FPT.
    regression_stride : int, default=1
        Step size for the regression window.
    auto_fpt_strategy : str, default="robust_threshold"
        Strategy used when ``gain_threshold`` is not provided.
    auto_fpt_top_k : int, default=5
        Number of top ratios used by the automatic strategy.
    dataset_name : str, default="UnknownDataset"
        Dataset name used for logging.
    unit_id_key : str, optional
        Optional key used to find the unit identifier.
    fpt_indexes_strategy : str, default="min"
        Strategy used when multiple FPT candidates are found.
    **kwargs
        Additional keyword arguments forwarded to the base transform.
    """

    def __init__(
        self,
        signal_key: str,
        features_keys: List[str],
        normal_period_end: int = 100,
        window_size: int = 70,
        gain_threshold: Optional[float] = None,
        regression_stride: int = 1,
        auto_fpt_strategy: str = "robust_threshold",
        auto_fpt_top_k: int = 5,
        dataset_name: str = "UnknownDataset",
        unit_id_key: Optional[str] = None,
        fpt_indexes_strategy: str = "min",
        **kwargs,
    ):
        """
        Store the FPT search configuration for later slicing.

        Parameters
        ----------
        signal_key : str
            Key for the 1D/2D signal array to find the FPT.
        features_keys : list[str]
            List of keys for arrays to be sliced.
        normal_period_end : int, default=100
            End index of the normal operation period.
        window_size : int, default=70
            Size of the sliding window used for regression.
        gain_threshold : float, optional
            Threshold to trigger FPT.
        regression_stride : int, default=1
            Step size for the regression window.
        auto_fpt_strategy : str, default="robust_threshold"
            Strategy used when ``gain_threshold`` is not provided.
        auto_fpt_top_k : int, default=5
            Number of top ratios used by the automatic strategy.
        dataset_name : str, default="UnknownDataset"
            Dataset name used for logging.
        unit_id_key : str, optional
            Optional key used to find the unit identifier.
        fpt_indexes_strategy : str, default="min"
            Strategy used when multiple FPT candidates are found.
        **kwargs
            Additional keyword arguments forwarded to the base transform.
        """
        super().__init__(**kwargs)
        self.signal_key = signal_key
        self.features_keys = features_keys
        self.normal_period_end = normal_period_end
        self.window_size = window_size
        self.gain_threshold = gain_threshold
        self.regression_stride = regression_stride
        self.auto_fpt_strategy = auto_fpt_strategy
        self.auto_fpt_top_k = auto_fpt_top_k
        self.dataset_name = dataset_name
        self.unit_id_key = unit_id_key
        self.fpt_indexes_strategy = fpt_indexes_strategy

        if not self.features_keys:
            raise ValueError("`features_keys` list cannot be empty.")
        if self.fpt_indexes_strategy not in ["min", "max", "mean"]:
            raise ValueError(
                f"Invalid fpt_indexes_strategy '{self.fpt_indexes_strategy}'."
            )
        if self.auto_fpt_strategy not in ["max_ratio", "robust_threshold"]:
            raise ValueError(f"Invalid auto_fpt_strategy '{self.auto_fpt_strategy}'.")

    def _linear_rectification_technique(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply the linear rectification smoothing used by the paper.

        Parameters
        ----------
        signal : numpy.ndarray
            One-dimensional signal to rectify.

        Returns
        -------
        numpy.ndarray
            Rectified signal.
        """
        n = len(signal)
        if n < 2:
            return signal

        growth_rate = np.nan_to_num(np.mean(np.diff(signal)), nan=0.0)
        smoothed = np.zeros(n)
        smoothed[0] = signal[0]

        for i in range(1, n):
            h_i = signal[i]
            h_i_min = smoothed[i - 1]

            if h_i_min <= h_i <= (1 + growth_rate) * h_i_min:
                smoothed[i] = h_i
            elif h_i < h_i_min or h_i > (1 + growth_rate) * h_i_min:
                smoothed[i] = h_i_min + growth_rate

        return smoothed

    def _find_fpt(self, signal_vector: np.ndarray, unit_id: str) -> int:
        """
        Find the first prediction time for a one-dimensional signal.

        Parameters
        ----------
        signal_vector : numpy.ndarray
            Signal values for a single column.
        unit_id : str
            Unit identifier used in log messages.

        Returns
        -------
        int
            Detected first prediction time.
        """
        if signal_vector.ndim != 1:
            raise ValueError("Signal vector provided to _find_fpt must be 1D.")

        if self.normal_period_end > len(signal_vector):
            logger.warning(
                f"normal_period_end ({self.normal_period_end}) is larger "
                f"than signal length ({len(signal_vector)}). Using available data."
            )

        # Calculate the RMS of the normal period to normalize the signal, as per Ahmad2019
        lin_reg = LinearRegression().fit(
            X=np.arange(self.normal_period_end).reshape(-1, 1),
            y=signal_vector[0 : self.normal_period_end].reshape(-1, 1),
        )

        index = [[x] for x in range(0, self.normal_period_end)]
        values = [[y] for y in signal_vector[0 : self.normal_period_end]]
        lin_reg = LinearRegression().fit(X=index, y=values)
        rms_normal = lin_reg.coef_[0][0]
        rms_normal = 1e-10 if rms_normal < 1e-10 else rms_normal

        health_indicator = signal_vector / rms_normal
        lrt = self._linear_rectification_technique(health_indicator)

        if self.window_size > len(lrt):
            raise ValueError(
                f"Window size {self.window_size} is longer than the "
                f"rectified signal length {len(lrt)}."
            )

        X_reg = np.arange(self.window_size).reshape(-1, 1)
        # Store (end_index, ratio)
        ratio_collector: List[Tuple[int, float]] = []

        # Use the specified regression_stride
        for i in range(
            self.normal_period_end,
            len(lrt) - self.window_size + 1,
            self.regression_stride,
        ):
            end_index = i + self.window_size
            y_reg = lrt[i:end_index].reshape(-1, 1)

            lin_reg = LinearRegression().fit(X=X_reg, y=y_reg)
            slope = lin_reg.coef_[0][0]
            intercept = lin_reg.intercept_[0]

            if intercept < 1e-10:
                ratio_collector.append((end_index, 0.0))
                continue

            gain = abs(slope * self.window_size)
            ratio = gain / intercept
            ratio_collector.append((end_index, ratio))

        # --- FPT Logic: Apply automatic or fixed threshold ---
        if not ratio_collector:
            raise ValueError(
                f"Ahmad2019 FPT not found for signal '{self.signal_key}'. "
                "The regression failed or the intercept was near zero for all windows."
            )

        # 1. Automatic Mode: Use the specified auto_fpt_strategy
        if self.gain_threshold is None:
            if self.auto_fpt_strategy == "max_ratio":
                fpt_index, max_ratio = max(ratio_collector, key=lambda x: x[1])
                logger.info(
                    f"Ahmad2019 FPT (Auto 'max_ratio') found at index {fpt_index} "
                    f"with max ratio {max_ratio:.4f}."
                )
                return fpt_index

            elif self.auto_fpt_strategy == "robust_threshold":
                k = min(self.auto_fpt_top_k, len(ratio_collector))
                sorted_ratios = sorted(
                    ratio_collector, key=lambda x: x[1], reverse=True
                )
                top_k_ratios = sorted_ratios[:k]
                # calculate average of the top-k ratios to determine a robust threshold, or use the minimum of the top-k ratios

                robust_threshold = np.mean([x[1] for x in top_k_ratios])

                # Find the first index that crosses this robust threshold
                for end_index, ratio in ratio_collector:
                    if ratio >= robust_threshold:
                        logger.info(
                            f"Ahmad2019 FPT (Auto 'robust_threshold') found at index {end_index}. "
                            f"(First point to cross robust threshold={robust_threshold:.4f}, "
                            f"derived from top {k} ratios)."
                        )
                        return end_index

                # Should be unreachable if ratio_collector is not empty
                return ratio_collector[0][0]

        # 2. Fixed Threshold Mode: Find the first index to cross the threshold
        else:
            fpt_candidates = [
                idx for idx, ratio in ratio_collector if ratio > self.gain_threshold
            ]

            if fpt_candidates:
                fpt_index = min(fpt_candidates)  # Get first prediction time
                logger.info(
                    f"Ahmad2019 FPT (Fixed) found at index {fpt_index} "
                    f"(ratio crossed {self.gain_threshold})."
                )
                return fpt_index

            # 3. Fixed Threshold Mode: Threshold was never met
            max_ratio = max(ratio_collector, key=lambda x: x[1])[1]
            min_ratio = min(ratio_collector, key=lambda x: x[1])[1]
            avg_ratio = np.mean([r[1] for r in ratio_collector])

            raise ValueError(
                f"Ahmad2019 FPT not found for signal '{self.signal_key}' "
                f"for unit '{unit_id}' in dataset '{self.dataset_name}'. "
                f"The sliding window regression never met the 'gain_threshold' "
                f"(currently {self.gain_threshold}). "
                f"Observed Ratios: max={max_ratio:.4f}, min={min_ratio:.4f}, "
                f"avg={avg_ratio:.4f}. Consider adjusting 'normal_period_end' "
                f"({self.normal_period_end}), 'window_size' ({self.window_size}), "
                "or 'gain_threshold' (or set to None for automatic mode)."
            )

    def _get_unit_id(self, data: NamedTransformInput, metadata: Dict) -> str:
        """
        Extract a printable unit identifier for logging.

        Parameters
        ----------
        data : NamedTransformInput
            Input data dictionary.
        metadata : dict
            Metadata dictionary, preserved for interface compatibility.

        Returns
        -------
        str
            Printable unit identifier.
        """
        if self.unit_id_key:
            if self.unit_id_key in data:
                # Handle array or scalar unit_id
                unit_id_data = data[self.unit_id_key]
                return str(unit_id_data)
        return "UnknownUnit"

    @override
    def transform_data(
        self, data: NamedTransformInput, metadata: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Find the FPT and slice every configured feature array.

        Parameters
        ----------
        data : NamedTransformInput
            Input data dictionary.
        metadata : dict
            Metadata dictionary, preserved for interface compatibility.

        Returns
        -------
        dict[str, numpy.ndarray]
            Sliced feature arrays.
        """
        unit_id = self._get_unit_id(data, metadata)

        if self.signal_key not in data:
            raise KeyError(
                f"signal_key '{self.signal_key}' not found in data. "
                f"Available keys: {data.keys()}"
            )

        signal_vector = convert_to_numpy(data[self.signal_key], ensure_2d=True)
        if signal_vector.ndim == 1:
            signal_vector = signal_vector.reshape(-1, 1)

        # 1. Find FPT index for each signal column
        fpt_indexes = []
        for col_idx in range(signal_vector.shape[1]):
            col_signal = signal_vector[:, col_idx]
            try:
                _fpt_index = self._find_fpt(col_signal, unit_id)
                fpt_indexes.append(_fpt_index)
            except ValueError as e:
                logger.error(
                    f"Failed to find FPT for col {col_idx} "
                    f"of '{self.signal_key}'. Error: {e}"
                )

        if not fpt_indexes:
            raise ValueError(
                f"Could not find FPT for any column in '{self.signal_key}'. "
                "Check signal data and transform parameters."
            )

        # 2. Apply strategy to get a single FPT index
        if len(fpt_indexes) > 1:
            if self.fpt_indexes_strategy == "min":
                fpt_index = min(fpt_indexes)
            elif self.fpt_indexes_strategy == "max":
                fpt_index = max(fpt_indexes)
            else:  # "mean"
                fpt_index = int(np.mean(fpt_indexes))
        else:
            fpt_index = fpt_indexes[0]

        logger.info(
            f"Final FPT index for unit '{unit_id}' set to {fpt_index} "
            f"using strategy '{self.fpt_indexes_strategy}' "
            f"from indexes {fpt_indexes}."
        )

        # 3. Slice all requested feature arrays
        sliced_data_dict: Dict[str, np.ndarray] = {}
        for key in self.features_keys:
            if key not in data:
                raise KeyError(
                    f"features_key '{key}' not found in data. "
                    f"Available keys: {data.keys()}"
                )

            if key == self.unit_id_key:
                logger.debug(f"Passing through key '{key}' without slicing.")
                sliced_data_dict[key] = data[key]
                continue

            features_array = convert_to_numpy(data[key], ensure_2d=False)

            if features_array.ndim < 1:
                raise ValueError(f"Features array from '{key}' must be at least 1D.")

            if signal_vector.shape[0] != features_array.shape[0]:
                raise ValueError(
                    f"Length of signal '{self.signal_key}' ({signal_vector.shape[0]}) "
                    f"does not match number of rows in '{key}' "
                    f"({features_array.shape[0]})."
                )

            sliced_features = features_array[fpt_index:, ...]

            if sliced_features.shape[0] == 0:
                logger.warning(
                    f"FPT cut at index {fpt_index} for key '{key}' "
                    "resulted in an empty array."
                )

            sliced_data_dict[key] = sliced_features

        return sliced_data_dict

    def __call__(self, data: Dict, metadata: Dict) -> Dict[str, np.ndarray]:
        return self.transform_data(data, metadata)
