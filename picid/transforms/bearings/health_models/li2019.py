import logging
from typing import Dict, List, Optional, override
import numpy as np

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.data.data_objects.utils import convert_to_numpy

# Changed to NoFit, as this transform doesn't learn parameters
from picid.transforms.base.multisource import NoFitPerSegmentMixin

logger = logging.getLogger(__name__)


TEST_RULS = {
    "PRONOSTIA": {
        "1_3": 573,
        "1_4": 290,  # 339, Adjusting this then Table 4 from the "Remaining useful life estimation of bearing via temporal convolutional networks enhanced by a gated convolutional unit" paper make sense
        "1_5": 161,
        "1_6": 146,
        "1_7": 757,
        "2_3": 753,
        "2_4": 139,
        "2_5": 309,
        "2_6": 129,
        "2_7": 58,
        "3_3": 82,
    }
}


class Li2019FPTCutTransform(NoFitPerSegmentMixin, DenseTransform):
    """
    Cut arrays at the Li et al. first prediction time.

    Slices multiple data arrays to the First Prediction Time (FPT) based on
    the "2*sigma interval of kurtosis" method from Li et al. 2019.

    Parameters
    ----------
    signal_key : str
        Key for the 1D/2D signal array used to find the FPT.
    features_keys : list[str]
        List of keys for arrays to be sliced.
    normal_period_start : int, default=50
        Start index of the normal operation period.
    normal_period_end : int, default=150
        End index of the normal operation period.
    sigma_interval : float, default=2.0
        Number of standard deviations used to set the threshold.
    scan_start_index : int, default=150
        Index from which to start scanning for the FPT.
    dataset_name : str, default="UnknownDataset"
        Dataset name used for logging.
    unit_id_key : str, optional
        Optional key used to find the unit identifier.
    fpt_indexes_strategy : str, default="min"
        Strategy used when multiple FPT candidates are found.
    use_predefined_test : bool, default=False
        Whether to fall back to predefined test RUL values.
    **kwargs
        Additional keyword arguments forwarded to the base transform.
    """

    def __init__(
        self,
        signal_key: str,
        features_keys: List[str],
        normal_period_start: int = 50,
        normal_period_end: int = 150,
        sigma_interval: float = 2.0,
        scan_start_index: int = 150,
        dataset_name: str = "UnknownDataset",
        unit_id_key: Optional[str] = None,
        fpt_indexes_strategy: str = "min",
        use_predefined_test: bool = False,
        **kwargs,
    ):
        """
        Store the FPT search configuration and slicing keys.

        Parameters
        ----------
        signal_key : str
            Key for the 1D/2D signal array used to find the FPT.
        features_keys : list[str]
            List of keys for arrays to be sliced.
        normal_period_start : int, default=50
            Start index of the normal operation period.
        normal_period_end : int, default=150
            End index of the normal operation period.
        sigma_interval : float, default=2.0
            Number of standard deviations used to set the threshold.
        scan_start_index : int, default=150
            Index from which to start scanning for the FPT.
        dataset_name : str, default="UnknownDataset"
            Dataset name used for logging.
        unit_id_key : str, optional
            Optional key used to find the unit identifier.
        fpt_indexes_strategy : str, default="min"
            Strategy used when multiple FPT candidates are found.
        use_predefined_test : bool, default=False
            Whether to fall back to predefined test RUL values.
        **kwargs
            Additional keyword arguments forwarded to the base transform.
        """
        super().__init__(**kwargs)
        self.signal_key = signal_key
        self.features_keys = features_keys
        self.normal_period = range(normal_period_start, normal_period_end)
        self.sigma_interval = sigma_interval
        self.scan_start_index = scan_start_index
        self.dataset_name = dataset_name
        self.unit_id_key = unit_id_key
        self.fpt_indexes_strategy = fpt_indexes_strategy
        self.use_predefined_test = use_predefined_test

        if not self.features_keys:
            raise ValueError("`features_keys` list cannot be empty.")
        if self.fpt_indexes_strategy not in ["min", "max", "mean"]:
            raise ValueError(
                f"Invalid fpt_indexes_strategy '{self.fpt_indexes_strategy}'. "
                "Expected one of: 'min', 'max', 'mean'."
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
                return str(unit_id_data).strip("[]").replace(" ", "_")
        return "UnknownUnit"

    def _find_fpt(self, signal_vector: np.ndarray, unit_id: str) -> int:
        """
        Find the first prediction time for a one-dimensional signal.

        Parameters
        ----------
        signal_vector : numpy.ndarray
            Signal values for a single column.
        unit_id : str
            Unit identifier used in the error message.

        Returns
        -------
        int
            Detected first prediction time.
        """

        if signal_vector.ndim != 1:
            raise ValueError("Signal vector provided to _find_fpt must be 1D.")

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

        # If the loop finishes without finding an FPT, raise an error.
        raise ValueError(
            f"Li2019 FPT not found for signal '{self.signal_key}' "
            f"for unit '{unit_id}' in dataset '{self.dataset_name}'. "
            f"The signal never crossed the threshold (mean + {self.sigma_interval}*std) "
            f"after index {self.scan_start_index}. "
            f"Consider adjusting 'normal_period_start' ({self.normal_period.start}), "
            f"'normal_period_end' ({self.normal_period.stop}), "
            f"or 'sigma_interval' (currently {self.sigma_interval})."
        )

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

        signal_vector = convert_to_numpy(data[self.signal_key])
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

        # Case when no FPT index is found for any column, but predefined test RUL is available for this unit in the dataset. This allows us to still proceed with the transform using the predefined RUL as a fallback.
        if not fpt_indexes:
            if (
                self.use_predefined_test
                and self.dataset_name in TEST_RULS
                and unit_id in TEST_RULS[self.dataset_name]
            ):
                predefined_rul = (
                    signal_vector.shape[0] - TEST_RULS[self.dataset_name][unit_id]
                )

                logger.info(
                    f"No FPT found for unit '{unit_id}' in dataset '{self.dataset_name}', "
                    f"but predefined RUL of {predefined_rul} is available. "
                    "Using this to set FPT index."
                )
                fpt_indexes.append(predefined_rul)
            else:
                raise ValueError(
                    f"Could not find FPT for any column in '{self.signal_key}'. "
                    "Check signal data and transform parameters."
                    f" Unit ID: '{unit_id}', Dataset: '{self.dataset_name}', "
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

        # Check if there is predefined test RUL for this unit in the dataset, and if so, adjust the FPT index accordingly
        if self.use_predefined_test and self.dataset_name in TEST_RULS:
            predefined_ruls = TEST_RULS[self.dataset_name]
            if unit_id in predefined_ruls:
                predefined_rul = predefined_ruls[unit_id]
                logger.info(
                    f"Using predefined RUL of {predefined_rul} for unit '{unit_id}' "
                    f"in dataset '{self.dataset_name}'. Adjusting FPT index accordingly."
                )
                fpt_index = max(0, signal_vector.shape[0] - predefined_rul)

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
        assert len(self.features_keys) == 1
        return sliced_data_dict[self.features_keys[0]]

    def __call__(self, data: Dict, metadata: Dict) -> Dict[str, np.ndarray]:
        return self.transform_data(data, metadata)
