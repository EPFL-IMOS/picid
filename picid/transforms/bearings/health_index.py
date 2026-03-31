import logging
from typing import Dict, List, Optional, Tuple, override, Union
import numpy as np

from picid.transforms.base.base_transform import DenseTransform
from picid.data.data_objects import NamedTransformInput
from picid.data.data_objects.utils import convert_to_numpy

from picid.transforms.base.multisource import (
    NoFitPerSegmentMixin,
    InverseTransformMixin,
)


logger = logging.getLogger(__name__)

# --- Type Aliases ---
UnitKey = Union[Tuple[int, int], str]
UnitLifeLookup = Dict[UnitKey, float]
DatasetLifeLookup = Dict[str, UnitLifeLookup]

# --- Default Lookup Dictionary ---
DEFAULT_TOTAL_LIFE_LOOKUP: DatasetLifeLookup = {
    "PRONOSTIA": {
        (1, 1): 28020.0,
        (1, 2): 8700.0,
        (1, 3): 18010.0 + 5730.0,
        (1, 4): 11280.0 + 2900.0,
        (1, 5): 23010.0 + 1610.0,
        (1, 6): 23010.0 + 1460.0,
        (1, 7): 15010.0 + 7570.0,
        (2, 1): 9100.0,
        (2, 2): 7960.0,
        (2, 3): 12010.0 + 7530.0,
        (2, 4): 6110.0 + 1390.0,
        (2, 5): 20010.0 + 3090.0,
        (2, 6): 5710.0 + 1290.0,
        (2, 7): 1710.0 + 580.0,
        (3, 1): 5140.0,
        (3, 2): 16360.0,
        (3, 3): 3510.0 + 820.0,
    },
    "XJTU-SY": {
        (1, 1): 123.0,
        (1, 2): 161.0,
        (1, 3): 158.0,
        (1, 4): 122.0,
        (1, 5): 52.0,
        (2, 1): 491.0,
        (2, 2): 161.0,
        (2, 3): 533.0,
        (2, 4): 42.0,
        (2, 5): 339.0,
        (3, 1): 2538.0,
        (3, 2): 2496.0,
        (3, 3): 371.0,
        (3, 4): 1515.0,
        (3, 5): 114.0,
    },
}

DECREASE_PERIOD = {"PRONOSTIA": 2560, "XJTU-SY": 32768}


def diff_at_step_n(vector, n):
    """
    Compute the difference between samples separated by a fixed step.

    Parameters
    ----------
    vector : numpy.ndarray
        One-dimensional sequence to compare.
    n : int
        Step size used to build the two aligned slices.

    Returns
    -------
    numpy.ndarray
        Difference between the forward-shifted and base slices.
    """

    # This slices the vector into:
    # [v[n], v[2n], v[3n], ...]
    v_i_plus_n = vector[n::n]

    # This slices the vector into:
    # [v[0], v[n], v[2n], ...]
    # We use -n to ensure this vector is the same
    # length as the one above.
    v_i = vector[:-n:n]

    return v_i_plus_n - v_i


class HealthIndexTransform(NoFitPerSegmentMixin, InverseTransformMixin, DenseTransform):
    """
    Convert a runtime channel into a unit-specific health index.

    The transform uses a dataset/unit lifetime lookup to map runtime values
    into a normalized health index and exposes the inverse mapping for
    downstream inspection or reconstruction.

    Parameters
    ----------
    runtime_key : str
        Key used to read the runtime vector from ``data``.
    unit_key : str
        Key used to read the unit identifier from ``metadata``.
    dataset_name : str
        Dataset name used to select the correct lifetime table.
    total_life_lookup : dict, optional
        Nested mapping of dataset name to unit lifetime values.
    **kwargs
        Additional keyword arguments forwarded to the base transform.

    Notes
    -----
    The forward mapping is ``HI = Runtime / Total_Life`` and the inverse
    mapping implemented here is ``Runtime = HI * Total_Life``. The transform
    emits a single-column array of shape ``(n_rows, 1)`` and validates that
    the computed health index stays within the ``[0, 1]`` range.
    """

    def __init__(
        self,
        runtime_key: str,
        unit_key: str,
        dataset_name: str,
        total_life_lookup: Optional[DatasetLifeLookup] = None,
        **kwargs,
    ):
        """
        Initialize the lifetime lookup and runtime key mapping.

        Parameters
        ----------
        runtime_key : str
            Key used to read the runtime vector from ``data``.
        unit_key : str
            Key used to read the unit identifier from ``metadata``.
        dataset_name : str
            Dataset name used to select the correct lifetime table.
        total_life_lookup : dict, optional
            Nested mapping of dataset name to unit lifetime values.
        **kwargs
            Additional keyword arguments forwarded to the base transform.
        """
        super().__init__(**kwargs)
        self.runtime_key = runtime_key
        self.unit_key = unit_key
        self.dataset_name = dataset_name

        if total_life_lookup is None:
            self.total_life_lookup = DEFAULT_TOTAL_LIFE_LOOKUP
            logger.info(
                "total_life_lookup not provided, using DEFAULT_TOTAL_LIFE_LOOKUP."
            )
        else:
            self.total_life_lookup = total_life_lookup

        if self.dataset_name not in self.total_life_lookup:
            raise KeyError(
                f"dataset_name '{self.dataset_name}' not found in "
                f"total_life_lookup. Available keys: {self.total_life_lookup.keys()}"
            )

        self.dataset_unit_lives = self.total_life_lookup[self.dataset_name]

        logger.debug(
            f"HealthIndexTransform initialized: "
            f"runtime_key='{self.runtime_key}', "
            f"unit_key='{self.unit_key}' (in metadata), "
            f"dataset_name='{self.dataset_name}'"
        )

    # def _get_total_life(self, unit_id: UnitKey) -> float:
    #     """Helper function to look up and validate the Total_Life."""
    #     if unit_id not in self.dataset_unit_lives:
    #         raise KeyError(
    #             f"Unit ID '{unit_id}' not found in total_life_lookup "
    #             f"for dataset '{self.dataset_name}'. "
    #             f"Available unit keys: {self.dataset_unit_lives.keys()}"
    #         )

    #     total_life = self.dataset_unit_lives[unit_id]

    #     if total_life <= 0:
    #         raise ValueError(
    #             f"Total_Life for dataset '{self.dataset_name}', unit '{unit_id}' "
    #             f"must be a positive number, but got {total_life}"
    #         )

    #     return total_life

    def _get_total_life_from_metadata(
        self, unit_id: Union[np.ndarray, List, Tuple]
    ) -> Union[float, np.ndarray]:
        """
        Resolve the lifetime value for one unit or a batch of units.

        Parameters
        ----------
        unit_id : numpy.ndarray or list or tuple
            Single unit identifier or a vectorized 2D array of identifiers.

        Returns
        -------
        float or numpy.ndarray
            Lifetime for the requested unit(s).
        """

        # --- VECTORIZED CASE (2D NumPy Array) ---
        if isinstance(unit_id, np.ndarray) and unit_id.ndim == 2:
            # Convert [[1,2], [1,2]] -> [(1,2), (1,2)]
            unit_keys = [tuple(row) for row in unit_id]

            # 1. Validate all keys exist first (No try/except)
            missing_keys = [k for k in unit_keys if k not in self.dataset_unit_lives]

            if missing_keys:
                # Log first few missing keys to avoid spamming logs
                unique_missing = list(set(missing_keys))[:5]
                raise KeyError(
                    f"Vectorized lookup failed. The following units were not found in "
                    f"dataset '{self.dataset_name}': {unique_missing}..."
                    f"(Total missing: {len(missing_keys)})"
                )

            # 2. Retrieve values
            lives = [self.dataset_unit_lives[k] for k in unit_keys]
            return np.array(lives)

        # --- SINGLE CASE (1D Array, List, or Tuple) ---

        # Convert 1D array to tuple
        if isinstance(unit_id, np.ndarray):
            unit_id = tuple(unit_id)
        # Convert list to tuple
        elif isinstance(unit_id, list):
            unit_id = tuple(unit_id)

        # Standard dictionary check
        if unit_id not in self.dataset_unit_lives:
            raise KeyError(
                f"Unit ID '{unit_id}' not found in dataset '{self.dataset_name}'. "
                f"Available keys: {list(self.dataset_unit_lives.keys())}"
            )

        total_life = self.dataset_unit_lives[unit_id]

        if total_life <= 0:
            raise ValueError(f"Total_Life must be positive, got {total_life}")

        return total_life

    @override
    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Map the runtime channel to a column vector of health index values.

        Parameters
        ----------
        data : NamedTransformInput
            Container holding the runtime vector.
        metadata : dict
            Metadata that identifies the unit being processed.

        Returns
        -------
        numpy.ndarray
            Two-dimensional array with shape ``(n_rows, 1)``.
        """
        # 1. Get Runtime vector from data
        if self.runtime_key not in data:
            raise KeyError(
                f"runtime_key '{self.runtime_key}' not found in data. "
                f"Available data keys: {data.keys()}"
            )
        runtime = data[self.runtime_key]
        init_shape = runtime.shape
        runtime_vector = convert_to_numpy(runtime).flatten()

        if runtime_vector.ndim != 1:
            raise ValueError(
                f"data['{self.runtime_key}'] must be a 1D vector, "
                f"but has shape {runtime_vector.shape}"
            )
        if np.any(np.isinf(runtime_vector)):
            raise ValueError(f"Infinite values found in '{self.runtime_key}'")

        # 2. Get Total_Life from metadata
        unit_id = data.get(self.unit_key)  # For error message
        total_life = self._get_total_life_from_metadata(unit_id)

        # 3. Calculate Health Index vector
        # HI = 1 - (Runtime / Total_Life)
        # As the Runtime goes from Total_Life left to 0 then we use (Runtime / Total_Life), hence HI goes from 1 to 0 as expected
        health_index_vector = runtime_vector / (total_life)

        # 4. Validate the HI is within the [0.0, 1.0] range
        min_hi = np.nanmin(health_index_vector)
        max_hi = np.nanmax(health_index_vector)

        min_hi_init = np.nanmin(runtime_vector)
        max_hi_init = np.nanmax(runtime_vector)

        # Find indexes of min and max values for health_index_vector
        agrmin = np.argmin(health_index_vector)
        argmax = np.argmax(health_index_vector)

        # Check that it is a monotonically decreasing function
        # 1) identify the step of decrease (how many records have the same value)
        step_decrease = diff_at_step_n(
            health_index_vector, n=DECREASE_PERIOD[self.dataset_name]
        )

        if not np.all(step_decrease <= 0):
            raise ValueError(
                f"Calculated HI for unit '{unit_id}' is not a monotonically "
                f"decreasing function. Check the runtime data in "
                f"'{self.runtime_key}'."
            )

        logger.info(
            f"Unit {unit_id} HI Validation: "
            f"Range=[{np.round(min_hi, 3)}, {np.round(max_hi, 3)}]. "
            f"Min HI at index {agrmin}, Max HI at index {argmax}. "
            f"Original runtime range=[{min_hi_init}, {max_hi_init}]."
        )

        if min_hi < 0.0 or max_hi > 1.0:
            raise ValueError(
                f"Calculated HI for unit '{unit_id}' is outside the valid "
                f"[0.0, 1.0] range. Found min={min_hi}, max={max_hi}. "
                f"Check if runtime data in '{self.runtime_key}' (max={np.nanmax(runtime_vector)}) "
                f"exceeds Total_Life ({total_life}) or is negative."
            )

        # 5. Reshape to (n_rows, 1)
        hi_column = health_index_vector.reshape(-1, 1)

        logger.debug(
            f"HealthIndexTransform result shape: {hi_column.shape} "
            f"(Unit: {unit_id}, Total_Life: {total_life})"
        )
        #
        assert (
            init_shape == hi_column.shape
        ), f"Shapes of initial {self.runtime_key} do not match after transform"

        return hi_column

    @override
    def inverse_transform(
        self,
        data: NamedTransformInput,
        metadata: dict = None,
    ) -> np.ndarray:
        """
        Reconstruct runtime values from the health index output.

        The inverse mapping is ``Runtime = HI * Total_Life``.

        Parameters
        ----------
        data : NamedTransformInput
            Output of :meth:`transform_data`.
        metadata : dict, optional
            Metadata containing the unit identifier needed for the lookup.

        Returns
        -------
        numpy.ndarray
            Reconstructed runtime as a two-dimensional column vector.
        """
        if self.unit_key not in metadata:
            raise ValueError(
                f"metadata must contain '{self.unit_key}' for inverse_transform."
            )

        # 1. Get Total_Life from metadata
        total_life = self._get_total_life_from_metadata(metadata["unit_id"])

        # 2. Get HI vector from data dict
        if not data:
            raise ValueError("No data provided to inverse_transform.")
        if len(data) > 1:
            logger.warning(
                f"inverse_transform received {len(data)} keys. "
                f"Only using the first key: '{list(data.keys())[0]}'"
            )

        hi_vector = convert_to_numpy(list(data.values())[0]).flatten()

        if hi_vector.ndim != 1:
            raise ValueError(
                f"Input HI data must be a 1D vector, "
                f"but has shape {hi_vector.shape}"
            )

        # 3. Calculate Runtime
        # Runtime = HI * Total_Life
        runtime_vector = hi_vector * total_life

        # 4. Reshape to (n_rows, 1)
        runtime_column = runtime_vector.reshape(-1, 1)

        return runtime_column

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)

    def get_feature_names(
        self, input_keys: List[str], input_shapes: Dict[str, Tuple[int, int]]
    ) -> List[str]:
        """
        Return the generated feature name for the health index column.

        Parameters
        ----------
        input_keys : list[str]
            Input feature names, preserved for interface compatibility.
        input_shapes : dict[str, tuple[int, int]]
            Input shapes, preserved for interface compatibility.

        Returns
        -------
        list[str]
            Generated output feature name.
        """
        return [f"HI_ds_{self.dataset_name}_from_{self.runtime_key}"]
