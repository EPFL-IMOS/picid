import logging
from typing import Any, Dict, List

import numpy as np

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin

logger = logging.getLogger(__name__)


class ConcatenateTransform(DenseTransform, ConcatinatedFitMultiSourceTransformMixin):
    def __init__(self, **kwargs):
        """
        Initialize the transform.

        Args:
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)

    def _convert_to_numpy(self, value: Any, key: str) -> np.ndarray:
        """Convert value to numpy array if it isn't already."""
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value)
                logger.info(f"Converted key '{key}' to numpy array")
            except Exception as e:
                raise ValueError(f"Could not convert key '{key}' to numpy array: {e}")
        return value

    def _ensure_2d(self, value: np.ndarray, key: str) -> np.ndarray:
        """Ensure array is 2D for concatenation."""
        if value.ndim == 1:
            value = value.reshape(-1, 1)
            logger.info(f"Reshaped 1D array '{key}' to 2D: {value.shape}")
        elif value.ndim > 2:
            raise ValueError(
                f"Key '{key}' has unsupported dimensions: {value.ndim}. "
                f"Only 1D and 2D arrays are supported."
            )
        return value

    def _check_length_consistency(
        self, arrays: List[np.ndarray], keys: List[str]
    ) -> int:
        """Check that all arrays have the same number of rows."""
        if not arrays:
            raise ValueError("No arrays to concatenate")

        lengths = [arr.shape[0] for arr in arrays]
        unique_lengths = set(lengths)

        if len(unique_lengths) > 1:
            length_info = {keys[i]: lengths[i] for i in range(len(keys))}
            raise ValueError(f"Length mismatch between arrays: {length_info}")

        return lengths[0]

    def _check_for_nans(self, arrays: List[np.ndarray], keys: List[str]) -> None:
        """Check for NaN values and raise error if found."""
        for arr, key in zip(arrays, keys):
            if np.any(np.isnan(arr)):
                raise ValueError(f"NaN values found in array '{key}'")

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Concatenate all data arrays horizontally.

        Args:
            data: Dictionary containing numpy arrays to concatenate
            metadata: Dictionary containing auxiliary metadata (unused in this transform)

        Returns:
            Concatenated numpy array
        """
        if not data:
            raise ValueError("No data provided for concatenation")

        # Collect arrays to concatenate
        arrays = []
        keys = []

        for key, value in data.items():
            # Convert to numpy array
            value = self._convert_to_numpy(value, key)

            # Ensure 2D shape
            value = self._ensure_2d(value, key)

            arrays.append(value)
            keys.append(key)
            logger.info(f"Added data '{key}' with shape {value.shape}")

        if not arrays:
            raise ValueError("No arrays to concatenate")

        # Check length consistency
        n_rows = self._check_length_consistency(arrays, keys)
        logger.info(f"All arrays have consistent length: {n_rows}")

        # Check for NaN values (will raise error if found)
        self._check_for_nans(arrays, keys)

        # Final concatenation
        result = np.concatenate(arrays, axis=1)
        logger.info(f"Final concatenated shape: {result.shape}")

        return result

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)
