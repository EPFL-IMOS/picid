import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin

logger = logging.getLogger(__name__)


class CutToFirstZero(BaseTransform, ConcatinatedFitMultiSourceTransformMixin):
    def __init__(self, key: str, **kwargs):
        """
        Initialize the transform.

        Args:
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.key: str = key

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

    def _check_for_nans(self, arrays: List[np.ndarray], keys: List[str]) -> None:
        """Check for NaN values and raise error if found."""
        for arr, key in zip(arrays, keys):
            if np.any(np.isnan(arr)):
                raise ValueError(f"NaN values found in array '{key}'")

    def transform_data(self, data: Dict, metadata: Dict) -> np.ndarray:
        """
        Filter targets correponding to the stftt

        Args:
            data: Dictionary containing numpy arrays to concatenate
            metadata: Dictionary containing auxiliary metadata (unused in this transform)

        Returns:
            Concatenated numpy array
        """

        if self.key == "target":
            # Identify how many zeros in target
            target = data["target"]
            target = self._convert_to_numpy(target, "target")
            target = self._ensure_2d(target, "target")

            num_zeros = np.sum(target == 0)
            logger.info(f"Number of zeros in target: {num_zeros}")
            # Find the first zero occurance and cut everything after
            first_zero_idx = np.argmax(target == 0)
            assert first_zero_idx > 0, "No zero found in target"
            result = target[: first_zero_idx + 1]
            logger.info(f"Cut target to first zero occurrence: {target.shape}")
            assert target[-1] == 0, "Last element of target is not zero after cutting"

        else:
            first_zero_idx = data["target"].shape[0]

            value = data[self.key]
            target = self._convert_to_numpy(value, self.key)
            target = self._ensure_2d(value, self.key)

            result = data[self.key][:first_zero_idx]
            assert (
                data["target"].shape[0] == result.shape[0]
            ), "Shape mismatch after cutting"

        return result

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)
