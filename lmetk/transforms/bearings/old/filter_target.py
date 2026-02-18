import logging
from typing import Any, Dict, List

import numpy as np

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import StatelessPerSegmentMixin

logger = logging.getLogger(__name__)


class FilterTarget(DenseTransform, StatelessPerSegmentMixin):
    def __init__(self, win_len: int, hop: int, **kwargs):
        """
        Initialize the transform.

        Args:
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.win_len = win_len
        self.hop = hop

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Filter targets correponding to the stftt

        Args:
            data: Dictionary containing numpy arrays to concatenate
            metadata: Dictionary containing auxiliary metadata (unused in this transform)

        Returns:
            Concatenated numpy array
        """
        features = data["features"]  # N, D --> n, <h1,h2> | subsample N to n for target
        assert "target" in data, "Missing 'target' key in data"

        if not data:
            raise ValueError("No data provided for concatenation")

        target, key = data["target"], "target"
        # Convert to numpy array
        target = self._convert_to_numpy(target, key)

        # Ensure 2D shape
        target = self._ensure_2d(target, key)
        logger.info(f"Added data '{key}' with shape {target.shape}")

        # Check for NaN values (will raise error if found)
        self._check_for_nans([target], ["target"])

        # Allign targets
        n_frames = features.shape[0]  # 103 here
        centers = np.arange(n_frames) * self.hop  # in samples
        end_idx = np.clip(centers + self.win_len // 2 - 1, 0, len(target) - 1)
        frame_targets = target[end_idx]  # aligned targets per time frame

        # Concatenate and reshape (-1,1)
        target = np.array(frame_targets).reshape(-1, 1)
        logger.info(f"Final target shape: {target.shape}")

        assert (
            target.shape[0] == features.shape[0]
        ), "Target and result shapes do not match"
        assert target[-1] == 0, "For the RUL the last value has to be 0"

        return target

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)
