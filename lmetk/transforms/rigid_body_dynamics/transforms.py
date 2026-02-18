import logging
from typing import Any, Dict, List

import numpy as np

from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import StatelessPerSegmentMixin

logger = logging.getLogger(__name__)


class First(DenseTransform, StatelessPerSegmentMixin):
    def __init__(self, **kwargs):
        """
        Initialize the transform.

        Args:
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)

    def transform_data(self, data: Dict, metadata: Dict) -> np.ndarray:
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
        print(data.keys())

        return data

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)
