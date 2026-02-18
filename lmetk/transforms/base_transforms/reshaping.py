import logging

import numpy as np
from einops import parse_shape, rearrange

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import StatelessPerSegmentMixin

logger = logging.getLogger(__name__)


class ReshapeTransform(DenseTransform, StatelessPerSegmentMixin):
    def __init__(self, pattern: str, **kwargs):
        """
        Initialize the transform.

        Args:
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.pattern = pattern

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> np.ndarray:
        """
        Concatenate all data arrays horizontally.

        Args:
            data: Dictionary containing numpy arrays to concatenate
            metadata: Dictionary containing auxiliary metadata (unused in this transform)

        Returns:
            Concatenated numpy array
        """

        assert len(data) == 1, "Data dictionary must contain exactly one entry."
        _, value = next(iter(data.items()))
        r_v = rearrange(value, self.pattern)
        return r_v

    def __call__(self, data: NamedTransformInput, metadata: dict) -> np.ndarray:
        return self.transform_data(data, metadata)
