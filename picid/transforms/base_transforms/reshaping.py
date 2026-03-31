import logging

import numpy as np
from einops import rearrange

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin

logger = logging.getLogger(__name__)


class ReshapeTransform(NoFitPerSegmentMixin, DenseTransform):
    def __init__(self, pattern: str, **kwargs):
        """
        Initialize the transform.

        Parameters
        ----------
        pattern : str
            Einops rearrangement pattern.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.pattern = pattern

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> np.ndarray:
        """
        Concatenate all data arrays horizontally.

        Parameters
        ----------
        data : dict
            Dictionary containing the input array to reshape.
        metadata : dict
            Dictionary containing auxiliary metadata (unused in this transform).

        Returns
        -------
        np.ndarray
            Reshaped numpy array.
        """

        assert len(data) == 1, "Data dictionary must contain exactly one entry."
        _, value = next(iter(data.items()))
        r_v = rearrange(value, self.pattern)
        return r_v

    def __call__(self, data: NamedTransformInput, metadata: dict) -> np.ndarray:
        return self.transform_data(data, metadata)
