import logging
from typing import Any


import awkward as ak

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import RaggedTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin

logger = logging.getLogger(__name__)


class RuggedDimensionSwap(NoFitPerSegmentMixin, RaggedTransform):
    """
    Swap one regular axis with one ragged axis in each input array.

    Parameters
    ----------
    axis_to_regularize : int
        Axis that should become regular.
    axis_to_irregularize : int
        Axis that should become ragged.
    **kwargs
        Additional keyword arguments forwarded to the base transform.
    """

    def __init__(self, axis_to_regularize, axis_to_irregularize, **kwargs):
        super().__init__()
        self.axis_to_regularize = axis_to_regularize
        self.axis_to_irregularize = axis_to_irregularize

    def transform_data(
        self,
        data: NamedTransformInput,
        metadata: dict,
    ) -> Any:
        """
        Convert the configured axis pair between regular and ragged form.

        Parameters
        ----------
        data : NamedTransformInput
            Input mapping containing Awkward arrays.
        metadata : dict
            Metadata dictionary, preserved for interface compatibility.

        Returns
        -------
        Any
            Input mapping with the requested axes converted.
        """
        for key, value in data.items():
            regularized = ak.to_regular(value, axis=self.axis_to_regularize)
            irregularized = ak.from_regular(regularized, axis=self.axis_to_irregularize)
            data[key] = irregularized
        return data
