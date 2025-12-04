import logging
from typing import Any

from sklearn.preprocessing import MinMaxScaler

import awkward as ak

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform, RaggedTransform
from lmetk.transforms.base.multisource import (
    ConcatinatedFitMultiSourceTransformMixin,
    InverseTransformMixin,
)
from lmetk.transforms.utils import _convert_to_numpy

logger = logging.getLogger(__name__)


class RuggedDimensionSwap(RaggedTransform, ConcatinatedFitMultiSourceTransformMixin):
    """
    Swap rugged dimension
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
        """Transforms a single data segment using the internal MinMaxScaler."""
        for key, value in data.items():
            regularized = ak.to_regular(value, axis=self.axis_to_regularize)
            irregularized = ak.from_regular(regularized, axis=self.axis_to_irregularize)
            data[key] = irregularized
        return data
