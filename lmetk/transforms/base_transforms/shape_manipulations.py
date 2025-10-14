import logging
from typing import Any, Dict, List

import numpy as np
import awkward as ak

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform, RaggedTransform
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin
from lmetk.utils.awkward_utils import ak_find_var_dims

logger = logging.getLogger(__name__)


class RegularizeRaggedDataTransform(
    RaggedTransform, ConcatinatedFitMultiSourceTransformMixin
):
    def __init__(self, **kwargs):
        """
        Initialize the transform.

        Args:
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Concatenate ragged arrays along a single ragged dimension
        """
        keys = list(data.keys())
        assert (
            len(keys) == 1
        ), "RegularizeRaggedDataTransform only supports single key data_segment."

        return ak.to_regular(data[keys[0]])
