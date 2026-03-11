import logging
from typing import Any, Dict, List

import numpy as np
import awkward as ak

from picid.data.data_objects.data import NamedTransformInput
from picid.transforms.base.base_transform import (
    DenseTransform,
    RaggedTransform,
    RaggedOrDenseTransform,
)
from picid.transforms.base.multisource import NoFitPerSegmentMixin
from picid.utils.awkward_utils import ak_find_var_dims

logger = logging.getLogger(__name__)


class RegularizeRaggedDataTransform(NoFitPerSegmentMixin, RaggedTransform):
    def __init__(self, **kwargs):
        """
        This transform regularizes ragged arrays to have uniform length.
        It is use

        Parameters
        ----------
        **kwargs
            Additional keyword arguments
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


class ExpandScalarToReferenceFeatureSize(
    NoFitPerSegmentMixin, DenseTransform
):
    def __init__(self, scalar_key: str, **kwargs):
        """
        Initialize the transform.

        Parameters
        ----------
        scalar_key : str
            The key in the input data that holds the scalar value
            (or N-D array) to be expanded.
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        self.scalar_key = scalar_key
        super().__init__(**kwargs)

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Expand a scalar/array to match a reference feature, ensuring 2D output.

        This transform takes a value from `scalar_key` and repeats it to create
        a new array. The length of the new array's first dimension is
        determined by the size of the first dimension (shape[0]) of a
        "reference" feature.

        The output is guaranteed to be at least 2-dimensional.

        Examples
        --------
        Example 1 (Scalar):
        - data[self.scalar_key] = 5
        - data[reference_key] = np.array([[1, 2], [3, 4], [5, 6]])  (shape (3, 2))
        - The target size will be 3.
        - This transform will return: np.array([[5], [5], [5]])  (shape (3, 1))

        Example 2 (Array):
        - data[self.scalar_key] = [5, 1]
        - data[reference_key] = np.array([[1, 2], [3, 4], [5, 6]])  (shape (3, 2))
        - The target size will be 3.
        - This transform will return: np.array([[5, 1], [5, 1], [5, 1]]) (shape (3, 2))
        """
        # Get the list of all keys this transform applies to from metadata
        apply_to_keys: List[str] = metadata["apply_to_keys"]

        # Identify the reference key (the one that isn't the scalar_key)
        reference_key = (
            apply_to_keys[0]
            if apply_to_keys[1] == self.scalar_key
            else apply_to_keys[1]
        )

        # Get the value to be repeated and ensure it's at least 1D
        fill_value = np.atleast_1d(data[self.scalar_key])

        # Get the reference feature
        reference_feature = data[reference_key]

        # Get the target size from the reference feature's shape
        target_size = reference_feature.shape[0]

        # --- THIS IS THE FIX ---

        # 1. Determine the final desired shape.
        # This will be (target_size, *shape_of_fill_value)
        # e.g., (30,) + (2,) -> (30, 2)
        # e.g., (30,) + (1,) -> (30, 1)
        output_shape = (target_size,) + fill_value.shape

        # 2. Create the array with the *full* desired shape.
        # np.full will broadcast the fill_value into this new shape.
        output_array = np.full(output_shape, fill_value)

        return output_array


class RaggedToDenseTransform(NoFitPerSegmentMixin, RaggedTransform):
    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Concatenate ragged arrays along a single ragged dimension
        """

        apply_to_keys = metadata["apply_to_keys"]
        out = {}
        for key in apply_to_keys:
            # convert to numpy
            out[key] = ak.to_numpy(data[key])

        return out
