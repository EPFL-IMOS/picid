import logging
from typing import Dict, List

import numpy as np
import awkward as ak

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import (
    DenseTransform,
    RaggedTransform,
)
from picid.transforms.base.multisource import NoFitPerSegmentMixin

logger = logging.getLogger(__name__)


class RegularizeRaggedDataTransform(NoFitPerSegmentMixin, RaggedTransform):
    def __init__(self, **kwargs):
        """
        Initialize the transform.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Regularize a ragged array to a dense regular array.

        Parameters
        ----------
        data : NamedTransformInput
            Input segment containing the ragged array.
        metadata : dict
            Transform metadata. Unused here but required by the interface.

        Returns
        -------
        numpy.ndarray
            The regularized array.
        """
        keys = list(data.keys())
        assert (
            len(keys) == 1
        ), "RegularizeRaggedDataTransform only supports single key data_segment."

        return ak.to_regular(data[keys[0]])


class ExpandScalarToReferenceFeatureSize(NoFitPerSegmentMixin, DenseTransform):
    def __init__(self, scalar_key: str, **kwargs):
        """
        Initialize the transform.

        Parameters
        ----------
        scalar_key : str
            Key in the input data holding the scalar value or array to expand.
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        self.scalar_key = scalar_key
        super().__init__(**kwargs)

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Expand a scalar/array to match a reference feature, ensuring 2D output.

        Parameters
        ----------
        data : NamedTransformInput
            Input segment containing the scalar and reference feature.
        metadata : dict
            Transform metadata containing ``apply_to_keys``.

        Returns
        -------
        numpy.ndarray
            Expanded array with at least two dimensions.

        Notes
        -----
        The value from ``scalar_key`` is repeated to match the first dimension
        of the reference feature.

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
        Convert ragged arrays to dense NumPy arrays.

        Parameters
        ----------
        data : NamedTransformInput
            Input segment containing ragged arrays.
        metadata : dict
            Transform metadata containing ``apply_to_keys``.

        Returns
        -------
        dict[str, numpy.ndarray]
            Dictionary of dense arrays keyed by ``apply_to_keys``.
        """

        apply_to_keys = metadata["apply_to_keys"]
        out = {}
        for key in apply_to_keys:
            # convert to numpy
            out[key] = ak.to_numpy(data[key])

        return out
