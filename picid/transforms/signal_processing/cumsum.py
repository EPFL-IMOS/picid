from typing import Dict, override
import numpy as np
import awkward as ak

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import RaggedTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin


class CumSumTransform(NoFitPerSegmentMixin, RaggedTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Apply a cumulative squared sum to the single ragged input array.

        Parameters
        ----------
        data : NamedTransformInput
            Mapping containing exactly one ragged array.
        metadata : dict
            Unused metadata dictionary, kept for the transform contract.

        Returns
        -------
        numpy.ndarray
            Ragged array with the same unit structure and cumulative power values.
        """
        assert len(data) == 1, "Data dictionary must contain exactly one entry."
        _, x = next(iter(data.items()))
        assert x.ndim == 3, "Input data must be 3-dimensional (units, time, features)."

        flatten = ak.flatten(x)
        cumsum_flatten = np.cumsum(flatten**2, axis=1)
        unflattened = ak.unflatten(cumsum_flatten, ak.num(x))

        assert ak.num(unflattened, axis=0).tolist() == ak.num(x, axis=0).tolist()
        assert ak.all(ak.num(unflattened, axis=1) == ak.num(x, axis=1))
        return unflattened
