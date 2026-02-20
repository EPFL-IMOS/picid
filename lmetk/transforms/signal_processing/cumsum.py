from typing import Any, Dict, List, Optional, Tuple, override
import numpy as np
import awkward as ak

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import RaggedTransform
from lmetk.transforms.base.multisource import NoFitPerSegmentMixin


class CumSumTransform(RaggedTransform, NoFitPerSegmentMixin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Apply cumulative sum to all signals in the input.
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
