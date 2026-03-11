"""
Multi-source (multi-unit) transform API.

Re-exports mixins (mixins.py) and utilities (utils.py). Use the mixins to
implement fit_multi_source / transform_multi_source; the pipeline calls these
when data is multi-unit. See mixins.py module docstring for composition guide.
"""

from picid.transforms.base.multisource.mixins import (
    ConcatenateBeforeTransformMixin,
    ConcatFitAndPerSegmentTransformMixin,
    FitByConcatenationMixin,
    InverseTransformMixin,
    MultiSourceTransformInterface,
    NoFitConcatAlongAxisMixin,
    NoFitMixin,
    NoFitPerSegmentMixin,
    PerSegmentTransformMixin,
)
from picid.transforms.base.multisource.utils import (
    find_singular_ragged_dim,
    tolist,
)

__all__ = [
    "ConcatenateBeforeTransformMixin",
    "ConcatFitAndPerSegmentTransformMixin",
    "FitByConcatenationMixin",
    "InverseTransformMixin",
    "MultiSourceTransformInterface",
    "NoFitConcatAlongAxisMixin",
    "NoFitMixin",
    "NoFitPerSegmentMixin",
    "PerSegmentTransformMixin",
    "find_singular_ragged_dim",
    "tolist",
]
