"""Data kind and transform capability for registry-based dispatching."""

from picid.transforms.base.data_kind.data_kind import (
    CAPABILITY_BOTH,
    CAPABILITY_DENSE,
    CAPABILITY_RAGGED,
    DATA_KIND_DENSE,
    DATA_KIND_RAGGED,
    DATA_KIND_RAGGED_REGULAR,
    DataKind,
    TransformCapability,
    get_capability,
    infer_data_kind,
)

__all__ = [
    "CAPABILITY_BOTH",
    "CAPABILITY_DENSE",
    "CAPABILITY_RAGGED",
    "DATA_KIND_DENSE",
    "DATA_KIND_RAGGED",
    "DATA_KIND_RAGGED_REGULAR",
    "DataKind",
    "TransformCapability",
    "get_capability",
    "infer_data_kind",
]
