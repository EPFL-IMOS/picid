"""
Data kind and transform capability for registry-based dispatching.

Provides:
- DataKind: classification of data (dense, ragged, ragged_regular)
- TransformCapability: what a transform supports (dense, ragged, both)
- infer_data_kind(): infer DataKind from segments and apply_to_keys
- get_capability(): infer TransformCapability from transform instance (marker classes)
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

import awkward as ak

from picid.data.data_objects import NamedTransformInput

# -----------------------------------------------------------------------------
# Data kind: classification of data for dispatch
# -----------------------------------------------------------------------------

DataKind = str  # "dense" | "ragged" | "ragged_regular"
DATA_KIND_DENSE = "dense"
DATA_KIND_RAGGED = "ragged"
DATA_KIND_RAGGED_REGULAR = "ragged_regular"

# -----------------------------------------------------------------------------
# Transform capability: what a transform supports (from marker classes)
# -----------------------------------------------------------------------------

TransformCapability = str  # "dense" | "ragged" | "both"
CAPABILITY_DENSE = "dense"
CAPABILITY_RAGGED = "ragged"
CAPABILITY_BOTH = "both"


def get_capability(transform_instance: Any) -> TransformCapability:
    """
    Infer transform capability from the transform instance's marker classes.

    Parameters
    ----------
    transform_instance : Any
        Transform instance whose class hierarchy is inspected.

    Returns
    -------
    TransformCapability
        Capability label used by the handler registry.
    """
    cls = transform_instance.__class__
    from picid.transforms.base.base_transform import (
        DenseTransform,
        RaggedOrDenseTransform,
        RaggedTransform,
    )

    if issubclass(cls, RaggedOrDenseTransform):
        return CAPABILITY_BOTH
    if issubclass(cls, RaggedTransform):
        return CAPABILITY_RAGGED
    if issubclass(cls, DenseTransform):
        return CAPABILITY_DENSE
    # Default: treat as dense-only for backward compatibility
    return CAPABILITY_DENSE


def infer_data_kind(
    segments: List[NamedTransformInput],
    apply_to_keys: Union[str, List[str]],
    find_singular_ragged_dim: Callable[[NamedTransformInput], Optional[int]],
) -> DataKind:
    """
    Infer the data kind from a list of segments and the keys we apply to.

    Parameters
    ----------
    segments : list[NamedTransformInput]
        Split-local segments used to inspect the payload shape.
    apply_to_keys : str or list[str]
        Keys considered when determining the data kind.
    find_singular_ragged_dim : Callable[[NamedTransformInput], Optional[int]]
        Helper used to detect whether ragged data is structurally regular.

    Returns
    -------
    DataKind
        One of ``dense``, ``ragged``, or ``ragged_regular``.
    """
    if not segments:
        return DATA_KIND_DENSE

    if isinstance(apply_to_keys, str):
        apply_to_keys = [apply_to_keys]

    # Single type across all segments for apply_to keys (same as multisource)
    data_types = set()
    for chunk in segments:
        cls_map = chunk.get_instance_cls()
        for k in apply_to_keys:
            if k in cls_map:
                data_types.add(cls_map[k])
    if len(data_types) != 1:
        return DATA_KIND_DENSE  # fallback; caller may raise

    is_ragged = next(iter(data_types)) is ak.Array
    if not is_ragged:
        return DATA_KIND_DENSE

    # Ragged: check if structurally regular (find_singular_ragged_dim returns None)
    ragged_dims = {find_singular_ragged_dim(ds) for ds in segments}
    if ragged_dims == {None}:
        return DATA_KIND_RAGGED_REGULAR
    return DATA_KIND_RAGGED
