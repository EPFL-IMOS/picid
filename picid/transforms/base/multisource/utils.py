"""
Utilities for multi-source (multi-unit) transforms.

Used by the multisource mixins (mixins.py) and the handler layer (handlers/):
- find_singular_ragged_dim: infer which axis is variable-length for data_kind.
- _concatenate_segments: merge segments along an axis (e.g. for ConcatenateBeforeTransformMixin).
- _assert_uniform_*: validate that segments share types/key count.
- _build_flags / _log_transform_error: diagnostics when transform_multi_source fails.

Data kind (ragged vs dense) and capability (transform supports ragged/dense/both)
drive handler selection in get_handler(); see data_kind and handlers packages.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import awkward as ak
import numpy as np

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.data_kind import (
    CAPABILITY_BOTH,
    CAPABILITY_DENSE,
    CAPABILITY_RAGGED,
    DATA_KIND_RAGGED,
    DATA_KIND_RAGGED_REGULAR,
)
from picid.utils.awkward_utils import (
    find_singular_ragged_dim as find_singular_ragged_dim_from_arrays,
    get_ak_shape,
)
from picid.utils.rich_output import (
    build_transform_error_renderables,
    describe_data_type,
)
from rich.console import Console
from sortedcontainers import SortedDict

logger = logging.getLogger(__name__)


def find_singular_ragged_dim(nti: NamedTransformInput) -> Optional[int]:
    """
    Find the single variable-length (ragged) dimension in a NamedTransformInput.

    Delegates to picid.utils.awkward_utils.find_singular_ragged_dim(nti.values()).

    Parameters
    ----------
    nti : NamedTransformInput
        Input mapping whose values are inspected for raggedness.

    Returns
    -------
    int or None
        The singular ragged dimension, or ``None`` if the input is not ragged.
    """
    return find_singular_ragged_dim_from_arrays(nti.values())


def tolist(d):
    """
    Recursively convert a nested dict into nested lists with sorted keys.

    Parameters
    ----------
    d : Any
        Nested mapping or scalar value to convert.

    Returns
    -------
    Any
        The recursively converted structure.
    """
    if isinstance(d, SortedDict):
        return [tolist(v) for v in d.values()]
    if isinstance(d, dict):
        return [tolist(d[i]) for i in sorted(d)]
    return d


def _normalize_keys(apply_to_keys) -> List[str]:
    """
    Ensure ``apply_to_keys`` is represented as a list.

    Parameters
    ----------
    apply_to_keys : Any
        A single key or a list of keys.

    Returns
    -------
    list[str]
        Normalized list of keys.
    """
    if isinstance(apply_to_keys, list):
        return apply_to_keys
    return [apply_to_keys]


def _assert_uniform_data_types(data_segments: List[NamedTransformInput]) -> None:
    """
    Require all segments to share the same data type.

    Parameters
    ----------
    data_segments : list[NamedTransformInput]
        Segments to validate.
    """
    data_types = {
        v for chunk in data_segments for v in chunk.get_instance_cls().values()
    }
    if len(data_types) != 1:
        raise ValueError(
            f"All data segments must have the same data type. Found: {data_types}"
        )


def _assert_uniform_key_count(data_segments: List[NamedTransformInput]) -> None:
    """
    Require all segments to have the same number of keys.

    Parameters
    ----------
    data_segments : list[NamedTransformInput]
        Segments to validate.
    """
    lengths = [len(chunk.keys()) for chunk in data_segments]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Data segments have different numbers of keys. Lengths: {lengths}"
        )


def _build_flags(data_kind: str, capability: str) -> Dict[str, bool]:
    """
    Build a small dict of data_kind/capability flags for logging and manifests.

    Parameters
    ----------
    data_kind : str
        Inferred data kind for the current transform call.
    capability : str
        Capability reported by the transform instance.

    Returns
    -------
    dict[str, bool]
        Boolean flags summarizing the data kind and capability.
    """
    return {
        "data_is_ragged": data_kind == DATA_KIND_RAGGED,
        "data_is_ragged_but_regular": data_kind == DATA_KIND_RAGGED_REGULAR,
        "transform_supports_ragged": capability in (CAPABILITY_RAGGED, CAPABILITY_BOTH),
        "transform_supports_dense": capability in (CAPABILITY_DENSE, CAPABILITY_BOTH),
    }


def _log_transform_error(
    transform_instance: Any,
    data_segments: List[NamedTransformInput],
    metadata: Dict[str, Any],
    data_kind: str,
    capability: str,
) -> None:
    """
    Log detailed diagnostics when ``transform_multi_source`` fails.

    Parameters
    ----------
    transform_instance : Any
        Transform instance that raised the error.
    data_segments : list[NamedTransformInput]
        Input segments passed to the transform.
    metadata : dict[str, Any]
        Pipeline metadata for the current transform call.
    data_kind : str
        Inferred data kind for the current transform call.
    capability : str
        Capability reported by the transform instance.
    """
    t_name = transform_instance.__class__.__name__
    flags = _build_flags(data_kind, capability)
    _seg = data_segments[0] if data_segments else None

    first_segment_rows: List[tuple[str, str, str]] = []
    if _seg is not None:
        for k, v in _seg.items():
            first_segment_rows.append(
                (
                    k,
                    type(v).__name__,
                    describe_data_type(v, calculate_stat=True),
                )
            )

    data_is_ragged = flags["data_is_ragged"]
    supports_ragged = flags["transform_supports_ragged"]
    supports_dense = flags["transform_supports_dense"]
    if not data_is_ragged and supports_dense:
        case_analysis_line = (
            "Dense data + dense-capable transform → data passed directly."
        )
    elif data_is_ragged and not supports_ragged:
        case_analysis_line = (
            "Ragged data + dense-only transform → "
            "processed block-by-block, converted to numpy, reassembled."
        )
    elif data_is_ragged and supports_ragged:
        case_analysis_line = (
            "Ragged data + ragged-capable transform → data passed directly."
        )
    else:
        case_analysis_line = "Unknown combination; check the handler registry."

    renderables = build_transform_error_renderables(
        t_name,
        flags,
        metadata,
        list(_seg.keys()) if _seg else None,
        first_segment_rows,
        case_analysis_line,
    )
    console = Console(stderr=True)
    for r in renderables:
        console.print(r)

    logger.error(
        "Transform error in %s (data_kind=%s, capability=%s); see stderr for Rich diagnostics.",
        t_name,
        data_kind,
        capability,
        exc_info=True,
    )


def _concatenate_segments(
    data_segments: List[NamedTransformInput],
    apply_to_keys: List[str],
    axis: int,
) -> NamedTransformInput:
    """
    Concatenate multiple data segments into a single NamedTransformInput along axis.

    Used by ConcatenateBeforeTransformMixin: merge units along axis, then run
    transform_data once. Keys in apply_to_keys must have consistent dtypes
    across segments (ak.Array or np.ndarray); other keys are taken from the
    first segment. Validates that concatenated lengths on axis agree across keys.

    Parameters
    ----------
    data_segments : list[NamedTransformInput]
        Segments to concatenate.
    apply_to_keys : list[str]
        Keys whose values should be concatenated.
    axis : int
        Axis along which concatenation is performed.

    Returns
    -------
    NamedTransformInput
        A single merged input structure.
    """
    if not data_segments:
        raise ValueError("_concatenate_segments requires at least one segment.")

    keys = _normalize_keys(apply_to_keys)
    first = data_segments[0]
    cls_map = first.get_instance_cls()
    merged: Dict[str, Any] = {}

    for key in keys:
        dtype = cls_map.get(key)

        for idx, seg in enumerate(data_segments[1:], start=1):
            other_dtype = seg.get_instance_cls().get(key)
            if other_dtype != dtype:
                raise ValueError(
                    f"Key '{key}': data type mismatch between segment 0 "
                    f"({dtype}) and segment {idx} ({other_dtype})."
                )

        arrays = [seg[key] for seg in data_segments]

        if dtype is ak.Array:
            merged[key] = ak.concatenate(arrays, axis=axis)
        elif dtype is np.ndarray:
            merged[key] = np.concatenate(arrays, axis=axis)
        else:
            raise ValueError(
                f"Key '{key}': unsupported data type {dtype}. "
                "Only np.ndarray and ak.Array are supported."
            )

    for key in first.keys():
        if key not in merged:
            merged[key] = first[key]

    lengths = []
    for key in keys:
        val = merged[key]
        if isinstance(val, ak.Array):
            shape = get_ak_shape(val)
            lengths.append(shape[axis] if axis < len(shape) else None)
        elif isinstance(val, np.ndarray):
            lengths.append(val.shape[axis] if axis < val.ndim else None)

    non_none = [l for l in lengths if l is not None]  # noqa: E741
    if non_none and len(set(non_none)) != 1:
        raise ValueError(
            f"After concatenation, arrays disagree on axis={axis}. "
            f"Per-key lengths: {dict(zip(keys, lengths))}"
        )

    return NamedTransformInput(**merged)
