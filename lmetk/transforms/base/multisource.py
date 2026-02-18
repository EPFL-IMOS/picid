"""
multisource.py
==============

Multi-source transform orchestration via composable mixins.


Mixin composition guide
-----------------------
Combine exactly ONE fit-policy mixin and ONE transform-policy mixin:

    class MyScaler(FitByConcatenationMixin, PerSegmentTransformMixin, BaseTransform):
        def fit_data(self, data, *, metadata): ...
        def transform_data(self, data, *, metadata): ...

    class MyStateless(NoFitMixin, PerSegmentTransformMixin, BaseTransform):
        def transform_data(self, data, *, metadata): ...

    class MyWindowTransform(NoFitMixin, ConcatenateBeforeTransformMixin, BaseTransform):
        def __init__(self, axis: int = 1):
            self.axis = axis        # must be set BEFORE super().__init__()
            super().__init__()
        def transform_data(self, data, *, metadata): ...

Handler behaviour per data kind
--------------------------------
The handler receives a single merged segment when ConcatenateBeforeTransformMixin
is used, so transform_data is always called exactly once in that policy. The
handler still runs to enforce type-compatibility:

    Data kind               Handler                 transform_data calls
    ─────────────────────── ─────────────────────── ────────────────────
    Dense / np.ndarray      DenseDenseHandler        1  (direct)
    Ragged-regular ak.Array DenseDenseHandler        1  (numpy roundtrip*)
    Ragged + ragged-capable RaggedRaggedHandler       1  (direct)
    Ragged + dense-only     RaggedDenseHandler        N  (one per block) ← bug-fix vs old

    (*) DenseDenseHandler converts ragged-regular ak.Array → numpy before calling
    transform_data, then wraps back to ak.Array. If transform_data must receive
    ak.Array directly, declare the transform as RaggedOrDenseTransform / RaggedTransform
    instead of DenseTransform.

Ready-made compositions (drop-in replacements for old mixins)
-------------------------------------------------------------
    StandardFitTransformMixin  ≡  old ConcatinatedFitMultiSourceTransformMixin
    StatelessPerSegmentMixin   ≡  old NoFitMultiSourceTransformMixin
    StatelessConcatenateMixin  ≡  old ConcatenateUnitsMixing
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import awkward as ak
import numpy as np

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.data_kind import (
    CAPABILITY_BOTH,
    CAPABILITY_DENSE,
    CAPABILITY_RAGGED,
    DATA_KIND_RAGGED,
    DATA_KIND_RAGGED_REGULAR,
    get_capability,
    infer_data_kind,
)
from lmetk.transforms.base.handlers import get_handler
from lmetk.utils.awkward_utils import ak_find_var_dims, get_ak_shape
from lmetk.utils.rich_output import describe_data_type
from sortedcontainers import SortedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def find_singular_ragged_dim(nti: NamedTransformInput) -> Optional[int]:
    """
    Find the single variable-length (ragged) dimension in a NamedTransformInput.

    This function iterates through all Awkward arrays in the input and identifies
    their variable-length dimensions. It expects to find at most one unique
    variable-length dimension across all arrays.

    Parameters
    ----------
    nti : NamedTransformInput
        A dictionary-like object containing Awkward arrays.

    Returns
    -------
    Optional[int]
        The singular variable-length dimension if found, otherwise None if all
        arrays are regular.

    Raises
    ------
    ValueError
        If more than one unique variable-length dimension is found.
    """
    var_dims = [ak_find_var_dims(arr) for arr in nti.values()]

    if any([len(dim) == 0 for dim in var_dims]):
        idx = [len(dim) == 0 for dim in var_dims].index(True)
        # TODO: this warning polutes a lot the outputs
        # logger.warning(
        #     f"One of the values of the {type(nti)} object has no ragge dim, key: {list(nti.keys())[idx]}, val: {list(nti.values())[idx].type}"
        # )
    unique_var_dims = set([x for row in var_dims for x in row])

    # TODO checks to warn
    if len(unique_var_dims) == 0:
        # No ragged dimensions found; the data is regular in structure.
        return None
    elif len(unique_var_dims) == 1:
        # Exactly one ragged dimension found, as expected.
        return unique_var_dims.pop()
    else:
        # More than one ragged dimension makes the concatenation axis ambiguous.
        raise ValueError(
            f"Expected at most one variable-length dimension, found {len(unique_var_dims)}: {unique_var_dims}"
        )


def tolist(d):
    """
    Recursively convert a nested dictionary to a nested list.

    The conversion is done by sorting the dictionary keys at each level
    to ensure a consistent order. Used by handlers (e.g. RaggedDenseHandler).
    """
    if isinstance(d, SortedDict):
        return [tolist(v) for v in d.values()]
    if isinstance(d, dict):
        return [tolist(d[i]) for i in sorted(d)]
    return d


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _normalize_keys(apply_to_keys) -> List[str]:
    """Always return a list, even when a bare string is passed."""
    if isinstance(apply_to_keys, list):
        return apply_to_keys
    return [apply_to_keys]


def _assert_uniform_data_types(data_segments: List[NamedTransformInput]) -> None:
    """Raise ``ValueError`` if segments carry more than one data type."""
    data_types = {
        v
        for chunk in data_segments
        for v in chunk.get_instance_cls().values()
    }
    if len(data_types) != 1:
        raise ValueError(
            f"All data segments must have the same data type. Found: {data_types}"
        )


def _assert_uniform_key_count(data_segments: List[NamedTransformInput]) -> None:
    """Raise ``ValueError`` if segments have different numbers of keys."""
    lengths = [len(chunk.keys()) for chunk in data_segments]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Data segments have different numbers of keys. Lengths: {lengths}"
        )


def _build_flags(data_kind: str, capability: str) -> Dict[str, bool]:
    """Build the diagnostic flags dict attached to every transform log."""
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
    Emit structured error diagnostics to the logger.

    Extracted once here so every mixin shares the same error format
    without any duplication.
    """
    t_name = transform_instance.__class__.__name__
    flags = _build_flags(data_kind, capability)
    sep = "-" * 120

    logger.error(sep)
    logger.error("Transform Error Details  (%s.transform_multi_source())", t_name)
    logger.error(sep)
    logger.error("Data type flags : %s", flags)
    logger.error("Metadata        : %s", metadata)
    logger.error(sep)

    _seg = data_segments[0] if data_segments else None
    if _seg is not None:
        logger.error("First segment keys: %s", list(_seg.keys()))
        logger.error("First segment contents:")
        for k, v in _seg.items():
            logger.error(
                "  key=%-20s  type=%-20s  %s",
                k,
                type(v).__name__,
                describe_data_type(v, calculate_stat=True),
            )
    logger.error(sep)

    data_is_ragged = flags["data_is_ragged"]
    supports_ragged = flags["transform_supports_ragged"]
    supports_dense = flags["transform_supports_dense"]

    logger.error("Case analysis:")
    if not data_is_ragged and supports_dense:
        logger.error("  Dense data + dense-capable transform → data passed directly.")
    elif data_is_ragged and not supports_ragged:
        logger.error(
            "  Ragged data + dense-only transform → "
            "processed block-by-block, converted to numpy, reassembled."
        )
    elif data_is_ragged and supports_ragged:
        logger.error("  Ragged data + ragged-capable transform → data passed directly.")
    else:
        logger.error("  Unknown combination; check the handler registry.")

    logger.error(sep)
    logger.error("Exception traceback:", exc_info=True)
    logger.error(sep)


def _concatenate_segments(
    data_segments: List[NamedTransformInput],
    apply_to_keys: List[str],
    axis: int,
) -> NamedTransformInput:
    """
    Concatenate multiple data segments into a single ``NamedTransformInput``
    along *axis*.
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

    non_none = [l for l in lengths if l is not None]
    if non_none and len(set(non_none)) != 1:
        raise ValueError(
            f"After concatenation, arrays disagree on axis={axis}. "
            f"Per-key lengths: {dict(zip(keys, lengths))}"
        )

    return NamedTransformInput(**merged)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class MultiSourceTransformInterface(ABC):
    """
    Contract every multi-source mixin must satisfy.
    """

    @abstractmethod
    def fit_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...  # pragma: no cover

    @abstractmethod
    def transform_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Layer 1 – shared dispatch engine
# ---------------------------------------------------------------------------

class _BaseDispatcher(MultiSourceTransformInterface, ABC):
    """Private base: handler resolution and dispatch for fit and transform."""

    def _dispatch_fit(
        self,
        data_segments: List[NamedTransformInput],
        fit_func: Callable,
        metadata: Dict[str, Any],
    ) -> None:
        apply_to_keys = _normalize_keys(metadata["apply_to_keys"])
        _assert_uniform_data_types(data_segments)

        data_kind = infer_data_kind(
            data_segments, apply_to_keys, find_singular_ragged_dim
        )
        capability = get_capability(self)

        try:
            handler = get_handler(data_kind, capability)
        except KeyError:
            raise ValueError(
                f"{self.__class__.__name__} does not support "
                f"data_kind={data_kind!r} with capability={capability!r}."
            )

        handler.fit_prepare(data_segments, apply_to_keys, metadata, fit_func)

    def _dispatch_transform(
        self,
        data_segments: List[NamedTransformInput],
        transform_func: Callable,
        metadata: Dict[str, Any],
    ) -> Tuple[List[Any], Dict[str, Any]]:
        apply_to_keys = _normalize_keys(metadata["apply_to_keys"])
        _assert_uniform_data_types(data_segments)

        data_kind = infer_data_kind(
            data_segments, apply_to_keys, find_singular_ragged_dim
        )
        capability = get_capability(self)

        try:
            handler = get_handler(data_kind, capability)
            out, log = handler.transform_apply(
                data_segments, transform_func, metadata, self.__class__.__name__
            )
        except KeyError:
            raise ValueError(
                f"{self.__class__.__name__} does not support "
                f"data_kind={data_kind!r} with capability={capability!r}."
            )
        except Exception:
            _log_transform_error(
                self, data_segments, metadata, data_kind, capability
            )
            raise

        log["flags"] = _build_flags(data_kind, capability)
        return out, log


# ---------------------------------------------------------------------------
# Layer 2a – fit-policy mixins
# ---------------------------------------------------------------------------

class FitByConcatenationMixin(_BaseDispatcher):
    """Fit on all segments concatenated into one. requires_fit = True."""

    requires_fit: bool = True

    def fit_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._dispatch_fit(
            data_segments, self.fit_data, copy.deepcopy(metadata)
        )


class NoFitMixin(_BaseDispatcher):
    """No fitting step. requires_fit = False."""

    requires_fit: bool = False

    def fit_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} is a stateless transform and does not "
            "support fitting.  Remove the fitting step from your pipeline, or "
            "switch to FitByConcatenationMixin if fitting is needed."
        )


# ---------------------------------------------------------------------------
# Layer 2b – transform-policy mixins
# ---------------------------------------------------------------------------

class PerSegmentTransformMixin(_BaseDispatcher):
    """Apply transform to each segment independently."""

    def transform_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        _assert_uniform_key_count(data_segments)
        return self._dispatch_transform(
            data_segments, self.transform_data, copy.deepcopy(metadata)
        )


class ConcatenateBeforeTransformMixin(_BaseDispatcher):
    """Concatenate all segments along self.axis, then transform once."""

    def transform_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        if not hasattr(self, "axis") or self.axis is None:
            raise AttributeError(
                f"{self.__class__.__name__} inherits from "
                "ConcatenateBeforeTransformMixin and must set self.axis "
                "before calling transform_multi_source."
            )

        _assert_uniform_key_count(data_segments)
        metadata_copy = copy.deepcopy(metadata)
        apply_to_keys = _normalize_keys(metadata_copy["apply_to_keys"])

        merged = _concatenate_segments(data_segments, apply_to_keys, self.axis)

        return self._dispatch_transform(
            [merged], self.transform_data, metadata_copy
        )


# ---------------------------------------------------------------------------
# Layer 3 – ready-made compositions (drop-in replacements for old mixins)
# ---------------------------------------------------------------------------

class StandardFitTransformMixin(FitByConcatenationMixin, PerSegmentTransformMixin):
    """Fit on concatenated segments; transform per segment. ≡ old ConcatinatedFitMultiSourceTransformMixin."""


class StatelessPerSegmentMixin(NoFitMixin, PerSegmentTransformMixin):
    """No fit; transform per segment. ≡ old NoFitMultiSourceTransformMixin."""


class StatelessConcatenateMixin(NoFitMixin, ConcatenateBeforeTransformMixin):
    """No fit; concatenate along axis then transform once. ≡ old ConcatenateUnitsMixing."""

    def __init__(self, axis: int, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Optional add-on: inverse transform support
# ---------------------------------------------------------------------------

class InverseTransformMixin(ABC):
    """Optional mixin for transforms that expose an inverse operation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, "inverse_transform") and not hasattr(self, "inverse_transform_data"):
            raise TypeError(
                "InverseTransformMixin requires the class to implement "
                "inverse_transform or inverse_transform_data."
            )

    def inverse_transform_data(
        self,
        data: NamedTransformInput,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Single-segment inverse; defaults to inverse_transform for backward compatibility."""
        return self.inverse_transform(data, metadata=metadata)

    @abstractmethod
    def inverse_transform(
        self,
        data: NamedTransformInput,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        ...  # pragma: no cover

    def inverse_transform_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        return [
            self.inverse_transform_data(seg, metadata=copy.deepcopy(metadata))
            for seg in data_segments
        ]


# ---------------------------------------------------------------------------
# Backward compatibility: old mixin names → new compositions
# ---------------------------------------------------------------------------

ConcatinatedFitMultiSourceTransformMixin = StandardFitTransformMixin
NoFitMultiSourceTransformMixin = StatelessPerSegmentMixin
ConcatenateUnitsMixing = StatelessConcatenateMixin

# Phase 1 refactor names (same behaviour as StandardFitTransformMixin)
ConcatenatedMultiSourceTransformMixin = StandardFitTransformMixin
FitMultiSourceTransformMixin = StandardFitTransformMixin
