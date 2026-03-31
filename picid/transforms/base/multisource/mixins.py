"""
Multi-source (multi-unit) transform mixins and compositions.

When data is multi-unit (e.g. multiple machines), the pipeline passes
List[NamedTransformInput] to fit_multi_source / transform_multi_source instead
of a single NamedTransformInput. These mixins implement that contract by:

- Inferring data kind (ragged vs dense) and transform capability (handlers).
- Dispatching to the right handler (get_handler) which prepares chunks and
  calls the underlying fit_data / transform_data.

Layering:
- Layer 1: Interfaces (MultiSourceFitInterface, MultiSourceTransformInterface)
- Layer 2: Dispatcher (internal _BaseDispatcher)
- Layer 3: Atomic fit strategies (FitByConcatenationMixin, NoFitMixin)
- Layer 4: Atomic transform strategies (PerSegmentTransformMixin, ConcatenateBeforeTransformMixin)
- Layer 5: Compound mixins (ConcatFitAndPerSegmentTransformMixin, NoFitPerSegmentMixin,
  NoFitConcatAlongAxisMixin) — public API for transforms.

InverseTransformMixin is optional and provides inverse_transform_data / inverse_transform_multi_source.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.data_kind import get_capability, infer_data_kind
from picid.transforms.base.handlers import get_handler

from picid.transforms.base.multisource.utils import (
    _assert_uniform_data_types,
    _assert_uniform_key_count,
    _build_flags,
    _concatenate_segments,
    _log_transform_error,
    _normalize_keys,
    find_singular_ragged_dim,
)


# -----------------------------------------------------------------------------
# Layer 1: Interfaces
# -----------------------------------------------------------------------------


class MultiSourceFitInterface(ABC):
    """Abstract contract for fit strategies in the multi-source pipeline."""

    @abstractmethod
    def fit_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...


class MultiSourceTransformInterface(ABC):
    """Abstract contract for transform strategies in the multi-source pipeline."""

    @abstractmethod
    def transform_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]: ...


# -----------------------------------------------------------------------------
# Layer 2: Dispatcher (internal)
# -----------------------------------------------------------------------------


class _BaseDispatcher(MultiSourceTransformInterface, ABC):
    """
    Private base: resolve data_kind + capability, get handler, and run fit/transform.

    Subclasses implement fit_multi_source / transform_multi_source and call
    _dispatch_fit or _dispatch_transform with their fit_data or transform_data.
    The handler (from get_handler) does the actual chunk preparation and call.
    """

    def _prepare_dispatch(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Dict[str, Any], str, str]:
        """
        Shared setup for fit and transform dispatch.

        Parameters
        ----------
        data_segments : list[NamedTransformInput]
            Input chunks grouped by split.
        metadata : dict[str, Any], optional
            Pipeline metadata for the current transform call.

        Returns
        -------
        tuple
            A tuple containing the resolved handler, a metadata copy, the
            inferred data kind, and the transform capability.
        """
        metadata = metadata or {}
        apply_to_keys = metadata.get("apply_to_keys")
        if not apply_to_keys:
            raise ValueError(
                "metadata must contain 'apply_to_keys' (list of keys to transform)."
            )
        apply_to_keys = _normalize_keys(apply_to_keys)
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

        return handler, metadata, data_kind, capability

    def _dispatch_fit(
        self,
        data_segments: List[NamedTransformInput],
        fit_func: Callable,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        handler, metadata_copy, _, _ = self._prepare_dispatch(data_segments, metadata)
        apply_to_keys = _normalize_keys(metadata_copy["apply_to_keys"])
        handler.fit_prepare(data_segments, apply_to_keys, metadata_copy, fit_func)

    def _dispatch_transform(
        self,
        data_segments: List[NamedTransformInput],
        transform_func: Callable,
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[List[Any], Dict[str, Any]]:
        handler, metadata_copy, data_kind, capability = self._prepare_dispatch(
            data_segments, metadata
        )

        try:
            out, log = handler.transform_apply(
                data_segments, transform_func, metadata_copy, self.__class__.__name__
            )
        except Exception:
            _log_transform_error(self, data_segments, metadata, data_kind, capability)
            raise

        log["flags"] = _build_flags(data_kind, capability)
        return out, log


# -----------------------------------------------------------------------------
# Layer 3: Atomic fit strategies
# -----------------------------------------------------------------------------


class FitByConcatenationMixin(MultiSourceFitInterface, _BaseDispatcher):
    """Fit on all segments concatenated into one (handler chooses how to merge)."""

    def fit_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata_copy = copy.deepcopy(metadata) if metadata is not None else {}
        self._dispatch_fit(data_segments, self.fit_data, metadata_copy)


class NoFitMixin(MultiSourceFitInterface, _BaseDispatcher):
    """No fitting step; fit_multi_source raises NotImplementedError."""

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


# -----------------------------------------------------------------------------
# Layer 4: Atomic transform strategies
# -----------------------------------------------------------------------------


class PerSegmentTransformMixin(_BaseDispatcher):
    """Apply transform to each segment independently (one transform_data call per segment)."""

    def transform_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        _assert_uniform_key_count(data_segments)
        metadata_copy = copy.deepcopy(metadata) if metadata is not None else {}
        return self._dispatch_transform(
            data_segments, self.transform_data, metadata_copy
        )


class ConcatenateBeforeTransformMixin(_BaseDispatcher):
    """
    Concatenate all segments along self.axis (e.g. units), then run transform_data once.

    Requires self.axis to be set before transform_multi_source.
    Typically used via NoFitConcatAlongAxisMixin, which sets axis in __init__.
    """

    def transform_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        if not hasattr(self, "axis"):
            raise AttributeError(
                f"{self.__class__.__name__} inherits from "
                "ConcatenateBeforeTransformMixin and must set self.axis "
                "before calling transform_multi_source."
            )
        if self.axis is None:
            raise AttributeError(
                f"{self.__class__.__name__} inherits from "
                "ConcatenateBeforeTransformMixin and must set self.axis "
                "before calling transform_multi_source."
            )

        _assert_uniform_key_count(data_segments)
        metadata_copy = copy.deepcopy(metadata) if metadata is not None else {}
        apply_to_keys = metadata_copy.get("apply_to_keys")
        if not apply_to_keys:
            raise ValueError(
                "metadata must contain 'apply_to_keys' (list of keys to transform)."
            )
        apply_to_keys = _normalize_keys(apply_to_keys)

        merged = _concatenate_segments(data_segments, apply_to_keys, self.axis)

        return self._dispatch_transform([merged], self.transform_data, metadata_copy)


# -----------------------------------------------------------------------------
# Layer 5: Compound mixins (public API for transforms)
# -----------------------------------------------------------------------------


class ConcatFitAndPerSegmentTransformMixin(
    FitByConcatenationMixin, PerSegmentTransformMixin
):
    """
    Fit on concatenated segments; transform per segment.

    Composes FitByConcatenationMixin + PerSegmentTransformMixin.
    Common for fittable per-unit transforms (e.g. scalers).
    """


class NoFitPerSegmentMixin(NoFitMixin, PerSegmentTransformMixin):
    """
    No fit; transform each segment independently.

    Composes NoFitMixin + PerSegmentTransformMixin.
    Use for stateless transforms.
    """


class NoFitConcatAlongAxisMixin(NoFitMixin, ConcatenateBeforeTransformMixin):
    """
    No fit; concatenate along axis then transform once.

    Composes NoFitMixin + ConcatenateBeforeTransformMixin.
    Requires self.axis to be set before transform_multi_source.
    This mixin's __init__(axis, **kwargs) sets self.axis; subclasses must call it
    (e.g. via super().__init__(axis=axis, **kwargs)).

    Parameters
    ----------
    axis : int
        Axis along which the input segments are concatenated.
    **kwargs
        Additional keyword arguments forwarded to the parent class.
    """

    def __init__(self, axis: int, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)


class InverseTransformMixin(ABC):
    """
    Optional mixin for transforms that expose an inverse operation.

    Requires inverse_transform (or inverse_transform_data) to be implemented.
    Provides inverse_transform_data and inverse_transform_multi_source (per segment).

    Parameters
    ----------
    **kwargs
        Additional keyword arguments forwarded to the parent class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, "inverse_transform") and not hasattr(
            self, "inverse_transform_data"
        ):
            raise TypeError(
                "InverseTransformMixin requires the class to implement "
                "inverse_transform or inverse_transform_data."
            )

    def inverse_transform_data(
        self,
        data: NamedTransformInput,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self.inverse_transform(data, metadata=metadata)

    @abstractmethod
    def inverse_transform(
        self,
        data: NamedTransformInput,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any: ...

    def inverse_transform_multi_source(
        self,
        data_segments: List[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        metadata_copy = copy.deepcopy(metadata) if metadata is not None else {}
        return [
            self.inverse_transform_data(seg, metadata=metadata_copy)
            for seg in data_segments
        ]
