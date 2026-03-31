"""
Transform handler protocol and registry for dispatch.

Handlers encapsulate fit and transform logic that was previously spread
across multisource mixins. The registry maps ``(data_kind, capability)``
pairs to concrete handler implementations.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Protocol, Tuple

import numpy as np
import awkward as ak
from sortedcontainers import SortedDict

from picid.data.data_objects import NamedTransformInput
from picid.data.data_objects import BaseDataObjectWithMetadata

from picid.transforms.base.data_kind import (
    CAPABILITY_BOTH,
    CAPABILITY_DENSE,
    CAPABILITY_RAGGED,
    DATA_KIND_DENSE,
    DATA_KIND_RAGGED,
    DATA_KIND_RAGGED_REGULAR,
)

logger = logging.getLogger(__name__)


class TransformHandler(Protocol):
    """
    Protocol for handlers that prepare fit input and apply transforms.
    """

    def fit_prepare(
        self,
        data_segments: List[NamedTransformInput],
        apply_to_keys: List[str],
        metadata: Dict[str, Any],
        fit_func: Callable[..., None],
    ) -> None:
        """
        Prepare fit input from segments and call ``fit_func``.

        Parameters
        ----------
        data_segments : list[NamedTransformInput]
            Split-local segments that should be combined for fitting.
        apply_to_keys : list[str]
            Keys whose arrays are concatenated into the fit payload.
        metadata : dict[str, Any]
            Transform metadata passed through to the fit function.
        fit_func : Callable[..., None]
            Callable invoked with the prepared fit payload.
        """
        ...

    def transform_apply(
        self,
        data_segments: List[NamedTransformInput],
        transform_func: Callable[..., Any],
        metadata: Dict[str, Any],
        transform_name: str,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Apply a transform to the provided segments.

        Parameters
        ----------
        data_segments : list[NamedTransformInput]
            Split-local segments to transform.
        transform_func : Callable[..., Any]
            Callable that applies the actual transform logic.
        metadata : dict[str, Any]
            Transform metadata passed through to the transform function.
        transform_name : str
            Human-readable transform name used in warnings and logs.

        Returns
        -------
        list[Any], dict[str, Any]
            The transformed outputs and a small transform log.
        """
        ...


def _postprocess():
    from picid.transforms.base.strategy import postprocess_transformed_data

    return postprocess_transformed_data


def _find_singular_ragged_dim():
    from picid.transforms.base.multisource import find_singular_ragged_dim

    return find_singular_ragged_dim


def _tolist():
    from picid.transforms.base.multisource import tolist

    return tolist


def _ragged_index_tuples():
    from picid.utils.awkward_utils import ragged_index_tuples

    return ragged_index_tuples


def _ak_utils():
    from picid.utils.awkward_utils import (
        ak_find_var_dims,
        ak_regularize_regular_axes,
        get_ak_shape,
    )

    return ak_find_var_dims, ak_regularize_regular_axes, get_ak_shape


# -----------------------------------------------------------------------------
# DenseDenseHandler: dense data or ragged_regular; dense or both capability
# -----------------------------------------------------------------------------


class DenseDenseHandler:
    """
    Handler for dense data and ragged-regular data.

    Ragged-regular inputs are converted to NumPy before the transform is
    applied and wrapped back into Awkward arrays afterward.
    """

    def fit_prepare(
        self,
        data_segments: List[NamedTransformInput],
        apply_to_keys: List[str],
        metadata: Dict[str, Any],
        fit_func: Callable[..., None],
    ) -> None:
        train = {}
        for key in apply_to_keys:
            train[key] = np.concatenate([chunk[key] for chunk in data_segments], axis=0)
        fit_func(train, metadata=metadata)

    def transform_apply(
        self,
        data_segments: List[NamedTransformInput],
        transform_func: Callable[..., Any],
        metadata: Dict[str, Any],
        transform_name: str,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        transform_log = {"mode": "dense"}
        out = []
        for data_segment in data_segments:
            # ragged_regular: convert to numpy for transform, wrap back after
            segment_dict = dict(data_segment)
            converted_to_numpy = False
            if hasattr(data_segment, "get_instance_cls"):
                cls_map = data_segment.get_instance_cls()
                if ak.Array in cls_map.values():
                    converted_to_numpy = True
                    for key in list(segment_dict.keys()):
                        val = segment_dict[key]
                        if hasattr(val, "to_numpy"):
                            segment_dict[key] = val.to_numpy()
            seg_input = (
                NamedTransformInput(
                    # Dense-only transforms may receive awkward-backed inputs
                    # here after an on-the-fly conversion to NumPy. Preserve the
                    # original per-unit metadata while rebuilding the temporary
                    # NamedTransformInput so metadata-aware transforms still see
                    # the same unit context.
                    metadata=(
                        copy.deepcopy(data_segment.metadata)
                        if data_segment.metadata is not None
                        else None
                    ),
                    **segment_dict,
                )
                if segment_dict
                else data_segment
            )
            out_segment = transform_func(seg_input, metadata=metadata)
            if converted_to_numpy:
                if isinstance(out_segment, BaseDataObjectWithMetadata):
                    for key, val in out_segment.items():
                        out_segment[key] = ak.Array(val)
                else:
                    out_segment = ak.Array(out_segment)
            out.append(out_segment)
        return out, transform_log


# -----------------------------------------------------------------------------
# RaggedDenseHandler: ragged data, dense-only transform (ragged -> dense -> reassemble)
# -----------------------------------------------------------------------------


class RaggedDenseHandler:
    """
    Handler for ragged data with a dense-only transform.

    Ragged segments are flattened into dense arrays before the transform is
    applied, then reconstructed back into ragged form.
    """

    def fit_prepare(
        self,
        data_segments: List[NamedTransformInput],
        apply_to_keys: List[str],
        metadata: Dict[str, Any],
        fit_func: Callable[..., None],
    ) -> None:
        find_singular_ragged_dim = _find_singular_ragged_dim()
        var_dim = set(find_singular_ragged_dim(ds) for ds in data_segments)
        assert (
            len(var_dim) == 1
        ), f"Expected exactly one variable-length dimension, found {var_dim}"
        var_dim = var_dim.pop()
        train = {}
        for key in apply_to_keys:
            train[key] = ak.concatenate(
                [
                    ak.concatenate(chunk[key], axis=var_dim - 1)
                    for chunk in data_segments
                ],
                axis=0,
            )
        train = {k: v.to_numpy() for k, v in train.items()}
        fit_func(train, metadata=metadata)

    def transform_apply(
        self,
        data_segments: List[NamedTransformInput],
        transform_func: Callable[..., Any],
        metadata: Dict[str, Any],
        transform_name: str,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        find_singular_ragged_dim = _find_singular_ragged_dim()
        tolist_fn = _tolist()
        ragged_index_tuples_fn = _ragged_index_tuples()
        postprocess = _postprocess()
        ak_find_var_dims, ak_regularize_regular_axes, get_ak_shape = _ak_utils()

        logger.warning(
            f"Transform {transform_name} does not support ragged arrays natively. "
            "This might be slow for large datasets."
        )
        transform_log = {"mode": "ragged_to_dense"}
        out = []

        for data_segment in data_segments:
            assert len(set(len(val) for val in data_segment.values())) == 1, (
                f"All keys in data_segment must have the same length at the first dimension. "
                f"Keys: {list(data_segment.keys())}, Lengths: {[len(val) for val in data_segment.values()]}"
            )
            var_dim = find_singular_ragged_dim(data_segment)
            multi_index = ragged_index_tuples_fn(
                next(iter(data_segment.values())), var_dim
            )
            index_depth = len(multi_index[0])
            out_segment = SortedDict()
            use_tqdm = len(multi_index) > 100
            try:
                from tqdm import tqdm as _tqdm

                iterator = (
                    _tqdm(
                        multi_index,
                        desc=f"{transform_name}: transforming ragged segments",
                    )
                    if use_tqdm
                    else multi_index
                )
            except ImportError:
                iterator = multi_index

            for coord in iterator:
                subsegment = {
                    k: data_segment[k][coord].to_numpy() for k in data_segment.keys()
                }
                out_subsegment = [
                    transform_func(
                        NamedTransformInput(
                            metadata=(
                                copy.deepcopy(data_segment.metadata)
                                if data_segment.metadata is not None
                                else None
                            ),
                            **subsegment,
                        ),
                        metadata=metadata,
                    )
                ]
                out_subsegment = postprocess(out_subsegment, metadata)
                assert (
                    len(out_subsegment) == 1
                ), "Expected single output from transform."
                for k, block in out_subsegment[0].items():
                    d = out_segment.setdefault(k, SortedDict())
                    for i in coord[:-1]:
                        d = d.setdefault(i, SortedDict())
                    d[coord[-1]] = block

            for k, v in out_segment.items():
                try:
                    if index_depth == 1:
                        subarrays = [np.asarray(arr) for arr in v.values()]
                        lengths = np.fromiter(
                            (len(a) for a in subarrays), dtype=np.int64
                        )
                        offsets = np.concatenate([[0], np.cumsum(lengths)])
                        if len(subarrays) == 0:
                            aka = ak.Array([])
                        else:
                            content = ak.from_numpy(np.concatenate(subarrays))
                            layout = ak.contents.ListOffsetArray(
                                ak.index.Index64(offsets), content.layout
                            )
                            aka = ak.Array(layout)
                    else:
                        aka = ak.Array(tolist_fn(v))
                except Exception:
                    logger.error(
                        f"{transform_name} Fast reconstruction of ragged array from dense transform output failed."
                    )
                    aka = ak.Array(tolist_fn(v))
                regularized = ak_regularize_regular_axes(aka)
                var_dims = ak_find_var_dims(regularized)
                if len(var_dims) == 0:
                    out_segment[k] = ak.from_regular(regularized, axis=var_dim)
                elif len(var_dims) > 1:
                    raise ValueError(
                        f"Expected at most one variable-length dimension after reconstruction, found {len(var_dims)}: {var_dims}"
                    )
                else:
                    out_segment[k] = regularized

            out.append(out_segment)

        return out, transform_log


# -----------------------------------------------------------------------------
# RaggedRaggedHandler: ragged data, ragged-supporting transform
# -----------------------------------------------------------------------------


class RaggedRaggedHandler:
    """
    Handler for ragged data with a ragged-supporting transform.
    """

    def fit_prepare(
        self,
        data_segments: List[NamedTransformInput],
        apply_to_keys: List[str],
        metadata: Dict[str, Any],
        fit_func: Callable[..., None],
    ) -> None:
        find_singular_ragged_dim = _find_singular_ragged_dim()
        var_dim = set(find_singular_ragged_dim(ds) for ds in data_segments)
        assert (
            len(var_dim) == 1
        ), f"Expected exactly one variable-length dimension, found {var_dim}"
        var_dim = var_dim.pop()
        train = {}
        for key in apply_to_keys:
            train[key] = ak.concatenate(
                [chunk[key] for chunk in data_segments], axis=var_dim - 1
            )
        fit_func(train, metadata=metadata)

    def transform_apply(
        self,
        data_segments: List[NamedTransformInput],
        transform_func: Callable[..., Any],
        metadata: Dict[str, Any],
        transform_name: str,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        transform_log = {"mode": "ragged"}
        out = [transform_func(seg, metadata=metadata) for seg in data_segments]
        return out, transform_log


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

_TRANSFORM_HANDLER_REGISTRY: Dict[Tuple[str, str], TransformHandler] = {}


def _build_registry() -> Dict[Tuple[str, str], TransformHandler]:
    dense_dense = DenseDenseHandler()
    ragged_dense = RaggedDenseHandler()
    ragged_ragged = RaggedRaggedHandler()
    return {
        (DATA_KIND_DENSE, CAPABILITY_DENSE): dense_dense,
        (DATA_KIND_DENSE, CAPABILITY_BOTH): dense_dense,
        (DATA_KIND_RAGGED_REGULAR, CAPABILITY_DENSE): dense_dense,
        (DATA_KIND_RAGGED_REGULAR, CAPABILITY_BOTH): dense_dense,
        (DATA_KIND_RAGGED_REGULAR, CAPABILITY_RAGGED): ragged_ragged,
        (DATA_KIND_RAGGED, CAPABILITY_DENSE): ragged_dense,
        (DATA_KIND_RAGGED, CAPABILITY_RAGGED): ragged_ragged,
        (DATA_KIND_RAGGED, CAPABILITY_BOTH): ragged_ragged,
    }


def get_handler(data_kind: str, capability: str) -> TransformHandler:
    """
    Return the handler registered for a ``(data_kind, capability)`` pair.

    Parameters
    ----------
    data_kind : str
        Data-kind label used to select the handler family.
    capability : str
        Transform capability label used to select the handler implementation.

    Returns
    -------
    TransformHandler
        The handler associated with the requested pair.

    Raises
    ------
    KeyError
        If no handler is registered for the requested pair.
    """
    if not _TRANSFORM_HANDLER_REGISTRY:
        _TRANSFORM_HANDLER_REGISTRY.update(_build_registry())
    key = (data_kind, capability)
    if key not in _TRANSFORM_HANDLER_REGISTRY:
        raise KeyError(
            f"No handler registered for (data_kind={data_kind!r}, capability={capability!r})"
        )
    return _TRANSFORM_HANDLER_REGISTRY[key]
