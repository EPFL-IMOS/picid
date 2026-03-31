"""
Base transform interface and data-type markers.

This module defines the abstract transform contract together with lightweight
marker classes that advertise whether a transform works on ragged, dense, or
mixed array representations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from picid.data.data_objects import NamedTransformInput
from picid.utils.decorators import check_transform_output_consistency


class BaseTransform(ABC):
    """
    Abstract base class for all transforms.

    Parameters
    ----------
    exclude_keys : list[str], optional
        Keys that should be skipped by the transform.
    **kwargs
        Extra constructor arguments kept for introspection and logging.
    """

    def __init__(self, exclude_keys: list[str] = None, **kwargs):
        self.exclude_keys = exclude_keys if exclude_keys is not None else []
        self._init_kwargs = kwargs
        # Set requires_fit from fit-policy mixin (avoids pipeline needing to know mixin types)
        from picid.transforms.base.multisource import (
            FitByConcatenationMixin,
            NoFitMixin,
        )

        if issubclass(type(self), FitByConcatenationMixin):
            self.requires_fit = True
        elif issubclass(type(self), NoFitMixin):
            self.requires_fit = False
        else:
            self.requires_fit = getattr(self, "requires_fit", False)

    @check_transform_output_consistency
    @abstractmethod
    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> Any:
        """
        Transform a single data segment.

        Parameters
        ----------
        data : Any
            The data to transform.
        metadata : Dict[str, Any]
            Additional data needed for transformation. May contain:
            mode : str
                Transformation mode (e.g. 'train', 'val', 'test').
            apply_to_keys, assign_to_keys, assign_to_map
                Pipeline key configuration.
            slice_info : dict, optional
                Serialized slice (split, unit_ids, cycle_ids, bounds, index_map).
                Only present when the transform's config sets
                metadata.include_slice_info_in_metadata: true and the container
                has slice_info set.
        """
        pass

    @abstractmethod
    def transform_multi_source(
        self,
        chunks: List[NamedTransformInput],
        metadata: Dict[str, Any],
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Transform all units for a given split.

        TransformStep always calls this method. Subclasses must provide it,
        either by inheriting a mixin or by implementing it directly.

        Parameters
        ----------
        chunks : List[NamedTransformInput]
            One chunk per unit for this split, in unit order.
        metadata : dict
            Pipeline metadata (same as transform_data receives).

        Returns
        -------
        results : List[Any]
            One transformed result per input chunk, in the same order.
        log : dict
            Per-split log information. Use {} if nothing to log.
        """
        raise NotImplementedError

    def fit_data(self, data: NamedTransformInput, metadata: Dict[str, Any]) -> Any:
        """
        Fit the transform on data.

        Parameters
        ----------
        data : NamedTransformInput
            The data to fit on.
        metadata : Dict[str, Any]
            Same keys as for transform_data (mode, apply_to_keys, assign_to_map,
            and optionally slice_info when metadata.include_slice_info_in_metadata is true).
        """
        pass

    def propagate_unit_metadata(
        self,
        *,
        unit_metadata_by_split: Dict[str, List[Dict[str, Any]]],
        transformed_results_for_new_key: Dict[str, Dict[str, List[Any]]],
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Optionally return replacement unit metadata after a transform runs.

        Default behaviour is a no-op: return ``None`` and let the pipeline keep
        the existing ``unit_metadata`` unchanged. Transforms that change unit
        cardinality can override this hook and return a replacement split-keyed
        mapping.

        Parameters
        ----------
        unit_metadata_by_split : dict[str, list[dict[str, Any]]]
            Current metadata grouped by split before the transform result is merged.
        transformed_results_for_new_key : dict[str, dict[str, list[Any]]]
            Postprocessed transformed values grouped by assign-to key and split.
        metadata : dict[str, Any]
            The per-transform metadata dictionary built by the pipeline.

        Returns
        -------
        dict[str, list[dict[str, Any]]] or None
            Replacement unit metadata, or ``None`` to preserve the current mapping.

        Notes
        -----
        The hook lives on the transform because a cardinality change is part of
        the transform's semantics, not a global pipeline concern. The transform
        knows whether it preserved one unit per output item, collapsed many units
        into one, or intentionally destroyed unit identity altogether.
        """
        return None

    def __call__(self, data: Any) -> Any:
        return self.transform_data(data, None)

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        exclude_str = f"exclude_keys={self.exclude_keys!r}"
        if args:
            return f"{self.__class__.__name__}({exclude_str}, {args})"
        return f"{self.__class__.__name__}({exclude_str})"


class RaggedTransform(BaseTransform):
    """Marker base class for transforms that support ragged (awkward) arrays."""

    pass


class DenseTransform(BaseTransform):
    """Marker base class for transforms that support only dense (NumPy) arrays."""

    pass


class RaggedOrDenseTransform(BaseTransform):
    """Marker base class for transforms that support both ragged and dense arrays."""

    pass
