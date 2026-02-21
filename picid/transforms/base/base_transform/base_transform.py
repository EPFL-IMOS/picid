"""
Base transform interface and data-type markers.

BaseTransform is the abstract base for all transforms: it defines
transform_data (and optional fit_data). Subclasses combine with multisource
mixins (e.g. ConcatFitAndPerSegmentTransformMixin) and optionally with RaggedTransform /
DenseTransform / RaggedOrDenseTransform to declare capability for handler dispatch.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from picid.data.data_objects.data import NamedTransformInput
from picid.utils.decorators import check_transform_output_consistency


class BaseTransform(ABC):
    """
    Simplified base transform interface. Transforms only need to implement
    the core transformation logic, not the fitting strategies.
    """

    def __init__(self, exclude_keys: list[str] = None, **kwargs):
        self.exclude_keys = exclude_keys if exclude_keys is not None else []
        self._init_kwargs = kwargs
        # Set requires_fit from fit-policy mixin (avoids pipeline needing to know mixin types)
        from picid.transforms.base.multisource import FitByConcatenationMixin, NoFitMixin
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
            The data to transform
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
        Fit the transform on data. Default implementation does nothing (for stateless transforms).
        Override this method for transforms that need fitting.

        Parameters
        ----------
        data : NamedTransformInput
            The data to fit on.
        metadata : Dict[str, Any]
            Same keys as for transform_data (mode, apply_to_keys, assign_to_map,
            and optionally slice_info when metadata.include_slice_info_in_metadata is true).
        """
        pass

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
