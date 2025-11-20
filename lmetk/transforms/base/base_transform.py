from abc import ABC, abstractmethod
from typing import Any, Dict

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.utils.decorators import check_transform_output_consistency


class BaseTransform(ABC):
    """
    Simplified base transform interface. Transforms only need to implement
    the core transformation logic, not the fitting strategies.
    """

    def __init__(self, exclude_keys: list[str] = None, **kwargs):
        self.exclude_keys = exclude_keys if exclude_keys is not None else []
        self._init_kwargs = kwargs

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
        metadata : Optional[Dict[str, Any]]
            Additional data needed for transformation, Transformation mode (e.g., 'train', 'val', 'test')
        """
        pass

    def fit_data(self, data: NamedTransformInput, metadata: Dict[str, Any]) -> Any:
        """
        Fit the transform on data. Default implementation does nothing (for stateless transforms).
        Override this method for transforms that need fitting.
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
