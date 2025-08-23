from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from numpy import ndarray



class BaseTransform(ABC):
    """
    Simplified base transform interface. Transforms only need to implement
    the core transformation logic, not the fitting strategies.
    """

    @abstractmethod
    def transform_data(self, 
                      data_segment: ndarray | list[ndarray] , 
                      metadata: Optional[Dict[str, Any]] = None
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
    
    
    def fit_data(self, 
                 data_segment: ndarray | list[ndarray], 
                 metadata: Optional[Dict[str, Any]] = None) -> 'BaseTransform':
        """
        Fit the transform on data. Default implementation does nothing (for stateless transforms).
        Override this method for transforms that need fitting.
        """
        return self

    def supports_multi_source_fit(self) -> bool:
        """
        Whether this transform can handle multi-source fitting.
        Override to return True if the transform can handle aggregated data from multiple sources.
        """
        return True  # Most transforms can handle concatenated data

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        exclude_str = f"exclude_keys={self.exclude_keys!r}"
        if args:
            return f"{self.__class__.__name__}({exclude_str}, {args})"
        return f"{self.__class__.__name__}({exclude_str})"
