from abc import ABC
from typing import Any, Dict, Optional

import numpy as np
from numpy import ndarray

from lmetk.transforms.base.base_transform import BaseTransform


class BaseMultiSourceUnitWiseTransformMixin(ABC):
    """
    Stateless mixin for transforming data from multiple sources.
    """

    def __init__(self):
        """
        Initialize the MultiSourceTransformMixin.
        """
        super().__init__()
        # Check if the class is a subclass of TransformerMixin
        if not hasattr(self, "fit_data") or not hasattr(self, "transform_data"):
            raise TypeError(
                "MultiSourceTransformMixin requires the class to implement fit and transform methods."
            )

    def fit_multi_source(
        self,
        data_segments: list[ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Get all train sources and fit the scalers to them.
        """

        train = np.concatenate(data_segments, axis=0)
        self.fit_data(train, metadata=metadata)

    def transform_multi_source(
        self,
        data_segments: list[ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list[ndarray]:
        """
        Transform the data from all sources using the fitted scalers.
        """
        return [
            self.transform_data(data_segment, metadata=metadata)
            for data_segment in data_segments
        ]


class ConstantScaler(BaseTransform, ConcatinatedFitMultiSourceTransformMixin):
    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor

    def fit_data(
        self,
        data_segment: ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """This transform do not require fitting."""
        return self

    def transform_data(
        self,
        data_segment: ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        return data_segment * self.factor
