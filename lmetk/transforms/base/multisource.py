from abc import ABC
from typing import Any, Dict, Optional

import numpy as np
from numpy import ndarray

import logging

logger = logging.getLogger(__name__)


class ConcatinatedFitMultiSourceTransformMixin(ABC):
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
                "ConcatinatedFitMultiSourceTransformMixin requires the class to implement fit and transform methods."
            )

    def fit_multi_source(
        self,
        data_segments: list[ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Get all train sources and fit the scalers to them.
        """
        # Go over all data segments dict keys and concatenate the data from all sources
        train = {}
        for key, val in data_segments.items():
            train[key] = np.concatenate(val, axis=0)
        self.fit_data(train, metadata=metadata)

    def transform_multi_source(
        self,
        data_segments: list[ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list[ndarray]:
        """
        Transform the data from all sources using the fitted scalers.
        """

        lengths = [len(val) for val in data_segments.values()]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"The lengths of the data segments are not the same. Lengths: {lengths}"
            )

        out = []
        for i in range(lengths[0]):
            data_segment = {}

            # Construct current data segment from each source
            for key, val in data_segments.items():
                data_segment[key] = val[i]

            # Transform the current data segment
            out.append(self.transform_data(data_segment, metadata=metadata))

        return out


class NoFitMultiSourceTransformMixin(ConcatinatedFitMultiSourceTransformMixin):
    """
    A stateless mixin for transforming data from multiple sources without fitting.
    This mixin is intended for use with transforms that do not require or support
    fitting on multiple data sources. Any attempt to fit the transform using
    `fit_multi_source` will raise a `NotImplementedError`.
    Inheritance:
        ConcatinatedFitMultiSourceTransformMixin
    Methods:
        fit_multi_source(data_segments, metadata=None):
            Raises NotImplementedError to indicate fitting is not supported.
    Attributes:
        None
    Usage:
        Use this mixin when implementing transforms that operate on multiple sources
        but do not require a fitting step.
    """

    def fit_multi_source(
        self,
        data_segments: list[ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError(
            "fit_multi_source is not implemented for this transform."
        )
