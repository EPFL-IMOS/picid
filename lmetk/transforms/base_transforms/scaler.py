import logging
from typing import Any, Dict, Optional

import numpy as np
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin
from lmetk.transforms.utils import _convert_from_numpy, _convert_to_numpy

logger = logging.getLogger(__name__)


class ConstantScaler(BaseTransform, ConcatinatedFitMultiSourceTransformMixin):
    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor

    def fit_data(
        self,
        data_segment: Dict,
        metadata: Optional[Dict] = None,
    ):
        """This transform do not require fitting."""
        return self

    def transform_data(
        self,
        data_segment: Dict,
        metadata: Optional[Dict] = None,
    ):
        return data_segment * self.factor


class MinMaxScalerSklearn(BaseTransform, ConcatinatedFitMultiSourceTransformMixin):
    """
    An adapter for sklearn.preprocessing.MinMaxScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_data(self, data_segment: Any, metadata: Optional[Dict]):
        """Fits the internal MinMaxScaler on a single NumPy array."""
        keys = list(data_segment.keys())
        assert (
            len(keys) == 1
        ), "MinMaxScalerSklearn only supports single key data_segment."

        np_data, _ = _convert_to_numpy(data_segment[keys[0]])

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for MinMaxScaler.")

        self.scaler.fit(np_data)

    def transform_data(
        self,
        data_segment: Any,
        metadata: Optional[Dict],
    ) -> Any:
        """Transforms a single data segment using the internal MinMaxScaler."""
        keys = list(data_segment.keys())
        assert (
            len(keys) == 1
        ), "MinMaxScalerSklearn only supports single key data_segment."

        np_data, _ = _convert_to_numpy(data_segment[keys[0]])

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for MinMaxScaler.")

        transformed_np_data = self.scaler.transform(np_data)
        return transformed_np_data


class StandardScalerSklearn(BaseTransform, ConcatinatedFitMultiSourceTransformMixin):
    """
    An adapter for sklearn.preprocessing.StandardScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_data(self, data_segment: Any, metadata: Optional[Dict]):
        """Fits the internal StandardScaler on a single NumPy array."""
        keys = list(data_segment.keys())
        assert len(keys) == 1, "StandardScaler only supports single key data_segment."

        np_data, _ = _convert_to_numpy(data_segment[keys[0]])

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for MinMaxScaler.")

        self.scaler.fit(np_data)

    def transform_data(
        self,
        data_segment: Any,
        metadata: Optional[Dict],
    ) -> Any:
        """Transforms a single data segment using the internal StandardScaler."""
        keys = list(data_segment.keys())
        assert len(keys) == 1, "StandardScaler only supports single key data_segment."

        np_data, _ = _convert_to_numpy(data_segment[keys[0]])

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for MinMaxScaler.")

        transformed_np_data = self.scaler.transform(np_data)
        return transformed_np_data
