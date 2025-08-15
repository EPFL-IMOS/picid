import logging
from typing import Any, Dict, Optional

import numpy as np
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.multisource import BaseMultiSourceTransformMixin
from lmetk.transforms.utils import _convert_from_numpy, _convert_to_numpy

logger = logging.getLogger(__name__)


class ConstantScaler(BaseTransform, BaseMultiSourceTransformMixin):
    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor

    def fit_data(
        self,
        data_segment: ndarray,
        mode: str,
        auxiliary_data: Optional[Dict[str, Any]] = None,
    ):
        """This transform do not require fitting."""
        return self

    def transform_data(
        self,
        data_segment: ndarray,
        mode: str,
        auxiliary_data: Optional[Dict[str, Any]] = None,
    ):
        return data_segment * self.factor


class MinMaxScalerSklearn(BaseTransform, BaseMultiSourceTransformMixin):
    """
    An adapter for sklearn.preprocessing.MinMaxScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_data(
        self, data_segment: Any, auxiliary_data: Optional[Dict], mode: Optional[str]
    ):
        """Fits the internal MinMaxScaler on a single NumPy array."""
        np_data, _ = _convert_to_numpy(data_segment)

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for MinMaxScaler.")

        self.scaler.fit(np_data)

    def transform_data(
        self, data_segment: Any, auxiliary_data: Optional[Dict], mode: Optional[str]
    ) -> Any:
        """Transforms a single data segment using the internal MinMaxScaler."""
        np_data, original_type = _convert_to_numpy(data_segment)

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for MinMaxScaler.")

        transformed_np_data = self.scaler.transform(np_data)
        return transformed_np_data


class SklearnStandardScalerAdapter(BaseTransform, BaseMultiSourceTransformMixin):
    """
    An adapter for sklearn.preprocessing.StandardScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_data(
        self, data_segment: Any, auxiliary_data: Optional[Dict], mode: Optional[str]
    ):
        """Fits the internal StandardScaler on a single NumPy array."""
        np_data, _ = _convert_to_numpy(data_segment)
        self.scaler.fit(np_data)

    def transform_data(
        self, data_segment: Any, auxiliary_data: Optional[Dict], mode: Optional[str]
    ) -> Any:
        """Transforms a single data segment using the internal StandardScaler."""
        np_data, original_type = _convert_to_numpy(data_segment)
        transformed_np_data = self.scaler.transform(np_data)
        return transformed_np_data  # _convert_from_numpy(transformed_np_data, original_type, data_segment)


# Example: Adapter for sklearn.preprocessing.MinMaxScaler


class SklearnMinMaxScalerAdapter(BaseTransform):
    """
    An adapter for sklearn.preprocessing.MinMaxScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_data(
        self, data_segment: Any, auxiliary_data: Optional[Dict], mode: Optional[str]
    ):
        """Fits the internal MinMaxScaler on a single NumPy array."""
        np_data, _ = _convert_to_numpy(data_segment)
        self.scaler.fit(np_data)

    def transform_data(
        self, data_segment: Any, auxiliary_data: Optional[Dict], mode: Optional[str]
    ) -> Any:
        """Transforms a single data segment using the internal MinMaxScaler."""
        np_data, original_type = _convert_to_numpy(data_segment)
        transformed_np_data = self.scaler.transform(np_data)
        return transformed_np_data  # _convert_from_numpy(transformed_np_data, original_type, data_segment)


# Example: A simple custom direct transform (no fitting)
class LogTransformAdapter(BaseTransform):
    """
    A simple adapter for a logarithmic transformation, conforming to BaseTransform.
    This is a direct transform (no fitting required). Handles various input data types
    by converting to numpy.ndarray internally and converting back.
    """

    def fit_data(
        self, data_segment: Any, auxiliary_data: Optional[Dict], mode: Optional[str]
    ):
        """No fitting needed for a direct log transform."""
        pass

    def transform_data(
        self, data_segment: Any, auxiliary_data: Optional[Dict], mode: Optional[str]
    ) -> Any:
        """Applies a natural logarithm transformation."""
        np_data, original_type = _convert_to_numpy(data_segment)

        # Add a small constant to avoid log(0) issues for non-negative data
        # Use np.maximum to handle negative numbers gracefully if they occur
        transformed_np_data = np.log1p(np.maximum(0, np_data))  # log(1 + x) for x >= 0

        return transformed_np_data  # _convert_from_numpy(transformed_np_data, original_type, data_segment)
