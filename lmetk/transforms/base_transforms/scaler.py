import logging
from typing import Any

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import (
    ConcatinatedFitMultiSourceTransformMixin,
    InverseTransformMixin,
)
from lmetk.transforms.utils import _convert_to_numpy

logger = logging.getLogger(__name__)


class ConstantScaler(
    DenseTransform, ConcatinatedFitMultiSourceTransformMixin, InverseTransformMixin
):
    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        """This transform do not require fitting."""
        return self

    def transform_data(
        self,
        data: NamedTransformInput,
        metadata: dict,
    ):
        assert len(data) == 1, "ConstantScaler only supports single key data_segment."

        data = list(data.values())[0]
        return data * self.factor

    def inverse_transform(
        self,
        data: NamedTransformInput,
        metadata: dict = None,
    ):
        assert len(data) == 1, "ConstantScaler only supports single key data_segment."

        data = list(data.values())[0]
        return data / self.factor


class MinMaxScalerSklearn(
    DenseTransform, ConcatinatedFitMultiSourceTransformMixin, InverseTransformMixin
):
    """
    An adapter for sklearn.preprocessing.MinMaxScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        """Fits the internal MinMaxScaler on a single NumPy array."""
        keys = list(data.keys())
        assert (
            len(keys) == 1
        ), "MinMaxScalerSklearn only supports single key data_segment."

        np_data, _ = _convert_to_numpy(data[keys[0]])

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for MinMaxScaler.")

        self.scaler.fit(np_data)

    def transform_data(
        self,
        data: NamedTransformInput,
        metadata: dict,
    ) -> Any:
        """Transforms a single data segment using the internal MinMaxScaler."""
        keys = list(data.keys())
        assert (
            len(keys) == 1
        ), "MinMaxScalerSklearn only supports single key data_segment."

        np_data, _ = _convert_to_numpy(data[keys[0]])

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for MinMaxScaler.")

        transformed_np_data = self.scaler.transform(np_data)
        return transformed_np_data

    def inverse_transform(
        self,
        data: NamedTransformInput,
        metadata: dict = None,
    ) -> Any:
        """Inverse transforms a single data segment using the internal MinMaxScaler."""
        keys = list(data.keys())
        assert (
            len(keys) == 1
        ), "MinMaxScalerSklearn only supports single key data_segment."

        np_data, _ = _convert_to_numpy(data[keys[0]])

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for MinMaxScaler.")

        inverse_transformed_np_data = self.scaler.inverse_transform(np_data)
        return inverse_transformed_np_data


class StandardScalerSklearn(
    DenseTransform, ConcatinatedFitMultiSourceTransformMixin, InverseTransformMixin
):
    """
    An adapter for sklearn.preprocessing.StandardScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        """Fits the internal StandardScaler on a single NumPy array."""
        keys = list(data.keys())
        assert len(keys) == 1, "StandardScaler only supports single key data_segment."

        np_data = data[keys[0]]

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for StandardScaler.")

        self.scaler.fit(np_data)

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any:
        """Transforms a single data segment using the internal StandardScaler."""
        keys = list(data.keys())
        assert len(keys) == 1, "StandardScaler only supports single key data_segment."

        np_data = data[keys[0]]

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for StandardScaler.")

        assert len(
            metadata["assign_to_map"]
        ), "StandardScaler only supports single assign_to_map map."

        transformed_np_data = self.scaler.transform(np_data)
        return transformed_np_data

    def inverse_transform(
        self, data: NamedTransformInput, metadata: dict = None
    ) -> Any:
        """Inverse transforms a single data segment using the internal StandardScaler."""
        keys = list(data.keys())
        assert len(keys) == 1, "StandardScaler only supports single key data_segment."

        np_data = data[keys[0]]

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for StandardScaler.")

        inverse_transformed_np_data = self.scaler.inverse_transform(np_data)
        return inverse_transformed_np_data
