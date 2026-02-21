import logging
from typing import Any

from sklearn.preprocessing import MinMaxScaler

import awkward as ak

from picid.data.data_objects.data import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import (
    ConcatFitAndPerSegmentTransformMixin,
    InverseTransformMixin,
)
from picid.transforms.utils import _convert_to_numpy

logger = logging.getLogger(__name__)


class MinMaxScalerMZVAV(
    ConcatFitAndPerSegmentTransformMixin, InverseTransformMixin, DenseTransform
):
    """
    An adapter for sklearn.preprocessing.MinMaxScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """

    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        """Fits the internal MinMaxScaler on a single NumPy array."""
        keys = list(data.keys())
        assert (
            len(keys) == 1
        ), "MinMaxScalerSklearn only supports single key data_segment."

        np_data = ak.to_numpy(data[keys[0]])
        np_data = np_data.reshape(-1, np_data.shape[-1])

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

        original_shape = np_data.shape
        np_data = np_data.reshape(-1, np_data.shape[-1])

        transformed_np_data = self.scaler.transform(np_data)

        ak_data = ak.from_numpy(transformed_np_data.reshape(original_shape))
        ak_ragged = ak.from_regular(ak_data, axis=1)
        return ak_ragged

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

        original_shape = np_data.shape
        np_data = np_data.reshape(np_data.shape[0], -1)

        inverse_transformed_np_data = self.scaler.inverse_transform(np_data)

        ak_data = ak.from_numpy(inverse_transformed_np_data.reshape(original_shape))
        ak_ragged = ak.from_regular(ak_data, axis=1)
        return ak_ragged
