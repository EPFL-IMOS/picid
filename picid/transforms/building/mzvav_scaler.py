"""Min-max scaler transform for the MZVAV building dataset."""

import logging
from typing import Any

from sklearn.preprocessing import MinMaxScaler

import awkward as ak

from picid.data.data_objects import NamedTransformInput
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
    """Adapt sklearn MinMaxScaler to the BaseTransform interface using Awkward arrays."""

    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        """
        Fit the internal scaler on the single provided array.

        Parameters
        ----------
        data : NamedTransformInput
            Input container with exactly one array-like entry.
        metadata : dict
            Metadata dictionary, preserved for interface compatibility.
        """
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
        """
        Scale the single array and return it in Awkward form.

        Parameters
        ----------
        data : NamedTransformInput
            Input container with exactly one array-like entry.
        metadata : dict
            Metadata dictionary, preserved for interface compatibility.

        Returns
        -------
        Any
            Scaled array converted back to Awkward form.
        """
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
        """
        Invert the min-max scaling for the single provided array.

        Parameters
        ----------
        data : NamedTransformInput
            Input container with exactly one array-like entry.
        metadata : dict, optional
            Metadata dictionary, preserved for interface compatibility.

        Returns
        -------
        Any
            Inverse-transformed array converted back to Awkward form.
        """
        keys = list(data.keys())
        assert (
            len(keys) == 1
        ), "MinMaxScalerSklearn only supports single key data_segment."

        np_data, _ = _convert_to_numpy(data[keys[0]])

        original_shape = np_data.shape
        np_data = np_data.reshape(-1, np_data.shape[-1])

        inverse_transformed_np_data = self.scaler.inverse_transform(np_data)

        ak_data = ak.from_numpy(inverse_transformed_np_data.reshape(original_shape))
        ak_ragged = ak.from_regular(ak_data, axis=1)
        return ak_ragged
