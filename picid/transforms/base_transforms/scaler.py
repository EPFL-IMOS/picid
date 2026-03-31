import logging
from typing import Any

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import (
    ConcatFitAndPerSegmentTransformMixin,
    NoFitPerSegmentMixin,
    InverseTransformMixin,
)
from picid.transforms.utils import _convert_to_numpy

logger = logging.getLogger(__name__)


class ConstantScaler(NoFitPerSegmentMixin, InverseTransformMixin, DenseTransform):
    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        """
        Return ``self`` because this transform does not require fitting.

        Parameters
        ----------
        data : NamedTransformInput
            Input segments for the transform.
        metadata : dict
            Auxiliary transform metadata.

        Returns
        -------
        ConstantScaler
            The current instance.
        """
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
        """
        Fit the internal scaler on the single input segment.

        Parameters
        ----------
        data : NamedTransformInput
            Input segments for the transform.
        metadata : dict
            Auxiliary transform metadata.
        """
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
        """
        Transform the single input segment with min-max scaling.

        Parameters
        ----------
        data : NamedTransformInput
            Input segments for the transform.
        metadata : dict
            Auxiliary transform metadata.

        Returns
        -------
        Any
            The min-max scaled segment.
        """
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
        """
        Invert the min-max scaling for the single input segment.

        Parameters
        ----------
        data : NamedTransformInput
            Input segments for the inverse transform.
        metadata : dict, optional
            Auxiliary transform metadata.

        Returns
        -------
        Any
            The recovered single input segment.
        """
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
    ConcatFitAndPerSegmentTransformMixin, InverseTransformMixin, DenseTransform
):
    """
    An adapter for sklearn.preprocessing.StandardScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """

    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        """
        Fit the internal scaler on the single input segment.

        Parameters
        ----------
        data : NamedTransformInput
            Input segments for the transform.
        metadata : dict
            Auxiliary transform metadata.
        """
        keys = list(data.keys())
        assert len(keys) == 1, "StandardScaler only supports single key data_segment."

        np_data = data[keys[0]]

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for StandardScaler.")

        self.scaler.fit(np_data)

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any:
        """
        Transform the single input segment with standard scaling.

        Parameters
        ----------
        data : NamedTransformInput
            Input segments for the transform.
        metadata : dict
            Auxiliary transform metadata.

        Returns
        -------
        Any
            The standardized segment.
        """
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
        """
        Invert the standard scaling for the single input segment.

        Parameters
        ----------
        data : NamedTransformInput
            Input segments for the inverse transform.
        metadata : dict, optional
            Auxiliary transform metadata.

        Returns
        -------
        Any
            The recovered single input segment.
        """
        keys = list(data.keys())
        assert len(keys) == 1, "StandardScaler only supports single key data_segment."

        np_data = data[keys[0]]

        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for StandardScaler.")

        inverse_transformed_np_data = self.scaler.inverse_transform(np_data)
        return inverse_transformed_np_data
