from typing import Any, Dict
import numpy as np
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin
import logging

logger = logging.getLogger(__name__)


class PadToLength(NoFitPerSegmentMixin, DenseTransform):
    """
    Pad a NumPy array along one axis to reach a minimum length.

    Parameters
    ----------
    target_length : int
        Minimum length along the specified axis after padding.
    axis : int, default=0
        Axis along which to pad, supporting negative indexing.
    pad_value : float, default=0.0
        Value used for padding.

    Examples
    --------
    Input shape: ``(time, channels)``, ``axis=0`` pads time steps.
    Input shape: ``(batch, time, ch)``, ``axis=1`` pads time steps.
    """

    def __init__(self, target_length: int, axis: int = 0, pad_value: float = 0.0):
        super().__init__()
        self.target_length = target_length
        self.axis = axis
        self.pad_value = pad_value

    def transform_data(self, data: Any, metadata: Dict) -> Any:
        """
        Pad the array in ``data["features"]`` along the configured axis.

        Parameters
        ----------
        data : Any
            Input mapping containing a ``"features"`` NumPy array.
        metadata : Dict
            Auxiliary transform metadata.

        Returns
        -------
        Any
            The original or padded NumPy array.
        """
        arr = data["features"]

        if not isinstance(arr, np.ndarray):
            raise TypeError(f"PadToLength expects numpy array, got {type(arr)}")

        # logger.info(f"Original shape: {arr.shape}")

        # Resolve axis if negative
        axis = self.axis if self.axis >= 0 else arr.ndim + self.axis
        if axis < 0 or axis >= arr.ndim:
            raise ValueError(
                f"Invalid axis {self.axis} for array with shape {arr.shape}"
            )

        current_len = arr.shape[axis]
        if current_len >= self.target_length:
            # logger.info(
            #     f"No padding needed (current length {current_len} >= target {self.target_length})"
            # )
            return arr  # already long enough

        pad_amount = self.target_length - current_len
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, pad_amount)

        # logger.info(
        #     f"Padding axis {axis} from {current_len} to {self.target_length} "
        #     f"with pad value {self.pad_value}"
        # )

        padded = np.pad(arr, pad_width, mode="constant", constant_values=self.pad_value)
        return padded

    def fit_data(self, data: Any, metadata: Dict):
        """
        Skip fitting because this is a stateless transform.

        Parameters
        ----------
        data : Any
            Input data provided by the pipeline.
        metadata : Dict
            Auxiliary transform metadata.
        """
        pass
