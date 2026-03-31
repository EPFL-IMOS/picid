import logging
from typing import Any, Dict

import numpy as np

# Assuming this import path is correct for your project
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin
from picid.utils.decorators import check_transform_output_consistency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubsampleTransform(NoFitPerSegmentMixin, DenseTransform):
    def __init__(
        self,
        step: int,
        **kwargs,
    ):
        super().__init__()
        self.step = step

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> Any:
        lengths = [len(segment) for segment in data.values()]
        for key, value in data.items():
            data[key] = value[:: self.step]

        if len(set(lengths)) > 1:
            logger.warning(
                f"SubsampleTransform: The input data segments have different lengths {lengths}. "
                "This might lead to misalignment after subsampling."
            )
        else:
            lengths = [len(segment) for segment in data.values()]
            assert len(set(lengths)) == 1, (
                f"SubsampleTransform: The input data segments have different lengths {lengths} "
                "after subsampling, even though they had same length before. This might lead to misalignment."
            )
        return data

    # Keeping your original __call__ and transform methods for compatibility.
    # They simply delegate to the new transform_data method.
    def __call__(self, data: Any) -> Any:
        return self.transform_data(data, None)


class WindowedAggregationTransform(NoFitPerSegmentMixin, DenseTransform):
    """
    Apply non-overlapping window aggregation to each input segment.

    Parameters
    ----------
    window_size : int or str
        Window size or ``"full"`` to aggregate over the entire axis.
    step : int
        Sliding step between windows.
    agg : str, default="mean"
        Aggregation method.
    dim : int, default=0
        Axis along which to build windows.
    **kwargs
        Additional keyword arguments forwarded to the base transform.
    """

    def __init__(
        self,
        window_size: int | str,
        step: int,
        agg: str = "mean",
        dim: int = 0,
        **kwargs,
    ):
        super().__init__()
        if isinstance(window_size, int):
            assert window_size > 0, "window_size must be a positive integer."
        elif isinstance(window_size, str):
            assert window_size == "full", "window_size string must be 'full'."

        self.window_size = window_size
        self.step = step
        self.agg = agg
        self.dim = dim

        # Map string to numpy aggregation function
        self._agg_func = {
            "mean": np.mean,
            "sum": np.sum,
            "min": np.min,
            "max": np.max,
            "median": np.median,
            "std": np.std,
            "first": lambda x: x,  # Handled separately
            "last": lambda x: x,  # Handled separately
        }.get(agg)

        if self._agg_func is None:
            raise ValueError(f"Unsupported aggregation: {agg}")

    @check_transform_output_consistency
    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> Any:
        lengths = [len(segment) for segment in data.values()]

        for key, value in data.items():
            if self.window_size == "full":
                window_size = value.shape[self.dim]
            else:
                window_size = self.window_size

            arr = np.asarray(value)
            arr_moved = np.moveaxis(arr, self.dim, 0)
            windows = np.lib.stride_tricks.sliding_window_view(
                arr_moved, window_shape=window_size, axis=0
            )
            # the reason why we switch axis is that
            # sliding_window_view does not support step and they
            # recommend this trick, which works easiest if its always
            # in the same dimension (here first):
            strided_windows = windows[:: self.step, ...]
            strided_windows = np.moveaxis(strided_windows, 0, self.dim)

            # Stride tricks adds the window in the last dimension.
            # so we apply aggregation in dim -1.
            if self.agg == "first":
                data[key] = np.take(strided_windows, 0, axis=-1)
            elif self.agg == "last":
                data[key] = np.take(strided_windows, -1, axis=-1)
            else:
                data[key] = self._agg_func(strided_windows, axis=-1)

        if len(set(lengths)) > 1:
            logger.warning(
                f"SubsampleTransform: The input data segments have different lengths {lengths}. "
                "This might lead to misalignment after subsampling."
            )
        else:
            lengths = [len(segment) for segment in data.values()]
            assert len(set(lengths)) == 1, (
                f"SubsampleTransform: The input data segments have different lengths {lengths} "
                "after subsampling, even though they had same length before. This might lead to misalignment."
            )
        return data
