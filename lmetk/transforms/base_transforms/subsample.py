import logging
from typing import Any, Dict

import numpy as np

# Assuming this import path is correct for your project
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubsampleTransform(DenseTransform, NoFitMultiSourceTransformMixin):

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


class WindowedAggregationTransform(DenseTransform, NoFitMultiSourceTransformMixin):
    """
    Applies a sliding window aggregation to the input data.
    """

    def __init__(
        self,
        window_size: int,
        step: int,
        agg: str = "mean",
        **kwargs,
    ):
        super().__init__()
        self.window_size = window_size
        self.step = step
        self.agg = agg

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

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> Any:

        lengths = [len(segment) for segment in data.values()]

        for key, value in data.items():
            arr = np.asarray(value)
            n_windows = (len(arr) - self.window_size) // self.step + 1
            if n_windows <= 0:
                data[key] = np.array([])
                continue

            if self.agg not in ["first", "last"]:
                windows = [
                    arr[i * self.step : i * self.step + self.window_size]
                    for i in range(n_windows)
                ]
                data[key] = self._agg_func(np.stack(windows), axis=1)
            elif self.agg == "first":
                data[key] = arr[:: self.step][:n_windows]
            elif self.agg == "last":
                data[key] = arr[self.window_size - 1 :: self.step][:n_windows]

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
