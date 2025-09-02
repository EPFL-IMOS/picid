import logging
from typing import Any, Dict

# Assuming this import path is correct for your project
from lmetk.data.data_objects.data import OrderedInput
from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubsampleTransform(BaseTransform, ConcatinatedFitMultiSourceTransformMixin):

    def __init__(
        self,
        step: int,
        **kwargs,
    ):

        super().__init__()
        self.step = step

    def transform_data(self, data_segment: OrderedInput, metadata: Dict) -> Any:

        lengths = [len(segment) for segment in data_segment.values()]
        for key, value in data_segment.items():
            data_segment[key] = value[:: self.step]

        if len(set(lengths)) > 1:
            logger.warning(
                f"SubsampleTransform: The input data segments have different lengths {lengths}. "
                "This might lead to misalignment after subsampling."
            )
        else:
            lengths = [len(segment) for segment in data_segment.values()]
            assert len(set(lengths)) == 1, (
                f"SubsampleTransform: The input data segments have different lengths {lengths} "
                "after subsampling, even though they had same length before. This might lead to misalignment."
            )
        return data_segment

    # Keeping your original __call__ and transform methods for compatibility.
    # They simply delegate to the new transform_data method.
    def __call__(self, data: Any) -> Any:
        return self.transform_data(data, None)
