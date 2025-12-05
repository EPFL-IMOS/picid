import logging
from typing import Any, Dict

import numpy as np

# Assuming this import path is correct for your project
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import RaggedOrDenseTransform, DenseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
from lmetk.utils.decorators import check_transform_output_consistency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptClassesBuilder(DenseTransform, NoFitMultiSourceTransformMixin):

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__()

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> Any:

        concepts = data["concepts"].copy()

        if not np.isin(concepts, [0, 1]).all():
            concepts = np.rint(concepts)
            logger.warning(
                "Non-binary concept was detected and is now rounded to 0 or 1"
            )

        assert np.isin(
            concepts, [0, 1]
        ).all(), "Non-binary concept is remaining and not supported"
        assert (
            concepts.sum(axis=1) <= 1
        ).all(), "Combined error modes detected and not supported"

        max_index = np.argmax(concepts, axis=1)
        all_zero = np.all(concepts == 0, axis=1)
        classes = max_index + 1
        classes[all_zero] = 0

        data["concepts"] = classes[:, None]

        return data

    # Keeping your original __call__ and transform methods for compatibility.
    # They simply delegate to the new transform_data method.
    def __call__(self, data: Any) -> Any:
        return self.transform_data(data, None)
