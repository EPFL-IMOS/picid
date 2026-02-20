import logging
from typing import Any, Dict

import numpy as np

from picid.data.data_objects.data import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin

# Assuming this import path is correct for your project

logger = logging.getLogger(__name__)


class FillTransform(DenseTransform, NoFitPerSegmentMixin):

    def __init__(self, **kwargs):
        self._init_kwargs = kwargs

    def __repr__(self):
        # Ensure 'freq' is included in the representation, along with any other init args
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}(freq={self.freq!r}, {args})"

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any:

        # Your original working logic: Convert input to DataFrame and extract 'date' values.
        # This assumes the 'data_segment' can be directly converted to a DataFrame
        # and has a 'date' column, or can be implicitly handled by pd.to_datetime.

        for key, value in data.items():
            data[key] = value.ffill()

        # Return the values as a NumPy array.
        return data

    # Keeping your original __call__ and transform methods for compatibility.
    # They simply delegate to the new transform_data method.
    def __call__(self, data_segment: Any, auxiliary_data: Dict, mode: str) -> Any:
        return self.transform_data(data_segment, auxiliary_data, mode)