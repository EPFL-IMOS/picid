import logging
from typing import Any, Dict

import numpy as np

from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin

# Assuming this import path is correct for your project


logger = logging.getLogger(__name__)


class RailwayDummyTransform(NoFitPerSegmentMixin, DenseTransform):
    def __init__(self, **kwargs):
        self._init_kwargs = kwargs
        self.shift = kwargs["shift"]

    def __repr__(self):
        # Ensure 'freq' is included in the representation, along with any other init args
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}(freq={self.freq!r}, {args})"

    def transform_data(self, data_segment: Any, auxiliary_data: Dict, mode: str) -> Any:
        # Your original working logic: Convert input to DataFrame and extract 'date' values.
        # This assumes the 'data_segment' can be directly converted to a DataFrame
        # and has a 'date' column, or can be implicitly handled by pd.to_datetime.

        assert (
            self.shift
            in [
                24,
                168,
                100,
            ]
        ), "dummy shift must be 24 for the timepfn, and 168 for the tabpfn for railway and 100 for the battery!"

        dummy_leak = np.concatenate(
            [np.zeros((self.shift, 1)), auxiliary_data["target"][: -self.shift]]
        )
        assert dummy_leak.shape[0] == data_segment.shape[0]
        initial_dates = list(auxiliary_data["initial_dates"])
        initial_dates_shifted = initial_dates[self.shift :]
        logger.info(initial_dates[:5])
        logger.info(initial_dates_shifted[:5])

        data_segment = np.concatenate([data_segment, dummy_leak], axis=1)

        # Return the values as a NumPy array.
        return data_segment

    # Keeping your original __call__ and transform methods for compatibility.
    # They simply delegate to the new transform_data method.
    def __call__(self, data_segment: Any, auxiliary_data: Dict, mode: str) -> Any:
        return self.transform_data(data_segment, auxiliary_data, mode)
