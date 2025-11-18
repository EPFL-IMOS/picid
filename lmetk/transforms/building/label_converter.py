import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin

# Assuming this import path is correct for your project

logger = logging.getLogger(__name__)


class BinaryToMultiClassTransform(DenseTransform, ConcatinatedFitMultiSourceTransformMixin):

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
        new_target_data = data['target'].copy()
        new_target_data = new_target_data.set_index(pd.to_datetime(data['timestamps'], format="%m/%d/%Y %H:%M"))

        # fault_number_by_dates = {"2/12/2008": 1, "5/7/2008": 1, "5/8/2008": 1, "9/5/2007": 1, "9/6/2007": 1,  # Damper
        #                          "8/28/2007": 2, "8/29/2007": 2, "8/30/2007": 2,                           # Heating Coil
        #                          "5/6/2008": 3, "8/31/2007": 3, "5/15/2008": 3, "9/1/2007": 3, "9/2/2007": 3} # Cooling Coil

        fault_number_by_dates = {"2/12/2008": 1, "5/7/2008": 1, "5/8/2008": 2, "9/5/2007": 3, "9/6/2007": 4,  # Damper
                                 "8/28/2007": 5, "8/29/2007": 6, "8/30/2007": 7,                           # Heating Coil
                                 "5/6/2008": 8, "8/31/2007": 9, "5/15/2008": 9, "9/1/2007": 10, "9/2/2007": 11} # Cooling Coil

        for date, number in fault_number_by_dates.items():
            if date in new_target_data.index:
                new_target_data.loc[date, data.target.columns] = number

        data['target'] = new_target_data.reset_index(drop=True)

        return data

    # Keeping your original __call__ and transform methods for compatibility.
    # They simply delegate to the new transform_data method.
    def __call__(self, data_segment: Any, auxiliary_data: Dict, mode: str) -> Any:
        return self.transform_data(data_segment, auxiliary_data, mode)