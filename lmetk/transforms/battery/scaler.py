from typing import Any, List, Union, Tuple, Type, Optional, Dict
from lmetk.transforms.base_transform import BaseTransform 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lmetk.transforms.utils import _convert_to_numpy
from lmetk.transforms.multisource import BaseMultiSourceTransformMixin

import logging
logger = logging.getLogger(__name__) 

class MinMaxScalerSklearn(BaseTransform, BaseMultiSourceTransformMixin):
    """
    An adapter for sklearn.preprocessing.MinMaxScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def fit_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]):
        """Fits the internal MinMaxScaler on a single NumPy array."""
        np_data, _ = _convert_to_numpy(data_segment)
        
        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for MinMaxScaler.")

        self.scaler.fit(np_data)
        
    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Transforms a single data segment using the internal MinMaxScaler."""
        np_data, original_type = _convert_to_numpy(data_segment)
        
        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
            logger.info("Reshaped 1D array to 2D for MinMaxScaler.")

        transformed_np_data = self.scaler.transform(np_data)
        return transformed_np_data 


class SklearnStandardScalerAdapter(BaseTransform, BaseMultiSourceTransformMixin):
    """
    An adapter for sklearn.preprocessing.StandardScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]):
        """Fits the internal StandardScaler on a single NumPy array."""
        np_data, _ = _convert_to_numpy(data_segment)
        self.scaler.fit(np_data) 
        
    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Transforms a single data segment using the internal StandardScaler."""
        np_data, original_type = _convert_to_numpy(data_segment)
        transformed_np_data = self.scaler.transform(np_data)
        return transformed_np_data #_convert_from_numpy(transformed_np_data, original_type, data_segment)