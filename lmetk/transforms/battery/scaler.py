from typing import Any, List, Union, Tuple, Type, Optional, Dict
from lmetk.transforms.base_transform import BaseTransform 
from sklearn.preprocessing import MinMaxScaler
from lmetk.transforms.utils import _convert_to_numpy
from lmetk.transforms.multisource import BaseMultiSourceTransformMixin

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
        self.scaler.fit(np_data)
        
    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Transforms a single data segment using the internal MinMaxScaler."""
        np_data, original_type = _convert_to_numpy(data_segment)
        transformed_np_data = self.scaler.transform(np_data)
        return transformed_np_data 