import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional, Union

from lmetk.transforms.base_transform import BaseTransform 
from lmetk.transforms.multisource import BaseMultiSourceTransformMixin

import logging
logger = logging.getLogger(__name__) 

class TabularizationTransform(BaseTransform, BaseMultiSourceTransformMixin):

    def __init__(self, exclude_keys: Optional[List[str]] = None, **kwargs):
        """
        Initialize the transform.
        
        Args:
            exclude_keys: List of auxiliary_data keys to exclude from concatenation
            **kwargs: Additional keyword arguments
        """
        self._init_kwargs = kwargs 
        self.exclude_keys = exclude_keys or []

    def transform_data(self, data_segment: Any, auxiliary_data: Dict,  mode: str) -> Any:
        """
        Concatenate all auxiliary_data values to the data_segment.
        
        Args:
            data_segment: The main data segment (assumed to be numpy array)
            auxiliary_data: Dictionary containing auxiliary data to concatenate
            
        Returns:
            Concatenated data segment with auxiliary data
        """

        data_segment # 

        self.feature_sequencer = AutoregressiveSequencer(
            features=data_dict["features"],
            time_features=data_dict["time_features"],
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
        )

        # pull the full dataset from the sequencer
        # rearrange into 2d

        return result


    def __call__(self, data_segment: Any, auxiliary_data: Dict,  mode: str) -> Any:
        return self.transform_data(data_segment, auxiliary_data,  mode=mode)