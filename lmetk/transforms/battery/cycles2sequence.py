from typing import Any, Dict
import pandas as pd
import numpy as np
from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
from lmetk.data.datasources.nb14.utils import unflatten_cycles
import logging
logger = logging.getLogger(__name__)


class Cycles2Sequence(BaseTransform, NoFitMultiSourceTransformMixin):
    """
    Compress cycles to the mean/std representations of the cycles for NB1 dataset
    """      
    def __init__(self, win_len=None, pad_value=0):
        super().__init__()
        self.win_len = win_len

        if pad_value == "np.nan":
            pad_value = np.nan
        
        self.pad_value = pad_value

    def transform_data(self, data: Any, metadata: Dict) -> Any:
        """Transforms a single data segment using the internal ."""

        features, target = unflatten_cycles(
            flattened_x=data['features'],
            flattened_y=data['target'],
            cycle_indices=data['metadata']['cycle_indices'],
            win_len=self.win_len,
            pad_value=self.pad_value,
        )

        # Check if the size of target is (N, 1), if not then it should be either (N, 1, 1) or (N, ) and convert to (N, 1)
        if target.ndim == 3 and target.shape[2] == 1:
            target = target.squeeze(axis=2)
        elif target.ndim == 1:
            target = target.reshape(-1, 1)
        elif target.ndim == 0:
            target = np.array([target]).reshape(-1, 1)
        
        
        logger.info(f"Features shape after convertion to sequences: {features.shape}")
        logger.info(f"Target shape after convertion to sequences: {target.shape}")

        return {'a': features, 'b': target}

    def fit_data(
        self,
        data: Any,
        metadata: Dict,
    ):

        pass