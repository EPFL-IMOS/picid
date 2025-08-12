from typing import Any, List, Union, Tuple, Type, Optional, Dict
from lmetk.transforms.base_transform import BaseTransform 
from sklearn.preprocessing import MinMaxScaler

from lmetk.transforms.multisource import BaseMultiSourceTransformMixin
from lmetk.transforms.battery.utils import get_mode_change_right_indices
import pandas as pd

class NB1_filtering(BaseTransform, BaseMultiSourceTransformMixin):
    """
    Filter features of NB1 dataset (NASA Randomized Battery Usage Dataset) according to the Bosello work. keep solely charge/discharge and voltage, current and temperature as features.
    """
    def __init__(self,columns_to_keep: str):
        self.columns_to_keep = columns_to_keep

    def fit_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]):
        pass
        
    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Transforms a single data segment using the internal ."""
        # Indexes
        # idx = data_segment['mode'].apply(lambda x: x in [0,1,2])

        # Keep mode:  0: charge, 1: discharge
        # data_segment = data_segment[idx]
        
        # Keep fetures: voltage (V), current (I), and temperature (T)
        data_segment = data_segment[self.columns_to_keep]

        # Reset index
        data_segment.reset_index(drop=True, inplace=True)

        return data_segment 

class UpdateTarget(BaseTransform, BaseMultiSourceTransformMixin):
    """
    Filter features of NB1 dataset (NASA Randomized Battery Usage Dataset) according to the Bosello work. keep solely charge/discharge and voltage, current and temperature as features.
    """
    def __init__(self,target_name: str):
        self.target_name = target_name
        
    def fit_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]):
        pass
        
    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Transforms a single data segment using the internal ."""
        assert 'features' in auxiliary_data, 'features features are not passed in auxiliary_data'
        data_segment = auxiliary_data['features'][self.target_name].values.reshape(-1,1).astype(float)
        return data_segment 

import numpy as np
class FixTimeFeatures(BaseTransform, BaseMultiSourceTransformMixin):
    """
    Filter features of NB1 dataset (NASA Randomized Battery Usage Dataset) according to the Bosello work. keep solely charge/discharge and voltage, current and temperature as features.
    """
        
    def fit_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]):
        pass
        
    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Transforms a single data segment using the internal ."""
        assert 'features' in auxiliary_data, 'features features are not passed in auxiliary_data'
        data_segment = np.arange(auxiliary_data['features'].shape[0]).reshape(-1,1).astype(float)
        return data_segment 


class Cycles2Statistics(BaseTransform, BaseMultiSourceTransformMixin):
    """
    Compress cycles to the mean/std representations of the cycles for NB1 dataset
    """
    def __init__(self,target_name: str, battery_mode_state: str=None, cycles_identifier: str=None):
        self.target_name = target_name
        self.mode_col = battery_mode_state
        self.cycles_col = cycles_identifier

    def fit_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]):
        
        pass
        
    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Transforms a single data segment using the internal ."""
        # Based on the rul column extract the stat
        target_col = self.target_name

        cycles_column = data_segment[self.cycles_col].values
        boundaries = get_mode_change_right_indices(cycles_column)

        # Get the feature columns
        feature_columns = auxiliary_data['metadata']['features_columns'].copy()
        
        # Prevent mode columns to be in the features_columns
        mode_is_present = False
        if self.mode_col is not None and self.mode_col in feature_columns:
            feature_columns.remove(self.mode_col)
            mode_is_present = True
        
        # Prevent cycles columns to be in the features_columns
        cycles_is_present = False
        
        if self.cycles_col is not None and self.cycles_col in data_segment.columns:
            if self.cycles_col in feature_columns:
                feature_columns.remove(self.cycles_col)
            cycles_is_present = True

        # Ensure there is no leakage of target data
        if target_col in feature_columns:
            feature_columns.remove(target_col)

        stats_list = []
        for i in range(len(boundaries)):
            start = 0 if i == 0 else boundaries[i - 1] 
            end = boundaries[i]
            segment = data_segment.iloc[start:end]
            
            assert len(segment[target_col].unique())==1, "Rul should be the same within a segment"
            stats = {target_col: segment[target_col].iloc[0]}
            
            if mode_is_present:
                assert len(segment['mode'].unique())==1, "Mode should be the same within a segment"
                stats['mode'] = segment[self.mode_col].iloc[0]
            if cycles_is_present:
                assert len(segment[self.cycles_col].unique())==1, "Cycles should be the same within a segment"
                stats[self.cycles_col] = segment[self.cycles_col].iloc[0]
            
            for feature in feature_columns:
                stats[f'{feature}_mean'] = segment[feature].mean()
                stats[f'{feature}_var'] = segment[feature].var()

            stats_list.append(stats)
        
        data_segment = pd.DataFrame(stats_list)
        
        return data_segment 