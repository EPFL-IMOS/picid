from typing import Any, List, Union, Tuple, Type, Optional, Dict
from lmetk.transforms.base_transform import BaseTransform 
from sklearn.preprocessing import MinMaxScaler
from lmetk.transforms.utils import _convert_to_numpy
from lmetk.transforms.multisource import BaseMultiSourceTransformMixin

class CycleSubsample(BaseTransform, BaseMultiSourceTransformMixin):
    """
    An adapter for sklearn.preprocessing.MinMaxScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """
        
    def fit_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]):
        """Fits the internal MinMaxScaler on a single NumPy array."""
        pass
        
    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Transforms a single data segment using the internal MinMaxScaler."""
        # Logic here
        # NB1 "0: charge, 1: discharge,
        transformed_np_data = data_segment 
        return transformed_np_data 
    


class NB1_filtering(BaseTransform, BaseMultiSourceTransformMixin):
    """
    Filter features of NB1 dataset (NASA Randomized Battery Usage Dataset) according to the Bosello work. keep solely charge/discharge and voltage, current and temperature as features.
    """
        
    def fit_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]):
        pass
        
    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Transforms a single data segment using the internal ."""
        # Keep mode:  0: charge, 1: discharge
        data_segment = data_segment[data_segment['mode'].apply(lambda x: x in [0,1])]
        # Keep fetures: voltage (V), current (I), and temperature (T)
        data_segment = data_segment[['mode', 'voltage_battery','current_battery','temp_battery']]
        # Reset index
        data_segment.reset_index(drop=True,inplace=True)

        return data_segment 



class Cycles2Statistics(BaseTransform, BaseMultiSourceTransformMixin):
    """
    Compress cycles to the mean/std representations of the cycles.
    """
        
    def fit_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]):
        
        pass
        
    def transform_data(self, data_segment: Any, auxiliary_data: Optional[Dict],  mode: Optional[str]) -> Any:
        """Transforms a single data segment using the internal ."""
        mode_col = 'mode'
        mode_column = data_segment[mode_col].values
        boundaries = get_mode_change_right_indices(mode_column)

        feature_columns = [col for col in data_segment.columns if col != mode_col]
        stats_list = []
        for i in range(len(boundaries)):
            start = 0 if i == 0 else boundaries[i - 1] 
            end = boundaries[i]
            segment = data_segment.iloc[start:end]
            assert len(segment['mode'].unique())==1, "Mode should be the same within a segment"
            
            stats = {
                #'start': start,
                #'end': end,
                # Leghts seems to be very strong feature that is not possible to measure in real-world fairly
                #'lenght': end - start,
                'mode': segment[mode_col].iloc[0]
            }
            for feature in feature_columns:
                stats[f'{feature}_mean'] = segment[feature].mean()
                stats[f'{feature}_var'] = segment[feature].var()
            stats_list.append(stats)
        
        data_segment = pd.DataFrame(stats_list)
        
        return data_segment 

import pandas as pd
import numpy as np
from typing import List

def get_mode_change_right_indices(x: np.ndarray) -> List[int]:
    """
    Identifies the right-exclusive boundaries (i.e., start of new regimes)
    of contiguous mode segments in the input array `x`.

    Parameters
    ----------
    x : np.ndarray
        A 1D NumPy array of mode values (e.g., 0s and 1s).

    Returns
    -------
    List[int]
        A list of indices marking the start of new segments, to be used as exclusive slice boundaries.

    Example
    -------
    >>> get_mode_change_right_indices(np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1]))
    [3, 7, 9, 12]
    """
    change_indices = np.where(x[:-1] != x[1:])[0]
    # +1 gives the exclusive right index where new mode starts
    right_boundaries = (change_indices + 1).tolist()
    # Add final end index (exclusive)
    right_boundaries.append(len(x))
    return right_boundaries

