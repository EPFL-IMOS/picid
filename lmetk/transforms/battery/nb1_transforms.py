from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.multisource import BaseMultiSourceTransformMixin
from lmetk.transforms.battery.utils import get_mode_change_right_indices


class FilterDataFrameColumns(BaseTransform, BaseMultiSourceTransformMixin):
    """
    Filter features of NB1 dataset (NASA Randomized Battery Usage Dataset) according to the Bosello work. keep solely charge/discharge and voltage, current and temperature as features.
    """

    def __init__(self, columns_to_keep: str, apply_to_key: str = 'features'):
        self.columns_to_keep = columns_to_keep
        self.apply_to_key = apply_to_key

    def fit_data(
        self, data_segment: Any, metadata: Dict
    ):
        pass

    def transform_data(
        self, data_segment_dict: Any, metadata: Dict
    ) -> Any:
        """Transforms a single data segment using the internal ."""
        
        data_segment = data_segment_dict[self.apply_to_key][self.columns_to_keep]

        # Reset index
        data_segment.reset_index(drop=True, inplace=True)

        return data_segment


class UpdateTarget(BaseTransform, BaseMultiSourceTransformMixin):
    """
    Update the target column to be the one specified in the config.
    This update takes a column from the "features" and returns it
    """

    def __init__(self, target_name: str, apply_to_key: str = None):
        self.target_name = target_name
        assert apply_to_key is not None, "apply_to_key should be specified under the transform"
        self.apply_to_key = apply_to_key

    def fit_data(
        self, data_segment_dict: Any, metadata: Dict
    ):
        pass

    def transform_data(
        self, data_segment_dict: Any, metadata: Dict
    ) -> Any:
        """Transforms a single data segment using the internal ."""
        assert self.apply_to_key in data_segment_dict, f"{self.apply_to_key}  are not passed in data_segment_dict"
        
        out = (
            data_segment_dict[self.apply_to_key][self.target_name]
            .values.reshape(-1, 1)
            .astype(float)
        )
        return out


import numpy as np


class AssignNaiveTimefeatures(BaseTransform, BaseMultiSourceTransformMixin):
    """
    Replace the time features with a range from 0 to len(feature)
    """

    # def fit_data(
    #     self, data_segment_dict: Any, metadata: Dict
    # ):
    #     pass

    def transform_data(
        self, data_segment_dict: Any, metadata: Dict
    ) -> Any:
        """Transforms a single data segment using the internal ."""
        assert (
            "features" in data_segment_dict
        ), "features features are not passed in auxiliary_data"
        data_segment = (
            np.arange(data_segment_dict["features"].shape[0]).reshape(-1, 1).astype(float)
        )
        return data_segment


class Cycles2Statistics(BaseTransform, BaseMultiSourceTransformMixin):
    """
    Compress cycles to the mean/std representations of the cycles for NB1 dataset
    """

    def __init__(
        self,
        target_name: str,
        battery_mode_state: str = None,
        cycles_identifier: str = None,
    ):
        self.target_name = target_name
        self.mode_col = battery_mode_state
        self.cycles_col = cycles_identifier

    def fit_data(
        self, data_segment: Any, metadata: Dict,
    ):

        pass

    def transform_data(
        self, data_segment_dict: Any, metadata: Dict
    ) -> Any:
        """Transforms a single data segment using the internal ."""
        
        data_segment = data_segment_dict['features']
        
        # Based on the rul column extract the stat
        target_col = self.target_name

        cycles_column = data_segment[self.cycles_col].values
        boundaries = get_mode_change_right_indices(cycles_column)

        # Get the feature columns
        feature_columns = data_segment_dict["metadata"]["features_columns"].copy()

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

            assert (
                len(segment[target_col].unique()) == 1
            ), "Rul should be the same within a segment"
            stats = {target_col: segment[target_col].iloc[0]}

            if mode_is_present:
                assert (
                    len(segment["mode"].unique()) == 1
                ), "Mode should be the same within a segment"
                stats["mode"] = segment[self.mode_col].iloc[0]
            if cycles_is_present:
                assert (
                    len(segment[self.cycles_col].unique()) == 1
                ), "Cycles should be the same within a segment"
                stats[self.cycles_col] = segment[self.cycles_col].iloc[0]

            for feature in feature_columns:
                stats[f"{feature}_mean"] = segment[feature].mean()
                stats[f"{feature}_var"] = segment[feature].var()

            stats_list.append(stats)

        data_segment = pd.DataFrame(stats_list)

        return data_segment
        