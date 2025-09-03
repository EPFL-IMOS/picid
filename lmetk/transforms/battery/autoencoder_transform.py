from typing import Any, Dict

import pandas as pd
import numpy as np
from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin, NoFitMultiSourceTransformMixin
from lmetk.transforms.battery.utils import get_mode_change_right_indices


class AutoencoderTransform(BaseTransform, NoFitMultiSourceTransformMixin):
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
        self,
        data_segment: Any,
        metadata: Dict,
    ):

        pass

    def transform_data(self, data_segment_obj: Any, metadata: Dict) -> Any:
        """Transforms a single data segment using the internal ."""

        data_segment = data_segment_obj["features"]

        # Based on the rul column extract the stat
        target_col = self.target_name

        cycles_column = data_segment[self.cycles_col].values
        boundaries = get_mode_change_right_indices(cycles_column)

        # Get the feature columns
        feature_columns = data_segment_obj["metadata"]["features_columns"].copy()

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
