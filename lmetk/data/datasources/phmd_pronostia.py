import logging
import pandas as pd
import numpy as np

from lmetk.data.datasources.base.phmd_loader import PHMDMultiSourceLoader

logger = logging.getLogger(__name__)

UNIT_NAMES_TO_ID = {
    "1_1": (1, 1),
    "1_2": (1, 2),
    "1_3": (1, 3),
    "1_4": (1, 4),
    "1_5": (1, 5),
    "1_6": (1, 6),
    "1_7": (1, 7),
    "2_1": (2, 1),
    "2_2": (2, 2),
    "2_3": (2, 3),
    "2_4": (2, 4),
    "2_5": (2, 5),
    "2_6": (2, 6),
    "2_7": (2, 7),
    "3_1": (3, 1),
    "3_2": (3, 2),
    "3_3": (3, 3),
}


class PronostiaLoader(PHMDMultiSourceLoader):

    def _get_features_columns(self):
        return self.meta_data["features"]

    def _get_target_column(self):
        return self.task_mode

    def _get_unit_column(self):
        return self.meta_data["identifier"]

    def _preprocess_fold(self, main_fold, auxiliary_tasks_fold_dict):
        return main_fold

    def _process_unit(self, df_unit, unit_name, features_col, target_col):

        df_features = df_unit[features_col].copy().reset_index(drop=True)
        df_target = df_unit[target_col].copy().reset_index(drop=True)
        # repeat the same id for every step
        unit_id = UNIT_NAMES_TO_ID[unit_name]
        return {
            "features": df_features,
            "time_features": np.array(df_features.index),
            "target": pd.DataFrame(df_target),
            "metadata": {
                "unit_name": f"Bearing {unit_name}",
                "unit_id": unit_id,
                "unit_length": len(df_features),
                "features_columns": features_col,
                "target_col": target_col,
                "target_in_the_featurs": True,
            },
        }
