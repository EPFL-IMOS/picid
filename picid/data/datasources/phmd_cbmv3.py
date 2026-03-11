import logging

import numpy as np

from picid.data.datasources.base.phmd_loader import PHMDMultiSourceLoader

logger = logging.getLogger(__name__)


class CBMv3Loader(PHMDMultiSourceLoader):
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

        unit_id = np.array(np.astype(unit_name, int))
        unit_name = f"Unit_{unit_id}"

        return {
            "features": df_features.values,
            "time_features": np.array(df_features.index),
            "unit_id": unit_id,
            "rul": df_target.values,
            "metadata": {
                "unit_id": unit_id,
                "unit_name": unit_name,
                "unit_length": len(df_features),
                "features_columns": features_col,
                "target_col": target_col,
                "target_in_the_featurs": True,
            },
        }
