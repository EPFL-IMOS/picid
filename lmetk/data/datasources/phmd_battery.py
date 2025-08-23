import logging

import numpy as np
import pandas as pd
from lmetk.data.data_objects.data import DatasetChunk
from lmetk.data.datasources.base.phmd_loader import PHMDMultiSourceLoader

logger = logging.getLogger(__name__)


class PHMDBatteryLoader(PHMDMultiSourceLoader):
    """This class instantiates each of the soure"""

    def __init__(self, **kwargs):

        data_name = kwargs["data_name"]
        task_mode = kwargs["task_mode"]

        # In case of ahRul, we also include rul in the task
        if data_name == "NB14" and task_mode == "ahRul":
            kwargs["auxiliary_tasks"] = [*kwargs.get("auxiliary_tasks", []), "rul"]

        super().__init__(**kwargs)

    def _get_features_columns(self):
        features_col = self.meta_data["features"]
        if self.data_name == "NB1":
            features_col = ["mode", "voltage", "current", "temperature"]
        return features_col

    def _get_target_column(self):
        return self.task_mode

    def _get_unit_column(self):
        unit_id = self.meta_data["identifier"]
        assert len(unit_id) == 1, "There should be only one identifier"
        unit_col = unit_id[0]

        if unit_col == "test_name":
            unit_col = "unit"

        return unit_col

    def _preprocess_fold(self, main_fold, auxiliary_tasks_fold_dict):

        fold = main_fold

        # Make the ahRul dataset to have rul values as well
        if self.data_name == "NB14" and self.task_mode == "ahRul":
            folds_rul = auxiliary_tasks_fold_dict["rul"]
            assert all(
                [
                    all(fold[key]["unit"] == folds_rul[key]["unit"])
                    for key in ["train", "val", "test"]
                ]
            ), "The units in NB14 rul and ahrul are not the same"

            # Add rul column to the ahrul task to construct the y
            for key in ["train", "val", "test"]:
                fold[key] = pd.concat([fold[key], folds_rul[key]["rul"]], axis=1)

            del folds_rul

        # allign UNIBO dataset with the NB1 and NB14
        original_unit_col = self.meta_data["identifier"][0]
        if original_unit_col == "test_name":
            fold["train"] = fold["train"].rename(columns={"test_name": "unit"})
            fold["val"] = fold["val"].rename(columns={"test_name": "unit"})
            fold["test"] = fold["test"].rename(columns={"test_name": "unit"})

        if self.data_name == "NB1":
            # The features_col in 'NB1' is quite weird, so allign to UNIBO and NB14
            fold["train"] = fold["train"].rename(
                columns={
                    "voltage_battery": "voltage",
                    "current_battery": "current",
                    "temp_battery": "temperature",
                }
            )
            fold["val"] = fold["val"].rename(
                columns={
                    "voltage_battery": "voltage",
                    "current_battery": "current",
                    "temp_battery": "temperature",
                }
            )
            fold["test"] = fold["test"].rename(
                columns={
                    "voltage_battery": "voltage",
                    "current_battery": "current",
                    "temp_battery": "temperature",
                }
            )

        return fold

    # Helper function to process a single unit
    def _process_unit(self, df_unit, unit_name, features_col, target_col):

        df_features = df_unit.reset_index(drop=True)
        df_target = df_unit[target_col].copy().reset_index(drop=True)

        if df_unit.isna().any().any():
            logger.info(
                f"NaN values detected in unit {unit_name}. Details:\n{df_unit.isna()}"
            )
            # Optionally handle df_target too
            df_features = df_features.dropna()

        # check if df_target has infinity values
        if np.isinf(df_target).any():
            logger.info(
                f"Infinity values detected in unit {unit_name}. Details:\n{df_target}"
            )
            # Replace inf with nan
            df_target = df_target.replace([np.inf, -np.inf], np.nan)
            # Find nan indexes
            nan_indexes = df_target[df_target.isna()].index
            # Drop nan rows from both target and features
            df_target = df_target.drop(nan_indexes)
            df_features = df_features.drop(nan_indexes)
            # Ensure the number of rows is the same
            assert len(df_target) == len(
                df_features
            ), "Mismatch after dropping nan/infinity rows"

        data_dict = {
            "features": df_features,
            "time_features": np.array(df_features.index),
            "target": df_target,
            "metadata": {
                "unit_id": unit_name,
                "unit_length": len(df_features),
                "features_columns": features_col,
                "target_col": target_col,
                "target_in_the_featurs": True,
            },
        }
        return DatasetChunk(**data_dict)  # data_dict
