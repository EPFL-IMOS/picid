import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from lmetk.data.datasources.base.phmd_loader import PHMDMultiSourceLoader

logger = logging.getLogger(__name__)


class XJTU_SYLoader(PHMDMultiSourceLoader):
    """This class instantiates each of the soure"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fold = kwargs["fold"]
        self.task_mode = kwargs["task_mode"]
        self.cache_dir = kwargs["cache_dir"]

        cache_path = Path(self.cache_dir).expanduser()
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = str(cache_path)

        # Instantiate multisource_data_splitter if is not none
        assert (
            kwargs.get("multisource_data_splitter") == None
        ), "For PHMD, multisource_data_splitter is not supported. PHMD are given with splits."
        self.multisource_data_splitter = "For PHMD, multisource_data_splitter is not supported. PHMD are given with splits."

    def __repr__(self) -> str:
        return f"MultiSource(sources={len(self.data_source_dict)} units)"

    def get_data(self) -> dict | list[dict]:
        """Return data sources.

        Returns:
            dict | list[dict] : either dict in case we use multisource_data_splitter or list of dicts in case we use data_splitter

        """

        # Load data from each source
        # Load sources
        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_isplitted = True

        return self.data_dict

    def get_meta_data(self) -> Dict[str, Any]:
        """
        Return metadata about the data source.
        This can include information like data source name, version, etc.
        """
        assert self._is_loaded, "Data must be loaded before getting metadata."
        return {}

    def get_data_name(self) -> list[str]:
        """Return source data names."""
        return self.data_dict["metadata"]

    def get_multisource_data_splitter(self):
        """Return multisource data splitter."""
        return self.multisource_data_splitter

    def split_data(self, data_lst: list[Dict], source_names: list[str]):
        "Predefined splits"
        pass

    def _rename_columns(self, folds):

        # allign UNIBO dataset with the NB1 and NB14
        if unit_col == "test_name":
            folds["train"] = folds["train"].rename(columns={"test_name": "unit"})
            folds["val"] = folds["val"].rename(columns={"test_name": "unit"})
            folds["test"] = folds["test"].rename(columns={"test_name": "unit"})
            unit_col = "unit"

        if self.data_name == "NB1":
            # The features_col in 'NB1' is quite weird, so allign to UNIBO and NB14
            folds["train"] = folds["train"].rename(
                columns={
                    "voltage_battery": "voltage",
                    "current_battery": "current",
                    "temp_battery": "temperature",
                }
            )
            folds["val"] = folds["val"].rename(
                columns={
                    "voltage_battery": "voltage",
                    "current_battery": "current",
                    "temp_battery": "temperature",
                }
            )
            folds["test"] = folds["test"].rename(
                columns={
                    "voltage_battery": "voltage",
                    "current_battery": "current",
                    "temp_battery": "temperature",
                }
            )
            features_col = ["mode", "voltage", "current", "temperature"]

        return out_dict

    # Helper function to process a single unit
    def _process_unit(self, df_unit, unit_name, features_col, target_col):
        df_features = df_unit.reset_index(
            drop=True
        )  # df_unit[features_col].copy().reset_index(drop=True)
        df_target = df_unit[target_col].copy().reset_index(drop=True)
        # df_features = df_unit.copy().reset_index(drop=True)

        if df_unit.isna().any().any():
            logger.info(
                f"NaN values detected in unit {unit_name}. Details:\n{df_unit.isna()}"
            )
            df_features = df_features.dropna()  # Optionally handle df_target too

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

        return {
            "features": df_features,
            "timestamps": np.array(df_features.index),
            "target": df_target,
            "metadata": {
                "unit_id": unit_name,
                "unit_length": len(df_features),
                "features_columns": features_col,
                "target_col": target_col,
                "target_in_the_featurs": True,
            },
        }
