# picid_report/core/preprocess.py
"""
Preprocessing helpers for the analysis pipeline.

clean_and_rename_models: normalizes model names for display and grouping:
- Strips a common package prefix (e.g. picid.model.forecasters. -> forecasters.) so names are short.
- Appends (linear) or (exponential) from model.model_type when present (e.g. StatisticalBaselineWrapper).
Defaults use config.COLUMN_CONFIG for column names; override via optional arguments.
"""

import logging
import os
from typing import Optional

import pandas as pd

from picid_report import config

logger = logging.getLogger(__name__)


def clean_and_rename_models(
    df: pd.DataFrame,
    model_target_col: Optional[str] = None,
    model_type_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Strip common prefix from model names and distinguish StatisticalBaselineWrapper
    as "(linear)" vs "(exponential)" using model.model_type.

    Matches the logic in table.ipynb so that:
    - "picid.model.forecasters.patchtst_model.PatchTST_Forecaster"
      -> "patchtst_model.PatchTST_Forecaster"
    - "picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper"
      + type "linear"
      -> "wrapper.StatisticalBaselineWrapper (linear)"

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a model target column (and optionally model type).
    model_target_col : str, optional
        Column holding the model identifier. Default: config.COLUMN_CONFIG["model_target"].
    model_type_col : str, optional
        Column holding model type (e.g. "linear", "exponential"). Default: "model.model_type".

    Returns
    -------
    pd.DataFrame
        Copy of df with the model target column updated.
    """
    df = df.copy()
    target_col = model_target_col or config.COLUMN_CONFIG["model_target"]
    type_col = model_type_col or "model.model_type"
    logger.debug("Preprocess: target_col=%s, type_col=%s", target_col, type_col)

    unique_models = df[target_col].astype(str).unique()
    if len(unique_models) > 1:
        prefix = os.path.commonprefix(list(unique_models))
        if prefix and "." in prefix:
            prefix = prefix.rsplit(".", 1)[0] + "."
            df[target_col] = (
                df[target_col].astype(str).str.replace(prefix, "", regex=False)
            )

    if type_col in df.columns:
        mask = df[type_col].notna()
        if mask.any():
            df.loc[mask, target_col] = (
                df.loc[mask, target_col].astype(str)
                + " ("
                + df.loc[mask, type_col].astype(str)
                + ")"
            )
            logger.debug(
                "Preprocess: appended (linear)/(exponential) for %d rows", mask.sum()
            )

    logger.info(
        "Preprocess: cleaned model names for %d unique models", len(unique_models)
    )
    return df
