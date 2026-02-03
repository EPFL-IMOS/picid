"""
Preprocessing helpers for the analysis pipeline (e.g. model name cleaning).
"""

import os
from typing import Optional

import pandas as pd

from lmetk_report import config


def clean_and_rename_models(
    df: pd.DataFrame,
    model_target_col: Optional[str] = None,
    model_type_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Strip common prefix from model names and distinguish StatisticalBaselineWrapper
    as "(linear)" vs "(exponential)" using model.model_type.

    Matches the logic in table.ipynb so that:
    - "lmetk.baselines.patchtst_model.PatchTST_Forecaster" -> "patchtst_model.PatchTST_Forecaster"
    - "lmetk.model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper" + type "linear"
      -> "statistical_models_wrapper.StatisticalBaselineWrapper (linear)"

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

    unique_models = df[target_col].astype(str).unique()
    if len(unique_models) > 1:
        prefix = os.path.commonprefix(list(unique_models))
        if prefix and "." in prefix:
            prefix = prefix.rsplit(".", 1)[0] + "."
            df[target_col] = df[target_col].astype(str).str.replace(prefix, "", regex=False)

    if type_col in df.columns:
        mask = df[type_col].notna()
        if mask.any():
            df.loc[mask, target_col] = (
                df.loc[mask, target_col].astype(str)
                + " ("
                + df.loc[mask, type_col].astype(str)
                + ")"
            )

    return df
