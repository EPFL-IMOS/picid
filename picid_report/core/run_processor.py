# picid_report/core/run_processor.py
"""
Handles fetching, caching, and preprocessing of wandb experiment run data.

This module is responsible for the first step of the analysis pipeline:
getting the raw data from Weights & Biases (or a local cache) and transforming
it into a clean, flat pandas DataFrame ready for analysis.
"""

import ast
import logging
import os
import warnings
from typing import List, Optional, Tuple

import pandas as pd
import wandb
from picid_report import config
from picid_report.config import PipelineConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Suppress common pandas warnings for a cleaner output.
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")


def _filter_columns(df: pd.DataFrame, filters: List[str]) -> List[str]:
    """Helper function to find columns that contain any of the filter strings."""
    return [col for col in df.columns if any(f in col for f in filters)]


def _normalize_column(
    df: pd.DataFrame, column_to_normalize: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Flattens a column containing nested dictionaries into separate columns."""
    if column_to_normalize not in df.columns:
        return df, []

    # Safely evaluate string-formatted dictionaries.
    def safe_load(item):
        if isinstance(item, str):
            try:
                return ast.literal_eval(item)
            except (ValueError, SyntaxError):
                return {}
        return item if isinstance(item, dict) else {}

    valid_entries = df[column_to_normalize].dropna().apply(safe_load)
    if valid_entries.empty:
        df = df.drop(columns=[column_to_normalize], errors="ignore")
        return df, []

    # Use pandas' json_normalize to flatten the dictionary structures.
    flattened_df = pd.json_normalize(valid_entries)
    flattened_df.columns = [
        f"{column_to_normalize}.{col}" for col in flattened_df.columns
    ]

    df = df.drop(columns=[column_to_normalize])
    result_df = pd.concat([df, flattened_df.set_index(valid_entries.index)], axis=1)
    return result_df, flattened_df.columns.tolist()


def load_runs_df(
    project_name: str,
    user: str,
    csv_cache_dir: str = "csv_files",
    pipeline_config: Optional[PipelineConfig] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load wandb run data (or from CSV cache) and return a flat, cleaned DataFrame.

    If pipeline_config is provided, its column_config, columns_to_normalize,
    and column filters are used; otherwise module-level config is used.

    Args:
        project_name: The name of the wandb project (e.g., 'my-awesome-project').
        user: The wandb username or entity (e.g., 'my-team').
        csv_cache_dir: The directory to store and look for cached CSV files.
        pipeline_config: Optional PipelineConfig to override column and filter config; None uses module config.

    Returns:
        A tuple containing:
        - pd.DataFrame: The prepared and cleaned DataFrame of all runs.
        - List[str]: All column names generated from the config (after normalization).
        - List[str]: Column names that were dropped during cleaning.
    """
    logger.info(
        "Load: project=%s, user=%s, cache_dir=%s", project_name, user, csv_cache_dir
    )
    os.makedirs(csv_cache_dir, exist_ok=True)
    csv_path = f"{csv_cache_dir}/{project_name}.csv"

    try:
        # Attempt to load data from the local cache to speed up re-runs.
        runs_df = pd.read_csv(csv_path, index_col=0)
        # The CSV stores dicts as strings, so we convert them back to Python objects.
        for col in ["summary", "config"]:
            runs_df[col] = runs_df[col].apply(ast.literal_eval)
        logger.info(f"Loaded {len(runs_df)} runs from cached file: {csv_path}")
    except FileNotFoundError:
        # If cache is not found, fetch fresh data from the wandb API.
        # Only create API client when needed (avoids auth failure in CI when cache exists).
        api = wandb.Api()
        logger.info(f"Fetching runs from wandb project: {user}/{project_name}...")
        runs = api.runs(f"{user}/{project_name}")
        summary_list, config_list, name_list = [], [], []
        for run in tqdm(runs):
            if run.state == "finished":
                summary_list.append(run.summary._json_dict)
                config_list.append(
                    {k: v for k, v in run.config.items() if not k.startswith("_")}
                )
                name_list.append(run.name)
            else:
                logger.info(f"Skipping run '{run.name}' with state '{run.state}'")
        runs_df = pd.DataFrame(
            {"summary": summary_list, "config": config_list, "run_name": name_list}
        )
        runs_df.to_csv(csv_path)
        logger.info(f"Saved {len(runs_df)} runs to cache: {csv_path}")

    # Merge the 'summary' (results) and 'config' (hyperparameters) dictionaries for each run.
    # Build merged records without mutating rows in place (iterrows mutation is discouraged).
    merged_records = [
        {**row["summary"], **row["config"], "run_name": row["run_name"]}
        for _, row in runs_df.iterrows()
    ]
    df = pd.DataFrame.from_records(merged_records)

    # --- Data Cleaning and Flattening from Config ---
    cols_to_normalize = (
        pipeline_config.columns_to_normalize
        if pipeline_config is not None
        else config.COLUMNS_TO_NORMALIZE
    )
    cols_to_drop_filter = (
        pipeline_config.column_filters_to_drop
        if pipeline_config is not None
        else config.COLUMN_FILTERS_TO_DROP
    )
    logger.debug("Normalizing columns: %s", cols_to_normalize)
    config_columns = []
    for column in cols_to_normalize:
        df, new_cols = _normalize_column(df, column)
        config_columns.extend(new_cols)
    logger.debug(
        "After normalize: shape=%s, config_columns count=%d",
        df.shape,
        len(config_columns),
    )

    # Drop columns that are mostly empty (>90% NaN) to reduce noise.
    nan_cols_to_drop = df.columns[df.isna().mean() > 0.9].tolist()
    df = df.drop(columns=nan_cols_to_drop)
    logger.info("Dropped %d columns with >90%% NaN", len(nan_cols_to_drop))
    logger.debug("Columns dropped (NaN): %s", nan_cols_to_drop)

    # Drop columns based on specific name patterns defined in the config file.
    cols_to_drop_by_filter = _filter_columns(df, cols_to_drop_filter)
    df = df.drop(columns=cols_to_drop_by_filter, errors="ignore")
    logger.info("Dropped %d columns by name filters", len(cols_to_drop_by_filter))
    logger.debug("Columns dropped (filter): %s", cols_to_drop_by_filter)

    # Keep track of all columns that have been dropped.
    all_dropped_columns = nan_cols_to_drop + cols_to_drop_by_filter
    logger.debug(
        "Load complete: final shape=%s, config_columns=%d, dropped=%d",
        df.shape,
        len(config_columns),
        len(all_dropped_columns),
    )
    return df, config_columns, all_dropped_columns
