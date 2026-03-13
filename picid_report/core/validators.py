# picid_report/core/validators.py
"""
Validation helpers for experiment consistency and scientific rigor.

- validate_schema: ensures required columns exist; raises ValueError if not.
- validate_seeds: filters to groups that contain all required data/model seeds; logs drops.
- filter_rows_with_valid_sort_metric: drops rows where the sort/optimization metric is missing or not a valid float.
- log_modification: logs DataFrame shape changes (e.g. after seed filter).
- check_hidden_variations: warns when a non-HP column varies within a (dataset, model) group.
"""

import logging
from typing import List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    Check if all columns required for processing are present in the DataFrame.

    Args:
        df: The normalized DataFrame.
        required_cols: List of column names that must be present.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        error_msg = (
            f"\n[SCHEMA ERROR] Missing {len(missing)} required columns!\n"
            f"The following columns were not found: {missing}\n"
            "Please check your WandB config or update picid_report.config."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(
        f"✅ Schema Validation Passed: All {len(required_cols)} required columns present."
    )


def log_modification(
    action: str,
    reason: str,
    input_shape: Tuple[int, int],
    output_shape: Tuple[int, int],
    context: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """
    Log a DataFrame modification (shape change).

    Args:
        action: A short description of the operation (e.g., "Seed Filter").
        reason: The justification for the modification.
        input_shape: The (rows, cols) of the DataFrame before the operation.
        output_shape: The (rows, cols) of the DataFrame after the operation.
        context: Optional string to describe the scope (e.g., Model/Dataset).
        level: Log level (default INFO). Use DEBUG for high-frequency notices (e.g. per-dataset/model subsetting).
    """
    rows_dropped = input_shape[0] - output_shape[0]
    cols_dropped = input_shape[1] - output_shape[1]

    # Only log if there was an actual change in dimensions.
    if rows_dropped == 0 and cols_dropped == 0:
        return

    logger.log(level, "\n  [DATAFRAME MODIFICATION] %s", action)

    # New: Print context separately if provided
    if context:
        logger.log(level, "    Context: %s", context)
    logger.log(level, "    Reason:  %s", reason)
    logger.log(level, "    Shape:   %s -> %s", input_shape, output_shape)
    changes = []
    if rows_dropped > 0:
        changes.append(f"{rows_dropped} rows dropped")
    if cols_dropped > 0:
        changes.append(f"{cols_dropped} cols dropped")
    logger.log(level, "    Change:  %s", ", ".join(changes))
    logger.log(level, "  " + "-" * 40)


def validate_seeds(
    df: pd.DataFrame,
    group_cols: List[str],
    seed_col: str,
    required_seeds: Optional[Set[int]] = None,
    allow_fallback: bool = False,
    context: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter the DataFrame to groups that contain all required seeds.

    Groups are defined by group_cols (e.g. dataset + model). If a group is
    missing any required seed value in seed_col, that group is dropped.
    When allow_fallback is True and the filter would remove all rows, the
    original df is returned and a warning is logged.

    Args:
        df: DataFrame with seed_col and group_cols.
        group_cols: Column names used to group runs (e.g. dataset, model).
        seed_col: Column name for the seed (e.g. "seed", "datasource.parameters.data_seed").
        required_seeds: Set of seed values that must all be present in each group; None skips filtering.
        allow_fallback: If True, when no group has all required seeds, return df unchanged and log a warning.
        context: Optional string (e.g. "dataset=X, model=Y") to include in warning messages for clarity.

    Returns:
        Filtered DataFrame (or original df if no required_seeds, seed_col missing, or fallback triggered).
    """
    if not required_seeds:
        return df

    start_shape = df.shape

    if seed_col not in df.columns:
        logger.warning(
            f"[Validator] Warning: Seed column '{seed_col}' not found. "
            "Cannot filter by seeds."
        )
        return df

    def has_required_seeds(group):
        try:
            current = set(group[seed_col].dropna().unique())
            current_int = {int(float(x)) for x in current}
            req_int = {int(x) for x in required_seeds}
            return req_int.issubset(current_int)
        except Exception:
            return False

    temp_df = df.copy()
    for col in group_cols:
        temp_df[col] = temp_df[col].astype(str)

    try:
        valid_subset = temp_df.groupby(group_cols).filter(has_required_seeds)
    except TypeError:
        logger.warning(
            "[Validator] Warning: Groupby filter failed. Skipping seed validation."
        )
        return df

    filtered_df = df.loc[valid_subset.index].copy()

    if filtered_df.empty and not df.empty and allow_fallback:
        ctx_suffix = f" ({context})" if context else ""
        logger.warning(
            "  [Validator] NOTE: Strict filter on '%s' removed all rows.%s",
            seed_col,
            ctx_suffix,
        )
        logger.warning(
            "  [Validator] Fallback triggered: Ignoring seed filter for this (dataset, model).%s",
            ctx_suffix,
        )
        return df

    log_modification(
        action=f"Strict Seed Filter applied on '{seed_col}'",
        reason=f"Dropping groups that do not contain all required seeds: {required_seeds}",
        input_shape=start_shape,
        output_shape=filtered_df.shape,
    )

    return filtered_df


def filter_rows_with_valid_sort_metric(
    df: pd.DataFrame,
    sort_metric_col: Optional[str] = None,
    context: Optional[str] = None,
) -> pd.DataFrame:
    """
    Drop rows where the sort/optimization metric column is missing or not a valid float.

    Used before model-seed validation so that only runs with a valid metric value
    count toward "all required seeds present". Rows with NaN, non-numeric, or
    missing values in sort_metric_col are removed.

    Args:
        df: DataFrame with at least sort_metric_col (if present).
        sort_metric_col: Column name for the metric used for ranking/selection (e.g. "val_best_rerun/loss").
            If None or not in df.columns, no filter is applied and df is returned unchanged.
        context: Optional string (e.g. "dataset=X, model=Y") for log messages.

    Returns:
        Filtered DataFrame (or df unchanged if sort_metric_col is None or missing).
    """
    if sort_metric_col is None:
        logger.warning(
            "[Validator] Sort metric column not set; skipping metric-validity filter.%s",
            f" ({context})" if context else "",
        )
        return df

    if sort_metric_col not in df.columns:
        logger.warning(
            "[Validator] Sort metric column %r not found in DataFrame; skipping metric-validity filter.%s",
            sort_metric_col,
            f" ({context})" if context else "",
        )
        return df

    start_rows = len(df)
    # Coerce to numeric; invalid/NaN become NaN
    series = pd.to_numeric(df[sort_metric_col], errors="coerce")
    valid_mask = series.notna()
    filtered_df = df.loc[valid_mask].copy()
    dropped = start_rows - len(filtered_df)

    if dropped > 0:
        ctx_msg = f" ({context})" if context else ""
        logger.info(
            "[Validator] Metric-validity filter: column=%s, rows before=%s, after=%s, dropped=%s%s",
            sort_metric_col,
            start_rows,
            len(filtered_df),
            dropped,
            ctx_msg,
        )

    return filtered_df


def check_hidden_variations(
    df: pd.DataFrame,
    group_cols: List[str],
    base_ignored_cols: List[str],
    additional_ignored_cols: Optional[List[str]] = None,
) -> None:
    """
    Warn when a non-HP column varies within a (dataset, model) group.

    Columns that are not in group_cols and not in ignored lists are checked:
    if any such column has more than one unique value within a group, a
    warning is logged (hidden variation). Numeric columns are skipped.

    Args:
        df: DataFrame with group and candidate columns.
        group_cols: Columns defining the group (e.g. dataset, model).
        base_ignored_cols: Substrings; columns whose name contains any of these are ignored.
        additional_ignored_cols: Extra column names to ignore (e.g. known HPs).
    """
    ignored = set()
    if additional_ignored_cols:
        ignored.update(additional_ignored_cols)

    candidates = []
    for c in df.columns:
        if c in group_cols:
            continue
        if c in ignored:
            continue
        if any(f in c for f in base_ignored_cols):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        candidates.append(c)

    if not candidates:
        return

    grouped = df.groupby(group_cols)

    for col in candidates:
        try:
            if grouped[col].apply(lambda x: x.nunique() > 1).any():
                logger.warning(
                    f"  [WARNING] Hidden variation detected in '{col}'! "
                    "(Varies within grouped config)"
                )
        except Exception:
            pass
