# lmetk_report/validators.py
"""
Module for validating experiment consistency and scientific rigor.

This module provides tools to:
1. Filter experiments based on required seeds (Data or Model).
2. Detect "hidden variations" (non-hyperparameters that change silently).
3. Log explicit modifications to the DataFrame for transparency.
"""

import warnings
from typing import List, Optional, Set, Tuple

import pandas as pd


def log_modification(
    action: str,
    reason: str,
    input_shape: Tuple[int, int],
    output_shape: Tuple[int, int],
) -> None:
    """
    Print an explicit notice about a DataFrame modification to stdout.

    Args:
        action: A short description of the operation (e.g., "Seed Filter").
        reason: The justification for the modification.
        input_shape: The (rows, cols) of the DataFrame before the operation.
        output_shape: The (rows, cols) of the DataFrame after the operation.
    """
    rows_dropped = input_shape[0] - output_shape[0]
    cols_dropped = input_shape[1] - output_shape[1]

    # Only log if there was an actual change in dimensions.
    if rows_dropped == 0 and cols_dropped == 0:
        return

    print(f"\n  [DATAFRAME MODIFICATION] {action}")
    print(f"    Reason: {reason}")
    print(f"    Input Shape: {input_shape} -> Output Shape: {output_shape}")

    changes = []
    if rows_dropped > 0:
        changes.append(f"{rows_dropped} rows dropped")
    if cols_dropped > 0:
        changes.append(f"{cols_dropped} cols dropped")

    print(f"    Change: {', '.join(changes)}")
    print("  " + "-" * 30)


def validate_seeds(
    df: pd.DataFrame,
    group_cols: List[str],
    seed_col: str,
    required_seeds: Optional[Set[int]] = None,
    allow_fallback: bool = False,
) -> pd.DataFrame:
    """
    Filter the DataFrame to ensure experiment groups contain required seeds.

    Args:
        df: The DataFrame containing experiment results.
        group_cols: Columns defining a unique configuration.
        seed_col: The column name containing the random seed.
        required_seeds: A set of integers (e.g., {0, 1, 2}).
        allow_fallback: If True, reverts filter if result is empty.

    Returns:
        pd.DataFrame: A filtered DataFrame.
    """
    # Case 1: Flexible mode (no strict requirement) -> Return data as is.
    if not required_seeds:
        return df

    start_shape = df.shape

    # Case 2: Strict mode.
    if seed_col not in df.columns:
        print(
            f"[Validator] Warning: Seed column '{seed_col}' not found. "
            "Cannot filter by seeds."
        )
        return df

    def has_required_seeds(group):
        try:
            # Drop NaNs and get unique seeds present in this group.
            current = set(group[seed_col].dropna().unique())
            # Robust conversion to int to handle 1.0 vs 1.
            current_int = {int(float(x)) for x in current}
            req_int = {int(x) for x in required_seeds}
            # Check if current seeds are a superset of required seeds.
            return req_int.issubset(current_int)
        except Exception:
            return False

    # Ensure grouping cols are strings to avoid pandas bugs with list objects.
    temp_df = df.copy()
    for col in group_cols:
        temp_df[col] = temp_df[col].astype(str)

    try:
        valid_subset = temp_df.groupby(group_cols).filter(has_required_seeds)
    except TypeError:
        print("[Validator] Warning: Groupby filter failed. Skipping seed validation.")
        return df

    filtered_df = df.loc[valid_subset.index].copy()

    # --- FALLBACK LOGIC ---
    # If the filter removed EVERYTHING, and fallback is allowed:
    if filtered_df.empty and not df.empty and allow_fallback:
        print(f"  [Validator] NOTE: Strict filter on '{seed_col}' removed all rows.")
        print(
            "  [Validator] Fallback triggered: Ignoring seed filter for this model "
            "(assuming it does not use seeds)."
        )
        return df

    # Log the explicit modification.
    log_modification(
        action=f"Strict Seed Filter applied on '{seed_col}'",
        reason=f"Ensuring groups contain required seeds: {required_seeds}",
        input_shape=start_shape,
        output_shape=filtered_df.shape,
    )

    return filtered_df


def check_hidden_variations(
    df: pd.DataFrame,
    group_cols: List[str],
    base_ignored_cols: List[str],
    additional_ignored_cols: Optional[List[str]] = None,
) -> None:
    """
    Check for non-hyperparameters that vary silently within a single group.
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
                print(
                    f"  [WARNING] Hidden variation detected in '{col}'! "
                    "(Varies within grouped config)"
                )
        except Exception:
            pass
