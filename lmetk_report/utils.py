# lmetk_report/utils.py
"""
Shared utilities for analysis and reporting.

- format_mean_std_count: display string 'mean ± std (n=count)' for aggregate stats; NaN std -> 0.
- flatten_aggregated_columns: turn MultiIndex columns (metric, 'mean'|'std'|'count') into
  metric_mean, metric_std, metric_count so downstream code can sort/display by metric.
"""

from typing import Union

import pandas as pd


def format_mean_std_count(
    mean: float,
    std: float,
    count: Union[int, str],
    precision: int = 4,
) -> str:
    """Format aggregate stats as 'mean ± std (n=count)' for display.

    Args:
        mean: Mean value.
        std: Std value; NaN is displayed as 0.
        count: Sample count; non-numeric is passed through (e.g. 'n/a').
        precision: Decimal places for mean and std.

    Returns:
        String like "1.2345 ± 0.0012 (n=3)" or "1.23 ± 0.00 (n=n/a)".
    """
    std_val = std if not pd.isna(std) else 0.0
    cnt_val = int(count) if not isinstance(count, str) else count
    return (
        f"{mean:.{precision}f} ± {std_val:.{precision}f} (n={cnt_val})"
    )


def flatten_aggregated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns to 'colname_stat' strings.

    Aggregated results have columns like (metric, 'mean'), (metric, 'std'), (metric, 'count').
    This converts them to metric_mean, metric_std, metric_count. Plain string columns
    are left unchanged.

    Args:
        df: DataFrame possibly with MultiIndex (tuple) columns.

    Returns:
        A copy of df with flattened column names.
    """
    result = df.copy()
    new_cols = []
    for col in result.columns:
        if isinstance(col, tuple):
            col_name, stat = col
            new_cols.append(f"{col_name}_{stat}" if stat else col_name)
        else:
            new_cols.append(col)
    result.columns = new_cols
    return result
