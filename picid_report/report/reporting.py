# picid_report/report/reporting.py
"""
Presentation-ready summaries and tables from analysis results.

- create_summary_table: pivoted table of best metric values per (dataset, model); optional
  re-selection by sort_metric (e.g. val_best_rerun/loss) instead of optimization metric.
- HP impact: per (dataset, model) tables of aggregated runs per HP config, with optional
  re-sort by sort_metric; format is mean ± std (n=count).
- Export: CSV/LaTeX and experiment stats; write_tables_tex for LaTeX tables.
All functions that take sort_metric accept either a single metric name or a dict
(dataset, model) -> metric for per-combination control.
"""

import logging
import os
import re
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import display

from picid_report.core.analysis import _ds_is_empty
from picid_report.utils import format_mean_std_count

logger = logging.getLogger(__name__)


def _infer_metric_mode(metric_name: str) -> str:
    """
    Infer whether a metric should be minimized or maximized based on its name.

    Parameters
    ----------
    metric_name : str
        Metric name (e.g., "test/mse", "val/accuracy", "loss")

    Returns
    -------
    str
        "min" for metrics to minimize (loss, mse, mae, rmse, etc.)
        "max" for metrics to maximize (accuracy, f1, auc, precision, recall, etc.)
        Defaults to "min" if uncertain
    """
    metric_lower = metric_name.lower()

    # Metrics to maximize (higher is better)
    maximize_keywords = [
        "accuracy",
        "acc",
        "f1",
        "f1_score",
        "f_score",
        "auc",
        "roc_auc",
        "precision",
        "recall",
        "r2",
        "r_squared",
        "spearman",
        "pearson",
    ]

    # Metrics to minimize (lower is better)
    minimize_keywords = [
        "loss",
        "error",
        "mse",
        "mean_squared_error",
        "mae",
        "mean_absolute_error",
        "rmse",
        "root_mean_squared_error",
        "mape",
        "mean_absolute_percentage_error",
        "log_loss",
        "cross_entropy",
    ]

    # Check for maximize keywords
    for keyword in maximize_keywords:
        if keyword in metric_lower:
            return "max"

    # Check for minimize keywords
    for keyword in minimize_keywords:
        if keyword in metric_lower:
            return "min"

    # Default to minimize if uncertain
    return "min"


def create_summary_table(
    all_results: defaultdict, precision: int = 4, sort_metric: str = None
) -> pd.DataFrame:
    """
    Create a single pivoted summary DataFrame for all results.

    Parameters
    ----------
    all_results : defaultdict
        Result of analyze_results (dataset -> model -> result dict).
    precision : int
        Decimal precision for formatting.
    sort_metric : str, optional
        Metric to use for selecting best run (e.g., "test/mse"). If None, uses optimization metric.
        Can also be a dict mapping (dataset, model) -> metric for per-combination control.

    Returns
    -------
    pd.DataFrame
        Pivoted summary table with models as rows and (Dataset, Metric) as columns.
    """
    records = []

    for dataset, models in all_results.items():
        for model, results in models.items():
            # Get sort_metric for this specific combination if dict provided
            if isinstance(sort_metric, dict):
                combo_sort_metric = sort_metric.get((dataset, model), None)
            else:
                combo_sort_metric = sort_metric

            # Determine which metric was used for selection
            opt_info = results.get("best_performance", {}).get("optimized_on", {})
            opt_metric = opt_info.get("metric", "")
            metric_used = (
                combo_sort_metric if combo_sort_metric is not None else opt_metric
            )

            # Get metrics data - either from best_performance or re-select from sorted_aggregated_results
            if combo_sort_metric is not None and combo_sort_metric != opt_metric:
                ds = results.get("sorted_aggregated_results")
                if not _ds_is_empty(ds) and combo_sort_metric in ds.coords["metric"].values:
                    sort_mode = _infer_metric_mode(combo_sort_metric)
                    sort_vals = ds.sel(metric=combo_sort_metric)["mean"].values
                    sort_idx  = np.argsort(sort_vals)
                    if sort_mode != "min":
                        sort_idx = sort_idx[::-1]
                    best_cfg = ds.isel(config=int(sort_idx[0]))

                    metrics_data = {}
                    for metric_full in ds.coords["metric"].values:
                        parts = str(metric_full).split("/", 1)
                        if len(parts) != 2:
                            continue
                        prefix, metric_name = parts
                        m         = best_cfg.sel(metric=metric_full)
                        mean_val  = float(m["mean"].values)
                        std_val   = float(m["std"].values) if not np.isnan(float(m["std"].values)) else 0.0
                        count_val = int(m["count"].values) if not np.isnan(float(m["count"].values)) else 0
                        metrics_data.setdefault(metric_name, {})[prefix] = {
                            "mean": mean_val, "std": std_val, "count": count_val,
                        }
                else:
                    metrics_data = results.get("best_performance", {}).get("metrics", {})
            else:
                # Use best_performance (selected by optimization metric)
                metrics_data = results.get("best_performance", {}).get("metrics", {})

            # Extract metrics for display
            if metrics_data:
                for metric_name, prefixes in metrics_data.items():
                    for prefix, values in prefixes.items():
                        mean = values.get("mean", float("nan"))
                        std = values.get("std", float("nan"))
                        count = values.get("count", "n/a")
                        fmt_val = format_mean_std_count(mean, std, count, precision)

                        # Fix: Explicitly add '/' separator for display
                        metric_display = f"{prefix}/{metric_name}"

                        records.append(
                            {
                                "Dataset": dataset,
                                "Model": model,
                                "Metric": metric_display,
                                "Value": fmt_val,
                            }
                        )

    if not records:
        logger.warning("Warning: No performance data found to create a summary table.")
        return pd.DataFrame()

    long_df = pd.DataFrame(records)
    summary_table = long_df.pivot_table(
        index="Model",
        columns=["Dataset", "Metric"],
        values="Value",
        aggfunc="first",
    )

    summary_table.sort_index(axis=1, inplace=True)
    summary_table.fillna("-", inplace=True)
    return summary_table


def export_summary_table(
    all_results: defaultdict,
    path: str,
    format: str = "csv",
    precision: int = 4,
    sort_metric: str = None,
) -> None:
    """Build the summary table and write it to a file (CSV or LaTeX).

    Parameters
    ----------
    all_results : defaultdict
        Result of analyze_results (dataset -> model -> result dict).
    path : str
        Output file path.
    format : str, optional
        "csv" or "latex". Default "csv".
    precision : int, optional
        Decimal precision for numeric values. Default 4.
    sort_metric : str, optional
        Metric to use for selecting best run. If None, uses optimization metric.
        Can also be a dict mapping (dataset, model) -> metric for per-combination control.
    """
    df = create_summary_table(all_results, precision=precision, sort_metric=sort_metric)
    if df.empty:
        logger.warning("No data to export; summary table is empty.")
        return
    if format == "csv":
        df.to_csv(path, index=True)
    elif format == "latex":
        df.to_latex(path, index=True, escape=False)
    else:
        raise ValueError(f"format must be 'csv' or 'latex', got {format!r}")


def get_experiment_stats_df(all_results: defaultdict) -> pd.DataFrame:
    """Build the experiment stats table (configs completed, runs, configs failed, seeds found). Returns empty DataFrame if no data."""
    data = []
    for dataset, models in all_results.items():
        for model_name, res in models.items():
            count = 0
            if "sorted_aggregated_results" in res:
                ds = res["sorted_aggregated_results"]
                if not _ds_is_empty(ds):
                    count = ds.sizes["config"]

            total_runs = res.get("total_runs", None)
            configs_failed = res.get("configs_failed_not_full_seed_set", 0)
            configs_failed_invalid_metric = res.get(
                "configs_failed_missing_invalid_metric", 0
            )

            seeds_info = res.get("seeds_info", {})
            d_seeds = seeds_info.get("data", "N/A")
            m_seeds = seeds_info.get("model", "N/A")

            seed_display = ""
            if d_seeds != "Not Found" and d_seeds != "Error":
                seed_display += f"Data: {d_seeds}"

            if m_seeds != "Not Found" and m_seeds != "Error":
                if seed_display:
                    seed_display += " | "
                seed_display += f"Model: {m_seeds}"

            if not seed_display:
                seed_display = "Unknown"

            row = {
                "Dataset": dataset,
                "Model": model_name,
                "Configs Completed": count,
                "Runs": total_runs if total_runs is not None else "",
                "Configs Failed (not full seed set)": configs_failed,
                "Configs Failed (missing/invalid metric)": configs_failed_invalid_metric,
                "Seeds Found": seed_display,
            }
            data.append(row)

    return pd.DataFrame(data) if data else pd.DataFrame()


def display_experiment_stats(all_results: defaultdict) -> None:
    """Display a table of completion counts and found seeds."""
    df = get_experiment_stats_df(all_results)
    if df.empty:
        logger.info("No results to display.")
        return

    logger.info(f"\n{'=' * 40}\n  EXPERIMENT STATISTICS\n{'=' * 40}")
    display(df.style.set_properties(**{"text-align": "left"}))


def export_experiment_stats(all_results: defaultdict, path: str) -> None:
    """Write experiment stats table to CSV."""
    df = get_experiment_stats_df(all_results)
    if df.empty:
        logger.warning("No experiment stats to export.")
        return
    df.to_csv(path, index=False)


def display_performance_tables(all_results: defaultdict, precision: int = 4) -> None:
    """Display separate pivot tables for each metric found in the results."""
    found_metrics = set()
    for dataset, models in all_results.items():
        for model, res in models.items():
            metrics_dict = res.get("best_performance", {}).get("metrics", {})
            for m_name, prefixes in metrics_dict.items():
                for prefix in prefixes:
                    # Fix: Explicitly add '/' separator
                    found_metrics.add(f"{prefix}/{m_name}")

    sorted_metrics = sorted(list(found_metrics))

    if not sorted_metrics:
        logger.info("No metrics found to display.")
        return

    for full_metric in sorted_metrics:
        logger.info(f"\n{'=' * 40}\n  METRIC: {full_metric}\n{'=' * 40}")
        records = []

        for dataset, models in all_results.items():
            for model, res in models.items():
                metrics_data = res.get("best_performance", {}).get("metrics", {})

                val_str = "-"
                # Parse "val/loss" back to "val" and "loss"
                parts = full_metric.split("/")
                # Handle standard case (prefix/metric)
                if len(parts) >= 2:
                    p_lookup = parts[0]
                    # Rejoin rest in case metric name has slashes
                    m_lookup = "/".join(parts[1:])

                    if m_lookup in metrics_data and p_lookup in metrics_data[m_lookup]:
                        vals = metrics_data[m_lookup][p_lookup]
                        mean = vals.get("mean", 0)
                        std = vals.get("std", 0)
                        count = vals.get("count", 0)

                        val_str = format_mean_std_count(mean, std, count, precision)

                records.append({"Dataset": dataset, "Model": model, "Value": val_str})

        if not records:
            continue

        df = pd.DataFrame(records)
        pivot = df.pivot(index="Model", columns="Dataset", values="Value").fillna("-")
        display(pivot)


def _build_hp_impact_df(res: dict, precision: int, sort_metric: str = None):
    """
    Build the HP impact DataFrame for one model/dataset result.

    Parameters
    ----------
    res : dict
        Result dict for a model/dataset combination from all_results
    precision : int
        Decimal precision for formatting
    sort_metric : str, optional
        Metric to use for sorting (e.g., "test/mse"). If None, uses optimization metric.

    Returns
    -------
    tuple or None
        (df, metric_used) if data exists, None otherwise
        df: DataFrame with HP impact table
        metric_used: str, the metric actually used for sorting
    """
    ds = res.get("sorted_aggregated_results")
    if _ds_is_empty(ds):
        return None

    opt_info       = res.get("best_performance", {}).get("optimized_on", {})
    opt_metric     = opt_info.get("metric", "")
    metric_to_sort = sort_metric if sort_metric is not None else opt_metric
    metric_used    = metric_to_sort

    # Re-sort by alternative metric if requested
    if sort_metric is not None and sort_metric != opt_metric:
        if sort_metric in ds.coords["metric"].values:
            sort_mode = _infer_metric_mode(sort_metric)
            sort_vals = ds.sel(metric=sort_metric)["mean"].values
            sort_idx  = np.argsort(sort_vals)
            if sort_mode != "min":
                sort_idx = sort_idx[::-1]
            ds = ds.isel(config=sort_idx.tolist())
        else:
            logger.warning(
                f"Sort metric '{sort_metric}' not in available metrics "
                f"{list(ds.coords['metric'].values)}. Falling back to '{opt_metric}'."
            )
            metric_used = opt_metric

    hp_coord_names = [c for c in ds.coords if c not in ("config", "metric")]
    metric_names   = [str(m) for m in ds.coords["metric"].values]

    rows = []
    for i in range(ds.sizes["config"]):
        cfg = ds.isel(config=i)
        row = {hp: cfg.coords[hp].item() for hp in hp_coord_names}
        for metric in metric_names:
            m         = cfg.sel(metric=metric)
            mean_val  = float(m["mean"].values)
            std_val   = float(m["std"].values)  if not np.isnan(float(m["std"].values))   else 0.0
            count_val = int(m["count"].values)  if not np.isnan(float(m["count"].values)) else 0
            row[metric] = "-" if np.isnan(mean_val) else format_mean_std_count(mean_val, std_val, count_val, precision)
        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Metric column ordering: sort/opt metric first, rest alphabetical
    ordered_metrics = sorted(metric_names)
    lead = metric_to_sort if metric_to_sort in ordered_metrics else opt_metric
    if lead in ordered_metrics:
        ordered_metrics.remove(lead)
        ordered_metrics = [lead] + ordered_metrics

    return df_out[hp_coord_names + ordered_metrics].fillna("-"), metric_used


def iter_hp_impact_tables(
    all_results: defaultdict, precision: int = 4, sort_metric: str = None
):
    """
    Yield (dataset, model, df, metric_used) for each model/dataset HP impact table.

    Parameters
    ----------
    all_results : defaultdict
        Result of analyze_results (dataset -> model -> result dict).
    precision : int
        Decimal precision for formatting.
    sort_metric : str, optional
        Metric to use for sorting. If None, uses optimization metric for each model/dataset.
        Can also be a dict mapping (dataset, model) -> metric for per-combination control.

    Yields
    ------
    tuple
        (dataset, model, df, metric_used) for each non-empty HP impact table
    """
    for dataset, models in all_results.items():
        for model, res in models.items():
            # Get sort_metric for this specific combination if dict provided
            if isinstance(sort_metric, dict):
                combo_sort_metric = sort_metric.get((dataset, model), None)
            else:
                combo_sort_metric = sort_metric

            # print(f"DEBUG: Processing {model} on {dataset}")
            # print(res.keys())
            result = _build_hp_impact_df(res, precision, sort_metric=combo_sort_metric)
            if result is not None:
                df, metric_used = result
                if res.get("sort_metric_is_fallback") and res.get(
                    "original_sort_metric"
                ):
                    metric_used = f"{metric_used} (fallback; original metric {res['original_sort_metric']} not found)"
                if not df.empty:
                    yield dataset, model, df, metric_used


def display_hp_impact(
    all_results: defaultdict, precision: int = 4, sort_metric: str = None
) -> None:
    """
    Display aggregated leaderboard for every model/dataset combination with (n=count).

    Parameters
    ----------
    all_results : defaultdict
        Result of analyze_results (dataset -> model -> result dict).
    precision : int
        Decimal precision for formatting.
    sort_metric : str, optional
        Metric to use for sorting. If None, uses optimization metric for each model/dataset.
        Can also be a dict mapping (dataset, model) -> metric for per-combination control.
    """
    logger.info(f"\n{'=' * 40}\n  HYPERPARAMETER IMPACT ANALYSIS\n{'=' * 40}")

    for dataset, model, df, metric_used in iter_hp_impact_tables(
        all_results, precision, sort_metric
    ):
        logger.info(f"\n>>> {model} on {dataset}")
        logger.info(f"    (Sorted by: {metric_used})")
        display(df)


def export_hp_impact_tables(
    all_results: defaultdict,
    output_dir: str,
    precision: int = 4,
    sort_metric: str = None,
) -> List[Tuple[str, str, str]]:
    """
    Save one CSV per model/dataset HP impact table. Returns list of (dataset, model, path).
    Filenames are sanitized for the filesystem.

    Parameters
    ----------
    all_results : defaultdict
        Result of analyze_results (dataset -> model -> result dict).
    output_dir : str
        Directory to save CSV files.
    precision : int
        Decimal precision for formatting.
    sort_metric : str, optional
        Metric to use for sorting. If None, uses optimization metric for each model/dataset.
        Can also be a dict mapping (dataset, model) -> metric for per-combination control.

    Returns
    -------
    List[Tuple[str, str, str]]
        List of (dataset, model, path) tuples for saved files.
    """
    os.makedirs(output_dir, exist_ok=True)
    out = []
    for dataset, model, df, metric_used in iter_hp_impact_tables(
        all_results, precision, sort_metric
    ):
        safe_ds = re.sub(r"[^\w\-.]", "_", str(dataset))[:80]
        safe_m = re.sub(r"[^\w\-.]", "_", str(model))[:80]
        path = os.path.join(output_dir, f"hp_impact_{safe_ds}_{safe_m}.csv")
        df.to_csv(path, index=True)
        out.append((dataset, model, path))
    return out



def write_tables_tex(
    path: str,
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    all_results: defaultdict,
    precision: int = 4,
    sort_metric: str = None,
) -> None:
    """
    Write a single LaTeX file containing all report tables for pasting into a paper.
    Sections: summary table, experiment stats, then one HP impact table per model/dataset.

    Parameters
    ----------
    path : str
        Output file path.
    summary_df : pd.DataFrame
        Summary table DataFrame.
    stats_df : pd.DataFrame
        Experiment stats DataFrame.
    all_results : defaultdict
        Result of analyze_results (dataset -> model -> result dict).
    precision : int
        Decimal precision for formatting.
    sort_metric : str, optional
        Metric used for sorting (for comments). If None, uses optimization metric.
        Can also be a dict mapping (dataset, model) -> metric for per-combination control.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    parts = [
        "% Generated by picid_report. Paste sections into your paper as needed.\n",
        "% Requires: \\usepackage{booktabs}\n\n",
    ]

    # Summary table
    parts.append("% --- Summary table (best metrics by model/dataset) ---\n")
    if sort_metric is not None:
        if isinstance(sort_metric, dict):
            parts.append(
                "% Metric used to select best results: varies by model/dataset\n"
            )
        else:
            parts.append(f"% Metric used to select best results: {sort_metric}\n")
    if not summary_df.empty:
        parts.append(summary_df.to_latex(index=True, escape=False))
        parts.append("\n")

    # Experiment stats
    parts.append("% --- Experiment statistics (configs completed, seeds) ---\n")
    if not stats_df.empty:
        parts.append(stats_df.to_latex(index=False, escape=False))
        parts.append("\n")

    # HP impact tables
    parts.append("% --- Hyperparameter impact (one table per model/dataset) ---\n")
    for dataset, model, df, metric_used in iter_hp_impact_tables(
        all_results, precision, sort_metric
    ):
        parts.append(
            f"% Dataset name: {dataset}, Model name: {model}, Metric used to select best results: {metric_used}\n"
        )
        parts.append(df.to_latex(index=True, escape=False))
        parts.append("\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
