# lmetk_report/reporting.py
"""
Module for creating presentation-ready summaries and visualizations.
"""

import logging
import os
import re
from collections import defaultdict
from typing import List, Tuple

import pandas as pd
from IPython.display import display

from lmetk_report.utils import flatten_aggregated_columns, format_mean_std_count

logger = logging.getLogger(__name__)


def create_summary_table(all_results: defaultdict, precision: int = 4) -> pd.DataFrame:
    """Create a single pivoted summary DataFrame for all results."""
    records = []

    for dataset, models in all_results.items():
        for model, results in models.items():
            if (
                "best_performance" in results
                and "metrics" in results["best_performance"]
            ):
                metrics_data = results["best_performance"]["metrics"]
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
    """
    df = create_summary_table(all_results, precision=precision)
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
    """Build the experiment stats table (configs completed, seeds found). Returns empty DataFrame if no data."""
    data = []
    for dataset, models in all_results.items():
        for model_name, res in models.items():
            count = 0
            if "sorted_aggregated_results" in res:
                count = len(res["sorted_aggregated_results"])

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

            data.append(
                {
                    "Dataset": dataset,
                    "Model": model_name,
                    "Configs Completed": count,
                    "Seeds Found": seed_display,
                }
            )

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


def _build_hp_impact_df(res: dict, precision: int):
    """Build the HP impact DataFrame for one model/dataset result. Returns None if no data."""
    if "sorted_aggregated_results" not in res or res["sorted_aggregated_results"].empty:
        return None
    df = res["sorted_aggregated_results"].copy()
    df = flatten_aggregated_columns(df)
    metric_bases = set(
        c.rsplit("_", 1)[0] for c in df.columns if c.endswith("_mean")
    )
    for base in metric_bases:
        m_col, s_col, c_col = f"{base}_mean", f"{base}_std", f"{base}_count"
        if m_col in df.columns and c_col in df.columns:

            def format_impact_row(row):
                if pd.isna(row[m_col]):
                    return "-"
                std_val = (
                    row[s_col]
                    if s_col in row and not pd.isna(row[s_col])
                    else 0.0
                )
                return format_mean_std_count(
                    row[m_col], std_val, int(row[c_col]), precision
                )

            df[base] = df.apply(format_impact_row, axis=1)

    opt_info = res.get("best_performance", {}).get("optimized_on", {})
    opt_metric = opt_info.get("metric", "")
    raw_stats, ignore = ["_mean", "_std", "_count"], [
        "run_name",
        "run_name_list",
        "Model",
    ]
    hp_cols = [
        c
        for c in df.columns
        if not any(c.endswith(s) for s in raw_stats)
        and c not in ignore
        and c not in metric_bases
    ]
    final_metrics = sorted(list(metric_bases))
    if opt_metric in final_metrics:
        final_metrics.remove(opt_metric)
        final_metrics = [opt_metric] + final_metrics
    return df[hp_cols + final_metrics].fillna("-")


def iter_hp_impact_tables(all_results: defaultdict, precision: int = 4):
    """Yield (dataset, model, df) for each model/dataset HP impact table."""
    for dataset, models in all_results.items():
        for model, res in models.items():
            df = _build_hp_impact_df(res, precision)
            if df is not None and not df.empty:
                yield dataset, model, df


def display_hp_impact(all_results: defaultdict, precision: int = 4) -> None:
    """Display aggregated leaderboard for every model/dataset combination with (n=count)."""
    logger.info(f"\n{'=' * 40}\n  HYPERPARAMETER IMPACT ANALYSIS\n{'=' * 40}")

    for dataset, models in all_results.items():
        for model, res in models.items():
            df = _build_hp_impact_df(res, precision)
            if df is None or df.empty:
                continue
            opt_metric = res.get("best_performance", {}).get("optimized_on", {}).get("metric", "")
            logger.info(f"\n>>> {model} on {dataset}")
            logger.info(f"    (Sorted by optimization target: {opt_metric})")
            display(df)


def export_hp_impact_tables(
    all_results: defaultdict,
    output_dir: str,
    precision: int = 4,
) -> List[Tuple[str, str, str]]:
    """
    Save one CSV per model/dataset HP impact table. Returns list of (dataset, model, path).
    Filenames are sanitized for the filesystem.
    """
    os.makedirs(output_dir, exist_ok=True)
    out = []
    for dataset, model, df in iter_hp_impact_tables(all_results, precision):
        safe_ds = re.sub(r"[^\w\-.]", "_", str(dataset))[:80]
        safe_m = re.sub(r"[^\w\-.]", "_", str(model))[:80]
        path = os.path.join(output_dir, f"hp_impact_{safe_ds}_{safe_m}.csv")
        df.to_csv(path, index=True)
        out.append((dataset, model, path))
    return out
