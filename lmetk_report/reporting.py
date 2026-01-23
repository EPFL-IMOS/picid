# lmetk_report/reporting.py
"""
Module for creating presentation-ready summaries and visualizations.
"""

from collections import defaultdict
from typing import Optional

import pandas as pd
from IPython.display import display


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
                        fmt_val = (
                            f"{mean:.{precision}f} ± "
                            f"{std:.{precision}f} (n={count})"
                        )
                        records.append(
                            {
                                "Dataset": dataset,
                                "Model": model,
                                "Metric": f"{prefix}{metric_name}",
                                "Value": fmt_val,
                            }
                        )

    if not records:
        print("Warning: No performance data found to create a summary table.")
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


def display_experiment_stats(all_results: defaultdict) -> None:
    """Display a table of completion counts and found seeds."""
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

    if not data:
        print("No results to display.")
        return

    print(f"\n{'=' * 40}\n  EXPERIMENT STATISTICS\n{'=' * 40}")
    df = pd.DataFrame(data)
    display(df.style.set_properties(**{"text-align": "left"}))


def display_performance_tables(all_results: defaultdict, precision: int = 2) -> None:
    """Display separate pivot tables for each metric found in the results."""
    found_metrics = set()
    for dataset, models in all_results.items():
        for model, res in models.items():
            metrics_dict = res.get("best_performance", {}).get("metrics", {})
            for m_name, prefixes in metrics_dict.items():
                for prefix in prefixes:
                    found_metrics.add(f"{prefix}{m_name}")

    sorted_metrics = sorted(list(found_metrics))

    if not sorted_metrics:
        print("No metrics found to display.")
        return

    for full_metric in sorted_metrics:
        print(f"\n{'=' * 40}\n  METRIC: {full_metric}\n{'=' * 40}")
        records = []

        for dataset, models in all_results.items():
            for model, res in models.items():
                metrics_data = res.get("best_performance", {}).get("metrics", {})

                val_str = "-"
                for m_name, prefixes in metrics_data.items():
                    for prefix, vals in prefixes.items():
                        current_full = f"{prefix}{m_name}"
                        if current_full == full_metric:
                            mean = vals.get("mean", 0)
                            std = vals.get("std", 0)
                            val_str = f"{mean:.{precision}f} ± {std:.{precision}f}"

                records.append({"Dataset": dataset, "Model": model, "Value": val_str})

        if not records:
            continue

        df = pd.DataFrame(records)
        pivot = df.pivot(index="Model", columns="Dataset", values="Value").fillna("-")
        display(pivot)


def display_hp_impact(all_results: defaultdict) -> None:
    """
    Display aggregated leaderboard for every model/dataset combination.
    Shows the mean of ALL available metrics.
    """
    print(f"\n{'=' * 40}\n  HYPERPARAMETER IMPACT ANALYSIS\n{'=' * 40}")

    for dataset, models in all_results.items():
        for model, res in models.items():
            if "sorted_aggregated_results" not in res:
                continue

            df = res["sorted_aggregated_results"].copy()
            if df.empty:
                continue

            # Flatten MultiIndex columns (col, stat) -> col_stat.
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    col_name, stat = col
                    if stat == "":
                        new_cols.append(col_name)  # Grouping key
                    else:
                        new_cols.append(f"{col_name}_{stat}")
                else:
                    new_cols.append(col)
            df.columns = new_cols

            opt_info = res.get("best_performance", {}).get("optimized_on", {})
            opt_metric = opt_info.get("metric", "")

            # --- UPDATED SELECTION LOGIC ---
            # Include:
            # 1. The specific optimized metric columns (including std, count)
            # 2. ANY column ending in '_mean' (to show all other metrics)

            metric_cols = [
                c for c in df.columns if (opt_metric in c) or c.endswith("_mean")
            ]

            ignore_cols = ["run_name_list", "run_name", "Model"]

            hp_cols = [
                c
                for c in df.columns
                if c not in metric_cols
                and c not in ignore_cols
                and "_std" not in c
                and "_mean" not in c
                and "_count" not in c
            ]

            final_cols = hp_cols + metric_cols
            # Filter to ensure columns actually exist in this specific DF
            final_cols = [c for c in final_cols if c in df.columns]

            print(f"\n>>> {model} on {dataset}")
            print(f"    (Sorted by {opt_metric})")

            display(df[final_cols].head(10))
