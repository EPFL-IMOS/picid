# lmetk_report/reporting.py
"""
Module for creating presentation-ready summaries and visualizations.
"""

from collections import defaultdict
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

                        # Fix: handle nans for clean display
                        std_val = std if not pd.isna(std) else 0.0
                        cnt_val = int(count) if not isinstance(count, str) else count

                        fmt_val = (
                            f"{mean:.{precision}f} ± "
                            f"{std_val:.{precision}f} (n={cnt_val})"
                        )

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
        print("No metrics found to display.")
        return

    for full_metric in sorted_metrics:
        print(f"\n{'=' * 40}\n  METRIC: {full_metric}\n{'=' * 40}")
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

                        # Handle potential NaN in std (e.g., if n=1)
                        std_val = std if not pd.isna(std) else 0.0

                        # Fix: Standardized output with count for consistency
                        val_str = f"{mean:.{precision}f} ± {std_val:.{precision}f} (n={int(count)})"

                records.append({"Dataset": dataset, "Model": model, "Value": val_str})

        if not records:
            continue

        df = pd.DataFrame(records)
        pivot = df.pivot(index="Model", columns="Dataset", values="Value").fillna("-")
        display(pivot)


def display_hp_impact(all_results: defaultdict, precision: int = 4) -> None:
    """Display aggregated leaderboard for every model/dataset combination with (n=count)."""
    print(f"\n{'=' * 40}\n  HYPERPARAMETER IMPACT ANALYSIS\n{'=' * 40}")

    for dataset, models in all_results.items():
        for model, res in models.items():
            if "sorted_aggregated_results" not in res:
                continue

            df = res["sorted_aggregated_results"].copy()
            if df.empty:
                continue

            # 1. Flatten MultiIndex columns
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    col_name, stat = col
                    new_cols.append(f"{col_name}_{stat}" if stat else col_name)
                else:
                    new_cols.append(col)
            df.columns = new_cols

            # 2. Identify and format metrics
            metric_bases = set(
                c.rsplit("_", 1)[0] for c in df.columns if c.endswith("_mean")
            )
            for base in metric_bases:
                m_col, s_col, c_col = f"{base}_mean", f"{base}_std", f"{base}_count"
                if m_col in df.columns and c_col in df.columns:

                    def format_impact_row(row):
                        if pd.isna(row[m_col]):
                            return "-"  # Missing grid entries

                        std_val = (
                            row[s_col]
                            if s_col in row and not pd.isna(row[s_col])
                            else 0.0
                        )
                        return f"{row[m_col]:.{precision}f} ± {std_val:.{precision}f} (n={int(row[c_col])})"

                    df[base] = df.apply(format_impact_row, axis=1)

            # 3. Final display logic
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

            print(f"\n>>> {model} on {dataset}")
            print(f"    (Sorted by optimization target: {opt_metric})")
            display(df[hp_cols + final_metrics].fillna("-"))
