# src/reporting.py
from collections import defaultdict
import pandas as pd


def create_summary_table(all_results: defaultdict) -> pd.DataFrame:
    """Processes the 'all_results' dictionary to create a pivoted summary DataFrame."""
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
                        formatted_value = f"{mean:.10f} ± {std:.10f}"
                        records.append(
                            {
                                "Dataset": dataset,
                                "Model": model,
                                "Metric": f"{prefix}/{metric_name}",
                                "Value": formatted_value,
                            }
                        )
    if not records:
        print("No performance data found to create a summary table.")
        return pd.DataFrame()

    long_df = pd.DataFrame(records)
    summary_table = long_df.pivot_table(
        index="Model", columns=["Dataset", "Metric"], values="Value", aggfunc="first"
    )

    summary_table.sort_index(axis=1, inplace=True)
    summary_table.fillna("-", inplace=True)
    return summary_table
