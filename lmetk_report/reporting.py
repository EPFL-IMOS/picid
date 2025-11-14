# lmetk_report/reporting.py
"""
Contains functions for creating presentation-ready summaries and reports
from the processed analysis results.
"""
from collections import defaultdict
import pandas as pd


def create_summary_table(all_results: defaultdict, precision=10) -> pd.DataFrame:
    """
    Processes the 'all_results' dictionary to create a pivoted summary DataFrame.

    This function unnests the results, formats performance metrics into a
    'mean ± std' string, and pivots the data into a wide-format table for
    easy comparison of models across datasets.

    Args:
        all_results: The nested dictionary from the analyze_results function.

    Returns:
        A pivoted summary table with models as rows and dataset/metrics as columns.
    """
    records = []
    # Step 1: Flatten the nested dictionary into a list of records.
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
                        formatted_value = (
                            f"{mean:.{precision}f} ± {std:.{precision}f} (n={count})"
                        )
                        records.append(
                            {
                                "Dataset": dataset,
                                "Model": model,
                                "Metric": f"{prefix}/{metric_name}",
                                "Value": formatted_value,
                            }
                        )
    if not records:
        print("Warning: No performance data found to create a summary table.")
        return pd.DataFrame()

    # Step 2: Create a long-format DataFrame and pivot it.
    long_df = pd.DataFrame(records)
    summary_table = long_df.pivot_table(
        index="Model", columns=["Dataset", "Metric"], values="Value", aggfunc="first"
    )

    # Step 3: Clean up the table for better presentation.
    summary_table.sort_index(axis=1, inplace=True)  # Sort columns alphabetically.
    summary_table.fillna("-", inplace=True)  # Fill missing values.
    return summary_table
