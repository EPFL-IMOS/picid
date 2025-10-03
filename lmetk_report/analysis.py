# lmetk_report/analysis.py
"""
Core analysis module for processing experiment results.

This module contains functions to group data, identify varying hyperparameters,
aggregate results, and find the best-performing configurations.
"""
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import pandas as pd
from lmetk_report import config


def get_unique_values(df: pd.DataFrame, column: str) -> List[Any]:
    """Extracts unique values from a column, handling lists and scalars."""
    if column not in df.columns:
        return []
    series = df[column].dropna()
    unique_items = set()
    for item in series:
        if isinstance(item, list):
            unique_items.update(item)
        else:
            unique_items.add(item)
    return list(unique_items)


def get_optimized_metric(
    df_subset: pd.DataFrame, model_name: str, dataset_name: str
) -> str:
    """Finds and returns the full name of the metric used for optimization."""
    metric_series = get_unique_values(
        df_subset, config.COLUMN_CONFIG["optimization_metric"]
    )
    if not metric_series:
        raise ValueError(f"No metric found for {model_name} on {dataset_name}.")
    assert len(metric_series) == 1, "There should be only one metric to optimize"
    full_metric_name = metric_series[0]
    print(f"Optimized metric for {model_name} on {dataset_name} is {full_metric_name}")
    return full_metric_name


def get_varying_hyperparameters(
    df_subset: pd.DataFrame, config_cols: List[str], special_cols_to_exclude: List[str]
) -> Dict[str, List[Any]]:
    """Identifies hyperparameters with more than one unique value, handling lists correctly."""
    varying_params = {}
    for col in config_cols:
        if col not in df_subset.columns:
            continue
        try:
            cleaned_series = df_subset[col].dropna()
            if cleaned_series.empty:
                continue

            is_list_like = isinstance(cleaned_series.iloc[0], (list, tuple))
            if is_list_like:
                series_of_tuples = cleaned_series.apply(
                    lambda x: tuple(x) if isinstance(x, list) else x
                )
                if series_of_tuples.nunique() > 1:
                    varying_params[col] = [list(t) for t in series_of_tuples.unique()]
            else:
                if cleaned_series.nunique() > 1:
                    varying_params[col] = cleaned_series.unique().tolist()
        except Exception as e:
            print(
                f"Warning: Could not process column '{col}' for varying hyperparameters. Error: {e}"
            )
            continue

    for col in special_cols_to_exclude:
        varying_params.pop(col, None)
    return varying_params


def aggregate_and_find_best(
    df_subset: pd.DataFrame,
    aggregation_cols: List[str],
    all_performance_cols: List[str],
    optimization_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs groupby aggregation and finds the best run based on the optimization metric.
    """
    subset_copy = df_subset.copy()
    for col in aggregation_cols:
        if (
            col in subset_copy.columns
            and not subset_copy[col].dropna().empty
            and isinstance(subset_copy[col].dropna().iloc[0], list)
        ):
            subset_copy[col] = subset_copy[col].apply(
                lambda x: tuple(x) if isinstance(x, list) else x
            )

    # Create an aggregation dictionary for all performance columns
    agg_dict = {
        col: ["mean", "std"]
        for col in all_performance_cols
        if col in subset_copy.columns
    }

    # <<< NEW: Add an aggregation to collect run_names into a list >>>
    if "run_name" in subset_copy.columns:
        agg_dict["run_name"] = list

    if not agg_dict:
        return pd.DataFrame(), pd.DataFrame()

    aggregated_df = subset_copy.groupby(aggregation_cols).agg(agg_dict).reset_index()

    # Clean up column names if pandas creates a multi-index for 'run_name'
    if ("run_name", "list") in aggregated_df.columns:
        aggregated_df.rename(columns={("run_name", "list"): "run_name"}, inplace=True)

    mode = subset_copy[config.COLUMN_CONFIG["optimization_mode"]].unique()[0]
    is_ascending = mode == "min"

    sorted_results = aggregated_df.sort_values(
        by=(optimization_col, "mean"), ascending=is_ascending
    )
    return sorted_results, sorted_results.head(1)


def analyze_results(
    df: pd.DataFrame,
    config_columns: List[str],
    dropped_columns: List[str],
    reporting_metrics: List[str],
    metric_prefixes: List[str],
) -> defaultdict:
    """Main function to orchestrate the analysis of experiment results."""
    config_columns = [col for col in config_columns if col not in dropped_columns]
    unique_models = get_unique_values(df, config.COLUMN_CONFIG["model_target"])
    unique_datasets = get_unique_values(df, config.COLUMN_CONFIG["dataset_name"])
    all_results = defaultdict(lambda: defaultdict(dict))

    for dataset in unique_datasets:
        for model in unique_models:
            subset = df[
                (df[config.COLUMN_CONFIG["dataset_name"]] == dataset)
                & (df[config.COLUMN_CONFIG["model_target"]] == model)
            ].copy()
            if subset.empty:
                print(f"---------\nNo results for {model} on {dataset}\n---------")
                continue

            subset["Model"] = model

            optimization_col = get_optimized_metric(subset, model, dataset)
            all_performance_cols = [
                f"{p}{m}"
                for p in metric_prefixes
                for m in reporting_metrics
                if f"{p}{m}" in subset.columns
            ]

            hp_search_columns = [
                col
                for col in config_columns
                if not any(
                    f in col for f in config.COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH
                )
            ]
            varying_hyperparams = get_varying_hyperparameters(
                subset, hp_search_columns, config.SPECIAL_COLUMNS
            )
            aggregation_cols = ["Model"] + list(varying_hyperparams.keys())

            print(f"Hyperparameters that vary for {model} on {dataset}:")
            if not varying_hyperparams:
                print("None.")
            else:
                for col, unique in varying_hyperparams.items():
                    print(f"{col}: {unique}")
            print("---------")

            sorted_aggregated, best_result = aggregate_and_find_best(
                subset, aggregation_cols, all_performance_cols, optimization_col
            )

            if best_result.empty:
                continue

            # --- Step 4: Collect and structure the final results ---

            # Extract the best hyperparameter values
            best_params = {}
            for hp_name in varying_hyperparams.keys():
                best_params[hp_name] = best_result[hp_name].item()

            # Extract the list of run names for the best configuration
            if "run_name" in best_result.columns:
                # <<< CHANGED: Use .iloc[0, 0] to get the single value from the DataFrame cell
                best_params["run_names"] = best_result["run_name"].iloc[0, 0]

            # (rest of the result collection logic is the same...)
            mode = subset[config.COLUMN_CONFIG["optimization_mode"]].unique()[0]
            performance_results = defaultdict(dict)
            for metric in reporting_metrics:
                for prefix in metric_prefixes:
                    col_name = f"{prefix}{metric}"
                    prefix_key = prefix.strip("/")
                    if (col_name, "mean") in best_result.columns:
                        mean_val = best_result[(col_name, "mean")].values[0]
                        std_val = best_result[(col_name, "std")].values[0]
                        if col_name == optimization_col and mode == "max":
                            mean_val, std_val = mean_val * 100, std_val * 100
                        performance_results[metric][prefix_key] = {
                            "mean": mean_val,
                            "std": std_val,
                        }

            all_results[dataset][model]["best_hyperparameters"] = best_params
            all_results[dataset][model]["best_performance"] = {
                "optimized_on": {"metric": optimization_col, "strategy": mode},
                "metrics": performance_results,
            }
            all_results[dataset][model]["avg_epoch_time"] = {
                "mean": subset["AvgTime/train_epoch_mean"].mean(),
                "std": subset["AvgTime/train_epoch_mean"].std(),
            }
            all_results[dataset][model]["avg_runtime"] = {
                "mean": subset["_runtime"].mean(),
                "std": subset["_runtime"].std(),
            }
            all_results[dataset][model]["sorted_aggregated_results"] = sorted_aggregated
            all_results[dataset][model]["non_aggregated_df"] = subset.copy()

    return all_results
