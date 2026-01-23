# lmetk_report/analysis.py
"""
Core analysis module for processing experiment results.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from lmetk_report import config, validators


def _trace_nans(df: pd.DataFrame, stage: str, prev_count: int) -> int:
    """Debug helper to track the introduction or removal of NaNs."""
    current_count = int(df.isna().sum().sum())
    diff = current_count - prev_count

    if diff != 0:
        if diff > 0:
            msg = f"🚨 [NaN TRACE] '{stage}' INTRODUCED {diff} NaNs!"
        else:
            msg = f"✅ [NaN TRACE] '{stage}' CLEANED {abs(diff)} NaNs."

        print(f"\n{msg}")
        print(f"    Total NaNs: {prev_count} -> {current_count}")
        print(f"    DataFrame Shape: {df.shape}")

        if diff > 0:
            nan_series = df.isna().sum()
            new_nan_cols = nan_series[nan_series > 0].sort_values(ascending=False)
            if not new_nan_cols.empty:
                top_3 = new_nan_cols.head(3).to_dict()
                print(f"    Top Columns with NaNs: {top_3}")
        print("-" * 40)

    return current_count


def find_checkpoint_path(best_runs_df: pd.DataFrame) -> Optional[str]:
    """Attempt to find the best checkpoint path from the best performing group."""
    configured_col = config.CHECKPOINT_PATH_COLUMN
    if (
        configured_col in best_runs_df.columns
        and not best_runs_df[configured_col].dropna().empty
    ):
        return best_runs_df[configured_col].dropna().iloc[0]

    common_alternatives = ["ckpt_path", "checkpoint_path"]
    for col in common_alternatives:
        if col in best_runs_df.columns and not best_runs_df[col].dropna().empty:
            print(f"Warning: Using fallback checkpoint path column '{col}'.")
            return best_runs_df[col].dropna().iloc[0]

    print(
        "Warning: Could not find a valid checkpoint path. "
        f"Checked '{configured_col}' and alternatives."
    )
    return None


def get_unique_values(df: pd.DataFrame, column: str) -> List[Any]:
    """Extract unique values from a column."""
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
    """
    Find and return the name of the metric used for optimization.
    Includes robust fallback logic if the configured metric is missing or NaN.
    """
    # 1. Try to get the metric defined in the config (e.g. 'val/loss')
    metric_series = get_unique_values(
        df_subset, config.COLUMN_CONFIG["optimization_metric"]
    )

    candidate = None
    if len(metric_series) == 1:
        candidate = metric_series[0]
    elif len(metric_series) > 1:
        print(
            f"Warning: Multiple optimization metrics found for {model_name}. "
            f"Picking first: {metric_series[0]}"
        )
        candidate = metric_series[0]

    # 2. Check if this candidate is valid (exists and has non-NaN values)
    if candidate and candidate in df_subset.columns:
        valid_count = df_subset[candidate].count()
        if valid_count > 0:
            print(
                f"Optimized metric for {model_name} on {dataset_name} "
                f"is {candidate}"
            )
            return candidate
        else:
            print(
                f"Warning: Configured metric '{candidate}' contains only NaNs "
                f"for {model_name}. Attempting fallback..."
            )

    # 3. Fallback Heuristics: Look for common metrics that actually have data
    # Priority: val/loss -> test/loss -> val/mse -> test/mse -> train/loss
    fallback_candidates = [
        "val/loss",
        "test/loss",
        "test/mse",
        "test/mae",
        "train/loss",
    ]

    for fb in fallback_candidates:
        if fb in df_subset.columns and df_subset[fb].count() > 0:
            print(
                f"Fallback: Switching optimization metric to '{fb}' "
                "(found valid data)."
            )
            return fb

    # 4. Final Hail Mary: Pick *any* numeric column with 'loss' or 'mse'
    for col in df_subset.columns:
        if ("loss" in col or "mse" in col) and df_subset[col].count() > 0:
            print(f"Fallback: Switching optimization metric to '{col}'.")
            return col

    # 5. If we reach here, we truly have no data to sort by
    raise ValueError(
        f"No valid metric found for {model_name} on {dataset_name}. "
        f"(Checked config '{candidate}' and common fallbacks)"
    )


def get_varying_hyperparameters(
    df_subset: pd.DataFrame,
    config_cols: List[str],
    special_cols_to_exclude: List[str],
) -> Dict[str, List[Any]]:
    """Identify hyperparameters that have more than one unique value."""
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
                f"Warning: Could not process column '{col}' for varying HPs. "
                f"Error: {e}"
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
    optimization_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform groupby aggregation and find the best run."""
    assert optimization_mode in ["min", "max"], "mode must be 'min' or 'max'"

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

    agg_dict = {
        col: ["mean", "std", "count"]
        for col in all_performance_cols
        if col in subset_copy.columns
    }
    if "run_name" in subset_copy.columns:
        agg_dict["run_name"] = list
    if not agg_dict:
        return pd.DataFrame(), pd.DataFrame()

    aggregated_df = subset_copy.groupby(aggregation_cols).agg(agg_dict).reset_index()
    if ("run_name", "list") in aggregated_df.columns:
        aggregated_df.rename(columns={("run_name", "list"): "run_name"}, inplace=True)

    is_ascending = optimization_mode == "min"
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
    optimization_col: Optional[str] = None,
    optimization_mode: Optional[str] = None,
    required_data_seeds: Optional[Set[int]] = None,
    data_seed_col: str = config.DEFAULT_DATA_SEED_COL,
    required_model_seeds: Optional[Set[int]] = None,
    model_seed_col: str = config.DEFAULT_MODEL_SEED_COL,
    additional_ignored_cols: Optional[List[str]] = None,
) -> defaultdict:
    """Orchestrate the full analysis pipeline for experiment results."""

    config_columns = [col for col in config_columns if col not in dropped_columns]
    unique_models = get_unique_values(df, config.COLUMN_CONFIG["model_target"])
    unique_datasets = get_unique_values(df, config.COLUMN_CONFIG["dataset_name"])
    all_results = defaultdict(lambda: defaultdict(dict))

    for dataset in unique_datasets:
        for model in unique_models:

            # --- SUBSETTING ---
            subset_start_shape = df.shape
            subset = df[
                (df[config.COLUMN_CONFIG["dataset_name"]] == dataset)
                & (df[config.COLUMN_CONFIG["model_target"]] == model)
            ].copy()

            validators.log_modification(
                action="Selecting Data for Analysis",
                reason=f"Isolating Model='{model}' and Dataset='{dataset}'",
                input_shape=subset_start_shape,
                output_shape=subset.shape,
            )

            nan_count = int(subset.isna().sum().sum())

            if subset.empty:
                print(f"---------\nNo results for {model} on {dataset}\n---------")
                continue

            subset["Model"] = model

            if optimization_col is None:
                try:
                    optimization_col = get_optimized_metric(subset, model, dataset)
                except Exception as e:
                    print(f"Warning: {e}. Falling back to default 'test/mse'.")
                    optimization_col = "test/mse"

            all_performance_cols = [
                f"{p}{m}"
                for p in metric_prefixes
                for m in reporting_metrics
                if f"{p}{m}" in subset.columns
            ]

            # 1. Identify Varying Hyperparameters.
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

            nan_count = _trace_nans(subset, "HP Extraction", nan_count)

            # --- VALIDATION 1: DATA SEEDS ---
            subset = validators.validate_seeds(
                subset,
                aggregation_cols,
                data_seed_col,
                required_data_seeds,
                allow_fallback=False,  # Strict
            )

            nan_count = _trace_nans(subset, "Data Seed Filter", nan_count)

            # --- VALIDATION 2: MODEL SEEDS ---
            subset = validators.validate_seeds(
                subset,
                aggregation_cols,
                model_seed_col,
                required_model_seeds,
                allow_fallback=True,  # Flexible
            )

            nan_count = _trace_nans(subset, "Model Seed Filter", nan_count)

            if subset.empty:
                print(
                    f"Skipping {model} on {dataset}: "
                    "No configs matched seed requirements."
                )
                continue

            # --- CAPTURE SEED INFO ---
            def get_seeds_str(column_name):
                if column_name in subset.columns:
                    try:
                        s_vals = subset[column_name].dropna().unique()
                        s_list = sorted([int(float(x)) for x in s_vals])
                        return ", ".join(map(str, s_list))
                    except Exception:
                        return "Error"
                return "Not Found"

            data_seeds_str = get_seeds_str(data_seed_col)
            model_seeds_str = get_seeds_str(model_seed_col)

            # --- VALIDATION 3: HIDDEN VARIATIONS ---
            ignore_vars = config.COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH + [
                data_seed_col,
                model_seed_col,
            ]
            validators.check_hidden_variations(
                subset,
                aggregation_cols,
                base_ignored_cols=ignore_vars,
                additional_ignored_cols=additional_ignored_cols,
            )

            nan_count = _trace_nans(subset, "Variation Check", nan_count)

            # --- STANDARD LOGIC ---
            print(f"Hyperparameters that vary for {model} on {dataset}:")
            if not varying_hyperparams:
                print("None.")
            else:
                for col, unique in varying_hyperparams.items():
                    print(f"{col}: {unique}")
            print("---------")

            if optimization_mode is None:
                modes = (
                    subset[config.COLUMN_CONFIG["optimization_mode"]].dropna().unique()
                )
                if len(modes) > 0:
                    optimization_mode = modes[0]
                else:
                    print("Warning: Optimization mode not found. Defaulting to 'min'.")
                    optimization_mode = "min"

            nan_count = _trace_nans(subset, "Pre-Aggregation", nan_count)

            sorted_aggregated, best_result = aggregate_and_find_best(
                subset,
                aggregation_cols,
                all_performance_cols,
                optimization_col=optimization_col,
                optimization_mode=optimization_mode,
            )

            if best_result.empty:
                continue

            # --- COLLECT RESULTS ---
            best_params = {}
            for hp_name in varying_hyperparams.keys():
                best_params[hp_name] = best_result[hp_name].item()

            if "run_name" in best_result.columns:
                best_params["run_names"] = best_result["run_name"].iloc[0, 0]

            best_checkpoint_path = None
            if "run_names" in best_params and best_params["run_names"]:
                best_run_name = best_params["run_names"][0]
                run_row = subset[subset["run_name"] == best_run_name]
                if not run_row.empty:
                    path_col = config.CHECKPOINT_PATH_COLUMN
                    if (
                        path_col in run_row.columns
                        and not run_row[path_col].isna().iloc[0]
                    ):
                        best_checkpoint_path = run_row[path_col].iloc[0]

            mode = optimization_mode
            performance_results = defaultdict(dict)
            for metric in reporting_metrics:
                for prefix in metric_prefixes:
                    col_name = f"{prefix}{metric}"
                    prefix_key = prefix.strip("/")
                    if (col_name, "mean") in best_result.columns:
                        mean_val = best_result[(col_name, "mean")].values[0]
                        std_val = best_result[(col_name, "std")].values[0]
                        count_val = best_result[(col_name, "count")].values[0]

                        # Scale Max metrics to % (heuristic).
                        if col_name == optimization_col and mode == "max":
                            mean_val = mean_val * 100
                            std_val = std_val * 100

                        performance_results[metric][prefix_key] = {
                            "mean": mean_val,
                            "std": std_val,
                            "count": count_val,
                        }

            res_entry = all_results[dataset][model]
            res_entry["best_hyperparameters"] = best_params
            res_entry["best_performance"] = {
                "optimized_on": {"metric": optimization_col, "strategy": mode},
                "metrics": performance_results,
            }
            if "run_names" in best_params:
                res_entry["best_run_names"] = best_params["run_names"]

            res_entry["best_checkpoint_path"] = best_checkpoint_path

            if "AvgTime/train_epoch_mean" in subset.columns:
                res_entry["avg_epoch_time"] = {
                    "mean": subset["AvgTime/train_epoch_mean"].mean(),
                    "std": subset["AvgTime/train_epoch_mean"].std(),
                }

            if "_runtime" in subset.columns:
                res_entry["avg_runtime"] = {
                    "mean": subset["_runtime"].mean(),
                    "std": subset["_runtime"].std(),
                }

            res_entry["sorted_aggregated_results"] = sorted_aggregated
            res_entry["non_aggregated_df"] = subset.copy()
            res_entry["seeds_info"] = {
                "data": data_seeds_str,
                "model": model_seeds_str,
            }

    return all_results
