# lmetk_report/analysis.py
"""
Core analysis module for processing experiment results.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

from lmetk_report import config, validators


def _trace_nans(df: pd.DataFrame, stage: str, prev_count: int) -> int:
    """Debug helper to track the introduction or removal of NaNs."""
    current_count = int(df.isna().sum().sum())
    # ... (Keep existing trace logic) ...
    return current_count


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


def get_dynamic_metrics(df_subset: pd.DataFrame) -> List[str]:
    """
    Extracts the list of metric names (e.g., ['f1', 'accuracy']) from the
    'evaluator.train.metric_names' column, which contains lists of strings.
    """
    col_name = config.COLUMN_CONFIG["evaluator_metrics"]
    if col_name not in df_subset.columns:
        return []

    # Get all unique lists from the column
    # The column contains lists like ['f1', 'acc'], so we need to unpack them
    found_metrics = set()

    # Drop NAs and iterate
    for entry in df_subset[col_name].dropna():
        if isinstance(entry, (list, tuple, np.ndarray)):
            found_metrics.update(entry)
        elif isinstance(entry, str):
            # Handle case where it might be a string representation of a list
            # or a single string
            import ast

            try:
                parsed = ast.literal_eval(entry)
                if isinstance(parsed, (list, tuple)):
                    found_metrics.update(parsed)
                else:
                    found_metrics.add(entry)
            except:
                found_metrics.add(entry)

    return sorted(list(found_metrics))


def get_optimized_metric(
    df_subset: pd.DataFrame, model_name: str, dataset_name: str
) -> str:
    """
    Finds the metric used for optimization.
    Priority:
    1. task_definition.target_metric (Your new request)
    2. callbacks.early_stopping.monitor (Legacy)
    3. Fallback heuristics
    """

    # --- PRIORITY 1: Task Definition ---
    target_col = config.COLUMN_CONFIG["target_metric"]
    if target_col in df_subset.columns:
        targets = get_unique_values(df_subset, target_col)
        if len(targets) == 1:
            # The config usually just says "f1", but the column in DF is "val/f1"
            # We need to find the matching column.
            metric_base = targets[0]

            # Try to find the actual column in the DF (prefer validation)
            candidates = [
                f"val/{metric_base}",
                f"test/{metric_base}",
                f"val_best_rerun/{metric_base}",
            ]
            for c in candidates:
                if c in df_subset.columns and df_subset[c].count() > 0:
                    print(f"Optimized metric (from task def) for {model_name}: {c}")
                    return c

            # If no prefix match, return base if it exists
            if metric_base in df_subset.columns:
                return metric_base

    # --- PRIORITY 2: Early Stopping Config ---
    metric_series = get_unique_values(
        df_subset, config.COLUMN_CONFIG["optimization_metric"]
    )
    if len(metric_series) >= 1:
        candidate = metric_series[0]
        if candidate in df_subset.columns and df_subset[candidate].count() > 0:
            print(
                f"Optimized metric (from early stopping) for {model_name}: {candidate}"
            )
            return candidate

    # --- PRIORITY 3: Fallbacks ---
    fallback_candidates = [
        "val/loss",
        "test/loss",
        "val/f1",
        "test/f1",
        "val/accuracy",
        "test/accuracy",
        "test/mse",
        "test/mae",
    ]
    for fb in fallback_candidates:
        if fb in df_subset.columns and df_subset[fb].count() > 0:
            print(f"Fallback metric for {model_name}: {fb}")
            return fb

    raise ValueError(f"No valid metric found for {model_name}.")


# ... (Keep get_varying_hyperparameters and aggregate_and_find_best as they were) ...
def get_varying_hyperparameters(df_subset, config_cols, special_cols_to_exclude):
    varying_params = {}
    for col in config_cols:
        if col not in df_subset.columns:
            continue
        try:
            cleaned = df_subset[col].dropna()
            if cleaned.empty:
                continue
            if isinstance(cleaned.iloc[0], (list, tuple)):
                s = cleaned.apply(lambda x: tuple(x) if isinstance(x, list) else x)
                if s.nunique() > 1:
                    varying_params[col] = [list(t) for t in s.unique()]
            else:
                if cleaned.nunique() > 1:
                    varying_params[col] = cleaned.unique().tolist()
        except:
            pass
    for col in special_cols_to_exclude:
        varying_params.pop(col, None)
    return varying_params


def aggregate_and_find_best(
    df_subset,
    aggregation_cols,
    all_performance_cols,
    optimization_col,
    optimization_mode,
):
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

    # Ensure optimization column exists
    if (optimization_col, "mean") not in aggregated_df.columns:
        # Fallback to first available metric if specific opt metric is missing
        if len(all_performance_cols) > 0:
            optimization_col = all_performance_cols[0]
        else:
            return pd.DataFrame(), pd.DataFrame()

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
    """Orchestrate the full analysis pipeline."""

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

            if subset.empty:
                continue

            validators.log_modification(
                action="Subsetting DataFrame",
                context=f"Model='{model}' | Dataset='{dataset}'",
                reason="Filtering data to keep only runs matching the current target configuration.",
                input_shape=subset_start_shape,
                output_shape=subset.shape,
            )
            subset["Model"] = model

            # --- 1. DYNAMIC METRIC DISCOVERY ---
            # Extract metrics from 'evaluator.train.metric_names'
            dynamic_metrics = get_dynamic_metrics(subset)

            # Combine hardcoded metrics with dynamically discovered ones
            # Use set to avoid duplicates
            all_metrics_to_report = list(set(reporting_metrics + dynamic_metrics))

            if not dynamic_metrics:
                print(
                    f"Info: No dynamic metrics found in '{config.COLUMN_CONFIG['evaluator_metrics']}'. Using defaults."
                )
            else:
                print(f"Dynamic metrics found for {model}: {dynamic_metrics}")

            # --- 2. DETERMINE OPTIMIZATION METRIC ---
            if optimization_col is None:
                try:
                    optimization_col = get_optimized_metric(subset, model, dataset)
                except Exception as e:
                    print(f"Warning: {e}. Fallback to test/mse")
                    optimization_col = "test/mse"

            # --- 3. DETERMINE OPTIMIZATION MODE ---
            # Check task_definition.target_metric_mode first
            if optimization_mode is None:
                target_mode_col = config.COLUMN_CONFIG["target_metric_mode"]
                if target_mode_col in subset.columns:
                    modes = subset[target_mode_col].dropna().unique()
                    if len(modes) > 0:
                        optimization_mode = modes[0]

                # Fallback to early stopping mode
                if optimization_mode is None:
                    es_mode_col = config.COLUMN_CONFIG["optimization_mode"]
                    if es_mode_col in subset.columns:
                        modes = subset[es_mode_col].dropna().unique()
                        if len(modes) > 0:
                            optimization_mode = modes[0]

                # Final default
                if optimization_mode is None:
                    optimization_mode = "min"

            # Build list of all columns to calculate (Mean/Std/Count) for
            all_performance_cols = []

            # Ensure the optimization column is ALWAYS calculated
            if optimization_col not in all_performance_cols:
                all_performance_cols.append(optimization_col)

            # Add all other metrics with prefixes
            for metric in all_metrics_to_report:
                for prefix in metric_prefixes:
                    col_name = f"{prefix}{metric}"
                    if col_name in subset.columns:
                        all_performance_cols.append(col_name)

            # Deduplicate list
            all_performance_cols = list(set(all_performance_cols))

            # --- 4. IDENTIFY HPs ---
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

            # --- 5. VALIDATION (Seeds & Vars) ---
            subset = validators.validate_seeds(
                subset, aggregation_cols, data_seed_col, required_data_seeds, False
            )
            subset = validators.validate_seeds(
                subset, aggregation_cols, model_seed_col, required_model_seeds, True
            )

            if subset.empty:
                continue

            ignore_vars = config.COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH + [
                data_seed_col,
                model_seed_col,
            ]
            validators.check_hidden_variations(
                subset, aggregation_cols, ignore_vars, additional_ignored_cols
            )

            # --- 6. AGGREGATION ---
            print(f"Hyperparameters for {model}: {list(varying_hyperparams.keys())}")

            sorted_aggregated, best_result = aggregate_and_find_best(
                subset,
                aggregation_cols,
                all_performance_cols,
                optimization_col,
                optimization_mode,
            )

            if best_result.empty:
                continue

            # --- 7. COLLECT RESULTS ---
            best_params = {hp: best_result[hp].item() for hp in varying_hyperparams}
            if "run_name" in best_result.columns:
                best_params["run_names"] = best_result["run_name"].iloc[0, 0]

            performance_results = defaultdict(dict)
            mode = optimization_mode

            # Structure results for report: metrics[metric_name][prefix]
            # Use the combined list of all discovered metrics
            for col_full in all_performance_cols:
                # Reverse engineer: split "val/f1" into "val" and "f1"
                # This is imperfect if metric name has '/', so we rely on prefixes
                matched_prefix = None
                metric_name = None

                for p in metric_prefixes:
                    if col_full.startswith(p):
                        matched_prefix = p.strip("/")
                        metric_name = col_full[len(p) :]  # remainder is metric name
                        break

                if matched_prefix and metric_name:
                    if (col_full, "mean") in best_result.columns:
                        mean = best_result[(col_full, "mean")].values[0]
                        std = best_result[(col_full, "std")].values[0]
                        cnt = best_result[(col_full, "count")].values[0]

                        # Scale if it's the target metric and we are maximizing (e.g. accuracy)
                        if col_full == optimization_col and mode == "max":
                            mean, std = mean * 100, std * 100

                        performance_results[metric_name][matched_prefix] = {
                            "mean": mean,
                            "std": std,
                            "count": cnt,
                        }

            res_entry = all_results[dataset][model]
            res_entry["best_hyperparameters"] = best_params
            res_entry["best_performance"] = {
                "optimized_on": {"metric": optimization_col, "strategy": mode},
                "metrics": performance_results,
            }
            res_entry["sorted_aggregated_results"] = sorted_aggregated
            res_entry["non_aggregated_df"] = subset.copy()

            # Seed info
            try:
                d_seeds = sorted(subset[data_seed_col].dropna().unique().astype(int))
                m_seeds = sorted(subset[model_seed_col].dropna().unique().astype(int))
                res_entry["seeds_info"] = {"data": str(d_seeds), "model": str(m_seeds)}
            except:
                res_entry["seeds_info"] = {"data": "?", "model": "?"}

    return all_results
