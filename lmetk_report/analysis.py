# lmetk_report/analysis.py
"""
Core analysis module for processing experiment results.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import re  # Added regex for cleaner filtering

import pandas as pd
import numpy as np

from lmetk_report import config, validators
import itertools


def get_search_grid_df(search_space: dict) -> pd.DataFrame:
    """Creates a DataFrame from the Cartesian product of all HP values."""
    if not search_space:
        return pd.DataFrame()

    keys = list(search_space.keys())
    values = list(search_space.values())

    # Generate Cartesian product
    grid = list(itertools.product(*values))
    return pd.DataFrame(grid, columns=keys)


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
    """Extracts the list of metric names from evaluator config."""
    col_name = config.COLUMN_CONFIG["evaluator_metrics"]
    if col_name not in df_subset.columns:
        return []

    found_metrics = set()
    for entry in df_subset[col_name].dropna():
        if isinstance(entry, (list, tuple, np.ndarray)):
            found_metrics.update(entry)
        elif isinstance(entry, str):
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
    """Finds the metric used for optimization."""
    # Priority 1: Task Definition
    target_col = config.COLUMN_CONFIG["target_metric"]
    if target_col in df_subset.columns:
        targets = get_unique_values(df_subset, target_col)
        if len(targets) == 1:
            metric_base = targets[0]
            candidates = [
                f"val/{metric_base}",
                f"test/{metric_base}",
                f"val_best_rerun/{metric_base}",
            ]
            for c in candidates:
                if c in df_subset.columns and df_subset[c].count() > 0:
                    print(f"Optimized metric (from task def) for {model_name}: {c}")
                    return c
            if metric_base in df_subset.columns:
                return metric_base

    # Priority 2: Early Stopping
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

    # Priority 3: Fallbacks
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
        except Exception as e:
            import logging

            logging.warning(f"Could not analyze hyperparameter column '{col}': {e}")
            pass
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
    subset_copy = df_subset.copy()

    # --- BUG FIX: Ensure all performance columns are strictly numeric ---
    for col in all_performance_cols:
        if col in subset_copy.columns:
            subset_copy[col] = pd.to_numeric(subset_copy[col], errors="coerce")

    for col in aggregation_cols:
        if (
            col in subset_copy.columns
            and not subset_copy[col].dropna().empty
            and isinstance(subset_copy[col].dropna().iloc[0], (list, np.ndarray))
        ):
            subset_copy[col] = subset_copy[col].apply(
                lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x
            )

    # Only include valid numeric performance columns in aggregation
    agg_dict = {
        col: ["mean", "std", "count"]
        for col in all_performance_cols
        if col in subset_copy.columns
        and pd.api.types.is_numeric_dtype(subset_copy[col])
    }

    if "run_name" in subset_copy.columns:
        agg_dict["run_name"] = list
    if not agg_dict:
        return pd.DataFrame(), pd.DataFrame()

    aggregated_df = subset_copy.groupby(aggregation_cols).agg(agg_dict).reset_index()
    if ("run_name", "list") in aggregated_df.columns:
        aggregated_df.rename(columns={("run_name", "list"): "run_name"}, inplace=True)

    is_ascending = optimization_mode == "min"

    if (optimization_col, "mean") not in aggregated_df.columns:
        if len(all_performance_cols) > 0:
            # Fallback to the first available mean column
            valid_means = [
                c
                for c in aggregated_df.columns
                if isinstance(c, tuple) and c[1] == "mean"
            ]
            if valid_means:
                optimization_col = valid_means[0][0]
            else:
                return aggregated_df, pd.DataFrame()
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
            # Extract metrics for logging/info
            dynamic_metrics = get_dynamic_metrics(subset)
            if dynamic_metrics:
                print(f"Dynamic metrics found for {model}: {dynamic_metrics}")

            # --- 2. OPTIMIZATION METRIC ---
            if optimization_col is None:
                try:
                    from lmetk_report.analysis import get_optimized_metric

                    cur_optimization_col = get_optimized_metric(subset, model, dataset)
                except Exception as e:
                    print(f"Warning: {e}. Fallback to test/mse")
                    cur_optimization_col = "test/mse"
            else:
                cur_optimization_col = optimization_col

            # --- 3. OPTIMIZATION MODE ---
            cur_optimization_mode = optimization_mode
            if cur_optimization_mode is None:
                target_mode_col = config.COLUMN_CONFIG["target_metric_mode"]
                if target_mode_col in subset.columns:
                    modes = subset[target_mode_col].dropna().unique()
                    if len(modes) > 0:
                        cur_optimization_mode = modes[0]

                if cur_optimization_mode is None:
                    es_mode_col = config.COLUMN_CONFIG["optimization_mode"]
                    if es_mode_col in subset.columns:
                        modes = subset[es_mode_col].dropna().unique()
                        if len(modes) > 0:
                            cur_optimization_mode = modes[0]

                if cur_optimization_mode is None:
                    cur_optimization_mode = "min"

            # --- 4. ROBUST & CLEAN COLUMN COLLECTION ---
            all_performance_cols = []

            # 1. Always include optimization col
            if cur_optimization_col in subset.columns:
                all_performance_cols.append(cur_optimization_col)

            # 2. Scan numeric columns, but FILTER for high-level aggregates
            # include object type because it might contain coerced strings
            numeric_cols = subset.select_dtypes(include=[np.number, "object"]).columns
            for col in numeric_cols:
                # Must start with a known prefix
                if not any(col.startswith(p) for p in metric_prefixes):
                    continue

                is_aggregate = False

                metric_part = col
                for p in metric_prefixes:
                    if col.startswith(p):
                        metric_part = col[len(p) :]
                        break

                if metric_part in reporting_metrics:
                    is_aggregate = True

                # Case B: Contains "mean" substring (as requested)
                elif "mean" in col.lower():
                    is_aggregate = True

                # Case C: Is a "simple" metric (no numbers/underscores indicating splits)
                elif not re.search(r"\d", metric_part):
                    is_aggregate = True

                if is_aggregate:
                    all_performance_cols.append(col)

            # Deduplicate
            all_performance_cols = list(set(all_performance_cols))

            # --- 5. HP & VALIDATION ---
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

            subset = validators.validate_seeds(
                subset, aggregation_cols, data_seed_col, required_data_seeds, False
            )
            subset = validators.validate_seeds(
                subset, aggregation_cols, model_seed_col, required_model_seeds, True
            )

            if subset.empty:
                continue

            # Updated ignore_vars to exclude val_best_rerun/ metrics from warnings
            ignore_vars = config.COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH + [
                data_seed_col,
                model_seed_col,
                "val_best_rerun/",
            ]
            validators.check_hidden_variations(
                subset, aggregation_cols, ignore_vars, additional_ignored_cols
            )

            # --- 6. AGGREGATION ---
            sorted_aggregated, best_result = aggregate_and_find_best(
                subset,
                aggregation_cols,
                all_performance_cols,
                cur_optimization_col,
                cur_optimization_mode,
            )

            # --- NEW: PER-MODEL GRID INTEGRATION FIX ---
            model_grid = None
            if (
                hasattr(config, "EXPECTED_SEARCH_SPACE")
                and config.EXPECTED_SEARCH_SPACE
            ):
                model_grid = config.EXPECTED_SEARCH_SPACE.get(model)

            # Flatten MultiIndex regardless of grid presence for downstream consistency
            actual_results = sorted_aggregated.copy()
            new_cols = []
            for col in actual_results.columns:
                if isinstance(col, tuple):
                    col_name, stat = col
                    new_cols.append(f"{col_name}_{stat}" if stat else col_name)
                else:
                    new_cols.append(col)
            actual_results.columns = new_cols
            opt_col_flattened = f"{cur_optimization_col}_mean"

            if model_grid:
                # FIX: Clear notification for Schema-First mode
                print(f"\n[PIPELINE STATUS] Mode: SCHEMA-FIRST (Target Search Space)")
                print(f"    - Merging results with expected grid for model: '{model}'")

                grid_df = get_search_grid_df(model_grid)
                grid_df["Model"] = model
                join_keys = ["Model"] + list(model_grid.keys())

                # Normalization: Ensure all join keys are the same type (string)
                # This prevents '150' != '150.0' merge failures
                grid_df_merged = grid_df.copy()
                actual_results_merged = actual_results.copy()

                for k in join_keys:
                    if (
                        k in grid_df_merged.columns
                        and k in actual_results_merged.columns
                    ):
                        grid_df_merged[k] = grid_df_merged[k].astype(str)
                        actual_results_merged[k] = actual_results_merged[k].astype(str)

                # Join actual results onto the skeleton grid
                merged_df = pd.merge(
                    grid_df_merged, actual_results_merged, on=join_keys, how="left"
                )

                # Restore original types to hyperparameter columns for display
                for k in list(model_grid.keys()):
                    merged_df[k] = grid_df[k].values

                sorted_aggregated = merged_df.sort_values(
                    by=opt_col_flattened,
                    ascending=(cur_optimization_mode == "min"),
                    na_position="last",
                )
            else:
                # FIX: Clear notification for Data-First mode
                print(f"\n[PIPELINE STATUS] Mode: DATA-FIRST (Auto-Discovery)")
                print(f"    - No grid defined in config for: '{model}'")
                print(
                    f"    - Auto-discovered varying HPs: {list(varying_hyperparams.keys())}"
                )

                sorted_aggregated = actual_results.sort_values(
                    by=opt_col_flattened, ascending=(cur_optimization_mode == "min")
                )

            if best_result.empty:
                continue

            # --- 7. COLLECT RESULTS ---
            best_params = {hp: best_result[hp].item() for hp in varying_hyperparams}
            if "run_name" in best_result.columns:
                best_params["run_names"] = best_result["run_name"].iloc[0, 0]

            performance_results = defaultdict(dict)
            mode = cur_optimization_mode

            # Sort prefixes by length (longest first) to prevent partial matching bugs
            sorted_prefixes = sorted(metric_prefixes, key=len, reverse=True)

            for col_full in all_performance_cols:
                matched_prefix = ""
                metric_name = ""

                # Reverse engineer the prefix and metric name
                for p in sorted_prefixes:
                    if col_full.startswith(p):
                        matched_prefix = p.strip("/")
                        metric_name = col_full[len(p) :]
                        break

                # If we couldn't match a known prefix, skip
                if not matched_prefix:
                    continue

                if (col_full, "mean") in best_result.columns:
                    mean = best_result[(col_full, "mean")].values[0]
                    std = best_result[(col_full, "std")].values[0]
                    cnt = best_result[(col_full, "count")].values[0]

                    # Scale if it's the target metric and we are maximizing
                    if col_full == cur_optimization_col and mode == "max":
                        mean, std = mean * 100, std * 100

                    performance_results[metric_name][matched_prefix] = {
                        "mean": mean,
                        "std": std,
                        "count": cnt,
                    }

            res_entry = all_results[dataset][model]
            res_entry["best_hyperparameters"] = best_params
            res_entry["best_performance"] = {
                "optimized_on": {"metric": cur_optimization_col, "strategy": mode},
                "metrics": performance_results,
            }
            res_entry["sorted_aggregated_results"] = sorted_aggregated
            res_entry["non_aggregated_df"] = subset.copy()

            # Fix: Use native Python ints to avoid np.int64 wrappers in seed display
            def get_formatted_seeds(col_name):
                if col_name not in subset.columns:
                    return "?"
                try:
                    unique_seeds = sorted(
                        [int(x) for x in subset[col_name].dropna().unique()]
                    )
                    return str(unique_seeds) if unique_seeds else "?"
                except (ValueError, TypeError):
                    return "?"

            res_entry["seeds_info"] = {
                "data": get_formatted_seeds(data_seed_col),
                "model": get_formatted_seeds(model_seed_col),
            }

    return all_results
