# lmetk_report/core/analysis.py
"""
Core analysis module for processing experiment results.

This module turns a flat DataFrame of runs (one row per run) into a nested structure
all_results[dataset][model] with:
- best_hyperparameters / best_performance: best run per (dataset, model) by optimization metric
- sorted_aggregated_results: aggregated stats (mean/std/count) per HP config, sorted by optimization metric
- optional sort_metric_used: metric used for ranking when different from optimization metric

Two modes:
- Schema-first: when a search space grid is provided, results are left-joined to the grid so
  missing configs appear as rows with NaN; sorting uses the configured grid order.
- Data-first: when no grid is found, varying hyperparameters are discovered from the data
  and only completed configs are shown.
"""

import ast
import itertools
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from lmetk_report import config
from lmetk_report.config import PipelineConfig
from lmetk_report.core import validators
from lmetk_report.utils import flatten_aggregated_columns

# Import search space helper: one function handles both legacy and new structure
try:
    from lmetk_report.configs.search_space import get_model_grid_from_search_space
except ImportError:
    def get_model_grid_from_search_space(_dataset: str, _model: str, search_space):
        if not search_space or not isinstance(search_space, dict):
            return None
        sample_val = next(iter(search_space.values()), None)
        if not sample_val or not isinstance(sample_val, dict):
            return None
        sample_inner = next(iter(sample_val.values()), None)
        if isinstance(sample_inner, list):
            return search_space.get(_model)
        if _dataset in search_space and _model in search_space[_dataset]:
            return search_space[_dataset][_model]
        return None

logger = logging.getLogger(__name__)


def get_search_grid_df(search_space: dict) -> pd.DataFrame:
    """
    Build a DataFrame whose rows are all combinations of HP values (Cartesian product).

    Args:
        search_space: Dict mapping HP column name -> list of values (e.g. {"lr": [0.01, 0.001], "seq_len": [10, 50]}).

    Returns:
        DataFrame with one column per HP and one row per combination; empty DataFrame if search_space is empty.
    """
    if not search_space:
        return pd.DataFrame()

    keys = list(search_space.keys())
    values = list(search_space.values())

    # Generate Cartesian product
    grid = list(itertools.product(*values))
    return pd.DataFrame(grid, columns=keys)


def _count_grid_configs_with_all_seeds(
    subset: pd.DataFrame,
    grid_df: pd.DataFrame,
    join_keys: List[str],
    model_seed_col: str,
    required_model_seeds: Set[int],
) -> int:
    """
    Count how many grid configs (rows in grid_df) have all required_model_seeds present in subset.

    Used for schema-first stats: y = count before metric filter, z = count after metric filter.
    """
    if subset.empty or grid_df.empty or not required_model_seeds or model_seed_col not in subset.columns:
        return 0
    join_keys = [k for k in join_keys if k in subset.columns and k in grid_df.columns]
    if not join_keys:
        return 0
    try:
        sub = subset.copy()
        gr = grid_df.copy()
        for k in join_keys:
            sub[k] = sub[k].astype(str)
            gr[k] = gr[k].astype(str)
        grouped = sub.groupby(join_keys)[model_seed_col].apply(
            lambda s: {int(float(x)) for x in s.dropna().unique()}
        )
        count = 0
        for _, row in gr.iterrows():
            key = tuple(row[k] for k in join_keys)
            if key in grouped.index and required_model_seeds.issubset(grouped[key]):
                count += 1
        return count
    except Exception:
        return 0


def _trace_nans(df: pd.DataFrame, stage: str, prev_count: int) -> int:
    """Debug helper to track the introduction or removal of NaNs."""
    current_count = int(df.isna().sum().sum())
    # ... (Keep existing trace logic) ...
    return current_count


def get_unique_values(df: pd.DataFrame, column: str) -> List[Any]:
    """
    Extract unique values from a column; list cells are flattened so each element counts as one value.

    Args:
        df: DataFrame containing the column.
        column: Column name; if missing, returns [].

    Returns:
        Sorted list of unique values (lists in cells are expanded into individual items).
    """
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


def get_dynamic_metrics(
    df_subset: pd.DataFrame,
    column_config: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Extract metric names from the evaluator config column (e.g. evaluator.train.metric_names).

    Handles list/tuple/string values in that column and returns a sorted list of unique metric names.

    Args:
        df_subset: DataFrame slice (e.g. for one dataset/model).
        column_config: Optional column config; uses config.COLUMN_CONFIG if None.

    Returns:
        Sorted list of metric name strings, or [] if the column is missing or empty.
    """
    col_cfg = column_config if column_config is not None else config.COLUMN_CONFIG
    col_name = col_cfg["evaluator_metrics"]
    if col_name not in df_subset.columns:
        return []

    found_metrics = set()
    for entry in df_subset[col_name].dropna():
        if isinstance(entry, (list, tuple, np.ndarray)):
            found_metrics.update(entry)
        elif isinstance(entry, str):
            try:
                parsed = ast.literal_eval(entry)
                if isinstance(parsed, (list, tuple)):
                    found_metrics.update(parsed)
                else:
                    found_metrics.add(entry)
            except Exception:
                found_metrics.add(entry)
    return sorted(list(found_metrics))


def get_optimized_metric(
    df_subset: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    column_config: Optional[Dict[str, str]] = None,
) -> str:
    """
    Resolve the metric used for optimization (best-run selection) for this dataset/model.

    Priority: (1) task_definition.target_metric, (2) early-stopping monitor,
    (3) fallbacks (val/loss, test/mse, etc.). The first column that exists and has data is returned.

    Args:
        df_subset: DataFrame slice for this dataset/model.
        model_name: Model identifier (for logging).
        dataset_name: Dataset identifier (for logging).
        column_config: Optional column config; uses config.COLUMN_CONFIG if None.

    Returns:
        Column name of the optimization metric (e.g. "val_best_rerun/loss", "test/mse").

    Raises:
        ValueError: If no valid metric column is found.
    """
    col_cfg = column_config if column_config is not None else config.COLUMN_CONFIG
    # Priority 1: Task Definition
    target_col = col_cfg["target_metric"]
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
                    logger.info(f"Optimized metric (from task def) for {model_name}: {c}")
                    return c
            if metric_base in df_subset.columns:
                return metric_base

    # Priority 2: Early Stopping
    metric_series = get_unique_values(
        df_subset, col_cfg["optimization_metric"]
    )
    if len(metric_series) >= 1:
        candidate = metric_series[0]
        if candidate in df_subset.columns and df_subset[candidate].count() > 0:
            logger.info(
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
            logger.info(f"Fallback metric for {model_name}: {fb}")
            return fb

    raise ValueError(f"No valid metric found for {model_name}.")


def get_varying_hyperparameters(
    df_subset: pd.DataFrame,
    config_cols: List[str],
    special_cols_to_exclude: List[str],
) -> Dict[str, List[Any]]:
    """
    Identify config columns that vary (more than one unique value) in the subset.

    Used for data-first mode when no search space grid is provided: these columns
    become the aggregation dimensions and define the HP grid from the data.

    Args:
        df_subset: DataFrame slice (e.g. one dataset/model).
        config_cols: Candidate column names (e.g. all config columns after filters).
        special_cols_to_exclude: Column names to exclude from varying HPs (e.g. seeds).

    Returns:
        Dict mapping column name -> list of unique values for each varying column.
    """
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
            logger.warning("Could not analyze hyperparameter column '%s': %s", col, e)
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
    """
    Aggregate runs by HP config (mean/std/count per metric) and pick the best config row.

    Args:
        df_subset: Runs for one dataset/model.
        aggregation_cols: Columns to group by (HP names).
        all_performance_cols: Metric columns to aggregate (mean, std, count).
        optimization_col: Metric column used to rank configs.
        optimization_mode: "min" or "max" for ranking.

    Returns:
        (aggregated_df, best_row_df): Aggregated DataFrame and the single best row (by optimization_col).
    """
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
    column_config: Optional[Dict[str, str]] = None,
    expected_search_space: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    sort_metric_resolver: Optional[callable] = None,
) -> defaultdict:
    """Orchestrate the full analysis pipeline: validate, then for each (dataset, model) aggregate and rank.

    Config can come from pipeline_config (overrides) or from module-level config. When
    expected_search_space (or pipeline_config.expected_search_space) is set, grid is
    resolved via get_model_grid_from_search_space (supports both dataset/model and legacy
    model-only dict shape). Otherwise varying HPs are discovered from the data.

    Parameters
    ----------
    df : pd.DataFrame
        Flat DataFrame of runs (after load and optional preprocess).
    config_columns, dropped_columns : list
        Column lists from load_runs_df (used to determine config vs dropped).
    reporting_metrics, metric_prefixes : list
        Metric names and prefixes for performance columns.
    optimization_col, optimization_mode : str, optional
        Override for optimization metric; if None, inferred per dataset/model.
    required_data_seeds, required_model_seeds : set, optional
        If set, validate_seeds filters to groups containing all seeds.
    data_seed_col, model_seed_col : str
        Column names for seed columns.
    additional_ignored_cols : list, optional
        Extra columns to ignore when detecting varying HPs.
    column_config, expected_search_space : dict, optional
        Override column mapping and search space; ignored if pipeline_config is set.
    pipeline_config : PipelineConfig, optional
        If set, supplies column_config, expected_search_space, filters, and sort_metric_resolver.
    sort_metric_resolver : callable, optional
        Resolver(dataset, model, task_type=None, dataset_category=None) -> metric name or None.
        If None, optimization metric is used for sorting/ranking.

    Returns
    -------
    all_results : defaultdict(lambda: defaultdict(dict))
        Nested structure: all_results[dataset][model] is a dict with keys:
        - best_hyperparameters : dict
            Best run's hyperparameters (from the row that achieved best on the
            optimization metric).
        - best_performance : dict
            Contains "metrics" (dict of metric_name -> prefix -> mean/std/count),
            "optimized_on" (dict with "metric", "value", etc.), and optionally
            "run_name" or "run_name_list".
        - sorted_aggregated_results : pd.DataFrame
            Aggregated stats per HP configuration (mean/std/count for each
            performance column), sorted by the optimization metric.
        - non_aggregated_df : pd.DataFrame
            Raw (non-aggregated) runs for this model/dataset subset.
        - seeds_info : dict
            Typically "data" and "model" keys with seed info (e.g. found seeds,
            required seeds) for display.
        - sort_metric_used : str, optional
            Metric used for sorting/ranking (if sort_metric_resolver provided).
            If None or not present, optimization metric is used.
    """
    # Resolve config source: PipelineConfig overrides or module-level config
    config_columns = [col for col in config_columns if col not in dropped_columns]
    if pipeline_config is not None:
        _col_cfg = pipeline_config.column_config
        _search_space = pipeline_config.expected_search_space
        _hp_filters = pipeline_config.column_filters_to_ignore_for_hp_search
        _special_cols = pipeline_config.special_columns
        _sort_metric_resolver = pipeline_config.sort_metric_resolver
    else:
        _col_cfg = column_config if column_config is not None else config.COLUMN_CONFIG
        _search_space = (
            expected_search_space
            if expected_search_space is not None
            else config.EXPECTED_SEARCH_SPACE
        )
        _hp_filters = config.COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH
        _special_cols = config.SPECIAL_COLUMNS
        _sort_metric_resolver = sort_metric_resolver
    
    unique_models = get_unique_values(df, _col_cfg["model_target"])
    unique_datasets = get_unique_values(df, _col_cfg["dataset_name"])
    all_results = defaultdict(lambda: defaultdict(dict))
    logger.info("Analyze: %d dataset(s), %d model(s) -> %d (dataset, model) combinations",
                len(unique_datasets), len(unique_models), len(unique_datasets) * len(unique_models))

    for dataset in unique_datasets:
        for model in unique_models:

            # --- SUBSETTING ---
            subset_start_shape = df.shape
            subset = df[
                (df[_col_cfg["dataset_name"]] == dataset)
                & (df[_col_cfg["model_target"]] == model)
            ].copy()

            if subset.empty:
                continue

            validators.log_modification(
                action="Subsetting DataFrame",
                context=f"Model='{model}' | Dataset='{dataset}'",
                reason="Filtering data to keep only runs matching the current target configuration.",
                input_shape=subset_start_shape,
                output_shape=subset.shape,
                level=logging.DEBUG,
            )
            subset["Model"] = model

            # --- 1. DYNAMIC METRIC DISCOVERY ---
            # Extract metrics for logging/info
            dynamic_metrics = get_dynamic_metrics(subset, _col_cfg)
            logger.debug("(%s | %s) dynamic_metrics: %s", dataset, model, dynamic_metrics)

            # --- 2. OPTIMIZATION METRIC ---
            if optimization_col is None:
                try:
                    from lmetk_report.core.analysis import get_optimized_metric

                    cur_optimization_col = get_optimized_metric(
                        subset, model, dataset, _col_cfg
                    )
                except Exception as e:
                    logger.warning(f"Warning: {e}. Fallback to test/mse")
                    cur_optimization_col = "test/mse"
            else:
                cur_optimization_col = optimization_col

            # --- 3. OPTIMIZATION MODE ---
            cur_optimization_mode = optimization_mode
            if cur_optimization_mode is None:
                target_mode_col = _col_cfg["target_metric_mode"]
                if target_mode_col in subset.columns:
                    modes = subset[target_mode_col].dropna().unique()
                    if len(modes) > 0:
                        cur_optimization_mode = modes[0]

                if cur_optimization_mode is None:
                    es_mode_col = _col_cfg["optimization_mode"]
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
            logger.debug("(%s | %s) all_performance_cols: %s", dataset, model, all_performance_cols[:10] if len(all_performance_cols) > 10 else all_performance_cols)

            # --- 5. HP & VALIDATION ---
            hp_search_columns = [
                col
                for col in config_columns
                if not any(f in col for f in _hp_filters)
            ]
            varying_hyperparams = get_varying_hyperparameters(
                subset, hp_search_columns, _special_cols
            )
            aggregation_cols = ["Model"] + list(varying_hyperparams.keys())
            logger.debug("(%s | %s) varying_hyperparams: %s", dataset, model, list(varying_hyperparams.keys()))

            subset = validators.validate_seeds(
                subset, aggregation_cols, data_seed_col, required_data_seeds, False
            )

            # --- Resolve grid early (schema-first) for stats: x = expected configs, y = with all seeds before metric filter, z = after ---
            model_grid = None
            grid_df = None
            join_keys = None
            x_expected = None
            try:
                from lmetk_report.configs.search_space import get_search_space
                model_grid = get_search_space(dataset, model)
            except ImportError:
                pass
            if model_grid is None:
                model_grid = get_model_grid_from_search_space(dataset, model, _search_space)
            if model_grid:
                grid_df = get_search_grid_df(model_grid)
                grid_df = grid_df.copy()
                grid_df["Model"] = model
                join_keys = ["Model"] + list(model_grid.keys())
                x_expected = len(grid_df)
                logger.debug("(%s | %s) schema-first: x_expected=%s", dataset, model, x_expected)

            # --- Metric-validity filter: drop rows where sort metric is missing or not a valid float ---
            seed_context = f"dataset={dataset}, model={model}"
            configs_failed_missing_invalid_metric = 0
            y_grid_full_seed_before_metric = None  # only set when schema-first and seeds required
            if x_expected is not None and required_model_seeds and model_seed_col in subset.columns:
                y_grid_full_seed_before_metric = _count_grid_configs_with_all_seeds(
                    subset, grid_df, join_keys, model_seed_col, required_model_seeds
                )
                logger.info(
                    "  [Schema-first stats] %s | x=%s, y (grid configs with all seeds before metric filter)=%s",
                    seed_context, x_expected, y_grid_full_seed_before_metric,
                )
            if cur_optimization_col not in subset.columns and not subset.empty:
                logger.warning(
                    "[Metric filter] %s | sort metric column %r not in subset; skipping (configs_failed_missing_invalid_metric=0).",
                    seed_context, cur_optimization_col,
                )
            else:
                def _count_configs_with_all_seeds(subs):
                    if subs.empty or not required_model_seeds or model_seed_col not in subs.columns:
                        return 0
                    try:
                        temp = subs.copy()
                        for c in aggregation_cols:
                            if c in temp.columns:
                                temp[c] = temp[c].astype(str)
                        def has_all(g):
                            try:
                                current = set(g[model_seed_col].dropna().unique())
                                current_int = {int(float(x)) for x in current}
                                return required_model_seeds.issubset(current_int)
                            except Exception:
                                return False
                        return temp.groupby(aggregation_cols).filter(has_all).groupby(aggregation_cols).ngroups
                    except Exception:
                        return 0

                subset = validators.filter_rows_with_valid_sort_metric(
                    subset, cur_optimization_col, context=seed_context,
                )
                # Configs Failed (missing/invalid metric): schema-first x-z, else data-driven
                if x_expected is not None and required_model_seeds and model_seed_col in subset.columns:
                    z_grid_full_seed_after_metric = _count_grid_configs_with_all_seeds(
                        subset, grid_df, join_keys, model_seed_col, required_model_seeds
                    )
                    configs_failed_missing_invalid_metric = x_expected - z_grid_full_seed_after_metric
                    logger.info(
                        "  [Metric filter] %s | sort_metric_col=%s, z (grid configs with all seeds after filter)=%s, configs_failed_missing_invalid_metric=x-z=%s",
                        seed_context, cur_optimization_col, z_grid_full_seed_after_metric, configs_failed_missing_invalid_metric,
                    )
                elif required_model_seeds and model_seed_col in subset.columns:
                    n_distinct_configs_before_metric_filter = (
                        subset.groupby(aggregation_cols).ngroups if not subset.empty else 0
                    )
                    n_configs_full_seed_after_metric_filter = _count_configs_with_all_seeds(subset)
                    configs_failed_missing_invalid_metric = (
                        n_distinct_configs_before_metric_filter - n_configs_full_seed_after_metric_filter
                    )
                    logger.info(
                        "  [Metric filter] %s | (data-first) configs with all seeds after filter=%s, configs_failed_missing_invalid_metric=%s",
                        seed_context, n_configs_full_seed_after_metric_filter, configs_failed_missing_invalid_metric,
                    )

            n_configs_before_model_seed = (
                subset.groupby(aggregation_cols).ngroups if not subset.empty else 0
            )
            subset = validators.validate_seeds(
                subset, aggregation_cols, model_seed_col, required_model_seeds, True,
                context=seed_context,
            )
            n_configs_after_model_seed = (
                subset.groupby(aggregation_cols).ngroups if not subset.empty else 0
            )
            # Schema-first: x - y (y = grid configs with all seeds before metric filter). Data-first: data-driven.
            if y_grid_full_seed_before_metric is not None and x_expected is not None:
                configs_failed_not_full_seed_set = x_expected - y_grid_full_seed_before_metric
                logger.info(
                    "  [Schema-first stats] %s | configs_failed_not_full_seed_set=x-y=%s",
                    seed_context, configs_failed_not_full_seed_set,
                )
            else:
                configs_failed_not_full_seed_set = (
                    n_configs_before_model_seed - n_configs_after_model_seed
                )
            total_runs = len(subset)

            if subset.empty:
                continue

            # Updated ignore_vars to exclude val_best_rerun/ metrics from warnings
            ignore_vars = _hp_filters + [
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

            # --- PER-MODEL GRID: reuse if already resolved early (schema-first stats), else lookup ---
            grid_from_configs = False
            if model_grid is None:
                try:
                    from lmetk_report.configs.search_space import get_search_space
                    model_grid = get_search_space(dataset, model)
                    grid_from_configs = model_grid is not None
                except ImportError:
                    pass
                if model_grid is None:
                    model_grid = get_model_grid_from_search_space(dataset, model, _search_space)
                if model_grid:
                    grid_df = get_search_grid_df(model_grid)
                    grid_df = grid_df.copy()
                    grid_df["Model"] = model
                    join_keys = ["Model"] + list(model_grid.keys())
            grid_source = "configs" if grid_from_configs else ("legacy" if model_grid else "auto")
            logger.debug("(%s | %s) model_grid source=%s, keys=%s", dataset, model, grid_source, list(model_grid.keys()) if model_grid else None)

            # Flatten MultiIndex regardless of grid presence for downstream consistency
            actual_results = flatten_aggregated_columns(sorted_aggregated)
            opt_col_flattened = f"{cur_optimization_col}_mean"

            # Skip if no metric column to sort by (e.g. zero matching performance cols)
            if opt_col_flattened not in actual_results.columns:
                continue

            if model_grid:
                logger.debug("(%s | %s) Mode: SCHEMA-FIRST (merging with expected grid)", dataset, model)
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
                logger.debug("(%s | %s) Mode: DATA-FIRST, varying HPs: %s", dataset, model, list(varying_hyperparams.keys()))
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
            res_entry["total_runs"] = total_runs
            res_entry["configs_failed_not_full_seed_set"] = configs_failed_not_full_seed_set
            res_entry["configs_failed_missing_invalid_metric"] = configs_failed_missing_invalid_metric

            # --- 8. RESOLVE SORT METRIC (if resolver provided) ---
            # Use resolver from pipeline_config if available, otherwise use parameter
            # Note: _sort_metric_resolver is already set from either pipeline_config or parameter at lines 303/313
            active_resolver = _sort_metric_resolver
            logger.debug(
                f"Resolving sort metric for {model} on {dataset}: "
                f"pipeline_config={'present' if pipeline_config is not None else 'None'}, "
                f"_sort_metric_resolver={'present' if _sort_metric_resolver is not None else 'None'}, "
                f"sort_metric_resolver param={'present' if sort_metric_resolver is not None else 'None'}"
            )
            if active_resolver is not None:
                try:
                    # Try to infer task type and dataset category from data
                    task_type = None
                    dataset_category = None
                    
                    # Attempt to infer task type from column config
                    if _col_cfg.get("target_metric") in subset.columns:
                        # Could extract task type from other columns if available
                        pass
                    
                    # Call resolver to get sort metric
                    resolved_sort_metric = active_resolver(
                        dataset, model, task_type=task_type, dataset_category=dataset_category
                    )
                    res_entry["sort_metric_used"] = resolved_sort_metric
                    logger.debug("(%s | %s) sort_metric resolved: %s", dataset, model, resolved_sort_metric)
                except Exception as e:
                    logger.warning(
                        "Failed to resolve sort metric for %s on %s: %s. Using optimization metric.",
                        model, dataset, e,
                    )
                    res_entry["sort_metric_used"] = None
            else:
                res_entry["sort_metric_used"] = None

            logger.info(
                "  %s | %s: opt=%s mode=%s grid=%s sort=%s runs=%d configs_failed_seed=%s configs_failed_invalid_metric=%s",
                dataset, model, cur_optimization_col, cur_optimization_mode, grid_source,
                res_entry.get("sort_metric_used") or cur_optimization_col,
                total_runs, configs_failed_not_full_seed_set, configs_failed_missing_invalid_metric,
            )

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
