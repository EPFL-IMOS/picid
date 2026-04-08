# picid_report/core/analysis.py
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
import xarray as xr

from picid_report import config
from picid_report.config import PipelineConfig
from picid_report.core import validators

# Import search space helper: one function handles both legacy and new structure
try:
    from picid_report.configs.search_space import get_model_grid_from_search_space
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
    if (
        subset.empty
        or grid_df.empty
        or not required_model_seeds
        or model_seed_col not in subset.columns
    ):
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


def _grid_configs_with_all_seeds(
    subset: pd.DataFrame,
    grid_df: pd.DataFrame,
    join_keys: List[str],
    model_seed_col: str,
    required_model_seeds: Set[int],
) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame of grid rows (join_keys columns) that have all required_model_seeds in subset.
    Used in schema-first mode to list which grid configs passed; complement gives failed configs.
    """
    if (
        subset.empty
        or grid_df.empty
        or not required_model_seeds
        or model_seed_col not in subset.columns
    ):
        return None
    join_keys = [k for k in join_keys if k in subset.columns and k in grid_df.columns]
    if not join_keys:
        return None
    try:
        sub = subset.copy()
        gr = grid_df.copy()
        for k in join_keys:
            sub[k] = sub[k].astype(str)
            gr[k] = gr[k].astype(str)
        grouped = sub.groupby(join_keys)[model_seed_col].apply(
            lambda s: {int(float(x)) for x in s.dropna().unique()}
        )
        req_int = {int(x) for x in required_model_seeds}
        rows = []
        for _, row in gr.iterrows():
            key = tuple(row[k] for k in join_keys)
            if key in grouped.index and req_int.issubset(grouped[key]):
                rows.append(list(key))
        if not rows:
            return pd.DataFrame(columns=join_keys)
        return pd.DataFrame(rows, columns=join_keys)
    except Exception:
        return None


def _group_keys_with_all_seeds(
    df: pd.DataFrame,
    group_cols: List[str],
    seed_col: str,
    required_seeds: Set[int],
) -> Set[Tuple]:
    """Return set of group keys (tuples) for groups that have all required_seeds in seed_col."""
    if not required_seeds or seed_col not in df.columns or df.empty:
        return set()
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return set()
    try:
        temp = df.copy()
        for c in group_cols:
            temp[c] = temp[c].astype(str)

        def has_all(g):
            try:
                current = {int(float(x)) for x in g[seed_col].dropna().unique()}
                return required_seeds.issubset(current)
            except Exception:
                return False

        valid = temp.groupby(group_cols).filter(has_all)
        if valid.empty:
            return set()
        keys = valid.groupby(group_cols).groups.keys()
        return set((k,) if not isinstance(k, tuple) else k for k in keys)
    except Exception:
        return set()


def _configs_table_to_log_lines(
    df: Optional[pd.DataFrame], max_cell_width: int = 50
) -> List[str]:
    """Format a small config table (HP columns) as lines for log.txt; truncate long values."""
    if df is None or df.empty:
        return ["    (none)"]
    cols = [c for c in df.columns]
    if not cols:
        return ["    (no columns)"]
    out = []
    for _, row in df.iterrows():
        parts = []
        for c in cols:
            v = row[c]
            s = str(v)[:max_cell_width] + (
                "..." if len(str(v)) > max_cell_width else ""
            )
            parts.append(f"{c}={s}")
        line = "  ".join(parts)
        if len(line) > 140:
            line = line[:137] + "..."
        out.append("    " + line)
    return out


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


def _metric_varies(subset: pd.DataFrame, col: str) -> bool:
    """True if the column has at least two distinct numeric values (so ranking by it is meaningful)."""
    try:
        s = pd.to_numeric(subset[col], errors="coerce").dropna()
        return s.nunique() >= 2
    except Exception:
        return False


def _pick_fallback_sort_metric(
    subset: pd.DataFrame, metric_prefixes: List[str]
) -> Optional[str]:
    """
    Find a metric column that has at least one valid numeric value. Prefer val* then test*.
    Among those, prefer columns that vary (nunique >= 2) so HP ranking is meaningful.
    Returns column name or None. Logs a warning if the chosen fallback does not vary.
    """
    candidates = []
    for col in subset.columns:
        if not isinstance(col, str):
            continue
        for p in metric_prefixes:
            if col.startswith(p):
                try:
                    s = pd.to_numeric(subset[col], errors="coerce")
                    if s.notna().any():
                        candidates.append(col)
                except Exception:
                    pass
                break
    if not candidates:
        return None

    # Prefer val* then test*, then by name
    def key(c):
        if c.startswith("val"):
            return (0, c)
        if c.startswith("test"):
            return (1, c)
        return (2, c)

    varying = [c for c in candidates if _metric_varies(subset, c)]
    non_varying = [c for c in candidates if c not in varying]
    varying.sort(key=key)
    non_varying.sort(key=key)
    ordered = varying + non_varying
    chosen = ordered[0]
    if chosen in non_varying:
        logger.warning(
            "[Sort metric] Fallback metric %r does not vary across rows; HP ranking may be arbitrary.",
            chosen,
        )
    return chosen


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
                    logger.info(
                        f"Optimized metric (from task def) for {model_name}: {c}"
                    )
                    return c
            if metric_base in df_subset.columns:
                return metric_base

    # Priority 2: Early Stopping
    metric_series = get_unique_values(df_subset, col_cfg["optimization_metric"])
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


def _ds_is_empty(ds: xr.Dataset) -> bool:
    """Return True when ds carries no data.

    Handles two shapes:
    - Full Dataset  (dims include "config"): empty when config size is 0.
    - Best-config Dataset (isel(config=0), no "config" dim): non-empty when data vars exist.
    - Empty sentinel xr.Dataset(): no data vars → empty.
    """
    if not isinstance(ds, xr.Dataset):
        return True
    if len(ds.data_vars) == 0:
        return True
    # Full dataset: check config count via sizes (avoids FutureWarning from ds.dims)
    if "config" in ds.sizes:
        return ds.sizes["config"] == 0
    # Best-config dataset (config dropped by isel): non-empty if it has data vars
    return False


def _ds_to_wide_df(ds: xr.Dataset) -> pd.DataFrame:
    """Convert an aggregated xr.Dataset to a flat wide DataFrame for schema-first merging."""
    hp_coords = [c for c in ds.coords if c not in ("config", "metric")]
    rows = []
    for i in range(ds.sizes["config"]):
        cfg = ds.isel(config=i)
        row = {hp: cfg.coords[hp].item() for hp in hp_coords}
        for metric in ds.coords["metric"].values:
            m = cfg.sel(metric=metric)
            row[f"{metric}__mean"]  = float(m["mean"].values)
            row[f"{metric}__std"]   = float(m["std"].values)
            row[f"{metric}__count"] = float(m["count"].values)
        rows.append(row)
    return pd.DataFrame(rows)


def _wide_df_to_ds(
    df: pd.DataFrame,
    aggregation_cols: List[str],
    perf_cols: List[str],
    opt_col: str,
    opt_mode: str,
) -> xr.Dataset:
    """Rebuild an xr.Dataset from a wide merged DataFrame (schema-first path only)."""
    hp_coords = {
        col: ("config", df[col].tolist())
        for col in aggregation_cols
        if col in df.columns
    }
    means  = np.array([[df[f"{m}__mean"].iloc[i]  for m in perf_cols] for i in range(len(df))], dtype=float)
    stds   = np.array([[df[f"{m}__std"].iloc[i]   for m in perf_cols] for i in range(len(df))], dtype=float)
    counts = np.array([[df[f"{m}__count"].iloc[i] for m in perf_cols] for i in range(len(df))], dtype=float)

    is_ascending = opt_mode == "min"
    opt_idx  = perf_cols.index(opt_col) if opt_col in perf_cols else 0
    sort_idx = np.argsort(means[:, opt_idx])
    if not is_ascending:
        sort_idx = sort_idx[::-1]

    sorted_coords = {
        k: (dim, [vals[i] for i in sort_idx])
        for k, (dim, vals) in hp_coords.items()
    }

    ds = xr.Dataset(
        {
            "mean":  xr.DataArray(means[sort_idx],  dims=["config", "metric"]),
            "std":   xr.DataArray(stds[sort_idx],   dims=["config", "metric"]),
            "count": xr.DataArray(counts[sort_idx], dims=["config", "metric"]),
        },
        coords={
            "config": np.arange(len(df)),
            "metric": perf_cols,
            **sorted_coords,
        },
    )
    ds.attrs["optimization_col"]  = opt_col
    ds.attrs["optimization_mode"] = opt_mode
    return ds


def aggregate_and_find_best(
    df_subset: pd.DataFrame,
    aggregation_cols: List[str],
    all_performance_cols: List[str],
    optimization_col: str,
    optimization_mode: str,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Aggregate runs by HP config and return an xr.Dataset.

    Dims:   config (int, 0 = best), metric (str)
    Vars:   mean, std, count  — shape (config, metric)
            run_names         — shape (config,), list of run-name strings per config
    Coords: metric, config, + one coord per HP column (dim=config)
    Attrs:  optimization_col, optimization_mode

    Args:
        df_subset: Runs for one dataset/model.
        aggregation_cols: Columns to group by (HP names).
        all_performance_cols: Metric columns to aggregate.
        optimization_col: Metric column used to rank configs.
        optimization_mode: "min" or "max".

    Returns:
        (sorted_ds, best_ds): Dataset of all configs sorted best-first, and isel(config=0).
    """
    _EMPTY = xr.Dataset()

    subset_copy = df_subset.copy()

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

    perf_cols = [
        col for col in all_performance_cols
        if col in subset_copy.columns
        and pd.api.types.is_numeric_dtype(subset_copy[col])
    ]
    if not perf_cols:
        return _EMPTY, _EMPTY

    grouped = subset_copy.groupby(aggregation_cols)
    means  = grouped[perf_cols].mean()
    stds   = grouped[perf_cols].std()
    counts = grouped[perf_cols].count()

    # Fallback if the requested opt col wasn't in the numeric set
    if optimization_col not in means.columns:
        optimization_col = perf_cols[0]

    is_ascending = optimization_mode == "min"
    sort_idx = np.argsort(means[optimization_col].values)
    if not is_ascending:
        sort_idx = sort_idx[::-1]

    means  = means.iloc[sort_idx]
    stds   = stds.iloc[sort_idx]
    counts = counts.iloc[sort_idx]

    # HP values as per-config coordinates.
    # Use dtype=object for all HP coords to safely handle tuples/lists (e.g. tags columns),
    # which numpy would otherwise unpack into a 2D array causing xarray coordinate errors.
    idx = means.index
    hp_coords: Dict[str, Any] = {}
    if isinstance(idx, pd.MultiIndex):
        for i, col in enumerate(aggregation_cols):
            vals = np.empty(len(idx), dtype=object)
            for j, v in enumerate(idx.get_level_values(i).tolist()):
                vals[j] = v
            hp_coords[col] = ("config", vals)
    elif len(aggregation_cols) == 1:
        vals = np.empty(len(idx), dtype=object)
        for j, v in enumerate(idx.tolist()):
            vals[j] = v
        hp_coords[aggregation_cols[0]] = ("config", vals)

    n_configs = len(means)
    ds = xr.Dataset(
        {
            "mean":  xr.DataArray(means.values.astype(float),  dims=["config", "metric"]),
            "std":   xr.DataArray(stds.values.astype(float),   dims=["config", "metric"]),
            "count": xr.DataArray(counts.values.astype(float), dims=["config", "metric"]),
        },
        coords={
            "config": np.arange(n_configs),
            "metric": perf_cols,
            **hp_coords,
        },
    )

    if "run_name" in subset_copy.columns:
        run_name_lists = grouped["run_name"].apply(list).iloc[sort_idx].tolist()
        # Use object dtype so numpy doesn't unpack equal-length inner lists into a 2D array
        run_names_arr = np.empty(n_configs, dtype=object)
        for _i, _lst in enumerate(run_name_lists):
            run_names_arr[_i] = _lst
        ds["run_names"] = xr.DataArray(run_names_arr, dims=["config"])

    ds.attrs["optimization_col"]  = optimization_col
    ds.attrs["optimization_mode"] = optimization_mode

    return ds, ds.isel(config=0)


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
    use_legacy_search_space_fallback: bool = True,
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
    use_legacy_search_space_fallback : bool, default True
        When True, get_search_space(..., use_default=True) so the default-for-all-datasets grid
        (search_space.DEFAULT_SEARCH_SPACE or config.EXPECTED_SEARCH_SPACE) is tried when no
        per-dataset entry exists. Set to False (e.g. --data-first) to use only per-dataset grid
        or data-first when search_space.py has no per-dataset entry.

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
        # Only use caller-provided search space; default/model-only grid is resolved inside get_search_space (search_space.py)
        _search_space = expected_search_space
        _hp_filters = config.COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH
        _special_cols = config.SPECIAL_COLUMNS
        _sort_metric_resolver = sort_metric_resolver

    unique_models = get_unique_values(df, _col_cfg["model_target"])
    unique_datasets = get_unique_values(df, _col_cfg["dataset_name"])
    all_results = defaultdict(lambda: defaultdict(dict))
    logger.info(
        "Analyze: %d dataset(s), %d model(s) -> %d (dataset, model) combinations",
        len(unique_datasets),
        len(unique_models),
        len(unique_datasets) * len(unique_models),
    )

    for dataset in unique_datasets:
        for model in unique_models:
            _used_fallback_sort_metric = False
            _original_sort_metric = None
            failed_metric_configs_df = None
            failed_not_full_seed_configs_df = None
            failed_not_full_seed_grid_df = None  # schema-first: grid rows missing seeds
            failed_metric_grid_df = (
                None  # schema-first: grid rows that lost runs after metric filter
            )
            grid_ok_before = grid_ok_after = (
                None  # schema-first: grid configs with all seeds (before/after metric filter)
            )

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
            logger.debug(
                "(%s | %s) dynamic_metrics: %s", dataset, model, dynamic_metrics
            )

            # --- 2. OPTIMIZATION METRIC ---
            if optimization_col is None:
                try:
                    from picid_report.core.analysis import get_optimized_metric

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
                    # Infer from metric name: higher-is-better (accuracy, f1, etc.) -> "max", else "min"
                    col_lower = (cur_optimization_col or "").lower()
                    if any(
                        h in col_lower
                        for h in ("accuracy", "f1", "f1_", "auc", "precision", "recall")
                    ):
                        cur_optimization_mode = "max"
                    else:
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
            logger.debug(
                "(%s | %s) all_performance_cols: %s",
                dataset,
                model,
                all_performance_cols[:10]
                if len(all_performance_cols) > 10
                else all_performance_cols,
            )

            # --- 5. HP & VALIDATION ---
            hp_search_columns = [
                col for col in config_columns if not any(f in col for f in _hp_filters)
            ]
            varying_hyperparams = get_varying_hyperparameters(
                subset, hp_search_columns, _special_cols
            )
            aggregation_cols = ["Model"] + list(varying_hyperparams.keys())
            logger.debug(
                "(%s | %s) varying_hyperparams: %s",
                dataset,
                model,
                list(varying_hyperparams.keys()),
            )

            subset = validators.validate_seeds(
                subset, aggregation_cols, data_seed_col, required_data_seeds, False
            )

            # --- Resolve grid early (schema-first) for stats: x = expected configs, y = with all seeds before metric filter, z = after ---
            model_grid = None
            grid_df = None
            join_keys = None
            x_expected = None
            try:
                from picid_report.configs.search_space import get_search_space

                model_grid = get_search_space(
                    dataset, model, use_default=use_legacy_search_space_fallback
                )
            except ImportError:
                pass
            if model_grid is None and _search_space is not None:
                model_grid = get_model_grid_from_search_space(
                    dataset, model, _search_space
                )
            if model_grid:
                grid_df = get_search_grid_df(model_grid)
                grid_df = grid_df.copy()
                grid_df["Model"] = model
                join_keys = ["Model"] + list(model_grid.keys())
                x_expected = len(grid_df)
                logger.debug(
                    "(%s | %s) schema-first: x_expected=%s", dataset, model, x_expected
                )

            # --- Metric-validity filter: drop rows where sort metric is missing or not a valid float ---
            seed_context = f"dataset={dataset}, model={model}"
            configs_failed_missing_invalid_metric = 0
            y_grid_full_seed_before_metric = (
                None  # only set when schema-first and seeds required
            )
            z_grid_full_seed_after_metric = None  # only set when schema-first (grid configs with all seeds after metric filter)
            if (
                x_expected is not None
                and required_model_seeds
                and model_seed_col in subset.columns
            ):
                y_grid_full_seed_before_metric = _count_grid_configs_with_all_seeds(
                    subset, grid_df, join_keys, model_seed_col, required_model_seeds
                )
                logger.info(
                    "  [Schema-first stats] %s | x=%s, y (grid configs with all seeds before metric filter)=%s",
                    seed_context,
                    x_expected,
                    y_grid_full_seed_before_metric,
                )
            # If original sort metric would drop all rows, try fallback (prefer val*, then test*)
            would_drop_all = (
                cur_optimization_col not in subset.columns
                or not pd.to_numeric(subset[cur_optimization_col], errors="coerce")
                .notna()
                .any()
            )
            if would_drop_all and not subset.empty:
                logger.info(
                    "[Sort metric] %s | Original metric %r not found or all invalid (would drop all rows). Searching for fallback.",
                    seed_context,
                    cur_optimization_col,
                )
                fallback = _pick_fallback_sort_metric(subset, metric_prefixes)
                if fallback:
                    _original_sort_metric = cur_optimization_col
                    cur_optimization_col = fallback
                    _used_fallback_sort_metric = True
                    if cur_optimization_col not in all_performance_cols:
                        all_performance_cols.append(cur_optimization_col)
                    logger.info(
                        "[Sort metric] %s | Using fallback metric: %r",
                        seed_context,
                        cur_optimization_col,
                    )
                else:
                    logger.warning(
                        "[Sort metric] %s | No fallback metric found; model will be skipped.",
                        seed_context,
                    )
            if cur_optimization_col not in subset.columns and not subset.empty:
                logger.warning(
                    "[Metric filter] %s | sort metric column %r not in subset; skipping (configs_failed_missing_invalid_metric=0).",
                    seed_context,
                    cur_optimization_col,
                )
            else:

                def _count_configs_with_all_seeds(subs):
                    if (
                        subs.empty
                        or not required_model_seeds
                        or model_seed_col not in subs.columns
                    ):
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

                        return (
                            temp.groupby(aggregation_cols)
                            .filter(has_all)
                            .groupby(aggregation_cols)
                            .ngroups
                        )
                    except Exception:
                        return 0

                # Capture which configs have all seeds before metric filter (for failed-metric table)
                subset_before_metric = subset.copy()
                groups_before_metric = (
                    _group_keys_with_all_seeds(
                        subset_before_metric,
                        aggregation_cols,
                        model_seed_col,
                        required_model_seeds,
                    )
                    if (required_model_seeds and model_seed_col in subset.columns)
                    else set()
                )

                subset = validators.filter_rows_with_valid_sort_metric(
                    subset,
                    cur_optimization_col,
                    context=seed_context,
                )

                groups_after_metric = (
                    _group_keys_with_all_seeds(
                        subset, aggregation_cols, model_seed_col, required_model_seeds
                    )
                    if (required_model_seeds and model_seed_col in subset.columns)
                    else set()
                )
                failed_metric_keys = groups_before_metric - groups_after_metric
                failed_metric_configs_df = None  # one row per config that had all seeds before metric filter but not after
                if failed_metric_keys and not subset_before_metric.empty:
                    try:
                        ac = [
                            c
                            for c in aggregation_cols
                            if c in subset_before_metric.columns
                        ]
                        if ac:

                            def _key_in_failed(g):
                                k = (
                                    (g.name,)
                                    if not isinstance(g.name, tuple)
                                    else g.name
                                )
                                return k in failed_metric_keys

                            failed_metric_configs_df = (
                                subset_before_metric.groupby(ac)
                                .filter(_key_in_failed)
                                .drop_duplicates(ac)[ac]
                            )
                    except Exception:
                        pass

                # Configs Failed (missing/invalid metric): schema-first x-z, else data-driven
                if (
                    x_expected is not None
                    and required_model_seeds
                    and model_seed_col in subset.columns
                ):
                    z_grid_full_seed_after_metric = _count_grid_configs_with_all_seeds(
                        subset, grid_df, join_keys, model_seed_col, required_model_seeds
                    )
                    configs_failed_missing_invalid_metric = (
                        x_expected - z_grid_full_seed_after_metric
                    )
                    logger.info(
                        "  [Metric filter] %s | sort_metric_col=%s, z (grid configs with all seeds after filter)=%s, configs_failed_missing_invalid_metric=x-z=%s",
                        seed_context,
                        cur_optimization_col,
                        z_grid_full_seed_after_metric,
                        configs_failed_missing_invalid_metric,
                    )
                    # Schema-first: list which grid configs failed (for log.txt); only when data has grid columns
                    if grid_df is not None and all(
                        k in subset_before_metric.columns for k in join_keys
                    ):
                        grid_ok_before = _grid_configs_with_all_seeds(
                            subset_before_metric,
                            grid_df,
                            join_keys,
                            model_seed_col,
                            required_model_seeds,
                        )
                        grid_ok_after = _grid_configs_with_all_seeds(
                            subset,
                            grid_df,
                            join_keys,
                            model_seed_col,
                            required_model_seeds,
                        )
                    else:
                        grid_ok_before = grid_ok_after = None
                    if grid_df is not None and grid_ok_before is not None:
                        jk = [k for k in join_keys if k in grid_df.columns]
                        if jk:
                            ok_keys_before = (
                                set(
                                    tuple(row[k] for k in jk)
                                    for _, row in grid_ok_before.iterrows()
                                )
                                if grid_ok_before is not None
                                and not grid_ok_before.empty
                                else set()
                            )
                            # Grid rows that do NOT have all seeds (before metric filter)
                            mask_not_ok = grid_df[jk].apply(
                                lambda r: tuple(r[k] for k in jk) not in ok_keys_before,
                                axis=1,
                            )
                            failed_not_full_seed_grid_df = grid_df.loc[
                                mask_not_ok, jk
                            ].drop_duplicates()
                            ok_keys_after = (
                                set(
                                    tuple(row[k] for k in jk)
                                    for _, row in grid_ok_after.iterrows()
                                )
                                if grid_ok_after is not None and not grid_ok_after.empty
                                else set()
                            )
                            # Grid rows that had all seeds before metric but lost them after
                            if grid_ok_before is not None and not grid_ok_before.empty:
                                mask_lost = grid_ok_before.apply(
                                    lambda r: tuple(r[k] for k in jk)
                                    not in ok_keys_after,
                                    axis=1,
                                )
                                failed_metric_grid_df = grid_ok_before.loc[
                                    mask_lost, jk
                                ].copy()
                            else:
                                failed_metric_grid_df = None
                elif required_model_seeds and model_seed_col in subset.columns:
                    n_distinct_configs_before_metric_filter = (
                        subset.groupby(aggregation_cols).ngroups
                        if not subset.empty
                        else 0
                    )
                    n_configs_full_seed_after_metric_filter = (
                        _count_configs_with_all_seeds(subset)
                    )
                    configs_failed_missing_invalid_metric = (
                        n_distinct_configs_before_metric_filter
                        - n_configs_full_seed_after_metric_filter
                    )
                    logger.info(
                        "  [Metric filter] %s | (data-first) configs with all seeds after filter=%s, configs_failed_missing_invalid_metric=%s",
                        seed_context,
                        n_configs_full_seed_after_metric_filter,
                        configs_failed_missing_invalid_metric,
                    )

            n_configs_before_model_seed = (
                subset.groupby(aggregation_cols).ngroups if not subset.empty else 0
            )
            # Capture configs that will be dropped by seed validation (for log table)
            failed_not_full_seed_configs_df = None
            if (
                required_model_seeds
                and model_seed_col in subset.columns
                and not subset.empty
            ):
                try:
                    ac = [c for c in aggregation_cols if c in subset.columns]

                    def _missing_seeds(g):
                        try:
                            current = {
                                int(float(x))
                                for x in g[model_seed_col].dropna().unique()
                            }
                            return not required_model_seeds.issubset(current)
                        except Exception:
                            return True

                    if ac:
                        failed_not_full_seed_configs_df = (
                            subset.groupby(ac)
                            .filter(_missing_seeds)
                            .drop_duplicates(ac)[ac]
                        )
                except Exception:
                    pass

            subset = validators.validate_seeds(
                subset,
                aggregation_cols,
                model_seed_col,
                required_model_seeds,
                True,
                context=seed_context,
            )
            n_configs_after_model_seed = (
                subset.groupby(aggregation_cols).ngroups if not subset.empty else 0
            )
            # Schema-first: x - y (y = grid configs with all seeds before metric filter). Data-first: data-driven.
            if y_grid_full_seed_before_metric is not None and x_expected is not None:
                configs_failed_not_full_seed_set = (
                    x_expected - y_grid_full_seed_before_metric
                )
                logger.info(
                    "  [Schema-first stats] %s | configs_failed_not_full_seed_set=x-y=%s",
                    seed_context,
                    configs_failed_not_full_seed_set,
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
                    from picid_report.configs.search_space import get_search_space

                    model_grid = get_search_space(
                        dataset, model, use_default=use_legacy_search_space_fallback
                    )
                    grid_from_configs = model_grid is not None
                except ImportError:
                    pass
                if model_grid is None and _search_space is not None:
                    model_grid = get_model_grid_from_search_space(
                        dataset, model, _search_space
                    )
                if model_grid:
                    grid_df = get_search_grid_df(model_grid)
                    grid_df = grid_df.copy()
                    grid_df["Model"] = model
                    join_keys = ["Model"] + list(model_grid.keys())
            grid_source = (
                "configs" if grid_from_configs else ("legacy" if model_grid else "auto")
            )
            logger.debug(
                "(%s | %s) model_grid source=%s, keys=%s",
                dataset,
                model,
                grid_source,
                list(model_grid.keys()) if model_grid else None,
            )

            # sorted_aggregated is now an xr.Dataset (from aggregate_and_find_best)
            opt_col = sorted_aggregated.attrs.get("optimization_col", cur_optimization_col)

            # Skip if aggregation produced no results or opt metric is missing
            if _ds_is_empty(sorted_aggregated):
                continue
            if opt_col not in sorted_aggregated.coords["metric"].values:
                continue

            if model_grid:
                grid_df = get_search_grid_df(model_grid)
                grid_df["Model"] = model
                join_keys = ["Model"] + list(model_grid.keys())
                hp_coord_names = [
                    c for c in sorted_aggregated.coords if c not in ("config", "metric")
                ]
                missing_in_actual = [k for k in join_keys if k not in hp_coord_names]

                if missing_in_actual:
                    logger.debug(
                        "(%s | %s) Schema-first skipped: actual results missing grid keys %s; using DATA-FIRST",
                        dataset,
                        model,
                        missing_in_actual,
                    )
                    # sorted_aggregated already sorted — nothing to do
                else:
                    logger.debug(
                        "(%s | %s) Mode: SCHEMA-FIRST (merging with expected grid)",
                        dataset,
                        model,
                    )
                    flat = _ds_to_wide_df(sorted_aggregated)
                    grid_df_merged  = grid_df.copy()
                    flat_merged     = flat.copy()
                    for k in join_keys:
                        if k in grid_df_merged.columns and k in flat_merged.columns:
                            grid_df_merged[k] = grid_df_merged[k].astype(str)
                            flat_merged[k]    = flat_merged[k].astype(str)
                    merged_df = pd.merge(grid_df_merged, flat_merged, on=join_keys, how="left")
                    for k in list(model_grid.keys()):
                        merged_df[k] = grid_df[k].values
                    sorted_aggregated = _wide_df_to_ds(
                        merged_df,
                        aggregation_cols,
                        perf_cols=list(sorted_aggregated.coords["metric"].values),
                        opt_col=opt_col,
                        opt_mode=cur_optimization_mode,
                    )
                    best_result = sorted_aggregated.isel(config=0) if not _ds_is_empty(sorted_aggregated) else xr.Dataset()
            else:
                logger.debug(
                    "(%s | %s) Mode: DATA-FIRST, varying HPs: %s",
                    dataset,
                    model,
                    list(varying_hyperparams.keys()),
                )
                # sorted_aggregated already sorted — nothing to do

            if _ds_is_empty(best_result):
                continue

            # --- 7. COLLECT RESULTS ---
            best_params = {
                hp: best_result.coords[hp].item()
                for hp in varying_hyperparams
                if hp in best_result.coords
            }
            if "run_names" in best_result:
                run_names_val = best_result["run_names"].item()
                best_params["run_names"] = run_names_val[0] if run_names_val else None

            performance_results = defaultdict(dict)
            mode = cur_optimization_mode

            # Sort prefixes by length (longest first) to prevent partial matching bugs
            sorted_prefixes = sorted(metric_prefixes, key=len, reverse=True)

            for col_full in all_performance_cols:
                if col_full not in best_result.coords["metric"].values:
                    continue

                matched_prefix = ""
                metric_name    = ""
                for p in sorted_prefixes:
                    if col_full.startswith(p):
                        matched_prefix = p.strip("/")
                        metric_name    = col_full[len(p):]
                        break
                if not matched_prefix:
                    continue

                m    = best_result.sel(metric=col_full)
                mean = float(m["mean"].values)
                std  = float(m["std"].values)
                _cnt = m["count"].values
                cnt  = int(_cnt) if not np.isnan(float(_cnt)) else 0

                # Scale if it's the target metric and we are maximizing
                if col_full == cur_optimization_col and mode == "max":
                    mean, std = mean * 100, std * 100

                performance_results[metric_name][matched_prefix] = {
                    "mean": mean,
                    "std":  std,
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
            res_entry["configs_failed_not_full_seed_set"] = (
                configs_failed_not_full_seed_set
            )
            res_entry["configs_failed_missing_invalid_metric"] = (
                configs_failed_missing_invalid_metric
            )

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
                        dataset,
                        model,
                        task_type=task_type,
                        dataset_category=dataset_category,
                    )
                    res_entry["sort_metric_used"] = resolved_sort_metric
                    logger.debug(
                        "(%s | %s) sort_metric resolved: %s",
                        dataset,
                        model,
                        resolved_sort_metric,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to resolve sort metric for %s on %s: %s. Using optimization metric.",
                        model,
                        dataset,
                        e,
                    )
                    res_entry["sort_metric_used"] = None
            else:
                res_entry["sort_metric_used"] = None

            if _used_fallback_sort_metric:
                res_entry["sort_metric_used"] = cur_optimization_col
                res_entry["sort_metric_is_fallback"] = True
                res_entry["original_sort_metric"] = _original_sort_metric

            logger.info(
                "  %s | %s: opt=%s mode=%s grid=%s sort=%s runs=%d configs_failed_seed=%s configs_failed_invalid_metric=%s",
                dataset,
                model,
                cur_optimization_col,
                cur_optimization_mode,
                grid_source,
                res_entry.get("sort_metric_used") or cur_optimization_col,
                total_runs,
                configs_failed_not_full_seed_set,
                configs_failed_missing_invalid_metric,
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

            # Build a human-readable diagnostics log for this (dataset, model) for log.txt
            configs_completed = len(res_entry.get("sorted_aggregated_results", []))
            req_data = (
                str(sorted(required_data_seeds))
                if required_data_seeds
                else "not required"
            )
            req_model = (
                str(sorted(required_model_seeds))
                if required_model_seeds
                else "not required"
            )
            seeds_data = res_entry["seeds_info"].get("data", "?")
            seeds_model = res_entry["seeds_info"].get("model", "?")
            diag = [
                "",
                "=" * 60,
                f"Dataset: {dataset}  |  Model: {model}",
                "=" * 60,
                f"  Optimization metric: {cur_optimization_col}  (mode: {cur_optimization_mode})",
                f"  Sort metric used:    {res_entry.get('sort_metric_used') or cur_optimization_col}",
                f"  Grid source:         {grid_source}",
                "",
                "  Required seeds:",
                f"    Data:  {req_data}",
                f"    Model: {req_model}",
                "  Seeds found in data:",
                f"    Data:  {seeds_data}",
                f"    Model: {seeds_model}",
                "",
                "  Counts:",
                f"    Configs completed (used for reporting): {configs_completed}",
                f"    Total runs (rows after filters):      {total_runs}",
                f"    Configs failed (not full seed set):   {configs_failed_not_full_seed_set}",
                f"    Configs failed (missing/invalid metric): {configs_failed_missing_invalid_metric}",
            ]
            if x_expected is not None:
                y_val = (
                    y_grid_full_seed_before_metric
                    if y_grid_full_seed_before_metric is not None
                    else "N/A"
                )
                z_val = (
                    z_grid_full_seed_after_metric
                    if z_grid_full_seed_after_metric is not None
                    else "N/A"
                )
                diag.extend(
                    [
                        "",
                        "  Schema-first grid:",
                        f"    Expected configs (grid size):           x = {x_expected}",
                        f"    Configs with all seeds (before metric): y = {y_val}",
                        f"    Configs with all seeds (after metric):  z = {z_val}",
                        "    (Not full seed set = x - y;  missing/invalid metric = x - z)",
                    ]
                )
            diag.extend(
                [
                    "",
                    "  What the failures mean:",
                    "    - 'Not full seed set': that many grid configs had at least one run but were missing",
                    "      one or more required model seeds (so the config was dropped before aggregation).",
                    "    - 'Missing/invalid metric': that many configs had all seeds but the sort metric",
                    "      was missing or invalid for some runs (or never reached z in schema-first).",
                    "",
                ]
            )

            # Completed configs: HP columns from the aggregated results used for reporting
            completed_ds = res_entry.get("sorted_aggregated_results")
            if not _ds_is_empty(completed_ds):
                hp_coord_names = [
                    c for c in completed_ds.coords if c not in ("config", "metric")
                ]
                diag_opt_col = completed_ds.attrs.get("optimization_col", cur_optimization_col)
                rows = []
                for i in range(completed_ds.sizes["config"]):
                    cfg = completed_ds.isel(config=i)
                    row = {hp: cfg.coords[hp].item() for hp in hp_coord_names}
                    if diag_opt_col in completed_ds.coords["metric"].values:
                        row[diag_opt_col] = float(cfg.sel(metric=diag_opt_col)["mean"].values)
                    rows.append(row)
                if rows:
                    completed_snapshot = pd.DataFrame(rows)
                    diag.extend(
                        [
                            "",
                            "  --- Completed configs (used for reporting) ---",
                            "  (Config parameters and sort metric for each row.)",
                            "",
                        ]
                        + _configs_table_to_log_lines(completed_snapshot)
                    )
                else:
                    diag.extend(
                        ["", "  --- Completed configs ---", "    (no HP columns)", ""]
                    )
            else:
                diag.extend(["", "  --- Completed configs ---", "    (none)", ""])

            # Failed configs: not full seed set (schema-first: grid configs; else: data configs)
            failed_not_full_seed_table = (
                failed_not_full_seed_grid_df
                if (
                    failed_not_full_seed_grid_df is not None
                    and not failed_not_full_seed_grid_df.empty
                )
                else failed_not_full_seed_configs_df
            )
            diag.extend(
                [
                    "",
                    "  --- Configs FAILED (not full seed set) ---",
                    "  (These configs had at least one run but were missing one or more required model seeds.)",
                    "",
                ]
                + _configs_table_to_log_lines(failed_not_full_seed_table)
            )

            # Failed configs: missing/invalid metric (schema-first: grid configs; else: data configs)
            failed_metric_table = (
                failed_metric_grid_df
                if (
                    failed_metric_grid_df is not None
                    and not failed_metric_grid_df.empty
                )
                else failed_metric_configs_df
            )
            diag.extend(
                [
                    "",
                    "  --- Configs FAILED (missing/invalid metric) ---",
                    "  (These configs had all seeds before the metric filter but lost runs after dropping invalid/missing metric rows.)",
                    "",
                ]
                + _configs_table_to_log_lines(failed_metric_table)
            )

            diag.append("")
            res_entry["diagnostics_log"] = diag

    return all_results
