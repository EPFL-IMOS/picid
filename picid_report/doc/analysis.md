# analysis.py — Core analysis

**Navigation:** [← Documentation index](README.md)

**Module:** `picid_report.core.analysis`

This document describes the **analysis** stage: turning a flat DataFrame of runs into a nested structure **all_results[dataset][model]** with best run, aggregated stats, and optional sort metric. It covers **every important function** with examples.

---

## 1. Overview and two modes

The analysis module:

- Splits the DataFrame by **dataset** and **model** (using column config).
- For each (dataset, model): discovers metrics, resolves optimization metric/mode, finds varying hyperparameters, validates seeds, aggregates runs (mean/std/count), resolves the HP grid (from configs or from data), picks the best run, and optionally resolves a sort metric.

**Two modes:**

- **Schema-first:** When a search space grid is provided (from `configs.search_space` or legacy config), results are **left-joined** to the grid so missing configs appear as rows with NaN; sorting uses the configured grid order.
- **Data-first:** When no grid is found, varying HPs are **discovered from the data** and only completed configs are shown.

---

## 2. analyze_results()

**Signature:**

```python
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
    expected_search_space: Optional[Dict] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    sort_metric_resolver: Optional[callable] = None,
) -> defaultdict:
```

**Returns:** `all_results` — `defaultdict(lambda: defaultdict(dict))` where `all_results[dataset][model]` contains:

- **best_hyperparameters** — dict of best run’s HP values (and optionally `run_names`).
- **best_performance** — `optimized_on` (metric, strategy), `metrics` (metric → prefix → mean/std/count).
- **sorted_aggregated_results** — DataFrame of aggregated stats per HP config, sorted by optimization metric (and optionally merged with grid in schema-first mode).
- **non_aggregated_df** — raw runs for this dataset/model.
- **seeds_info** — e.g. `{"data": "...", "model": "..."}` for display.
- **sort_metric_used** — metric used for ranking (if resolver provided); else None.
- **total_runs**, **configs_failed_not_full_seed_set** — counts.

Config can come from **pipeline_config** (overrides) or from module-level config. Grid is resolved via `get_search_space()` first, then `get_model_grid_from_search_space()` with legacy or new shape; if still None, varying HPs are discovered from data.

---

### Example 1: After load and preprocess (typical use from run_pipeline)

```python
from picid_report import load_runs_df, analyze_results, config
from picid_report.preprocess import clean_and_rename_models

df, config_columns, dropped_columns = load_runs_df(
    project_name="my_project", user="my_team", csv_cache_dir="csv_files"
)
df = clean_and_rename_models(df)

all_results = analyze_results(
    df=df,
    config_columns=config_columns,
    dropped_columns=dropped_columns,
    reporting_metrics=["loss", "mse", "mae", "rmse", "f1", "accuracy"],
    metric_prefixes=["val/", "test/", "test_best_rerun/", "val_best_rerun/"],
)
# all_results["UNIBO21"]["LSTM"]["best_hyperparameters"], ["sorted_aggregated_results"], etc.
```

---

### Example 2: With sort_metric_resolver (from configs)

When `run_pipeline` is used, it builds a resolver from `get_sort_metric` and passes it (or uses the one in `PipelineConfig`). You can pass one explicitly:

```python
from picid_report.configs import get_sort_metric, infer_task_type_from_dataset, infer_dataset_category_from_name

def my_resolver(dataset, model, task_type=None, dataset_category=None):
    if task_type is None:
        task_type = infer_task_type_from_dataset(dataset)
    if dataset_category is None:
        dataset_category = infer_dataset_category_from_name(dataset)
    return get_sort_metric(dataset, model, task_type=task_type, dataset_category=dataset_category)

all_results = analyze_results(
    df=df,
    config_columns=config_columns,
    dropped_columns=dropped_columns,
    reporting_metrics=["loss", "mse", "mae"],
    metric_prefixes=["val/", "test/", "val_best_rerun/"],
    sort_metric_resolver=my_resolver,
)
# Each all_results[dataset][model]["sort_metric_used"] will be set (e.g. "val_best_rerun/loss")
```

---

### Example 3: Schema-first vs data-first

- **Schema-first:** For (dataset, model) present in `configs.search_space.EXPECTED_SEARCH_SPACE` (or in legacy `config.EXPECTED_SEARCH_SPACE`), the grid is used; merged DataFrame has one row per grid point (missing = NaN).
- **Data-first:** For (dataset, model) with no grid, `get_varying_hyperparameters()` discovers varying columns; only configs that appear in the data are in `sorted_aggregated_results`.

No extra code needed — the module chooses automatically per (dataset, model).

---

## 3. Helper functions

### 3.1 get_search_grid_df(search_space)

Builds a DataFrame whose rows are the Cartesian product of HP values.

**Example:**

```python
from picid_report.analysis import get_search_grid_df

grid = get_search_grid_df({
    "task_definition.seq_len": [1, 10, 50],
    "optimization.lr": [0.001, 0.0001],
})
# 6 rows (3*2), columns: task_definition.seq_len, optimization.lr
```

---

### 3.2 get_unique_values(df, column)

Returns sorted unique values from a column; list cells are expanded so each element counts as one value.

**Example:**

```python
from picid_report.analysis import get_unique_values

vals = get_unique_values(df, "model._target_")
# e.g. ["LSTM", "Transformer"]
# If column missing, returns []
```

---

### 3.3 get_dynamic_metrics(df_subset, column_config=None)

Extracts metric names from the evaluator config column (e.g. `evaluator.train.metric_names`). Handles list/tuple/string values.

**Example:**

```python
from picid_report.analysis import get_dynamic_metrics

metrics = get_dynamic_metrics(subset_df)
# e.g. ["loss", "mse", "mae"]
```

---

### 3.4 get_optimized_metric(df_subset, model_name, dataset_name, column_config=None)

Resolves the metric used for optimization (best-run selection). Priority: (1) task_definition.target_metric, (2) early-stopping monitor, (3) fallbacks (val/loss, test/mse, …).

**Example:**

```python
from picid_report.analysis import get_optimized_metric

opt_col = get_optimized_metric(subset, "LSTM", "UNIBO21")
# e.g. "val_best_rerun/loss" or "test/mse"
# Raises ValueError if no valid metric column found
```

---

### 3.5 get_varying_hyperparameters(df_subset, config_cols, special_cols_to_exclude)

Identifies config columns that have more than one unique value. Used for data-first mode to build the HP grid from data.

**Example:**

```python
from picid_report.analysis import get_varying_hyperparameters

varying = get_varying_hyperparameters(
    subset,
    config_columns,
    special_cols_to_exclude=["seed", "datasource.parameters.data_seed"],
)
# e.g. {"task_definition.seq_len": [1, 10, 50], "optimization.lr": [0.001, 0.0001]}
```

---

### 3.6 aggregate_and_find_best(df_subset, aggregation_cols, all_performance_cols, optimization_col, optimization_mode)

Aggregates runs by HP config (mean/std/count per metric) and returns the aggregated DataFrame and the single best row (by optimization metric and mode).

**Example:**

```python
from picid_report.analysis import aggregate_and_find_best

sorted_agg, best_row = aggregate_and_find_best(
    subset,
    aggregation_cols=["Model", "task_definition.seq_len", "optimization.lr"],
    all_performance_cols=["val_best_rerun/loss", "test/mse"],
    optimization_col="val_best_rerun/loss",
    optimization_mode="min",
)
# sorted_agg: DataFrame with multi-level agg columns; best_row: single row DataFrame
```

---

## 4. Flow summary (per dataset/model)

1. Subset by dataset and model.
2. **Dynamic metrics** — `get_dynamic_metrics()`.
3. **Optimization metric/mode** — `get_optimized_metric()` and target/early-stop mode columns or default `"min"`.
4. **Performance columns** — collect from metric_prefixes + reporting_metrics (and optimization col).
5. **Varying HPs** — `get_varying_hyperparameters()`; then **validate_seeds** (data and model).
6. **Hidden variations** — `check_hidden_variations()`.
7. **Aggregate and best** — `aggregate_and_find_best()`.
8. **Grid** — `get_search_space()` then `get_model_grid_from_search_space()`; if grid exists, merge with grid (schema-first); else sort only (data-first).
9. **Store** best_hyperparameters, best_performance, sorted_aggregated_results, seeds_info, **sort_metric_used** (if resolver provided).

This document covers the public and main internal functions; for config and grid resolution details see [config.md](config.md).

---

**Navigation:** [← Documentation index](README.md)
