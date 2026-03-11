# Configuration: config.py and configs/

**Navigation:** [← Documentation index](README.md)

**Modules:** `picid_report.config`, `picid_report.configs.search_space`, `picid_report.configs.sort_metrics`

This document explains the **logic** behind `config.py` and the `configs/` package (search space and sort metrics), and gives **examples** for every constant and function.

---

## 1. Why two places? (config.py vs configs/)

- **`config.py`** holds **global, module-level** settings: column names (from WandB), which columns to normalize/drop/ignore, required columns, and a **legacy** search space (model-only dict). It is the single place the pipeline reads when no override is passed.
- **`configs/`** holds **dataset/model-level** settings:
  - **`configs/search_space.py`**: expected HP grid **per (dataset, model)** (preferred over the legacy dict in config.py).
  - **`configs/sort_metrics.py`**: which metric to use for **sorting/ranking** best runs (hierarchy: override → task type → category → default).

The pipeline **first** consults `configs.search_space` for a grid; if none is found for a (dataset, model), it falls back to `config.EXPECTED_SEARCH_SPACE` (legacy). Sort metric is resolved via `configs.sort_metrics.get_sort_metric(...)` when a resolver is used.

---

## 2. config.py — In-depth

### 2.1 COLUMN_CONFIG

Maps **logical names** to the **actual column names** in your WandB logs (after flattening). The pipeline uses these keys everywhere (e.g. "model_target", "dataset_name").

| Key | Meaning | Example value |
|-----|--------|----------------|
| `model_target` | Model class/identifier | `"model._target_"` |
| `dataset_name` | Dataset name | `"datasource.data_name"` |
| `optimization_metric` | Early-stopping monitor | `"callbacks.early_stopping.monitor"` |
| `optimization_mode` | min/max | `"callbacks.early_stopping.mode"` |
| `target_metric` | Task metric (e.g. loss) | `"task_definition.target_metric"` |
| `target_metric_mode` | min/max for task | `"task_definition.target_metric_mode"` |
| `evaluator_metrics` | List of metric names | `"evaluator.train.metric_names"` |

**Example — reading columns in code:**
```python
from picid_report import config

model_col = config.COLUMN_CONFIG["model_target"]   # e.g. "model._target_"
dataset_col = config.COLUMN_CONFIG["dataset_name"]  # e.g. "datasource.data_name"
# Use in DataFrame: df[model_col], df[dataset_col]
```

**Example — overriding for a different project:**
```python
from picid_report.config import PipelineConfig

custom = PipelineConfig.from_default()
custom.column_config["model_target"] = "model.name"  # if your logs use model.name
# Then pass pipeline_config=custom to load_runs_df / analyze_results / run_pipeline
```

---

### 2.2 COLUMNS_TO_NORMALIZE

List of **top-level keys** in the raw WandB run config/summary that are **nested dicts** and will be flattened into columns with a `parent.key` prefix.

**Example:** If a run has `"model": {"_target_": "...", "lr": 0.01}`, after normalization you get columns `model._target_`, `model.lr`.

**Example — what gets flattened:**
```python
# config.COLUMNS_TO_NORMALIZE includes: "optimizer", "optimization", "datasource",
# "task_definition", "model", "dataset", "datamodule", "callbacks", "paths",
# "evaluator", "cache", "logger", "trainer", "_wandb"
# So a column "model" with value {"_target_": "x", "lr": 0.01} becomes
# columns: model._target_, model.lr
```

---

### 2.3 SPECIAL_COLUMNS

Column names that must **not** be treated as varying hyperparameters even if they vary across runs (e.g. seeds). They are excluded from the "varying HP" detection in analysis.

**Example:**
```python
# config.SPECIAL_COLUMNS = ["datasource.parameters.data_seed", "task_definition.subset_seed",
#                           "datamodule.subset_seed", "seed"]
# So "seed" and data/subset seeds are never listed as aggregation dimensions.
```

---

### 2.4 COLUMN_FILTERS_TO_DROP

Substrings: any column whose name **contains** one of these is **dropped** during load (e.g. `paths.*`).

**Example:** With `COLUMN_FILTERS_TO_DROP = ["paths."]`, columns like `paths.output_dir` are removed from the DataFrame.

---

### 2.5 COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH

Substrings: columns that **stay** in the DataFrame but are **ignored** when detecting varying hyperparameters (e.g. metrics, run_name, timers). Used to build the list of "config" columns that are scanned for variation.

**Example:** Columns containing `"val/"`, `"test/"`, `"run_name"`, `"epoch"` are not considered as HP dimensions.

---

### 2.6 REQUIRED_COLUMNS

Columns that **must** exist after load; otherwise `validate_schema()` raises. Derived from COLUMN_CONFIG and seed/checkpoint names.

**Example:**
```python
from picid_report import config
from picid_report.validators import validate_schema

validate_schema(df, config.REQUIRED_COLUMNS)  # Raises ValueError if any missing
```

---

### 2.7 EXPECTED_SEARCH_SPACE (legacy)

**Shape:** `{model: {hp_name: [values]}}`. Same grid for **all datasets**. Deprecated in favor of `configs.search_space.EXPECTED_SEARCH_SPACE`.

**Logic:** If the pipeline does not find a grid in `configs.search_space` for (dataset, model), it uses this dict via `get_model_grid_from_search_space(dataset, model, config.EXPECTED_SEARCH_SPACE)`. So it acts as a **fallback** when the (dataset, model) is not in the configs file.

**Example — legacy structure:**
```python
# config.EXPECTED_SEARCH_SPACE looks like:
{
    "baselines.lstm_model.LSTM_Forecaster": {
        "task_definition.seq_len": [1, 10, 50, 100],
        "optimization.lr": [0.001, 0.0005, 0.0001],
    },
}
# Every dataset uses this same grid for LSTM_Forecaster.
```

---

### 2.8 PipelineConfig

Dataclass that **bundles** all overridable config so you can pass one object instead of mutating globals. Used by `load_runs_df`, `analyze_results`, and `run_pipeline`.

**Fields:** `column_config`, `expected_search_space`, `column_filters_to_ignore_for_hp_search`, `special_columns`, `columns_to_normalize`, `column_filters_to_drop`, `sort_metric_resolver`.

**Example — build from current module config:**
```python
from picid_report.config import PipelineConfig

cfg = PipelineConfig.from_default()
# cfg.column_config is a copy of config.COLUMN_CONFIG
# cfg.expected_search_space is a copy of config.EXPECTED_SEARCH_SPACE (same legacy shape)
# cfg.sort_metric_resolver is get_sort_metric from configs (if available)
```

**Example — custom PipelineConfig for a different benchmark:**
```python
from picid_report import config
from picid_report.config import PipelineConfig

cfg = PipelineConfig(
    column_config=dict(config.COLUMN_CONFIG),
    expected_search_space={
        "MyModel": {"lr": [0.01, 0.001], "epochs": [10, 20]},
    },
    column_filters_to_ignore_for_hp_search=config.COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH,
    special_columns=config.SPECIAL_COLUMNS,
    columns_to_normalize=config.COLUMNS_TO_NORMALIZE,
    column_filters_to_drop=config.COLUMN_FILTERS_TO_DROP,
    sort_metric_resolver=None,
)
# Then: run_pipeline(..., pipeline_config=cfg)
```

---

## 3. configs/search_space.py — In-depth

### 3.1 Dict shapes

Two shapes are supported by `get_model_grid_from_search_space()`:

- **New (preferred):** `{dataset: {model: {hp_name: [values]}}}`
  Example: `EXPECTED_SEARCH_SPACE["UNIBO21"]["baselines.lstm_model.LSTM_Forecaster"]` → `{"task_definition.seq_len": [1, 10, 50, 100], ...}`

- **Legacy:** `{model: {hp_name: [values]}}`
  Example: `config.EXPECTED_SEARCH_SPACE["baselines.lstm_model.LSTM_Forecaster"]` → same value shape. No dataset key; same grid for every dataset.

**Resolution in the pipeline:** For each (dataset, model), the code first calls `get_search_space(dataset, model)` (reads `configs.search_space.EXPECTED_SEARCH_SPACE`). If that returns a dict, that grid is used. Otherwise it calls `get_model_grid_from_search_space(dataset, model, _search_space)` where `_search_space` is the legacy dict or a PipelineConfig’s expected_search_space. If that returns `None`, the pipeline uses **auto-discovery** (varying HPs from data only).

### 3.2 get_search_space(dataset, model)

Looks up **(dataset, model)** in the **module-level** `EXPECTED_SEARCH_SPACE` (new shape only). Returns `None` if not found (then fallback/auto-discovery applies).

**Example:**
```python
from picid_report.configs.search_space import get_search_space, EXPECTED_SEARCH_SPACE

# If EXPECTED_SEARCH_SPACE has "UNIBO21" and "baselines.lstm_model.LSTM_Forecaster":
grid = get_search_space("UNIBO21", "baselines.lstm_model.LSTM_Forecaster")
# grid = {"task_definition.seq_len": [1, 10, 50, 100], "optimization.lr": [0.001, 0.0005, 0.0001]}

grid = get_search_space("OTHER_DS", "baselines.lstm_model.LSTM_Forecaster")
# grid = None  (no key "OTHER_DS")
```

### 3.3 get_model_grid_from_search_space(dataset, model, search_space)

Returns the HP grid for (dataset, model) from **any** dict you pass: **new** or **legacy** shape. Returns `None` if not found or search_space is None/empty.

**Example — new shape:**
```python
from picid_report.configs.search_space import get_model_grid_from_search_space

space_new = {
    "UNIBO21": {
        "LSTM": {"lr": [0.01, 0.001], "seq_len": [10, 50]},
    },
}
get_model_grid_from_search_space("UNIBO21", "LSTM", space_new)
# {"lr": [0.01, 0.001], "seq_len": [10, 50]}
get_model_grid_from_search_space("OTHER", "LSTM", space_new)
# None
```

**Example — legacy shape:**
```python
space_legacy = {
    "LSTM": {"lr": [0.01, 0.001], "seq_len": [10, 50]},
}
get_model_grid_from_search_space("any_dataset", "LSTM", space_legacy)
# {"lr": [0.01, 0.001], "seq_len": [10, 50]}
get_model_grid_from_search_space("any_dataset", "UnknownModel", space_legacy)
# None
```

**Example — None/empty:**
```python
get_model_grid_from_search_space("UNIBO21", "LSTM", None)   # None
get_model_grid_from_search_space("UNIBO21", "LSTM", {})     # None
```

---

## 4. configs/sort_metrics.py — In-depth

### 4.1 Resolution order

`get_sort_metric(dataset, model, task_type=None, dataset_category=None)` resolves in this order:

1. **SORT_METRIC_OVERRIDES[(dataset, model)]** — per (dataset, model) override.
2. **SORT_METRIC_BY_TASK_TYPE[task_type]** — e.g. regression → `"val_best_rerun/loss"`.
3. **SORT_METRIC_BY_DATASET_CATEGORY[dataset_category]** — e.g. prognostics → `"val_best_rerun/loss"`.
4. **DEFAULT_SORT_METRIC** — e.g. `"val_best_rerun/loss"`.

So you get a metric name for sorting/ranking the best run and HP tables; if you don’t set overrides, task type and category defaults (or global default) are used.

### 4.2 Constants

- **DEFAULT_SORT_METRIC:** string, e.g. `"val_best_rerun/loss"`.
- **SORT_METRIC_BY_TASK_TYPE:** dict `task_type -> metric_name`.
- **SORT_METRIC_BY_DATASET_CATEGORY:** dict `category -> metric_name`.
- **SORT_METRIC_OVERRIDES:** dict `(dataset, model) -> metric_name`; only for exceptions.

**Example — override one (dataset, model):**
```python
# In configs/sort_metrics.py you could set:
SORT_METRIC_OVERRIDES = {
    ("nb14", "baselines.lstm_model.LSTM_Forecaster"): "test/mae",
}
# Then get_sort_metric("nb14", "baselines.lstm_model.LSTM_Forecaster", ...) returns "test/mae".
```

### 4.3 get_sort_metric(dataset, model, task_type=None, dataset_category=None, fallback_to_optimization=True)

Returns the metric name to use for sorting/ranking. See resolution order above. The parameter `fallback_to_optimization` is kept for backward compatibility but is no longer used; the function always returns a string (DEFAULT_SORT_METRIC) when no override/task/category match is found.

**Example:**
```python
from picid_report.configs.sort_metrics import get_sort_metric, DEFAULT_SORT_METRIC

get_sort_metric("UNIBO21", "LSTM", task_type="regression", dataset_category="prognostics")
# Typically "val_best_rerun/loss" (from task type or category or default)
get_sort_metric("unknown_ds", "UnknownModel", task_type=None, dataset_category=None)
# DEFAULT_SORT_METRIC, e.g. "val_best_rerun/loss"
```

### 4.4 infer_task_type_from_dataset(dataset)

Heuristic: infers "classification" or "regression" from dataset name (e.g. mzvav → classification, unibo → regression). Returns `None` if unknown.

**Example:**
```python
from picid_report.configs.sort_metrics import infer_task_type_from_dataset

infer_task_type_from_dataset("mzvav")   # "classification"
infer_task_type_from_dataset("UNIBO21") # "regression"
infer_task_type_from_dataset("xyz")     # None
```

### 4.5 infer_dataset_category_from_name(dataset)

Heuristic: infers "diagnostics" or "prognostics" from dataset name. Returns `None` if unknown.

**Example:**
```python
from picid_report.configs.sort_metrics import infer_dataset_category_from_name

infer_dataset_category_from_name("mzvav")   # "diagnostics"
infer_dataset_category_from_name("UNIBO21") # "prognostics"
infer_dataset_category_from_name("xyz")     # None
```

---

## 5. How config and configs interact in the pipeline

1. **Load:** Uses `config.COLUMN_CONFIG`, `COLUMNS_TO_NORMALIZE`, `COLUMN_FILTERS_TO_DROP` (or from `PipelineConfig` if provided).
2. **Validate:** Uses `config.REQUIRED_COLUMNS`.
3. **Analyze:** For each (dataset, model):
   - **Grid:** First `get_search_space(dataset, model)` (configs); if None, `get_model_grid_from_search_space(dataset, model, _search_space)` where `_search_space` is from config or PipelineConfig (legacy or new shape). If still None → auto-discovery.
   - **Sort metric:** If a resolver is used (e.g. from PipelineConfig or run.py), it calls `get_sort_metric(dataset, model, task_type, dataset_category)` (configs) and stores `sort_metric_used`.
4. **Reporting:** Uses `sort_metric_used` from analysis when building summary and HP impact tables (re-rank/re-select by that metric when provided).

This gives one place for column/global settings (`config.py`) and dataset/model-specific behavior (grid and sort metric in `configs/`), with a clear fallback and resolution order.

---

**Navigation:** [← Documentation index](README.md)
