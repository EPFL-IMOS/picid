# lmetk_report — Functionality Reference

This document describes all modules and functionality in the `lmetk_report` package. For quick start and usage, see [README.md](README.md).

---

## Table of contents

1. [run.sh — Run pipeline and serve report](#runsh)
2. [run.py — Main pipeline](#runpy)
3. [run_processor.py — Data loading](#run_processory)
4. [preprocess.py — Preprocessing](#preprocesspy)
5. [validators.py — Validation](#validatorspy)
6. [config.py — Configuration](#configpy)
7. [analysis.py — Analysis](#analysispy)
8. [reporting.py — Tables and export](#reportingpy)
9. [plots.py — Plotting](#plotspy)
10. [report_html.py — HTML report](#report_htmlpy)
11. [utils.py — Shared utilities](#utilspy)
12. [Public API (__init__.py)](#public-api)

---

## run.sh

**Purpose:** Run the full pipeline and start a local HTTP server so you can view the generated report in a browser.

**Location:** `lmetk_report/run.sh`

**Usage:**
```bash
./lmetk_report/run.sh          # default port 8000
./lmetk_report/run.sh 8080     # custom port
```

**Behavior:**
1. Changes to the **project root** (parent of `lmetk_report/`).
2. Runs: `uv run python -m lmetk_report.run -o report_output --export-latex`
3. Starts: `python -m http.server <port>` in the project root.

**Environment:**
- `OUTPUT_DIR` — Override output directory (default: `report_output`).
- First argument — HTTP port (default: `8000`).

**After starting:** Open **http://localhost:8000/report_output/report.html** (or your port). Press Ctrl+C to stop the server.

---

## run.py

**Purpose:** Main pipeline script: load → preprocess → validate → analyze → report (tables and optional plots). Mirrors the flow of `table.ipynb`.

### `run_pipeline(...)`

Runs the full pipeline and optionally writes all outputs to disk.

**Parameters (main):**
- `project_name`, `user` — W&B project and entity.
- `csv_cache_dir` — Directory for CSV cache (default: `"csv_files"`).
- `metric_prefixes`, `reporting_metrics` — Metric prefixes and metric names for reporting (defaults from module).
- `required_data_seeds`, `required_model_seeds` — Optional sets of required seeds; groups missing any are dropped.
- `additional_ignored_cols` — Extra columns to ignore when detecting varying HPs.
- `preprocess_df` — Callable `(DataFrame) -> DataFrame`. If `None`, applies `clean_and_rename_models`; use `lambda df: df` to skip preprocessing.
- `pipeline_config` — Optional `PipelineConfig` to override column config, search space, filters, etc.
- `show_performance_tables` — If True, print per-metric performance tables to stdout.
- `show_plots` — If True, build plots (and save to `output_dir/plots` when `output_dir` is set).
- `precision` — Decimal precision for tables.
- `output_dir` — If set, save tables (CSV, optionally LaTeX), plots, `report.html`, and `tables_tex.tex` here.
- `export_laTeX` — If True and `output_dir` set, also save the summary table as LaTeX.
- `plot_metric` — Metric for bar chart and HP impact plots (e.g. `"test/mse"`).

**Parameters (new):**
- `sort_metric` — Optional metric to use for sorting/ranking/selecting best results (str or dict mapping `(dataset, model) -> metric`). If `None`, uses resolved metrics from analysis phase or optimization metric.

**Returns:** `(df, all_results, summary_df)` — Raw DataFrame (after load and preprocess), `analyze_results` output, and summary table DataFrame.

### `main()`

CLI entry point. Arguments:

- `--project` — W&B project name.
- `--user` — W&B user/entity.
- `--cache-dir` — CSV cache directory.
- `-o` / `--output-dir` — Directory for tables, plots, and report.
- `--export-latex` — Also write summary table as LaTeX when using `-o`.
- `--no-plots` — Skip generating plots.
- `--performance-tables` — Print per-metric performance tables.
- `-q` / `--quiet` — Do not print tables to terminal; only write report and files.
- `-d` / `--debug` — Enable DEBUG logging (stage details, resolver in/out, grid source, per-dataset/model decisions).

**Invocation:** `uv run python -m lmetk_report.run [options]`

**Logging:** At startup, `main()` configures the package logger (`lmetk_report`). With default (INFO): stage headers, row/column counts, and one line per (dataset, model) summarizing optimization metric, grid source (configs/legacy/auto), sort metric, and run counts. With `--debug`: DEBUG-level messages from run_processor (columns normalized/dropped), analysis (varying HPs, performance cols, resolver result), configs (search_space/sort_metric resolution). See `lmetk_report/logging_config.py` to configure programmatically.

---

## run_processor.py

**Purpose:** Fetch, cache, and prepare W&B run data into a flat pandas DataFrame.

### `load_runs_df(project_name, user, csv_cache_dir="csv_files", pipeline_config=None)`

**Returns:** `(df, config_columns, dropped_columns)`.

**Behavior:**
1. **Cache:** Tries to load from `csv_cache_dir/{project_name}.csv`. If missing, fetches from W&B API (`user/project_name`), keeps only `state == "finished"` runs, and saves to that CSV.
2. **Merge:** Builds one record per run from `summary` + `config` + `run_name`.
3. **Normalize:** Flattens nested dict columns listed in config (`COLUMNS_TO_NORMALIZE` or `pipeline_config.columns_to_normalize`) via `pd.json_normalize`; new columns get a `parent.` prefix.
4. **Drop:** Removes columns with >90% NaN and columns whose names match `COLUMN_FILTERS_TO_DROP` (or `pipeline_config.column_filters_to_drop`).

**Internal helpers:**
- `_filter_columns(df, filters)` — Column names containing any of the filter strings.
- `_normalize_column(df, column_to_normalize)` — Flatten one nested column into `parent.key` columns.

---

## preprocess.py

**Purpose:** Preprocessing helpers (e.g. model name cleaning) for the pipeline.

### `clean_and_rename_models(df, model_target_col=None, model_type_col=None)`

**Returns:** A copy of `df` with the model target column updated.

**Behavior:**
1. **Common prefix:** Among unique values in the model target column, computes the longest common prefix that contains a `.`; strips that prefix (up to and including the last `.`) so short names stay readable (e.g. `lmetk.baselines.patchtst_model.PatchTST_Forecaster` → `patchtst_model.PatchTST_Forecaster`).
2. **StatisticalBaselineWrapper:** If a model type column exists (default `model.model_type`), appends ` (linear)` or ` (exponential)` to the model target value so the two variants are distinct in analysis and config.

**Parameters:**
- `model_target_col` — Column holding the model identifier (default: `config.COLUMN_CONFIG["model_target"]`).
- `model_type_col` — Column for model type (default: `"model.model_type"`).

---

## validators.py

**Purpose:** Validate experiment consistency and schema.

### `validate_schema(df, required_cols)`

Checks that all `required_cols` exist in `df`. Raises `ValueError` with a clear message if any are missing. Logs success otherwise.

### `log_modification(action, reason, input_shape, output_shape, context=None)`

Logs a short notice when a DataFrame’s shape changes (rows/cols dropped), with optional context (e.g. model/dataset).

### `validate_seeds(df, group_cols, seed_col, required_seeds=None, allow_fallback=False)`

Filters `df` to groups that contain **all** values in `required_seeds` for `seed_col`. Used for strict seed completeness. If `allow_fallback` is True and the filter would remove all rows, logs a warning and returns the original `df`.

### `check_hidden_variations(df, group_cols, base_ignored_cols, additional_ignored_cols=None)`

Warns when a non-numeric, non-ignored column varies within a group (by `group_cols`). Helps detect unintended variation (e.g. different config keys) within what should be a single HP configuration.

---

## config.py

**Purpose:** Central configuration for column names, search space, and pipeline behavior.

### Constants

- **COLUMN_CONFIG** — Map of logical names to actual column names: `model_target`, `dataset_name`, `optimization_metric`, `optimization_mode`, `target_metric`, `target_metric_mode`, `evaluator_metrics`.
- **COLUMNS_TO_NORMALIZE** — List of nested config keys to flatten (e.g. `optimizer`, `datasource`, `model`, `callbacks`).
- **SPECIAL_COLUMNS** — Columns excluded from “varying hyperparameter” detection (e.g. data/model seeds).
- **COLUMN_FILTERS_TO_DROP** — Name substrings for columns to drop at load (e.g. `paths.`).
- **COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH** — Name substrings for columns to keep but not treat as HPs (metrics, timers, paths, etc.).
- **REQUIRED_COLUMNS** — Columns that must exist after loading (used by `validate_schema`).
- **EXPECTED_SEARCH_SPACE** — ⚠️ **DEPRECATED**: Legacy per-model dict of HP name → list of values. Use `configs/search_space.py` instead. Kept for backward compatibility.
- **CHECKPOINT_PATH_COLUMN**, **DEFAULT_DATA_SEED_COL**, **DEFAULT_MODEL_SEED_COL** — Column names for checkpoint path and seeds.

### Config modules (configs/)

- **configs/search_space.py** — Defines the expected HP grid per (dataset, model). **Dict shape (preferred):** `{dataset: {model: {hp_name: [values]}}}` (e.g. `EXPECTED_SEARCH_SPACE["UNIBO21"]["baselines.lstm_model.LSTM_Forecaster"]`). **Legacy shape:** `{model: {hp_name: [values]}}` (still supported by `get_model_grid_from_search_space`). Analysis uses `get_model_grid_from_search_space(dataset, model, search_space)` so both shapes work; missing keys → auto-discovery from data.
- **configs/sort_metrics.py** — Defines which metric is used for sorting/ranking best runs. **Resolution order:** (1) `SORT_METRIC_OVERRIDES[(dataset, model)]`, (2) `SORT_METRIC_BY_TASK_TYPE[task_type]`, (3) `SORT_METRIC_BY_DATASET_CATEGORY[category]`, (4) `DEFAULT_SORT_METRIC` (e.g. `"val_best_rerun/loss"`). Entry point: `get_sort_metric(dataset, model, task_type=None, dataset_category=None)`.

### `PipelineConfig` (dataclass)

Bundles: `column_config`, `expected_search_space`, `column_filters_to_ignore_for_hp_search`, `special_columns`, `columns_to_normalize`, `column_filters_to_drop`, `sort_metric_resolver`.

- **`PipelineConfig.from_default()`** — Builds an instance from the module-level constants above. Automatically includes `sort_metric_resolver` using `get_sort_metric` from configs if available. Used to override config without mutating globals (e.g. for a different benchmark).

---

## analysis.py

**Purpose:** Core analysis: per (dataset, model) aggregation, best run, optimization metric resolution, and HP impact data.

### `analyze_results(...)`

**Returns:** `all_results` — Nested `defaultdict`: `all_results[dataset][model]` is a dict with:

- **best_hyperparameters** — Dict of the best run’s hyperparameters.
- **best_performance** — Contains:
  - `metrics`: `metric_name -> prefix -> {mean, std, count}`,
  - `optimized_on`: `{metric, value, ...}`,
  - optionally `run_name` / `run_name_list`.
- **sorted_aggregated_results** — DataFrame of aggregated stats per HP configuration (mean/std/count per metric), sorted by the optimization metric.
- **non_aggregated_df** — Raw runs for this model/dataset.
- **seeds_info** — Seed information (e.g. data/model seeds found) for display.

**Main parameters:** `df`, `config_columns`, `dropped_columns`, `reporting_metrics`, `metric_prefixes`, optional `optimization_col`/`optimization_mode`, seed-related args, `additional_ignored_cols`, `column_config`, `expected_search_space`, `pipeline_config`, `sort_metric_resolver`.

**Flow (per dataset/model):** Subset → dynamic metric discovery → resolve optimization metric (task definition → early stopping → fallback) → seed validation → varying HP detection → aggregate by HP config → pick best run → resolve sort metric (if resolver provided) → store all of the above. If the resolved `opt_col_flattened` is missing in the data, that model/dataset is skipped.

**Sort Metric Resolution:** If `sort_metric_resolver` is provided (callable `(dataset, model, task_type, dataset_category) -> metric`), it is called for each dataset/model combination and the result is stored in `all_results[dataset][model]["sort_metric_used"]`. This metric can then be used by reporting functions to sort/rank by a different metric than the optimization metric.

### Helpers (used by analysis)

- **get_search_grid_df(search_space)** — Cartesian product of HP values → DataFrame.
- **get_unique_values(df, column)** — Unique values from a column (flattens lists).
- **get_dynamic_metrics(df_subset, column_config)** — Metric names from evaluator config (e.g. list/dict column).
- **get_optimized_metric(df_subset, model_name, dataset_name, column_config)** — Resolves the optimization metric (task_definition → early_stopping → fallback list).
- **get_varying_hyperparameters(df_subset, config_cols, special_cols_to_exclude)** — Dict of column → list of unique values for columns with more than one value.
- **aggregate_and_find_best(...)** — Groupby aggregation (mean/std/count) and best row by optimization metric.

---

## reporting.py

**Purpose:** Build and export summary tables, experiment stats, and HP impact tables (display, CSV, LaTeX).

### Summary table

- **create_summary_table(all_results, precision=4)** — Builds a single pivoted DataFrame: rows = models, columns = (Dataset, Metric), values = formatted “mean ± std (n=count)”.
- **export_summary_table(all_results, path, format="csv"|"latex", precision=4)** — Writes that table to CSV or LaTeX.

### Experiment stats

- **get_experiment_stats_df(all_results)** — DataFrame with Dataset, Model, Configs Completed, Seeds Found.
- **display_experiment_stats(all_results)** — Prints that table (e.g. via IPython display).
- **export_experiment_stats(all_results, path)** — Writes experiment stats to CSV.

### Performance tables (per metric)

- **display_performance_tables(all_results, precision=4)** — Prints one pivot per metric (model vs dataset) with formatted values.

### HP impact

- **_build_hp_impact_df(res, precision, sort_metric=None)** — Builds one HP impact DataFrame for a model/dataset result (flattened aggregates, formatted metric columns). If `sort_metric` is provided, re-sorts by that metric. Returns `(df, metric_used)` tuple.
- **iter_hp_impact_tables(all_results, precision=4, sort_metric=None)** — Yields `(dataset, model, df, metric_used)` for each non-empty HP impact table. Supports `sort_metric` parameter (str or dict).
- **display_hp_impact(all_results, precision=4, sort_metric=None)** — Prints each HP impact table with a short header showing the metric used for sorting.
- **export_hp_impact_tables(all_results, output_dir, precision=4, sort_metric=None)** — Saves one CSV per (dataset, model) with sanitized filenames; returns list of `(dataset, model, path)`. Supports `sort_metric` parameter.

### LaTeX

- **write_tables_tex(path, summary_df, stats_df, all_results, precision=4, sort_metric=None)** — Writes one LaTeX file with: summary table, experiment stats, then one HP impact table per model/dataset. Comment lines indicate sections and metric information. Requires `\usepackage{booktabs}`. Supports `sort_metric` parameter.

---

## plots.py

**Purpose:** Paper-ready plots from `all_results`. Requires **matplotlib** (raises a clear `ImportError` with install hint if missing).

### `plot_best_metric_bars(all_results, metric="test/mse", save_path=None, figsize=None)`

Bar chart of best metric value per (model, dataset). Returns the matplotlib figure (or `None` if no data). If `save_path` is set, saves the figure.

### `plot_hp_impact(all_results, model, dataset, metric, save_path=None, figsize=None, max_configs=50)`

Horizontal bar plot of metric vs HP configuration (one bar per config, from `sorted_aggregated_results`) for one model/dataset. Returns the figure or `None`. Truncates to `max_configs` configs.

**Internal:** `_ensure_matplotlib()` — Raises `ImportError` with install instructions if matplotlib is not available.

---

## report_html.py

**Purpose:** Generate a single HTML report that embeds tables and plot images.

### `write_report_html(output_dir, summary_df, stats_df, hp_impact_entries, global_plots, title="Experiment Report", all_results=None, sort_metric=None)`

**Parameters:**
- `output_dir` — Directory containing `tables/` and `plots/`; report is written as `output_dir/report.html`.
- `summary_df` — From `create_summary_table`.
- `stats_df` — From `get_experiment_stats_df`.
- `hp_impact_entries` — List of `(dataset, model, df, plot_rel_path, metric_used)` for each HP impact section (5-tuple).
- `global_plots` — List of `(label, rel_path)` for global figures (e.g. best-metric bar chart).
- `title` — Page title.
- `all_results` — Optional `all_results` dict for extracting metric information.
- `sort_metric` — Optional sort metric (str or dict) for displaying metric information.

**Returns:** Path to the written report file.

**Behavior:** Renders HTML with inline CSS, experiment stats table, summary table (with metric information), global plot images, then one section per (dataset, model) with HP impact table (showing dataset name, model name, and metric used) and optional plot image. Paths to images are relative to `output_dir`.

---

## utils.py

**Purpose:** Shared formatting and column flattening.

### `format_mean_std_count(mean, std, count, precision=4)`

Returns a string like `"1.2345 ± 0.0012 (n=3)"`. Handles NaN std (shown as 0) and non-numeric count (e.g. `"n/a"`).

### `flatten_aggregated_columns(df)`

Converts MultiIndex columns `(metric, 'mean')`, `(metric, 'std')`, etc., to `metric_mean`, `metric_std`, … Plain string columns are unchanged. Returns a copy.

---

## Public API

The package exposes the following from `lmetk_report` (see `__init__.py`):

| Symbol | Description |
|--------|-------------|
| `analyze_results` | Full analysis → `all_results` |
| `clean_and_rename_models` | Strip model prefix, add (linear)/(exponential) |
| `config` | Module with COLUMN_CONFIG, REQUIRED_COLUMNS, etc. |
| `create_summary_table` | Pivot table of best metrics |
| `PipelineConfig` | Dataclass + `from_default()` for overrides |
| `display_experiment_stats` | Print config counts and seeds |
| `display_hp_impact` | Print HP impact tables |
| `display_performance_tables` | Print one pivot per metric |
| `export_summary_table` | Write summary to CSV or LaTeX |
| `load_runs_df` | Load W&B runs (or cache), flatten config |
| `plot_best_metric_bars` | Bar chart of best metric |
| `plot_hp_impact` | HP impact plot for one model/dataset |
| `validate_schema` | Check required columns |

Additional reporting helpers are in `lmetk_report.reporting`: `get_experiment_stats_df`, `iter_hp_impact_tables`, `export_experiment_stats`, `export_hp_impact_tables`, `write_tables_tex`. HTML report: `lmetk_report.report_html.write_report_html`.

---

## Data flow summary

1. **run.sh** or **run.py** → `run_pipeline()`
2. **load_runs_df()** → raw flat DataFrame + config/dropped column lists
3. **clean_and_rename_models()** (or custom `preprocess_df`) → preprocessed DataFrame
4. **validate_schema()** → raises if required columns missing
5. **analyze_results()** → `all_results` (per dataset/model: best run, aggregates, seeds)
6. **reporting** — summary table, experiment stats, HP impact (display + optional export)
7. If **output_dir**: export CSV/LaTeX, build plots, **write_report_html()** and **write_tables_tex()**

See **README.md** for quick start, CLI options, output layout, and programmatic examples.

---

## Migration Guide

### Migrating from Model-Only to Dataset/Model Search Space

**Old structure (deprecated):**
```python
EXPECTED_SEARCH_SPACE = {
    "baselines.lstm_model.LSTM_Forecaster": {
        "task_definition.seq_len": [1, 10, 50, 100],
        "optimization.lr": [0.001, 0.0005, 0.0001],
    },
}
```

**New structure (recommended):**
```python
# In lmetk_report/configs/search_space.py
EXPECTED_SEARCH_SPACE = {
    "nb14": {
        "baselines.lstm_model.LSTM_Forecaster": {
            "task_definition.seq_len": [1, 10, 50, 100],
            "optimization.lr": [0.001, 0.0005, 0.0001],
        },
    },
    "mzvav": {
        "baselines.lstm_model.LSTM_Forecaster": {
            "task_definition.seq_len": [1, 5, 10, 20],  # Different for diagnostics
            "optimization.lr": [0.001, 0.0005],
        },
    },
}
```

**Backward compatibility:** The old structure still works via automatic fallback. However, migrating to the new structure enables dataset-specific configurations.

### Using Sort Metrics

**Automatic resolution (recommended):**
The pipeline automatically resolves sort metrics using the hierarchical config system:
1. Check specific overrides in `SORT_METRIC_OVERRIDES[(dataset, model)]`
2. Check task type defaults in `SORT_METRIC_BY_TASK_TYPE[task_type]`
3. Check dataset category defaults in `SORT_METRIC_BY_DATASET_CATEGORY[category]`
4. Fallback to `DEFAULT_SORT_METRIC` (e.g. `"val_best_rerun/loss"`)

**Manual override:**
```python
from lmetk_report import run_pipeline

# Use specific metric for all model/dataset combinations
results = run_pipeline(
    project_name="my_project",
    user="my_user",
    sort_metric="test/accuracy",
)

# Use different metric per combination
results = run_pipeline(
    project_name="my_project",
    user="my_user",
    sort_metric={
        ("nb14", "baselines.lstm_model.LSTM_Forecaster"): "test/mse",
        ("mzvav", "baselines.lstm_model.LSTM_Forecaster"): "test/accuracy",
    },
)
```

**In reporting functions:**
```python
from lmetk_report import reporting

# Use resolved metrics automatically (from analysis phase)
summary_df = reporting.create_summary_table(all_results)

# Or override manually
summary_df = reporting.create_summary_table(
    all_results,
    sort_metric="test/accuracy"
)
```

### Configuring Sort Metrics

**Add task type default:**
```python
# In lmetk_report/configs/sort_metrics.py
SORT_METRIC_BY_TASK_TYPE = {
    "regression": "test/mse",
    "classification": "test/accuracy",
    "custom_task": "test/f1",  # Add your task type
}
```

**Add dataset category default:**
```python
# In lmetk_report/configs/sort_metrics.py
SORT_METRIC_BY_DATASET_CATEGORY = {
    "prognostics": "test/mse",
    "diagnostics": "test/accuracy",
    "custom_category": "test/mae",  # Add your category
}
```

**Add specific override:**
```python
# In lmetk_report/configs/sort_metrics.py
SORT_METRIC_OVERRIDES = {
    ("nb14", "baselines.lstm_model.LSTM_Forecaster"): "test/mae",
}
```
