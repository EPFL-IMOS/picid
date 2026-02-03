# lmetk_report

Pipeline and reporting utilities for experiment results (e.g. from Weights & Biases). Load runs, validate schema, analyze by model/dataset, and produce **tables** (CSV, LaTeX) and **plots** for papers, plus a single **HTML report** for local review.

---

## Quick start

**Generate report and serve it locally:**

```bash
# From project root
./lmetk_report/run.sh          # uses port 8000
./lmetk_report/run.sh 8080     # custom port
```

Then open **http://localhost:8000/report_output/report.html** (or the port you chose). The script runs the full pipeline (load → preprocess → validate → analyze → report), writes all outputs into `report_output/`, then starts an HTTP server so you can view the report in a browser.

**Generate report only (no server):**

```bash
uv run python -m lmetk_report.run -o report_output --export-latex
```

**CLI options:**

| Option | Description |
|--------|-------------|
| `-o`, `--output-dir` | Directory for tables, plots, and report (default: none; only display) |
| `--export-latex` | Also write summary table as LaTeX (when using `-o`) |
| `--project` | W&B project name |
| `--user` | W&B user/entity |
| `--cache-dir` | Directory for CSV cache (default: `csv_files`) |
| `--no-plots` | Skip generating plot images |
| `--performance-tables` | Print per-metric performance tables to stdout |

---

## Pipeline overview

1. **Load** – `load_runs_df(project_name, user)` fetches runs from W&B (or cache), flattens config, drops noisy columns.
2. **Preprocess** – By default `clean_and_rename_models(df)` strips a common prefix from model names and appends `(linear)` / `(exponential)` for StatisticalBaselineWrapper.
3. **Validate** – `validate_schema(df, config.REQUIRED_COLUMNS)`.
4. **Analyze** – `analyze_results(...)` builds per–(dataset, model) aggregates, best run, and HP impact tables.
5. **Report** – Tables (summary, experiment stats, HP impact) and optional plots; when `output_dir` is set, everything is written to disk and a single `report.html` + `tables_tex.tex` are produced.

---

## Output layout (when using `-o report_output`)

```
report_output/
├── report.html              # Single-file HTML report (open in browser)
├── tables_tex.tex           # All tables in one LaTeX file for pasting into a paper
├── tables/
│   ├── summary.csv
│   ├── summary.tex          # (if --export-latex)
│   ├── experiment_stats.csv
│   └── hp_impact/           # One CSV per (dataset, model)
│       └── hp_impact_<dataset>_<model>.csv
└── plots/
    ├── best_metric_bars.png
    └── hp_impact_<dataset>_<model>.png
```

- **report.html** – Experiment stats, summary table, plots, and HP impact sections (tables + images). No server required if you open the file directly; use `run.sh` to serve it over HTTP.
- **tables_tex.tex** – Summary table, experiment stats, and one LaTeX table per HP impact. Paste the sections you need into your paper; requires `\usepackage{booktabs}`.

---

## Programmatic usage

### Run the full pipeline

```python
from lmetk_report.run import run_pipeline

df, all_results, summary_df = run_pipeline(
    project_name="my_project",
    user="my_team",
    output_dir="report_output",   # optional: write all outputs here
    export_laTeX=True,
    plot_metric="test/mse",       # metric for bar chart and HP impact plots
)

# Optional: skip preprocessing (keep raw model names)
run_pipeline(..., preprocess_df=lambda df: df)
```

### Load and validate only

```python
from lmetk_report import load_runs_df, validate_schema, config

df, config_columns, dropped_columns = load_runs_df("my_project", "my_user")
df = clean_and_rename_models(df)  # optional
validate_schema(df, config.REQUIRED_COLUMNS)
```

### Analyze and report

```python
from lmetk_report import (
    analyze_results,
    create_summary_table,
    display_experiment_stats,
    display_hp_impact,
    export_summary_table,
)
from lmetk_report.reporting import (
    get_experiment_stats_df,
    iter_hp_impact_tables,
    export_experiment_stats,
    export_hp_impact_tables,
    write_tables_tex,
)

all_results = analyze_results(
    df, config_columns, dropped_columns,
    reporting_metrics=["loss", "mse", "mae", "rmse", "f1", "accuracy"],
    metric_prefixes=["val/", "test/", "test_best_rerun/", "val_best_rerun/"],
    required_model_seeds={72, 88, 101},  # or None
    additional_ignored_cols=["gpu_id"],
)

display_experiment_stats(all_results)
display_hp_impact(all_results, precision=4)
summary_df = create_summary_table(all_results, precision=4)

# Export to files
export_summary_table(all_results, "summary.csv", format="csv")
export_summary_table(all_results, "summary.tex", format="latex")
export_experiment_stats(all_results, "experiment_stats.csv")
export_hp_impact_tables(all_results, "tables/hp_impact", precision=4)

# Single LaTeX file for the paper
stats_df = get_experiment_stats_df(all_results)
write_tables_tex("tables_tex.tex", summary_df, stats_df, all_results, precision=4)
```

### Custom config (PipelineConfig)

```python
from lmetk_report import PipelineConfig, config

# Default from config.py
default_cfg = PipelineConfig.from_default()

# Override for a different benchmark
custom_cfg = PipelineConfig(
    column_config={**config.COLUMN_CONFIG, "model_target": "model.name"},
    expected_search_space={"ModelA": {"lr": [0.01, 0.001], "epochs": [10, 20]}},
    column_filters_to_ignore_for_hp_search=config.COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH,
    special_columns=config.SPECIAL_COLUMNS,
    columns_to_normalize=config.COLUMNS_TO_NORMALIZE,
    column_filters_to_drop=config.COLUMN_FILTERS_TO_DROP,
)

run_pipeline(..., pipeline_config=custom_cfg)
# or
df, config_columns, dropped_columns = load_runs_df(..., pipeline_config=custom_cfg)
all_results = analyze_results(..., pipeline_config=custom_cfg)
```

### Preprocessing

```python
from lmetk_report import clean_and_rename_models

# Strip common prefix and add (linear)/(exponential) for StatisticalBaselineWrapper
df = clean_and_rename_models(df)

# Custom column names
df = clean_and_rename_models(df, model_target_col="model.name", model_type_col="model.type")
```

### Plots (requires matplotlib)

```python
from lmetk_report import plot_best_metric_bars, plot_hp_impact

fig = plot_best_metric_bars(all_results, metric="test/mse", save_path="bars.png")

fig = plot_hp_impact(all_results, model="MyModel", dataset="DS1", metric="test/mse", save_path="hp.png")
```

If matplotlib is not installed, plot functions raise a clear `ImportError` with install instructions.

### HTML report (standalone)

```python
from lmetk_report.report_html import write_report_html

write_report_html(
    output_dir="report_output",
    summary_df=summary_df,
    stats_df=stats_df,
    hp_impact_entries=[(dataset, model, df, "plots/hp_impact_xy.png") for ...],
    global_plots=[("Best test/mse", "plots/best_metric_bars.png")],
    title="Experiment Report",
)
```

---

## Public API summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `load_runs_df` | run_processor | Load W&B runs (or cache), flatten config, return `(df, config_columns, dropped_columns)` |
| `analyze_results` | analysis | Run full analysis → `all_results` (dataset → model → dict with best_*, sorted_aggregated_results, etc.) |
| `create_summary_table` | reporting | Pivot table of best metrics by model/dataset |
| `display_experiment_stats` | reporting | Print config counts and seeds |
| `display_hp_impact` | reporting | Print HP impact tables per model/dataset |
| `display_performance_tables` | reporting | Print one pivot per metric |
| `export_summary_table` | reporting | Write summary table to CSV or LaTeX |
| `export_experiment_stats` | reporting | Write experiment stats to CSV |
| `export_hp_impact_tables` | reporting | Write one CSV per (dataset, model) HP impact |
| `write_tables_tex` | reporting | Write single LaTeX file with all tables |
| `get_experiment_stats_df` | reporting | Return experiment stats as DataFrame |
| `iter_hp_impact_tables` | reporting | Yield `(dataset, model, df)` for each HP impact table |
| `validate_schema` | validators | Check required columns; raise if missing |
| `clean_and_rename_models` | preprocess | Strip model name prefix, add (linear)/(exponential) |
| `plot_best_metric_bars` | plots | Bar chart of best metric per model/dataset |
| `plot_hp_impact` | plots | HP impact plot for one model/dataset |
| `config` | config | Module with COLUMN_CONFIG, EXPECTED_SEARCH_SPACE, REQUIRED_COLUMNS, etc. |
| `PipelineConfig` | config | Dataclass + `from_default()` for overrideable pipeline config |
| `run_pipeline` | run | Full pipeline: load → preprocess → validate → analyze → report (and optional save) |

---

## Configuration

- **config.py** – Column names, expected search space (per model), filters, seed columns, required columns. Used by default; override via `PipelineConfig` or optional `column_config` / `expected_search_space` arguments to `analyze_results`.
- **run.py** – Default project, user, metric prefixes, reporting metrics, required seeds, and additional ignored columns. Override by passing arguments to `run_pipeline()` or via CLI.

---

## Tests

From project root:

```bash
uv run pytest test/lmetk_report -v
```

---

## See also

- **table.ipynb** – Reference notebook that mirrors the pipeline (load, clean_and_rename_models, validate, analyze, display).
- **IMPROVEMENT_PLAN.md** – Refactoring and improvement log (logging, schema doc, exports, PipelineConfig, etc.).
