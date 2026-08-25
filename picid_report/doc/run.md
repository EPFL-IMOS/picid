# run.py — Main pipeline and CLI

**Navigation:** [← Documentation index](README.md)

**Module:** `picid_report.run` (root)

This document describes the main pipeline script: `run_pipeline()`, `main()`, and the CLI. It includes **examples** for every function and common usage patterns.

---

## 1. Overview

`run.py` orchestrates the full flow:

1. **Load** — `load_runs_df`: fetch runs from W&B or CSV cache; normalize and drop columns.
2. **Preprocess** — by default `clean_and_rename_models`; optional custom callable.
3. **Validate** — `validate_schema(df, config.REQUIRED_COLUMNS)`.
4. **Analyze** — `analyze_results`: per (dataset, model) aggregation, grid resolution, best run, sort metric.
5. **Report** — build summary table, experiment stats, HP impact; display and/or export.
6. **Save** — if `output_dir` is set: write tables (CSV, optionally LaTeX), plots, and `report.html`.

Sort metric is resolved from configs (or overridden via `sort_metric=` or `PipelineConfig`) and used to rank/select best config and to show "Metric used to select best results" in reports.

---

## 2. run_pipeline()

**Signature:**

```python
def run_pipeline(
    project_name: str = DEFAULT_PROJECT_NAME,
    user: str = DEFAULT_USER,
    csv_cache_dir: str = "csv_files",
    *,
    metric_prefixes: Optional[list] = None,
    reporting_metrics: Optional[list] = None,
    required_data_seeds: Optional[Set[int]] = None,
    required_model_seeds: Optional[Set[int]] = None,
    additional_ignored_cols: Optional[list] = None,
    preprocess_df: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    show_performance_tables: bool = False,
    show_plots: bool = True,
    precision: int = 4,
    output_dir: Optional[str] = None,
    export_laTeX: bool = False,
    plot_metric: str = "test/mse",
    sort_metric: Optional[str] = None,
    report_filename: Optional[str] = None,
    quiet: bool = False,
) -> Tuple[pd.DataFrame, defaultdict, pd.DataFrame]:
```

**Returns:** `(df, all_results, summary_df)` — raw DataFrame (after load and preprocess), nested analysis results, and the summary table DataFrame.

---

### Example 1: Full pipeline with default project and save to disk

```python
from picid_report.run import run_pipeline

df, all_results, summary_df = run_pipeline(
    project_name="29_01_2026_unibo_prognostics_combined",
    user="anonlab-buildingenergy-1",
    output_dir="report_output",
    export_laTeX=True,
)
# Tables and plots in report_output/tables/, report_output/plots/
# report_output/report.html contains the single-file report
# summary_df is the summary table (e.g. for further processing)
```

---

### Example 2: Quiet mode — no terminal tables, only files and report

```python
df, all_results, summary_df = run_pipeline(
    project_name="my_project",
    user="my_team",
    output_dir="out",
    quiet=True,
)
# Nothing printed to terminal except logs; out/report.html and out/tables/, out/plots/ are written
```

---

### Example 3: Custom preprocess — skip model-name cleaning

```python
# Use raw model names (no clean_and_rename_models)
df, all_results, summary_df = run_pipeline(
    project_name="my_project",
    user="my_team",
    preprocess_df=lambda df: df,
)
```

---

### Example 4: Custom preprocess — your own cleaning

```python
def my_preprocess(df):
    df = df.copy()
    df["model._target_"] = df["model._target_"].str.replace("old_prefix.", "", regex=False)
    return df

df, all_results, summary_df = run_pipeline(
    project_name="my_project",
    user="my_team",
    preprocess_df=my_preprocess,
)
```

---

### Example 5: Using PipelineConfig (custom column config / search space)

```python
from picid_report.run import run_pipeline
from picid_report.config import PipelineConfig

cfg = PipelineConfig.from_default()
cfg.column_config["model_target"] = "model.name"  # if your logs use model.name

df, all_results, summary_df = run_pipeline(
    project_name="my_project",
    user="my_team",
    pipeline_config=cfg,
)
```

---

### Example 6: Override sort metric globally or per (dataset, model)

```python
# Single metric for all
df, all_results, summary_df = run_pipeline(
    project_name="my_project",
    user="my_team",
    sort_metric="test/mae",
)

# Per (dataset, model): pass a dict (run_pipeline builds this from all_results if sort_metric is None)
# So normally you leave sort_metric=None and the pipeline fills it from sort_metric_used in all_results
```

---

### Example 7: No plots, no output dir — only in-memory tables

```python
df, all_results, summary_df = run_pipeline(
    project_name="my_project",
    user="my_team",
    show_plots=False,
    output_dir=None,
)
# summary_df and all_results available for your code; nothing saved
```

---

### Example 8: Custom report filename

```python
df, all_results, summary_df = run_pipeline(
    project_name="my_project",
    user="my_team",
    output_dir="out",
    report_filename="my_project_report.html",
)
# Report written to out/my_project_report.html
```

---

## 3. main() and CLI

`main()` is the entry point when the module is run as a script. It parses minimal CLI arguments and calls `run_pipeline()`.

**Usage:**

```bash
python -m picid_report.run [options]
```

**Arguments:**

| Argument | Short | Default | Description |
|----------|--------|---------|-------------|
| `--project` | | `29_01_2026_unibo_prognostics_combined` | W&B project name |
| `--user` | | `anonlab-buildingenergy-1` | W&B user/entity |
| `--cache-dir` | | `csv_files` | CSV cache directory |
| `--output-dir` | `-o` | None | Save tables, plots, and report here |
| `--report-name` | | None | Report HTML filename (e.g. `my_project.html`); default `report.html` |
| `--export-latex` | | False | Also save summary as LaTeX (when using `-o`) |
| `--no-plots` | | False | Skip building plots |
| `--performance-tables` | | False | Show per-metric performance tables |
| `--quiet` | `-q` | False | Do not print tables to terminal; only write report and files |
| `--debug` | `-d` | False | Enable DEBUG logging (resolver details, shapes, etc.) |

**Examples:**

```bash
# Default run (no save)
python -m picid_report.run

# Save everything to report_output/
python -m picid_report.run -o report_output

# Save with LaTeX and custom report name, quiet
python -m picid_report.run -o report_output --export-latex --report-name unibo_report.html -q

# Debug logging
python -m picid_report.run --debug -o report_output
```

---

## 4. Internal helpers (for reference)

- **`_safe_basename(dataset, model, max_len=80)`** — Sanitizes dataset and model strings for filenames (e.g. `hp_impact_UNIBO21_LSTM.png`).
- **`_save_outputs(...)`** — Writes tables (summary CSV/TeX, experiment stats, HP impact CSVs), plots (best-metric bars, HP impact per dataset/model), and calls `write_report_html()`.
- **`_run_plots(...)`** — Builds default plots in-memory when `output_dir` is not set but `show_plots` is True (one bar chart + one HP impact example).

These are used by `run_pipeline()`; you typically do not call them directly.

---

**Navigation:** [← Documentation index](README.md)
