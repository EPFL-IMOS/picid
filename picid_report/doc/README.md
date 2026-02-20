# picid_report — Documentation Index

This folder documents the **picid_report** package: pipeline for loading W&B experiment runs, validating and analyzing them by dataset/model, and producing tables, plots, and a single HTML report. Each document focuses on one area and includes **concrete examples** for every function.

**Navigation:** Each document in the table below links to its page; every other doc has a **← Documentation index** link at the top (and often at the bottom) to return here.

---

## 1. Pipeline overview and how parts interplay

**Flow:** Load → Preprocess → Validate → Analyze → Report → (optional) Save.

| Stage | Module | Purpose |
|--------|--------|--------|
| **Load** | `core/run_processor` | Fetch runs from W&B (or CSV cache), merge summary+config, flatten nested columns, drop noisy columns. Output: flat DataFrame + list of config column names. |
| **Preprocess** | `core/preprocess` | Optionally clean model names (strip prefix, add "(linear)" / "(exponential)"). |
| **Validate** | `core/validators` | Ensure required columns exist; optionally filter to groups with all required seeds; warn on hidden variation. |
| **Analyze** | `core/analysis` | For each (dataset, model): discover metrics, resolve optimization metric/mode, detect varying HPs, aggregate runs (mean/std/count), resolve grid (config or auto-discovery), pick best run, resolve sort metric. Output: `all_results[dataset][model]` with best_hyperparameters, sorted_aggregated_results, sort_metric_used, etc. |
| **Report** | `report/reporting` | Build summary table, experiment stats, HP impact tables; display and/or export to CSV/LaTeX. |
| **Save** | `run` + `report/report_html` | When `output_dir` is set: write tables, plots, and a single `report.html`. |

**Configuration** drives behavior at every stage: column names and required columns (`config.py`), which columns to normalize/drop/ignore for HP search (`config.py`), expected HP grid per dataset/model (`configs/search_space.py`), and which metric to use for sorting/ranking (`configs/sort_metrics.py`). See [Configuration (config and configs)](config.md) for in-depth logic.

**Entry points:**
- **Script:** `python -m picid_report.run` (or `run.sh`) runs the full pipeline; use `-o <dir>` to save outputs, `-q` to avoid printing tables to the terminal.
- **Programmatic:** `run_pipeline(...)` returns `(df, all_results, summary_df)`; you can also call `load_runs_df`, `analyze_results`, and reporting functions step by step.

---

## 2. Document map

| Document | Contents |
|----------|----------|
| **[config.md](config.md)** | **In-depth:** `config.py` (column names, filters, required columns, legacy search space, `PipelineConfig`) and `configs/` (search space dict shapes, resolution order, sort metrics hierarchy). Many examples for every constant and function. |
| **[run.md](run.md)** | `run.py`: `run_pipeline()`, `main()`, CLI, stage flow. Examples: full pipeline, quiet mode, custom preprocess, PipelineConfig. |
| **[run_processor.md](run_processor.md)** | `core/run_processor.py`: `load_runs_df()`. Examples: cache vs fetch, pipeline_config override. |
| **[analysis.md](analysis.md)** | `core/analysis.py`: `analyze_results()` and helpers. Examples: schema-first vs data-first, sort_metric_resolver. |
| **[reporting.md](reporting.md)** | `report/reporting.py`: summary table, experiment stats, HP impact (create, display, export, LaTeX). Examples for every public function. |
| **[preprocess.md](preprocess.md)** | `core/preprocess.py`: `clean_and_rename_models()`. Examples: default columns, custom columns. |
| **[validators.md](validators.md)** | `core/validators.py`: `validate_schema`, `validate_seeds`, `log_modification`, `check_hidden_variations`. Examples for each. |
| **[report_html.md](report_html.md)** | `report/report_html.py`: `write_report_html()`. Example: minimal and full call. |
| **[utils.md](utils.md)** | `utils.py`: `format_mean_std_count`, `flatten_aggregated_columns`. Examples. |
| **[logging_config.md](logging_config.md)** | `logging_config.py`: `configure_logging(debug=...)`. INFO vs DEBUG, CLI `--debug`. |
| **[plots.md](plots.md)** | `report/plots.py`: `plot_best_metric_bars`, `plot_hp_impact`. Examples. |

---

## 3. Quick example (end-to-end)

```python
from picid_report.run import run_pipeline

# Run full pipeline and save report to report_output/
df, all_results, summary_df = run_pipeline(
    project_name="my_wandb_project",
    user="my_team",
    output_dir="report_output",
    export_laTeX=True,
    quiet=True,
)

# all_results[dataset][model] contains:
# - best_hyperparameters, best_performance
# - sorted_aggregated_results (DataFrame)
# - sort_metric_used, seeds_info, etc.
```

For step-by-step and per-function examples, use the documents above.
