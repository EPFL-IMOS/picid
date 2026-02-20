# picid_report — Folder structure

The package uses a **function-based layout**: **core/** (pipeline engine), **report/** (output), **configs/** (dataset/model overrides), and root for entry point and shared modules. The public API is unchanged (`from picid_report import ...`).

---

## Current layout (implemented)

```
picid_report/
  __init__.py              # re-exports from subpackages → same public API
  run.py                   # entry point (python -m picid_report.run)
  run.sh, run_all_projects.sh, projects.sh
  config.py                # column/schema defaults, PipelineConfig
  configs/                 # dataset/model overrides (unchanged)
    __init__.py
    search_space.py
    sort_metrics.py
  core/                    # pipeline engine: load → preprocess → validate → analyze
    __init__.py
    run_processor.py
    preprocess.py
    validators.py
    analysis.py
  report/                  # output: tables, HTML, plots
    __init__.py
    reporting.py
    report_html.py
    plots.py
  utils.py                 # shared helpers
  logging_config.py        # startup logging
  doc/
  not_used/
  table.ipynb
```

---

## Rationale

| Folder / file | Role |
|---------------|------|
| **run.py** (root) | Single entry for CLI and `run_pipeline()`; stays at root so `python -m picid_report.run` is unchanged. |
| **config.py** + **configs/** | Configuration; no change. |
| **core/** | “Engine”: load W&B data, preprocess, validate, analyze. One place for the data → `all_results` flow. |
| **report/** | Everything that turns `all_results` into output: summary/stats/HP tables, HTML report, plots. |
| **utils.py**, **logging_config.py** (root) | Shared and startup; small, stay at root. |

---

## Public API (unchanged)

`__init__.py` will import from the new locations and re-export. Callers keep using:

```python
from picid_report import (
    config,
    PipelineConfig,
    load_runs_df,
    clean_and_rename_models,
    validate_schema,
    analyze_results,
    create_summary_table,
    display_experiment_stats,
    display_hp_impact,
    display_performance_tables,
    export_summary_table,
    plot_best_metric_bars,
    plot_hp_impact,
)
```

So: **no breaking changes** for code that uses `picid_report` as above.

---

## Import changes (internal only)

| From | To |
|------|----|
| `from picid_report.run_processor import load_runs_df` | `from picid_report.core.run_processor import load_runs_df` |
| `from picid_report.preprocess import clean_and_rename_models` | `from picid_report.core.preprocess import clean_and_rename_models` |
| `from picid_report.validators import validate_schema` | `from picid_report.core.validators import validate_schema` |
| `from picid_report.analysis import analyze_results` | `from picid_report.core.analysis import analyze_results` |
| `from picid_report.reporting import ...` | `from picid_report.report.reporting import ...` |
| `from picid_report.report_html import write_report_html` | `from picid_report.report.report_html import write_report_html` |
| `from picid_report.plots import ...` | `from picid_report.report.plots import ...` |

Within **core/** and **report/**, modules will use relative or package-qualified imports (e.g. `from picid_report.config import ...`, `from picid_report.utils import ...`) as they do today; only the path to the moved modules changes.

---

## Optional: even flatter “report”

If you prefer fewer folders, **report/** could be skipped and only **core/** introduced:

- Keep **reporting.py**, **report_html.py**, **plots.py** at root.
- Move only **run_processor**, **preprocess**, **validators**, **analysis** into **core/**.

That still groups the “engine” and leaves output modules at root.

---

## Summary

- **Recommended:** Add **core/** and **report/** as above; keep **config.py** and **configs/** as is; keep **run.py**, **utils.py**, **logging_config.py** at root.
- **Benefit:** Clear split between “pipeline engine” (core) and “output” (report); fewer top-level files.
- **Cost:** One-time refactor of internal imports and `__init__.py`; doc links to module paths can be updated in passing.

This structure is **implemented**. Internal imports use `picid_report.core.*` and `picid_report.report.*`; the root `__init__.py` re-exports so `from picid_report import load_runs_df`, `analyze_results`, etc. still works.
