# lmetk_report: Improvement & Refactoring Plan

**Scope:** All code in `lmetk_report/` excluding `not_used/`. `table.ipynb` is not modified; it is the reference for current pipeline usage.  
**Constraint:** All changes must keep `test/lmetk_report` passing and preserve the current `all_results` contract and pipeline entry points.  
**Goal:** Improve software quality and enable reliable generation of **tables and plots for papers**.

---

## Implementation status

| Priority | Status   | Notes |
|----------|----------|--------|
| **P0**   | **Done** | Typo and iterrows fix in `run_processor.py`; bare `except` → `except Exception`, top-level `ast`/`logging` and module `logger` in `analysis.py`. |
| **P1**   | **Done** | `lmetk_report/utils.py` added with `flatten_aggregated_columns()`; used in `analysis.py` and `reporting.py`. Guard for missing `opt_col_flattened` in `analysis.py`. |
| **P2**   | **Done** | Logging abstraction (logger in analysis, reporting, validators, run_processor); package logger in `__init__.py`; `all_results` schema documented in `analyze_results` docstring. |
| **P3**   | **Done** | Optional `column_config` / `expected_search_space` in `analyze_results`; `export_summary_table(path, format=csv|latex)`; `format_mean_std_count()` in `utils.py` used in reporting. |
| **P4**   | **Done** | Plot module (`plots.py`); public API in `__init__.py`; `PipelineConfig` dataclass in `config.py` with `from_default()`; pass-through in `analyze_results` and `load_runs_df`. |

---

## 1. Current Pipeline (Reference)

As used in `table.ipynb`:

1. **Load:** `load_runs_df(project_name, user)` → `(df, config_columns, dropped_columns)`
2. **Optional:** `clean_and_rename_models(df)` (not in lmetk_report; notebook-only)
3. **Validate:** `validators.validate_schema(df, config.REQUIRED_COLUMNS)`
4. **Analyze:** `analyze_results(df, config_columns, dropped_columns, reporting_metrics, metric_prefixes, required_*_seeds, additional_ignored_cols)` → `all_results`
5. **Report:** `display_experiment_stats(all_results)`, `display_hp_impact(all_results)`, `create_summary_table(all_results)` (+ optional `display_performance_tables`)

**Contract:** `all_results[dataset][model]` contains at least: `best_hyperparameters`, `best_performance` (with `metrics` and `optimized_on`), `sorted_aggregated_results`, `non_aggregated_df`, `seeds_info`.

---

## 2. Critical Issues (Software Engineering)

### 2.1 Observability and Side Effects

- **Problem:** Heavy use of `print()` for status, warnings, and validators. No log levels, no redirection, hard to reuse in scripts or headless export.
- **Impact:** Noisy in notebooks; difficult to turn off or capture when generating many tables/plots for papers.
- **Improvement:**
  - Introduce a small **logging abstraction**: e.g. a module-level `logger` (or optional callback) used instead of `print` in `analysis.py`, `reporting.py`, `validators.py`, `run_processor.py`.
  - Keep default behavior: log to stdout so current notebook output and tests remain unchanged.
  - Optional: `verbose: bool = True` or `log_level` on main entry points (`analyze_results`, `load_runs_df`) to reduce noise when batch-exporting.

### 2.2 Configuration Coupling

- **Problem:** `analysis.py` and others import `config` and read `COLUMN_CONFIG`, `EXPECTED_SEARCH_SPACE`, `COLUMN_FILTERS_*`, etc. directly. Adding a new project or paper variant requires editing `config.py` or mutating global state (e.g. `patch(EXPECTED_SEARCH_SPACE)` in tests).
- **Impact:** Hard to run multiple analyses with different column names or grids (e.g. different benchmarks) without code changes or fragile patching.
- **Improvement:**
  - **Phase 1 (non-breaking):** Add optional parameters to `analyze_results(..., column_config=None, expected_search_space=None)`. When `None`, use `config.*` as today. Tests and notebook can keep current behavior.
  - **Phase 2:** Introduce a small `PipelineConfig` (or similar) dataclass holding column names, filters, and expected grids; pass it through `analyze_results` and optionally `load_runs_df`. Keep `config.py` as the default source for that object.

### 2.3 Type and Data Contract

- **Problem:** `all_results` is `defaultdict(lambda: defaultdict(dict))` with informally defined nested keys. Reporting code assumes specific keys (`best_performance.metrics`, `sorted_aggregated_results`, etc.) without a single documented schema.
- **Impact:** Extensions (e.g. new exporters or plots) may break when keys are missing or renamed; refactoring is risky.
- **Improvement:**
  - Document the **canonical structure** of `all_results` in a single place (e.g. `docs/ or README` in lmetk_report, or docstring of `analyze_results`): dataset → model → `best_hyperparameters`, `best_performance`, `sorted_aggregated_results`, `non_aggregated_df`, `seeds_info`, with types (dict, DataFrame, etc.).
  - Optionally introduce **TypedDict** (or a small dataclass) for the per-model result entry so IDEs and type checkers help; keep the same runtime structure so tests and notebook need no change.

### 2.4 Bare `except` and Imports Inside Functions — **Done**

- **Problem:** `analysis.get_dynamic_metrics` used `except:` and `import ast` / `import logging` inside functions.
- **Improvement:** Use `except Exception:`; move `ast` and `logging` to top-level imports; use module-level `logger` in `analysis.py` for the varying-hyperparams warning. No change to function signatures or return values.

### 2.5 Duplicated Column-Flattening Logic — **Done**

- **Problem:** MultiIndex → `f"{col_name}_{stat}"` flattening appeared in both `analysis.py` and `reporting.display_hp_impact`.
- **Improvement:** Added `lmetk_report/utils.py` with `flatten_aggregated_columns(df) -> pd.DataFrame`; both `analysis.py` and `reporting.py` use it. Column names and behavior unchanged.

### 2.6 Run Processor: Typo and In-Place Mutation — **Done**

- **Problem:** "drropped" typo; `row["summary"].update(row["config"])` mutated rows while iterating with `iterrows()`.
- **Improvement:** Typo fixed; merged records are built as `[{**row["summary"], **row["config"], "run_name": row["run_name"]} for ...]` then `pd.DataFrame.from_records(...)`. Return type and column semantics unchanged.

### 2.7 Empty / Edge Paths — **Done**

- **Problem:** When `all_performance_cols` is empty or optimization column is missing, the pipeline can still call `sort_values(by=opt_col_flattened)` on a DataFrame that doesn’t have that column, leading to KeyError (observed in tests before adding a guard).
- **Improvement:** After building `actual_results` and `opt_col_flattened`, if `opt_col_flattened` not in `actual_results.columns`, skip appending to `all_results` for that (dataset, model) (same as when `best_result.empty`), and optionally log once. This avoids crashes and keeps the contract consistent; tests that expect “no entry” for that model/dataset remain valid. Implemented in P1 (guard in `analysis.py`).

---

## 3. Usability for Paper-Ready Output

### 3.1 Export to LaTeX / CSV (Tables)

- **Gap:** `create_summary_table` returns a DataFrame and the notebook uses `display()`. There is no built-in way to export that table (or HP impact tables) to LaTeX or CSV for inclusion in papers.
- **Improvement (additive, backward compatible):**
  - Add `export_summary_table(all_results, path, format="csv" | "latex", precision=4)` that builds the same DataFrame as `create_summary_table`, then writes:
    - CSV: `df.to_csv(path, index=True)` (or index=False depending on desired layout).
    - LaTeX: `df.to_latex(path, ...)` with sensible defaults (escape=False, column_format, caption/label optional).
  - Optionally add `export_hp_impact_tables(all_results, output_dir, format=...)` to export per-model/dataset HP tables. Implementation can reuse the same logic as `display_hp_impact` but write to files instead of `display()`. No change to `all_results` or to existing display functions.

### 3.2 Plots (New Capability)

- **Gap:** Package has no plotting. For papers, users need plots (e.g. bar charts of metrics, HP sensitivity, learning curves).
- **Improvement (additive):**
  - Add a small `plots` submodule (or `reporting/plots.py`) with functions that take `all_results` (and optionally `PipelineConfig`) and produce paper-ready figures:
    - **Bar chart of best metric per model/dataset:** e.g. `plot_best_metric_bars(all_results, metric="test/mse", save_path=...)`.
    - **HP impact heatmap or strip:** e.g. `plot_hp_impact(all_results, model, dataset, metric, save_path=...)` using `sorted_aggregated_results`.
  - Use a single optional dependency (e.g. `matplotlib` or `matplotlib + seaborn`) and document it; if absent, raise a clear error when plot functions are called. Do not change `analyze_results` or the structure of `all_results`; only read from it.

### 3.3 Configurable Precision and Formatting

- **Current:** `precision=4` (or 2) is passed to reporting; formatting of "mean ± std (n=count)" is hardcoded in several places.
- **Improvement:** Centralize the format string (e.g. `"{mean:.{precision}f} ± {std:.{precision}f} (n={count})"`) in one helper and use it in `create_summary_table`, `display_performance_tables`, and `display_hp_impact`. Allow optional custom formatter or precision per call. Same outputs for same inputs so tests remain valid.

---

## 4. Extensibility and Maintainability

### 4.1 Public API Surface

- **Problem:** `__init__.py` is empty; callers import from `run_processor`, `analysis`, `reporting`, `validators`, `config` directly. No clear “public” API.
- **Improvement:** Optionally expose a minimal public API in `__init__.py`, e.g.:
  - `load_runs_df`, `analyze_results`, `create_summary_table`, `display_experiment_stats`, `display_performance_tables`, `display_hp_impact`, `validate_schema`, and default `config`.
  - Internal helpers (e.g. `get_search_grid_df`, `aggregate_and_find_best`, `_trace_nans`) stay in their modules. This does not change behavior; only import paths could be simplified for new users.

### 4.2 Adding a New Metric or Display

- **Current:** New metrics are picked up automatically if they appear in data and match `metric_prefixes` / `reporting_metrics`. New display formats require editing `reporting.py` or the notebook.
- **Improvement:** Keep automatic metric discovery. For new output formats (e.g. another table style or plot type), add new functions that consume `all_results` rather than changing the structure of `all_results`. Document that `all_results` is the single “analysis result” object and that reporting/plotting are pure consumers.

### 4.3 Tests and Regression

- **Current:** `test/lmetk_report` covers analysis, reporting, validators, run_processor with mocks and fixtures; pipeline contract is implicitly tested via structure of `all_results`.
- **Improvement:**
  - Add a single **integration test** that runs the full pipeline (load from a tiny cached CSV or in-memory DataFrame, then `analyze_results` → `create_summary_table` / `display_hp_impact`) and asserts on the presence and shape of `all_results` keys and one summary table. Use the same `config_columns` / `dropped_columns` pattern as the notebook. This guards refactors and new features.
  - When adding exporters (e.g. LaTeX) or plot functions, add tests that build minimal `all_results`, call the new function, and assert output file exists and contains expected snippets (e.g. `\\begin{tabular}` or a known number of rows).

---

## 5. Implementation Priority (Backward Compatible)

| Priority | Item | Status | Risk to tests / notebook | Effort |
|----------|------|--------|----------------------------|--------|
| P0 | Fix typo "drropped" and iterrows mutation in run_processor | **Done** | None | Low |
| P0 | Replace bare `except` and move imports to top in analysis | **Done** | None | Low |
| P1 | Extract shared `flatten_aggregated_columns` and use in analysis + reporting | **Done** | Low (same column names) | Low |
| P1 | Guard missing `opt_col_flattened` to avoid KeyError when no metrics | **Done** | None | Low |
| P2 | Introduce logging abstraction; keep default = current prints | **Done** | None if default behavior unchanged | Medium |
| P2 | Document `all_results` schema (and optionally add TypedDict) | **Done** | None | Low |
| P3 | Optional `column_config` / `expected_search_space` override in `analyze_results` | **Done** | None if default = config | Medium |
| P3 | `export_summary_table(..., format="csv"\|"latex")` | **Done** | None | Low |
| P3 | Centralize "mean ± std (n=count)" formatter | **Done** | None (same strings) | Low |
| P4 | `PipelineConfig` dataclass and pass-through | **Done** | Low if default from config | Medium |
| P4 | Plot helpers (bars, HP impact) in a new module | **Done** | None | Medium |
| P4 | Public API in `__init__.py` | **Done** | None | Low |

---

## 6. Summary

- **Do not change:** The shape and semantics of `all_results`, the signatures and default behavior of `analyze_results`, `load_runs_df`, and the existing reporting functions, so that `table.ipynb` and `test/lmetk_report` keep working.
- **Improve robustness:** Logging abstraction (P2), config override (P3), edge-case guards (P1 done), no bare `except` (P0 done), single place for flattening (P1 done) and formatting (P3).
- **Add for papers:** Export summary (and optionally HP) tables to CSV/LaTeX (P3); add a small plot module that reads `all_results` and writes figures (P4). All of this can be additive and tested without altering the current pipeline contract.

**Completed (P0–P4):** Run processor typo and iterrows fix; analysis bare-except and top-level imports; shared `flatten_aggregated_columns` in `utils.py`; guard for missing `opt_col_flattened`. P2: logging abstraction and `all_results` schema doc. P3: config override, `export_summary_table`, `format_mean_std_count`. P4: `plots.py`, public API in `__init__.py`, and `PipelineConfig` dataclass with `PipelineConfig.from_default()`, optional `pipeline_config` in `analyze_results` and `load_runs_df`. All improvements backward compatible; no test changes required.
