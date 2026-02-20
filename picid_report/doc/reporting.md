# reporting.py — Summary tables, stats, and HP impact

**Navigation:** [← Documentation index](README.md)

**Module:** `picid_report.report.reporting`

This document describes every **reporting** function: summary table (create, export), experiment stats (get, display, export), HP impact (iterate, display, export), and LaTeX writing. **sort_metric** can be a single metric name or a dict `(dataset, model) -> metric` for per-combination control.

---

## 1. Summary table

### create_summary_table(all_results, precision=4, sort_metric=None)

Builds a single pivoted DataFrame: models as rows, (Dataset, Metric) as columns, values = formatted "mean ± std (n=count)". If **sort_metric** is provided (or dict per (dataset, model)), the best run is **re-selected** by that metric when it differs from the optimization metric.

**Example — use optimization metric (default):**
```python
from picid_report import create_summary_table

summary_df = create_summary_table(all_results, precision=4)
# summary_df.index = models, summary_df.columns = MultiIndex (Dataset, Metric)
```

**Example — re-select best by a single sort metric:**
```python
summary_df = create_summary_table(all_results, precision=4, sort_metric="val_best_rerun/loss")
```

**Example — per (dataset, model) sort metric (e.g. from pipeline):**
```python
sort_dict = {(d, m): res.get("sort_metric_used") for d, models in all_results.items() for m, res in models.items()}
sort_dict = {k: v for k, v in sort_dict.items() if v}
summary_df = create_summary_table(all_results, precision=4, sort_metric=sort_dict)
```

---

### export_summary_table(all_results, path, format="csv", precision=4, sort_metric=None)

Builds the summary table and writes it to **path** as CSV or LaTeX.

**Example:**
```python
from picid_report import export_summary_table

export_summary_table(all_results, "tables/summary.csv", format="csv", precision=4)
export_summary_table(all_results, "tables/summary.tex", format="latex", precision=4, sort_metric="test/mse")
```

---

## 2. Experiment stats

### get_experiment_stats_df(all_results)

Builds a DataFrame with one row per (dataset, model): Dataset, Model, Configs Completed, Runs, Configs Failed (not full seed set), Seeds Found. Returns empty DataFrame if no data.

**Example:**
```python
from picid_report.reporting import get_experiment_stats_df

stats_df = get_experiment_stats_df(all_results)
# stats_df columns: Dataset, Model, Configs Completed, Runs, Configs Failed..., Seeds Found
```

---

### display_experiment_stats(all_results)

Prints the experiment stats table (using `get_experiment_stats_df` + IPython display or logger).

**Example:**
```python
from picid_report import display_experiment_stats

display_experiment_stats(all_results)
```

---

### export_experiment_stats(all_results, path)

Writes the experiment stats table to CSV.

**Example:**
```python
from picid_report.reporting import export_experiment_stats

export_experiment_stats(all_results, "tables/experiment_stats.csv")
```

---

## 3. Performance tables (per-metric pivot)

### display_performance_tables(all_results, precision=4)

Displays one pivot table per metric (e.g. val/loss, test/mse): rows = Model, columns = Dataset, values = formatted mean ± std (n=count).

**Example:**
```python
from picid_report import display_performance_tables

display_performance_tables(all_results, precision=4)
```

---

## 4. HP impact

### iter_hp_impact_tables(all_results, precision=4, sort_metric=None)

Yields `(dataset, model, df, metric_used)` for each non-empty HP impact table. **df** is the formatted DataFrame (HP columns + metric columns as "mean ± std (n=count)"); **sort_metric** can be a string or dict `(dataset, model) -> metric`.

**Example:**
```python
from picid_report.reporting import iter_hp_impact_tables

for dataset, model, hp_df, metric_used in iter_hp_impact_tables(all_results, precision=4):
    print(dataset, model, metric_used)
    # hp_df: HP cols + metric columns (e.g. val_best_rerun/loss, test/mse)
```

**Example — with per-combo sort metric:**
```python
sort_dict = {(d, m): res.get("sort_metric_used") for d, models in all_results.items() for m, res in models.items()}
sort_dict = {k: v for k, v in sort_dict.items() if v}
for dataset, model, hp_df, metric_used in iter_hp_impact_tables(all_results, precision=4, sort_metric=sort_dict):
    ...
```

---

### display_hp_impact(all_results, precision=4, sort_metric=None)

Displays every HP impact table (one per dataset/model) with a header showing the metric used for sorting.

**Example:**
```python
from picid_report import display_hp_impact

display_hp_impact(all_results, precision=4)
```

---

### export_hp_impact_tables(all_results, output_dir, precision=4, sort_metric=None)

Saves one CSV per dataset/model HP impact table in **output_dir**. Filenames are sanitized (e.g. `hp_impact_UNIBO21_LSTM.csv`). Returns list of `(dataset, model, path)`.

**Example:**
```python
from picid_report.reporting import export_hp_impact_tables

paths = export_hp_impact_tables(all_results, "tables/hp_impact", precision=4)
for dataset, model, path in paths:
    print(f"Saved {path}")
```

---

## 5. LaTeX (single file for paper)

### write_tables_tex(path, summary_df, stats_df, all_results, precision=4, sort_metric=None)

Writes one LaTeX file containing: summary table, experiment stats table, then one HP impact table per dataset/model. Adds comments for “metric used to select best results”. Requires `\usepackage{booktabs}`.

**Example:**
```python
from picid_report.reporting import write_tables_tex, get_experiment_stats_df
from picid_report import create_summary_table

summary_df = create_summary_table(all_results, precision=4, sort_metric=sort_metric)
stats_df = get_experiment_stats_df(all_results)
write_tables_tex(
    "output/tables_tex.tex",
    summary_df=summary_df,
    stats_df=stats_df,
    all_results=all_results,
    precision=4,
    sort_metric=sort_metric,
)
```

---

## 6. Internal helper

- **\_infer_metric_mode(metric_name)** — Returns `"min"` or `"max"` from metric name (e.g. loss/mse → min, accuracy/f1 → max). Used when re-sorting by sort_metric.
- **\_build_hp_impact_df(res, precision, sort_metric)** — Builds one HP impact DataFrame for a single result dict; returns `(df, metric_used)` or None. Used by `iter_hp_impact_tables`.

You typically call the public functions above; the pipeline uses them in `run.py` when building and saving reports.

---

**Navigation:** [← Documentation index](README.md)
