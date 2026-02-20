# report_html.py — Single HTML report

**Navigation:** [← Documentation index](README.md)

**Module:** `picid_report.report.report_html`

This document describes **write_report_html()**, which produces a single HTML file that embeds experiment statistics, the summary table, optional plots, and per (dataset, model) HP impact tables with "Metric used to select best results".

---

## 1. write_report_html()

**Signature:**

```python
def write_report_html(
    output_dir: str,
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    hp_impact_entries: List[Tuple[str, str, pd.DataFrame, str, str]],
    global_plots: List[Tuple[str, str]],
    title: str = "Experiment Report",
    all_results: Optional[defaultdict] = None,
    sort_metric: Optional[str] = None,
    report_filename: str = "report.html",
) -> str:
```

**Parameters:** output_dir (directory for report and relative paths to tables/plots), summary_df (from create_summary_table), stats_df (from get_experiment_stats_df), hp_impact_entries (list of (dataset, model, df, plot_rel_path, metric_used)), global_plots (list of (label, rel_path)), title, all_results (optional, for metric display), sort_metric (optional, string or dict), report_filename (default "report.html").

**Returns:** Path to the written report file.

**Sections in the report:** (1) Experiment statistics, (2) Summary table with optional "Metric used to select best results", (3) Plots (global_plots), (4) Hyperparameter impact per dataset/model.

---

### Example 1: Minimal (tables only)

```python
from picid_report.report_html import write_report_html
from picid_report import create_summary_table
from picid_report.reporting import get_experiment_stats_df

summary_df = create_summary_table(all_results, precision=4)
stats_df = get_experiment_stats_df(all_results)
path = write_report_html(
    output_dir="report_output",
    summary_df=summary_df,
    stats_df=stats_df,
    hp_impact_entries=[],
    global_plots=[],
    title="My Report",
    report_filename="report.html",
)
```

---

### Example 2: Full report (with HP entries and plots)

Build hp_impact_entries from iter_hp_impact_tables and global_plots from plot paths. Pass all_results and sort_metric so the report shows "Metric used to select best results". See run.py _save_outputs() for the exact construction.

---

### Example 3: Custom report filename

```python
path = write_report_html(..., output_dir="out", report_filename="unibo_feb2026.html")
# Writes out/unibo_feb2026.html
```

---

## 2. Internal helper

**_dataframe_to_html_table(df, max_rows=100)** - Renders DataFrame as HTML table; empty returns "<p>No data.</p>". Used for stats, summary, and each HP table.

---

**Navigation:** [← Documentation index](README.md)
