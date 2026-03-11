# picid_report/report/report_html.py
"""
Generate a single HTML report for experiment results.

write_report_html: produces report.html in output_dir containing the summary table,
experiment stats, per (dataset, model) HP impact tables, and optional plot images.
Accepts sort_metric / all_results to display "Metric used to select best results"
in the summary and HP sections.
"""

import html
import logging
import os
from collections import defaultdict
from typing import List, Tuple, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Render a DataFrame as an HTML table (escape text)."""
    if df.empty:
        return "<p>No data.</p>"
    sub = df.head(max_rows)
    return sub.to_html(index=True, border=1, classes=["report-table"], escape=True)


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
    """
    Write a single HTML report file that embeds all tables and plot images.

    Parameters
    ----------
    output_dir : str
        Directory containing tables/ and plots/; report will be output_dir/report_filename.
    report_filename : str
        Name of the HTML file (default "report.html").
    summary_df : pd.DataFrame
        Main summary table (from create_summary_table).
    stats_df : pd.DataFrame
        Experiment stats (from get_experiment_stats_df).
    hp_impact_entries : list of (dataset, model, df, plot_rel_path, metric_used)
        Each HP impact table and its corresponding plot path (relative to output_dir),
        plus the metric used for sorting.
    global_plots : list of (label, rel_path)
        e.g. [("Best metric (test/mse)", "plots/best_metric_bars.png")].
    title : str
        Report title.
    all_results : defaultdict, optional
        Result of analyze_results (dataset -> model -> result dict).
        Used to extract metric information for display.
    sort_metric : str, optional
        Metric used for sorting/selecting. If None, uses optimization metric.
        Can also be a dict mapping (dataset, model) -> metric.

    Returns
    -------
    path : str
        Path to the written report file.
    """
    report_path = os.path.join(output_dir, report_filename)

    sections = []

    # Inline CSS for readable tables and layout
    sections.append(
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>"""
        + html.escape(title)
        + """</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1rem 2rem; max-width: 1400px; }
  h1 { border-bottom: 2px solid #333; }
  h2 { margin-top: 2rem; color: #444; }
  .report-table { border-collapse: collapse; width: 100%; margin: 0.5rem 0; font-size: 0.9rem; }
  .report-table th, .report-table td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
  .report-table th { background: #f0f0f0; }
  .report-table tr:nth-child(even) { background: #fafafa; }
  img { max-width: 100%; height: auto; border: 1px solid #ddd; margin: 0.5rem 0; }
  .plot-caption { font-style: italic; color: #555; margin-bottom: 1rem; }
  .hp-section { margin-bottom: 2rem; padding-bottom: 2rem; border-bottom: 1px solid #eee; }
</style>
</head>
<body>
<h1>"""
        + html.escape(title)
        + """</h1>
"""
    )

    # 1. Experiment statistics
    sections.append("<h2>1. Experiment statistics</h2>\n")
    sections.append(_dataframe_to_html_table(stats_df))

    # 2. Summary table
    sections.append("<h2>2. Summary table (best metric by model/dataset)</h2>\n")

    # Extract and display metric information
    if all_results is not None:
        # Collect metrics used for each model/dataset
        metrics_used = set()
        for dataset, models in all_results.items():
            for model, res in models.items():
                if isinstance(sort_metric, dict):
                    combo_sort_metric = sort_metric.get((dataset, model), None)
                else:
                    combo_sort_metric = sort_metric

                if combo_sort_metric is not None:
                    metrics_used.add(combo_sort_metric)
                else:
                    opt_info = res.get("best_performance", {}).get("optimized_on", {})
                    opt_metric = opt_info.get("metric", "")
                    if opt_metric:
                        metrics_used.add(opt_metric)

        if metrics_used:
            if len(metrics_used) == 1:
                metric_display = list(metrics_used)[0]
                sections.append(
                    f"<p><strong>Metric used to select best results:</strong> {html.escape(metric_display)}</p>\n"
                )
            else:
                metrics_list = ", ".join(sorted(metrics_used))
                sections.append(
                    f"<p><strong>Metric used to select best results:</strong> {html.escape(metrics_list)} (varies by model/dataset)</p>\n"
                )

    sections.append(_dataframe_to_html_table(summary_df))

    # 3. Global plots (e.g. best metric bars)
    if global_plots:
        sections.append("<h2>3. Plots</h2>\n")
        for label, rel_path in global_plots:
            path = os.path.join(output_dir, rel_path)
            if os.path.isfile(path):
                sections.append(f'<p class="plot-caption">{html.escape(label)}</p>\n')
                sections.append(
                    f'<img src="{html.escape(rel_path)}" alt="{html.escape(label)}" />\n'
                )

    # 4. HP impact (table + plot per model/dataset)
    sections.append("<h2>4. Hyperparameter impact</h2>\n")
    for dataset, model, df, plot_rel_path, metric_used in hp_impact_entries:
        safe_label = f"{dataset} / {model}"
        sections.append(f'<div class="hp-section"><h3>{html.escape(safe_label)}</h3>\n')
        sections.append(
            f"<p><strong>Dataset name:</strong> {html.escape(dataset)}, "
            f"<strong>Model name:</strong> {html.escape(model)}, "
            f"<strong>Metric used to select best results:</strong> {html.escape(metric_used)}</p>\n"
        )
        path = os.path.join(output_dir, plot_rel_path)
        if os.path.isfile(path):
            sections.append(
                f'<img src="{html.escape(plot_rel_path)}" alt="{html.escape(safe_label)}" />\n'
            )
        sections.append(_dataframe_to_html_table(df))
        sections.append("</div>\n")

    sections.append("</body>\n</html>")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        "Writing report to %s (%d HP impact sections)",
        report_path,
        len(hp_impact_entries),
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(sections))

    return report_path
