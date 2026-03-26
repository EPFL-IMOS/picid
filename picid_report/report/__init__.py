# picid_report/report: tables, HTML report, plots

import picid_report.report.reporting as reporting
import picid_report.report.report_html as report_html
from picid_report.report.reporting import (
    create_summary_table,
    display_experiment_stats,
    display_hp_impact,
    display_performance_tables,
    export_summary_table,
    get_experiment_stats_df,
    iter_hp_impact_tables,
    export_experiment_stats,
    export_hp_impact_tables,
    write_tables_tex,
)
from picid_report.report.report_html import write_report_html
from picid_report.report.plots import plot_best_metric_bars, plot_hp_impact, plot_summary

__all__ = [
    "reporting",
    "report_html",
    "create_summary_table",
    "display_experiment_stats",
    "display_hp_impact",
    "display_performance_tables",
    "export_summary_table",
    "get_experiment_stats_df",
    "iter_hp_impact_tables",
    "export_experiment_stats",
    "export_hp_impact_tables",
    "write_tables_tex",
    "write_report_html",
    "plot_best_metric_bars",
    "plot_hp_impact",
    "plot_summary",
]
