# Optional: configure package logger to emit to stdout by default (print-like behavior).
# Callers can reconfigure logging or attach their own handlers.
import logging
import sys

from picid_report import config
from picid_report.config import PipelineConfig
from picid_report.core import (
    load_runs_df,
    clean_and_rename_models,
    validate_schema,
    analyze_results,
)
from picid_report.report import (
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
    write_report_html,
    plot_best_metric_bars,
    plot_hp_impact,
    plot_summary,
)

_logger = logging.getLogger("picid_report")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.setLevel(logging.INFO)
    _logger.addHandler(_handler)

__all__ = [
    "analyze_results",
    "clean_and_rename_models",
    "config",
    "create_summary_table",
    "PipelineConfig",
    "display_experiment_stats",
    "display_hp_impact",
    "display_performance_tables",
    "export_summary_table",
    "export_experiment_stats",
    "export_hp_impact_tables",
    "get_experiment_stats_df",
    "iter_hp_impact_tables",
    "write_tables_tex",
    "write_report_html",
    "load_runs_df",
    "plot_best_metric_bars",
    "plot_hp_impact",
    "plot_summary",
    "validate_schema",
]
