# Optional: configure package logger to emit to stdout by default (print-like behavior).
# Callers can reconfigure logging or attach their own handlers.
import logging
import sys

from lmetk_report import config
from lmetk_report.analysis import analyze_results
from lmetk_report.config import PipelineConfig
from lmetk_report.plots import plot_best_metric_bars, plot_hp_impact
from lmetk_report.preprocess import clean_and_rename_models
from lmetk_report.reporting import (
    create_summary_table,
    display_experiment_stats,
    display_hp_impact,
    display_performance_tables,
    export_summary_table,
)
from lmetk_report.run_processor import load_runs_df
from lmetk_report.validators import validate_schema

_logger = logging.getLogger("lmetk_report")
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
    "load_runs_df",
    "plot_best_metric_bars",
    "plot_hp_impact",
    "validate_schema",
]
