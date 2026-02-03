"""
Main pipeline script: load → validate → analyze → report (tables & plots).

Mirrors the flow of table.ipynb. Run from the project root or with
PYTHONPATH including the package. Set output_dir to save all tables and
plots to disk and generate a single HTML report for visualization.
"""

import logging
import os
import re
from collections import defaultdict
from typing import Callable, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

from lmetk_report import (
    PipelineConfig,
    config,
    analyze_results,
    create_summary_table,
    display_experiment_stats,
    display_hp_impact,
    display_performance_tables,
    export_summary_table,
    load_runs_df,
    plot_best_metric_bars,
    plot_hp_impact,
    validate_schema,
)
from lmetk_report.reporting import (
    export_experiment_stats,
    export_hp_impact_tables,
    get_experiment_stats_df,
    iter_hp_impact_tables,
    write_tables_tex,
)
from lmetk_report.preprocess import clean_and_rename_models
from lmetk_report.report_html import write_report_html


# --- Default configuration (override via run_pipeline() or CLI) ---
DEFAULT_PROJECT_NAME = "29_01_2026_nb14_prognostics_combined"
DEFAULT_USER = "imos-buildingenergy-1"
DEFAULT_METRIC_PREFIXES = ["val/", "test/", "test_best_rerun/", "val_best_rerun/"]
DEFAULT_REPORTING_METRICS = ["loss", "mse", "mae", "rmse", "f1", "accuracy"]
DEFAULT_REQUIRED_DATA_SEEDS: Optional[Set[int]] = None
DEFAULT_REQUIRED_MODEL_SEEDS: Optional[Set[int]] = {72, 88, 101, 666, 226688}
DEFAULT_ADDITIONAL_IGNORED_COLS = ["gpu_id", "slurm_job_id"]


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
    preprocess_df: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,  # None => use clean_and_rename_models
    pipeline_config: Optional[PipelineConfig] = None,
    show_performance_tables: bool = False,
    show_plots: bool = True,
    precision: int = 4,
    output_dir: Optional[str] = None,
    export_laTeX: bool = False,
    plot_metric: str = "test/mse",
) -> Tuple[pd.DataFrame, defaultdict, pd.DataFrame]:
    """
    Run the full pipeline: load → optional preprocess → validate → analyze → report.

    Parameters
    ----------
    project_name : str
        W&B project name.
    user : str
        W&B user/entity.
    csv_cache_dir : str
        Directory for CSV cache.
    metric_prefixes, reporting_metrics, required_data_seeds, required_model_seeds,
    additional_ignored_cols
        Passed through to analyze_results; defaults from this module.
    preprocess_df : callable or None
        If None (default), apply clean_and_rename_models (strip common prefix,
        add "(linear)" / "(exponential)" for StatisticalBaselineWrapper). If set,
        called as preprocess_df(df) instead. Pass lambda df: df to skip preprocessing.
    pipeline_config : PipelineConfig or None
        If set, passed to load_runs_df and analyze_results; else module config is used.
    show_performance_tables : bool
        If True, call display_performance_tables (one table per metric).
    show_plots : bool
        If True, build plots (saved to output_dir/plots if output_dir set).
    precision : int
        Decimal precision for tables and display.
    output_dir : str or None
        If set, save all tables (CSV, optionally LaTeX) and plots here, and write
        output_dir/report.html for single-file visualization.
    export_laTeX : bool
        If True and output_dir set, also save summary table as LaTeX.
    plot_metric : str
        Metric for bar chart and HP impact plots (e.g. "test/mse").

    Returns
    -------
    (df, all_results, summary_df) : tuple
        Raw DataFrame after load (and preprocess), all_results from analyze_results,
        and the summary table DataFrame.
    """
    metric_prefixes = metric_prefixes or DEFAULT_METRIC_PREFIXES
    reporting_metrics = reporting_metrics or DEFAULT_REPORTING_METRICS
    required_data_seeds = required_data_seeds if required_data_seeds is not None else DEFAULT_REQUIRED_DATA_SEEDS
    required_model_seeds = required_model_seeds if required_model_seeds is not None else DEFAULT_REQUIRED_MODEL_SEEDS
    additional_ignored_cols = additional_ignored_cols or DEFAULT_ADDITIONAL_IGNORED_COLS

    # 1. Load
    df, config_columns, dropped_columns = load_runs_df(
        project_name=project_name,
        user=user,
        csv_cache_dir=csv_cache_dir,
        pipeline_config=pipeline_config,
    )

    # 2. Preprocessing: by default clean model names (strip prefix, add linear/exponential)
    apply_preprocess = preprocess_df if preprocess_df is not None else clean_and_rename_models
    df = apply_preprocess(df)

    # 3. Validate
    validate_schema(df, config.REQUIRED_COLUMNS)

    # 4. Analyze
    all_results = analyze_results(
        df=df,
        config_columns=config_columns,
        dropped_columns=dropped_columns,
        reporting_metrics=reporting_metrics,
        metric_prefixes=metric_prefixes,
        required_data_seeds=required_data_seeds,
        required_model_seeds=required_model_seeds,
        additional_ignored_cols=additional_ignored_cols,
        pipeline_config=pipeline_config,
    )

    # 5. Reporting — tables (display and optionally save)
    display_experiment_stats(all_results)
    if show_performance_tables:
        display_performance_tables(all_results, precision=precision)
    display_hp_impact(all_results, precision=precision)
    summary_df = create_summary_table(all_results, precision=precision)
    stats_df = get_experiment_stats_df(all_results)

    # 6. Save outputs and/or build plots
    if output_dir:
        _save_outputs(
            output_dir=output_dir,
            all_results=all_results,
            summary_df=summary_df,
            stats_df=stats_df,
            show_plots=show_plots,
            precision=precision,
            export_laTeX=export_laTeX,
            plot_metric=plot_metric,
        )
    elif show_plots:
        _run_plots(all_results, metric=plot_metric, precision=precision)

    return df, all_results, summary_df


def _safe_basename(dataset: str, model: str, max_len: int = 80) -> str:
    """Sanitize dataset and model for use in filenames."""
    safe_ds = re.sub(r"[^\w\-.]", "_", str(dataset))[:max_len]
    safe_m = re.sub(r"[^\w\-.]", "_", str(model))[:max_len]
    return f"{safe_ds}_{safe_m}"


def _save_outputs(
    output_dir: str,
    all_results: defaultdict,
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    show_plots: bool,
    precision: int,
    export_laTeX: bool,
    plot_metric: str,
) -> None:
    """Write all tables, plots, and report.html to output_dir."""
    tables_dir = os.path.join(output_dir, "tables")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Summary table
    export_summary_table(all_results, os.path.join(tables_dir, "summary.csv"), format="csv", precision=precision)
    if export_laTeX:
        export_summary_table(all_results, os.path.join(tables_dir, "summary.tex"), format="latex", precision=precision)

    # Experiment stats
    export_experiment_stats(all_results, os.path.join(tables_dir, "experiment_stats.csv"))

    # Single LaTeX file with all tables for pasting into paper
    write_tables_tex(
        os.path.join(output_dir, "tables_tex.tex"),
        summary_df=summary_df,
        stats_df=stats_df,
        all_results=all_results,
        precision=precision,
    )

    # HP impact tables (CSV per model/dataset) and collect entries for report
    hp_impact_dir = os.path.join(tables_dir, "hp_impact")
    export_hp_impact_tables(all_results, hp_impact_dir, precision=precision)

    global_plots: List[Tuple[str, str]] = []
    hp_impact_entries: List[Tuple[str, str, pd.DataFrame, str]] = []

    if show_plots:
        # Bar chart of best metric
        bars_path = os.path.join(plots_dir, "best_metric_bars.png")
        fig_bars = plot_best_metric_bars(all_results, metric=plot_metric, save_path=bars_path)
        if fig_bars is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig_bars)
            except Exception:
                pass
            global_plots.append((f"Best {plot_metric} by model/dataset", "plots/best_metric_bars.png"))

        # HP impact plot per model/dataset
        for dataset, model, hp_df in iter_hp_impact_tables(all_results, precision):
            base = _safe_basename(dataset, model)
            plot_path = os.path.join(plots_dir, f"hp_impact_{base}.png")
            fig_hp = plot_hp_impact(all_results, model=model, dataset=dataset, metric=plot_metric, save_path=plot_path)
            if fig_hp is not None:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig_hp)
                except Exception:
                    pass
            hp_impact_entries.append((dataset, model, hp_df, f"plots/hp_impact_{base}.png"))

    # Single HTML report for visualization
    report_path = write_report_html(
        output_dir=output_dir,
        summary_df=summary_df,
        stats_df=stats_df,
        hp_impact_entries=hp_impact_entries,
        global_plots=global_plots,
        title="Experiment Report",
    )
    logger.info("Report written to %s", report_path)


def _run_plots(
    all_results: defaultdict,
    metric: str = "test/mse",
    precision: int = 4,
) -> None:
    """Build default plots without saving. One bar chart + one HP impact example."""
    fig_bars = plot_best_metric_bars(all_results, metric=metric)
    if fig_bars is not None:
        try:
            import matplotlib.pyplot as plt
            plt.close(fig_bars)
        except Exception:
            pass

    for dataset, model, _ in iter_hp_impact_tables(all_results, precision):
        fig_hp = plot_hp_impact(all_results, model=model, dataset=dataset, metric=metric)
        if fig_hp is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig_hp)
            except Exception:
                pass
        return


def main() -> None:
    """Entry point when run as a script. Parses minimal CLI and runs the pipeline."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Run lmetk_report pipeline. Use --output-dir to save tables, plots, and an HTML report."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT_NAME, help="W&B project name")
    parser.add_argument("--user", default=DEFAULT_USER, help="W&B user/entity")
    parser.add_argument("--cache-dir", default="csv_files", help="CSV cache directory")
    parser.add_argument("--output-dir", "-o", default=None, help="Save all tables and plots here; generate report.html")
    parser.add_argument("--export-latex", action="store_true", help="Also save summary table as LaTeX (when using -o)")
    parser.add_argument("--no-plots", action="store_true", help="Skip building plots")
    parser.add_argument("--performance-tables", action="store_true", help="Show per-metric performance tables")
    args = parser.parse_args()

    run_pipeline(
        project_name=args.project,
        user=args.user,
        csv_cache_dir=args.cache_dir,
        show_plots=not args.no_plots,
        show_performance_tables=args.performance_tables,
        output_dir=args.output_dir,
        export_laTeX=args.export_latex,
    )


if __name__ == "__main__":
    main()
