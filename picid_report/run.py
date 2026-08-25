"""
Main pipeline script: load → validate → analyze → report (tables and optional plots).

Flow:
1. load_runs_df: fetch runs from wandb or CSV cache; normalize config columns.
2. Optional preprocess (e.g. clean_and_rename_models); validate_schema.
3. analyze_results: build all_results[dataset][model] with best run, aggregates, sort_metric_used.
4. Reporting: summary table, HP impact tables, experiment stats; optional export and HTML report.
5. If output_dir: save CSVs, LaTeX, plots, and report.html.

Sort metric: resolved automatically from configs (sort_metrics) per (dataset, model), or
overridden via run_pipeline(sort_metric=...) or PipelineConfig. Used to rank/select best
config and to display "Metric used to select best results" in reports.
"""

import logging
import os
import re
from collections import defaultdict
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from picid_report import (
    PipelineConfig,
    analyze_results,
    clean_and_rename_models,
    config,
    create_summary_table,
    display_experiment_stats,
    display_hp_impact,
    display_performance_tables,
    export_experiment_stats,
    export_hp_impact_tables,
    export_summary_table,
    get_experiment_stats_df,
    iter_hp_impact_tables,
    load_runs_df,
    plot_best_metric_bars,
    plot_hp_impact,
    plot_summary,
    validate_schema,
    write_report_html,
    write_tables_tex,
)
from picid_report.core.analysis import _ds_is_empty
from picid_report.logging_config import configure_logging

logger = logging.getLogger(__name__)

# --- Default configuration (override via run_pipeline() or CLI) ---
DEFAULT_PROJECT_NAME = "29_01_2026_unibo_prognostics_combined"
DEFAULT_USER = "anonlab-buildingenergy-1"
DEFAULT_METRIC_PREFIXES = ["val/", "test/", "test_best_rerun/", "val_best_rerun/"]
DEFAULT_REPORTING_METRICS = ["loss", "mse", "mae", "rmse", "f1", "accuracy"]
DEFAULT_REQUIRED_DATA_SEEDS: Optional[Set[int]] = None
DEFAULT_REQUIRED_MODEL_SEEDS: Optional[Set[int]] = {72, 88, 101, 666, 226688}
DEFAULT_ADDITIONAL_IGNORED_COLS = ["gpu_id", "slurm_job_id"]


def all_results_to_xarray(
    all_results: defaultdict,
    reporting_metrics: List[str],
    project_name: str = "",
) -> xr.Dataset:
    """Convert nested ``all_results`` into an ``xr.Dataset``.

    Variables with dims ``(dataset, model, metric_key)``:
        ``mean``, ``std``, ``n`` — best-performance stats for each combined
        ``"prefix/metric_name"`` key (e.g. ``"test_best_rerun/mae_normalized"``).

    Variables with dims ``(dataset, model)`` — per-entry metadata:
        ``sort_metric``  — metric key used to rank HP configs.
        ``opt_metric``   — metric the model was optimised on during training.
        ``opt_mode``     — ``"min"`` or ``"max"``.
        ``opt_value``    — best value of the optimisation metric.
        ``total_runs``   — total W&B runs for this (dataset, model).
        ``configs_failed_seed``   — HP configs dropped due to incomplete seeds.
        ``configs_failed_metric`` — HP configs dropped due to missing metric.

    Dataset-level ``.attrs``: ``project_name``, ``reporting_metrics``,
    ``created_at`` (ISO-8601 UTC).
    """
    import datetime

    datasets = sorted(all_results.keys())
    models = sorted({m for ds in all_results.values() for m in ds.keys()})

    # Build the flat metric_key set: "prefix/metric_name"
    metric_keys_set: set = set()
    for ds_results in all_results.values():
        for res_entry in ds_results.values():
            metrics_data = res_entry.get("best_performance", {}).get("metrics", {})
            for metric_name, prefix_dict in metrics_data.items():
                for prefix_name in prefix_dict:
                    metric_keys_set.add(f"{prefix_name}/{metric_name}")
    metric_keys = sorted(metric_keys_set)

    n_ds = len(datasets)
    n_m = len(models)
    n_k = len(metric_keys)

    dataset_idx = {name: i for i, name in enumerate(datasets)}
    model_idx = {name: i for i, name in enumerate(models)}
    mk_idx = {key: i for i, key in enumerate(metric_keys)}

    arr_mean = np.full((n_ds, n_m, n_k), np.nan)
    arr_std = np.full((n_ds, n_m, n_k), np.nan)
    arr_n = np.full((n_ds, n_m, n_k), np.nan)

    arr_sort_metric = np.full((n_ds, n_m), None, dtype=object)
    arr_opt_metric = np.full((n_ds, n_m), None, dtype=object)
    arr_opt_mode = np.full((n_ds, n_m), None, dtype=object)
    arr_opt_value = np.full((n_ds, n_m), np.nan)
    arr_total_runs = np.full((n_ds, n_m), np.nan)
    arr_failed_seed = np.full((n_ds, n_m), np.nan)
    arr_failed_metric = np.full((n_ds, n_m), np.nan)

    for dataset_name, model_dict in all_results.items():
        d_i = dataset_idx.get(dataset_name)
        if d_i is None:
            continue
        for model_name, res_entry in model_dict.items():
            m_i = model_idx.get(model_name)
            if m_i is None:
                continue

            # --- metric data ---
            metrics_data = res_entry.get("best_performance", {}).get("metrics", {})
            for metric_name, prefix_dict in metrics_data.items():
                for prefix_name, values in prefix_dict.items():
                    k_i = mk_idx.get(f"{prefix_name}/{metric_name}")
                    if k_i is None:
                        continue
                    for arr, key in (
                        (arr_mean, "mean"),
                        (arr_std, "std"),
                        (arr_n, "count"),  # stored as "count" in analysis.py
                    ):
                        val = values.get(key)
                        if val is not None:
                            try:
                                arr[d_i, m_i, k_i] = float(val)
                            except (TypeError, ValueError):
                                pass

            # --- (dataset, model) metadata ---
            arr_sort_metric[d_i, m_i] = res_entry.get("sort_metric_used")

            opt = res_entry.get("best_performance", {}).get("optimized_on", {})
            arr_opt_metric[d_i, m_i] = opt.get("metric")
            arr_opt_mode[d_i, m_i] = opt.get("mode")
            opt_val = opt.get("value")
            if opt_val is not None:
                try:
                    arr_opt_value[d_i, m_i] = float(opt_val)
                except (TypeError, ValueError):
                    pass

            for arr, key in (
                (arr_total_runs, "total_runs"),
                (arr_failed_seed, "configs_failed_not_full_seed_set"),
                (arr_failed_metric, "configs_failed_missing_invalid_metric"),
            ):
                val = res_entry.get(key)
                if val is not None:
                    try:
                        arr[d_i, m_i] = float(val)
                    except (TypeError, ValueError):
                        pass

    dims3 = ("dataset", "model", "metric_key")
    dims2 = ("dataset", "model")
    coords3 = {"dataset": datasets, "model": models, "metric_key": metric_keys}
    coords2 = {"dataset": datasets, "model": models}

    return xr.Dataset(
        {
            "mean": xr.DataArray(arr_mean, dims=dims3, coords=coords3),
            "std": xr.DataArray(arr_std, dims=dims3, coords=coords3),
            "n": xr.DataArray(arr_n, dims=dims3, coords=coords3),
            "sort_metric": xr.DataArray(arr_sort_metric, dims=dims2, coords=coords2),
            "opt_metric": xr.DataArray(arr_opt_metric, dims=dims2, coords=coords2),
            "opt_mode": xr.DataArray(arr_opt_mode, dims=dims2, coords=coords2),
            "opt_value": xr.DataArray(arr_opt_value, dims=dims2, coords=coords2),
            "total_runs": xr.DataArray(arr_total_runs, dims=dims2, coords=coords2),
            "configs_failed_seed": xr.DataArray(
                arr_failed_seed, dims=dims2, coords=coords2
            ),
            "configs_failed_metric": xr.DataArray(
                arr_failed_metric, dims=dims2, coords=coords2
            ),
        },
        attrs={
            "project_name": project_name,
            "reporting_metrics": list(reporting_metrics or []),
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )


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
    preprocess_df: Optional[
        Callable[[pd.DataFrame], pd.DataFrame]
    ] = None,  # None => use clean_and_rename_models
    pipeline_config: Optional[PipelineConfig] = None,
    show_performance_tables: bool = False,
    show_plots: bool = True,
    precision: int = 4,
    output_dir: Optional[str] = None,
    export_laTeX: bool = False,
    plot_metric: str = "test/mae_denormalized",
    sort_metric: Optional[str] = None,
    report_filename: Optional[str] = None,
    quiet: bool = False,
    use_legacy_search_space_fallback: bool = True,
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
        output_dir/report.html (or output_dir/report_filename if set) for single-file visualization.
    report_filename : str or None
        If set and output_dir is set, use this as the report HTML filename (e.g. "my_project.html").
        If None, defaults to "report.html".
    quiet : bool
        If True, do not print experiment stats or HP impact tables to the terminal; only the final
        report (and files when output_dir is set) are produced. Use when you only care about the
        saved report.
    export_laTeX : bool
        If True and output_dir set, also save summary table as LaTeX.
    plot_metric : str
        Metric for bar chart and HP impact plots (e.g. "test/mse").
    sort_metric : str, optional
        Metric to use for sorting/ranking/selecting best results in tables (e.g. "test/accuracy").
        If None, uses optimization metric for each model/dataset.
        Can also be a dict mapping (dataset, model) -> metric for per-combination control.
    use_legacy_search_space_fallback : bool, default True
        If True, when get_search_space(dataset, model) returns None (e.g. search_space.py has no
        entry), the pipeline falls back to config.EXPECTED_SEARCH_SPACE (legacy model-only grid).
        Set to False (e.g. --data-first) to use data-first mode in that case.

    Returns
    -------
    (df, all_results, summary_df) : tuple
        Raw DataFrame after load (and preprocess), all_results from analyze_results,
        and the summary table DataFrame.
    """
    metric_prefixes = metric_prefixes or DEFAULT_METRIC_PREFIXES
    reporting_metrics = reporting_metrics or DEFAULT_REPORTING_METRICS
    required_data_seeds = (
        required_data_seeds
        if required_data_seeds is not None
        else DEFAULT_REQUIRED_DATA_SEEDS
    )
    required_model_seeds = (
        required_model_seeds
        if required_model_seeds is not None
        else DEFAULT_REQUIRED_MODEL_SEEDS
    )
    additional_ignored_cols = additional_ignored_cols or DEFAULT_ADDITIONAL_IGNORED_COLS

    # 1. Load
    logger.info("[Stage 1/6] Load: fetching runs from W&B or CSV cache")
    df, config_columns, dropped_columns = load_runs_df(
        project_name=project_name,
        user=user,
        csv_cache_dir=csv_cache_dir,
        pipeline_config=pipeline_config,
    )
    logger.info(
        "[Stage 1/6] Load: %d rows, %d config columns, %d columns dropped",
        len(df),
        len(config_columns),
        len(dropped_columns),
    )

    # 2. Preprocessing: by default clean model names (strip prefix, add linear/exponential)
    logger.info(
        "[Stage 2/6] Preprocess: %s",
        "custom" if preprocess_df is not None else "clean_and_rename_models",
    )
    apply_preprocess = (
        preprocess_df if preprocess_df is not None else clean_and_rename_models
    )
    df = apply_preprocess(df)

    # 3. Validate
    logger.info("[Stage 3/6] Validate: schema check (required columns)")
    validate_schema(df, config.REQUIRED_COLUMNS)

    # 4. Analyze
    # Create sort metric resolver if configs available and not already in pipeline_config
    sort_metric_resolver = None
    if pipeline_config is not None and pipeline_config.sort_metric_resolver is not None:
        # Use resolver from pipeline_config
        sort_metric_resolver = pipeline_config.sort_metric_resolver
        logger.info("Using sort_metric_resolver from pipeline_config")
    else:
        # Try to create resolver using get_sort_metric from configs
        try:
            from picid_report.configs import (
                get_sort_metric,
                infer_dataset_category_from_name,
                infer_task_type_from_dataset,
            )

            def resolver(
                dataset: str, model: str, task_type=None, dataset_category=None
            ):
                """Resolve sort metric for a dataset/model combination."""
                # Infer task type if not provided
                if task_type is None:
                    task_type = infer_task_type_from_dataset(dataset)

                # Infer dataset category if not provided
                if dataset_category is None:
                    dataset_category = infer_dataset_category_from_name(dataset)

                # Call get_sort_metric with inferred values
                result = get_sort_metric(
                    dataset=dataset,
                    model=model,
                    task_type=task_type,
                    dataset_category=dataset_category,
                    fallback_to_optimization=True,
                )
                logger.debug(
                    "Sort metric resolver: dataset=%s, model=%s, task_type=%s, category=%s -> %s",
                    dataset,
                    model,
                    task_type,
                    dataset_category,
                    result,
                )
                return result

            sort_metric_resolver = resolver
            logger.info("Created sort_metric_resolver from configs")
        except ImportError as e:
            # Configs not available, use None (will use optimization metric)
            logger.warning(
                f"Configs not available, cannot create sort_metric_resolver: {e}"
            )
            sort_metric_resolver = None

    logger.info(
        "[Stage 4/6] Analyze: per (dataset, model) aggregation and grid resolution"
    )
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
        sort_metric_resolver=sort_metric_resolver,
        use_legacy_search_space_fallback=use_legacy_search_space_fallback,
    )
    n_entries = sum(len(m) for m in all_results.values()) if all_results else 0
    logger.info(
        "[Stage 4/6] Analyze: %d dataset(s) -> %d (dataset, model) result entries",
        len(all_results),
        n_entries,
    )

    # 5. Reporting — tables (display and optionally save)
    # Extract sort_metric from all_results if not explicitly provided
    # Build dict mapping (dataset, model) -> sort_metric_used
    if sort_metric is None:
        sort_metric_dict = {}
        for dataset, models in all_results.items():
            for model, res in models.items():
                sort_metric_used = res.get("sort_metric_used")
                logger.debug(
                    f"Extracted sort_metric_used for {model} on {dataset}: {sort_metric_used}"
                )
                if sort_metric_used is not None:
                    sort_metric_dict[(dataset, model)] = sort_metric_used
        # Use dict if we have any resolved metrics, otherwise None
        sort_metric = sort_metric_dict if sort_metric_dict else None
        if sort_metric_dict:
            logger.info(
                f"Using sort_metric dict with {len(sort_metric_dict)} entries: {sort_metric_dict}"
            )
        else:
            logger.warning(
                "No sort_metric_used found in results, will use optimization metric for all"
            )

    logger.info("[Stage 5/6] Report: building summary table and stats")
    if not quiet:
        display_experiment_stats(all_results)
        if show_performance_tables:
            display_performance_tables(all_results, precision=precision)
        display_hp_impact(all_results, precision=precision, sort_metric=sort_metric)

    all_results_xarr = all_results_to_xarray(
        all_results, reporting_metrics, project_name=project_name
    )

    summary_df = create_summary_table(
        all_results, precision=precision, sort_metric=sort_metric
    )
    stats_df = get_experiment_stats_df(all_results)

    # 6. Save outputs and/or build plots
    if output_dir:
        logger.info(
            "[Stage 6/6] Save: writing tables, plots, and report to %s", output_dir
        )
        _save_outputs(
            output_dir=output_dir,
            all_results=all_results,
            all_results_xarr=all_results_xarr,
            summary_df=summary_df,
            stats_df=stats_df,
            show_plots=show_plots,
            precision=precision,
            export_laTeX=export_laTeX,
            plot_metric=plot_metric,
            sort_metric=sort_metric,
            report_filename=report_filename,
        )
    elif show_plots:
        logger.info("[Stage 6/6] Plots only (no output_dir): in-memory plots")
        _run_plots(
            all_results,
            metric=plot_metric,
            precision=precision,
            sort_metric=sort_metric,
        )
    else:
        logger.info("[Stage 6/6] No output_dir and no plots: nothing to save")

    return df, all_results, summary_df


def combine_results_nc(output_base: str) -> xr.Dataset:
    """Load all ``results.nc`` files under *output_base* and merge into one Dataset.

    Each per-project ``results.nc`` typically covers a single dataset.  This
    function concatenates them along the ``dataset`` dimension using an outer
    join, so metric_keys that only exist in some projects (e.g. regression
    metrics absent from classification projects) are filled with NaN.

    Parameters
    ----------
    output_base : str
        Root directory to search, e.g. ``"report_output"``.  All files
        matching ``<output_base>/*/results.nc`` are loaded.

    Returns
    -------
    xr.Dataset
        Combined dataset with all datasets as coordinates.

    Raises
    ------
    FileNotFoundError
        When no ``results.nc`` files are found under *output_base*.
    """
    import glob as _glob

    paths = sorted(_glob.glob(os.path.join(output_base, "*", "results.nc")))
    if not paths:
        raise FileNotFoundError(
            f"No results.nc files found under {output_base!r}. "
            "Run the pipeline with --output-dir first."
        )

    datasets = [xr.open_dataset(p) for p in paths]
    logger.info("combine_results_nc: merging %d files from %s", len(paths), output_base)
    combined = xr.concat(
        datasets, dim="dataset", join="outer", combine_attrs="drop_conflicts"
    )
    for ds in datasets:
        ds.close()
    return combined


def _safe_basename(dataset: str, model: str, max_len: int = 80) -> str:
    """Sanitize dataset and model for use in filenames."""
    safe_ds = re.sub(r"[^\w\-.]", "_", str(dataset))[:max_len]
    safe_m = re.sub(r"[^\w\-.]", "_", str(model))[:max_len]
    return f"{safe_ds}_{safe_m}"


def _prepare_hp_dataset_for_netcdf(dataset: xr.Dataset) -> xr.Dataset:
    """Drop unsupported values and stringify object coordinates element by element."""
    prepared = dataset.drop_vars("run_names", errors="ignore")
    for coord_name in list(prepared.coords):
        coord = prepared.coords[coord_name]
        if coord.dtype != object:
            continue
        string_values = np.asarray(
            [str(value) for value in coord.values.flat], dtype=str
        ).reshape(coord.shape)
        prepared = prepared.assign_coords({coord_name: (coord.dims, string_values)})
    return prepared


def _save_outputs(
    output_dir: str,
    all_results: defaultdict,
    all_results_xarr: xr.Dataset,
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    show_plots: bool,
    precision: int,
    export_laTeX: bool,
    plot_metric: str,
    sort_metric: Optional[str] = None,
    report_filename: Optional[str] = None,
) -> None:
    """Write all tables, plots, and report HTML to output_dir."""
    tables_dir = os.path.join(output_dir, "tables")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # xarray Dataset
    xarr_path = os.path.join(output_dir, "results.nc")
    all_results_xarr.to_netcdf(xarr_path)
    logger.info("xarray Dataset written to %s", xarr_path)

    # Summary table
    export_summary_table(
        all_results,
        os.path.join(tables_dir, "summary.csv"),
        format="csv",
        precision=precision,
        sort_metric=sort_metric,
    )
    if export_laTeX:
        export_summary_table(
            all_results,
            os.path.join(tables_dir, "summary.tex"),
            format="latex",
            precision=precision,
            sort_metric=sort_metric,
        )

    # Experiment stats
    export_experiment_stats(
        all_results, os.path.join(tables_dir, "experiment_stats.csv")
    )

    # Diagnostics log: per (dataset, model) explanation of counts and failures for debugging
    log_path = os.path.join(output_dir, "log.txt")
    with open(log_path, "w") as f:
        f.write(
            "Report pipeline diagnostics — per dataset/model: optimization metric, seeds, and failure counts.\n"
        )
        f.write(
            "Use this to understand why 'Configs Failed (not full seed set)' or 'Configs Failed (missing/invalid metric)' appear.\n"
        )
        for dataset, models in sorted(all_results.items()):
            for model, res in sorted(models.items()):
                lines = res.get("diagnostics_log", [])
                if lines:
                    f.write("\n".join(lines))
                    f.write("\n")
    logger.info("Diagnostics log written to %s", log_path)

    # Single LaTeX file with all tables for pasting into paper
    write_tables_tex(
        os.path.join(output_dir, "tables_tex.tex"),
        summary_df=summary_df,
        stats_df=stats_df,
        all_results=all_results,
        precision=precision,
        sort_metric=sort_metric,
    )

    # HP impact tables (CSV per model/dataset) and collect entries for report
    hp_impact_dir = os.path.join(tables_dir, "hp_impact")
    export_hp_impact_tables(
        all_results, hp_impact_dir, precision=precision, sort_metric=sort_metric
    )

    # Save sorted_aggregated_results as .nc per (dataset, model) for dashboard
    hp_nc_dir = os.path.join(tables_dir, "hp_configs")
    os.makedirs(hp_nc_dir, exist_ok=True)
    for _ds_name, _models in all_results.items():
        for _model_name, _res in _models.items():
            _agg_ds = _res.get("sorted_aggregated_results")
            if _agg_ds is not None and not _ds_is_empty(_agg_ds):
                _safe_ds = re.sub(r"[^\w\-.]", "_", str(_ds_name))[:80]
                _safe_m = re.sub(r"[^\w\-.]", "_", str(_model_name))[:80]
                # netCDF cannot encode sequence-valued object coordinates directly.
                _to_save = _prepare_hp_dataset_for_netcdf(_agg_ds)
                _to_save.to_netcdf(os.path.join(hp_nc_dir, f"{_safe_ds}_{_safe_m}.nc"))

    global_plots: List[Tuple[str, str]] = []
    hp_impact_entries: List[Tuple[str, str, pd.DataFrame, str, str]] = []

    if show_plots:
        # Summary heatmap (model × dataset) from xarray
        _CLASSIFICATION_HINTS = ("accuracy", "auroc", "f1", "precision", "recall")
        summary_mode = (
            "max" if any(h in plot_metric for h in _CLASSIFICATION_HINTS) else "min"
        )
        # Derive a fallback prefix: strip _best_rerun so fit-predict models (XGBoost,
        # TabPFN, TabDPT) fill in from their plain test/* or val/* metrics.
        _fallback = (
            plot_metric.replace("_best_rerun", "")
            if "_best_rerun" in plot_metric
            else None
        )
        summary_path = os.path.join(plots_dir, "summary.png")
        fig_summary = plot_summary(
            all_results_xarr,
            metric_key=plot_metric,
            mode=summary_mode,
            fallback_metric_keys=[_fallback] if _fallback else None,
            save_path=summary_path,
        )
        if fig_summary is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig_summary)
            except Exception:
                pass
            global_plots.append((f"Summary: {plot_metric}", "plots/summary.png"))

        # Bar chart of best metric
        bars_path = os.path.join(plots_dir, "best_metric_bars.png")
        fig_bars = plot_best_metric_bars(
            all_results, metric=plot_metric, save_path=bars_path
        )
        if fig_bars is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig_bars)
            except Exception:
                pass
            global_plots.append(
                (f"Best {plot_metric} by model/dataset", "plots/best_metric_bars.png")
            )

        # HP impact plot per model/dataset
        for dataset, model, hp_df, metric_used in iter_hp_impact_tables(
            all_results, precision, sort_metric
        ):
            base = _safe_basename(dataset, model)
            plot_path = os.path.join(plots_dir, f"hp_impact_{base}.png")
            fig_hp = plot_hp_impact(
                all_results,
                model=model,
                dataset=dataset,
                metric=plot_metric,
                save_path=plot_path,
            )
            if fig_hp is not None:
                try:
                    import matplotlib.pyplot as plt

                    plt.close(fig_hp)
                except Exception:
                    pass
            hp_impact_entries.append(
                (dataset, model, hp_df, f"plots/hp_impact_{base}.png", metric_used)
            )

    # Single HTML report for visualization
    report_path = write_report_html(
        output_dir=output_dir,
        summary_df=summary_df,
        stats_df=stats_df,
        hp_impact_entries=hp_impact_entries,
        global_plots=global_plots,
        title="Experiment Report",
        all_results=all_results,
        sort_metric=sort_metric,
        report_filename=report_filename or "report.html",
    )
    logger.info("Report written to %s", report_path)


def _run_plots(
    all_results: defaultdict,
    metric: str = "test/mse",
    precision: int = 4,
    sort_metric: Optional[str] = None,
) -> None:
    """Build default plots without saving. One bar chart + one HP impact example."""
    fig_bars = plot_best_metric_bars(all_results, metric=metric)
    if fig_bars is not None:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig_bars)
        except Exception:
            pass

    for dataset, model, _, _ in iter_hp_impact_tables(
        all_results, precision, sort_metric
    ):
        fig_hp = plot_hp_impact(
            all_results, model=model, dataset=dataset, metric=metric
        )
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
        description="Run picid_report pipeline. Use --output-dir to save tables, plots, and an HTML report."
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable DEBUG logging (resolver details, shapes, etc.)",
    )
    parser.add_argument(
        "--project", default=DEFAULT_PROJECT_NAME, help="W&B project name"
    )
    parser.add_argument("--user", default=DEFAULT_USER, help="W&B user/entity")
    parser.add_argument("--cache-dir", default="csv_files", help="CSV cache directory")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Save all tables and plots here; generate report.html",
    )
    parser.add_argument(
        "--report-name",
        default=None,
        help="Report HTML filename (e.g. my_project.html); default report.html",
    )
    parser.add_argument(
        "--export-latex",
        action="store_true",
        help="Also save summary table as LaTeX (when using -o)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip building plots")
    parser.add_argument(
        "--performance-tables",
        action="store_true",
        help="Show per-metric performance tables",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Do not print tables to terminal; only write report and files",
    )
    parser.add_argument(
        "--data-first",
        action="store_true",
        help="Do not fall back to config.EXPECTED_SEARCH_SPACE when search_space.py has no entry; use data-first mode",
    )
    args = parser.parse_args()

    configure_logging(debug=args.debug)

    run_pipeline(
        project_name=args.project,
        user=args.user,
        csv_cache_dir=args.cache_dir,
        show_plots=not args.no_plots,
        show_performance_tables=args.performance_tables,
        output_dir=args.output_dir,
        export_laTeX=args.export_latex,
        report_filename=args.report_name,
        quiet=args.quiet,
        use_legacy_search_space_fallback=not args.data_first,
    )


if __name__ == "__main__":
    main()
