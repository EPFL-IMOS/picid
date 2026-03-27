"""
Panel component builders for the piCID dashboard.

Each function takes a ResultsLoader and widget selections, returns a Panel object.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import panel as pn
import param

from .data import ResultsLoader


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _shorten_model_name(name: str) -> str:
    """Extract a concise display label from a dotted Python module path.

    Examples
    --------
    'baselines.crossformer.model.Crossformer_forecaster'  -> 'Crossformer'
    'model.wrappers.statistical_models.wrapper.statistical.LinearWrapper (linear)'
        -> 'LinearWrapper (linear)'
    """
    # Preserve trailing variant annotation like " (linear)"
    variant = ""
    if " (" in name:
        name, rest = name.rsplit(" (", 1)
        variant = f" ({rest.rstrip(')')})"
    # Take the last dotted component (class/function name)
    short = name.split(".")[-1]
    # Strip common verbose suffixes, case-insensitively
    for suffix in ("_forecaster", "Forecaster", "Wrapper", "wrapper"):
        if short.lower().endswith(suffix.lower()) and len(short) > len(suffix):
            short = short[: -len(suffix)]
    return short.strip("_") + variant


def _shorten_dataset_name(name: str, max_len: int = 40) -> str:
    """Shorten a dataset label for display in chart axes.

    Handles the '[project_suffix]' appended by ResultsLoader for duplicates:
    'MyDataset [16_03_2026_very_long_project_name]' -> 'MyDataset [long_project]'
    """
    if len(name) <= max_len:
        return name
    # If there's a project suffix, keep the dataset name and abbreviate the suffix
    if " [" in name and name.endswith("]"):
        ds_part, bracket = name.rsplit(" [", 1)
        suffix = bracket.rstrip("]")
        # Take the last meaningful segment of the suffix (split by _ or -)
        parts = [p for p in suffix.replace("-", "_").split("_") if p]
        abbrev = "_".join(parts[-2:]) if len(parts) >= 2 else suffix[:12]
        short = f"{ds_part} [{abbrev}]"
        if len(short) <= max_len:
            return short
        # If dataset name itself is still too long, truncate it
        return ds_part[:max_len - len(abbrev) - 4] + f"… [{abbrev}]"
    # No bracket — just truncate with ellipsis
    return name[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Heatmap (model × dataset)
# ---------------------------------------------------------------------------

def build_heatmap(
    loader: ResultsLoader,
    metric_key: str,
    mode: str = "min",
    width: int = 900,
    height: int = 400,
) -> pn.viewable.Viewable:
    """Color-coded heatmap of mean metric values across datasets and models."""
    import holoviews as hv
    hv.extension("bokeh")

    df = loader.metric_matrix(metric_key, "mean")
    # Melt to long form
    long = df.reset_index().melt(id_vars="index", var_name="model", value_name="value")
    long = long.rename(columns={"index": "dataset"})
    long = long.dropna(subset=["value"])

    if long.empty:
        return pn.pane.Markdown(f"No data for metric **{metric_key}**.")

    # Shorten model and dataset names for readability
    long["model"] = long["model"].map(_shorten_model_name)
    long["dataset"] = long["dataset"].map(_shorten_dataset_name)

    # Clamp outliers for color range
    vmin = np.nanpercentile(long["value"], 2)
    vmax = np.nanpercentile(long["value"], 98)

    n_datasets = len(loader.datasets)
    n_models = long["model"].nunique()
    row_height = max(50, min(80, 400 // n_datasets))
    plot_height = max(300, row_height * n_datasets)
    # Reserve enough width per model column so labels don't crowd
    col_width = max(55, min(80, 900 // max(n_models, 1)))
    plot_width = max(600, col_width * n_models)

    heatmap = hv.HeatMap(long, kdims=["model", "dataset"], vdims=["value"]).opts(
        frame_width=plot_width,
        frame_height=plot_height,
        colorbar=True,
        clim=(vmin, vmax),
        cmap="RdYlGn_r" if mode == "min" else "RdYlGn",
        tools=["hover"],
        xrotation=45,
        title=f"{metric_key} (mean)",
        fontsize={"title": 13, "labels": 11, "ticks": 9},
        labelled=[],         # suppress redundant "model" / "dataset" axis titles
        toolbar="above",
        responsive=True,
    )
    return pn.pane.HoloViews(heatmap, sizing_mode="stretch_width", min_height=400)


# ---------------------------------------------------------------------------
# Bar chart — best metric per model for a single dataset
# ---------------------------------------------------------------------------

def build_bar_chart(
    loader: ResultsLoader,
    metric_key: str,
    dataset: str,
    mode: str = "min",
    width: int = 700,
    height: int = 400,
) -> pn.viewable.Viewable:
    """Horizontal bar chart with error bars for a single dataset."""
    import holoviews as hv
    hv.extension("bokeh")

    if dataset not in loader.datasets:
        return pn.pane.Markdown(f"Dataset **{dataset}** not found.")

    mean_row = loader.metric_matrix(metric_key, "mean").loc[dataset]
    std_row = loader.metric_matrix(metric_key, "std").loc[dataset]

    df = pd.DataFrame({"model": mean_row.index, "mean": mean_row.values, "std": std_row.values})
    df = df.dropna(subset=["mean"]).sort_values("mean", ascending=(mode == "min"))
    df["std"] = df["std"].fillna(0)

    if df.empty:
        return pn.pane.Markdown(f"No data for **{dataset}** / **{metric_key}**.")

    # Apply shortened display names
    df["model"] = df["model"].map(_shorten_model_name)

    n_models = len(df)
    bar_height = max(30, min(50, 400 // max(n_models, 1)))
    plot_height = max(300, bar_height * n_models)

    bars = hv.Bars(df, kdims=["model"], vdims=["mean"]).opts(
        frame_width=600,
        frame_height=plot_height,
        invert_axes=True,
        tools=["hover"],
        color="mean",
        cmap="RdYlGn_r" if mode == "min" else "RdYlGn",
        colorbar=True,
        title=f"{dataset} — {metric_key}",
        fontsize={"title": 12, "labels": 11, "ticks": 10},
        xrotation=0,
        toolbar="above",
        labelled=["x"],   # keep value axis label, suppress category label
    )
    errorbars = hv.ErrorBars(df, kdims=["model"], vdims=["mean", "std"]).opts(
        line_width=1.5, line_color="black"
    )
    return pn.pane.HoloViews(bars * errorbars, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_table(loader: ResultsLoader) -> pn.viewable.Viewable:
    df = loader.summary_df
    if df is None:
        return pn.pane.Markdown("No `summary.csv` found in report output.")
    return pn.widgets.Tabulator(
        df,
        pagination="local",
        page_size=25,
        show_index=False,
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# Experiment stats
# ---------------------------------------------------------------------------

def build_stats_table(loader: ResultsLoader) -> pn.viewable.Viewable:
    df = loader.stats_df
    if df is None:
        return pn.pane.Markdown("No `experiment_stats.csv` found in report output.")
    return pn.widgets.Tabulator(
        df,
        pagination="local",
        page_size=25,
        show_index=False,
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# HP impact table
# ---------------------------------------------------------------------------

def build_hp_impact_table(
    loader: ResultsLoader,
    dataset: str,
    model: str,
) -> pn.viewable.Viewable:
    df = loader.hp_impact_df(dataset, model)
    if df is None:
        return pn.pane.Markdown(
            f"No HP impact table found for **{dataset}** / **{model}**."
        )
    return pn.widgets.Tabulator(
        df,
        pagination="local",
        page_size=30,
        show_index=False,
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# Metadata panel (opt metric, total runs, failures)
# ---------------------------------------------------------------------------

def build_metadata_panel(
    loader: ResultsLoader,
    dataset: str,
    model: str,
) -> pn.viewable.Viewable:
    ds = loader.xarray_dataset
    if dataset not in loader.datasets or model not in loader.models:
        return pn.pane.Markdown("Select a valid dataset and model.")

    def _val(var: str) -> str:
        try:
            v = ds[var].sel(dataset=dataset, model=model).item()
            return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
        except Exception:
            return "—"

    rows = {
        "Sort metric": _val("sort_metric"),
        "Opt metric": _val("opt_metric"),
        "Opt mode": _val("opt_mode"),
        "Opt value": _val("opt_value"),
        "Total runs": _val("total_runs"),
        "Configs failed (seed)": _val("configs_failed_seed"),
        "Configs failed (metric)": _val("configs_failed_metric"),
    }
    df = pd.DataFrame(list(rows.items()), columns=["Property", "Value"])
    return pn.widgets.Tabulator(
        df,
        show_index=False,
        sizing_mode="stretch_width",
        width=400,
    )
