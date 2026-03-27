"""
Panel component builders for the piCID dashboard.

Each function takes a ResultsLoader and widget selections, returns a Panel object.
"""

from __future__ import annotations

from typing import Optional

import re

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

    Every dataset now carries a '[project_name]' suffix from ResultsLoader.
    Strip the date prefix (e.g. '16_03_2026_') and drop 'MultiSource_' to save space.
    'MultiSource_concepts_N-CMAPSS [16_03_2026_concepts_n_cmapss_prognostics_]'
        -> 'N-CMAPSS [concepts_n_cmapss_prognostics]'
    """
    if " [" in name and name.endswith("]"):
        ds_part, bracket = name.rsplit(" [", 1)
        suffix = bracket.rstrip("]")
        # Strip MultiSource_ prefix from dataset name
        ds_part = re.sub(r"^MultiSource_", "", ds_part)
        # Strip leading date stamp (dd_mm_yyyy_) from project suffix
        suffix = re.sub(r"^\d{2}_\d{2}_\d{4}_", "", suffix).rstrip("_")
        short = f"{ds_part} [{suffix}]"
        if len(short) <= max_len:
            return short
        # Abbreviate suffix to last 2 underscore-segments
        parts = [p for p in suffix.split("_") if p]
        abbrev = "_".join(parts[-2:]) if len(parts) >= 2 else suffix[:12]
        return f"{ds_part} [{abbrev}]"
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Heatmap (model × dataset)
# ---------------------------------------------------------------------------

_ALT_METRIC_MODELS = ("tabpfn", "tabdpt", "xgboost")


def _is_alt_model(name: str) -> bool:
    lower = name.lower()
    return any(p in lower for p in _ALT_METRIC_MODELS)


def build_heatmap(
    loader: ResultsLoader,
    metric_key: str,
    mode: str = "min",
    width: int = 900,
    height: int = 400,
    alt_metric_key: str | None = None,
    show_n: bool = True,
    show_rank: bool = False,
) -> pn.viewable.Viewable:
    """Color-coded heatmap of mean metric values across datasets and models."""
    import holoviews as hv
    hv.extension("bokeh")

    df = loader.metric_matrix(metric_key, "mean")
    n_df = loader.metric_matrix(metric_key, "n")
    # Melt to long form
    long = df.reset_index().melt(id_vars="index", var_name="model", value_name="value")
    long = long.rename(columns={"index": "dataset"})
    n_long = n_df.reset_index().melt(id_vars="index", var_name="model", value_name="n")
    n_long = n_long.rename(columns={"index": "dataset"})
    long = long.merge(n_long, on=["dataset", "model"], how="left")

    # Substitute alt metric values (and their n) before dropna so alt models aren't lost
    if alt_metric_key:
        alt_df = loader.metric_matrix(alt_metric_key, "mean")
        alt_n_df = loader.metric_matrix(alt_metric_key, "n")
        alt_long = alt_df.reset_index().melt(id_vars="index", var_name="model", value_name="alt_value")
        alt_long = alt_long.rename(columns={"index": "dataset"})
        alt_n_long = alt_n_df.reset_index().melt(id_vars="index", var_name="model", value_name="alt_n")
        alt_n_long = alt_n_long.rename(columns={"index": "dataset"})
        alt_map = alt_long.set_index(["dataset", "model"])["alt_value"]
        alt_n_map = alt_n_long.set_index(["dataset", "model"])["alt_n"]
        mask = long["model"].apply(_is_alt_model)
        long.loc[mask, "value"] = long.loc[mask].apply(
            lambda r: alt_map.get((r["dataset"], r["model"]), r["value"]), axis=1
        )
        long.loc[mask, "n"] = long.loc[mask].apply(
            lambda r: alt_n_map.get((r["dataset"], r["model"]), r["n"]), axis=1
        )

    long = long.dropna(subset=["value"])

    if long.empty:
        return pn.pane.Markdown(f"No data for metric **{metric_key}**.")

    # Shorten model and dataset names for readability
    long["model"] = long["model"].map(_shorten_model_name)
    long["dataset"] = long["dataset"].map(_shorten_dataset_name)

    # Column-normalise: within each dataset, rank models 0→1
    def _col_norm(x):
        lo, hi = x.min(), x.max()
        if hi == lo:
            return pd.Series(0.5, index=x.index)
        return (x - lo) / (hi - lo)

    long["color"] = long.groupby("dataset")["value"].transform(_col_norm)
    if mode == "min":
        long["color"] = 1.0 - long["color"]

    # Average column: mean value per model, normalised independently across models
    avg_val = long.groupby("model")["value"].mean()
    avg_n = long.groupby("model")["n"].mean()
    lo, hi = avg_val.min(), avg_val.max()
    if hi > lo:
        avg_color = (avg_val - lo) / (hi - lo)
        if mode == "min":
            avg_color = 1.0 - avg_color
    else:
        avg_color = pd.Series(0.5, index=avg_val.index)

    # Sort models by average color (best at top)
    model_order = avg_color.sort_values(ascending=False).index.tolist()

    # Rank within each dataset (1 = best); computed before average rows are added
    long["rank"] = long.groupby("dataset")["value"].rank(method="min", ascending=(mode == "min")).astype(int)

    # Append average rows (placed last on x-axis); no rank for the average column
    avg_rows = pd.DataFrame({
        "dataset": "Average",
        "model": avg_val.index,
        "value": avg_val.values,
        "n": avg_n.values,
        "color": avg_color.reindex(avg_val.index).values,
        "rank": pd.NA,
    })
    long = pd.concat([long, avg_rows], ignore_index=True)

    dataset_order = [d for d in long["dataset"].unique() if d != "Average"] + ["Average"]

    n_models = len(model_order)
    row_height = max(30, min(60, 400 // max(n_models, 1)))
    plot_height = max(300, row_height * n_models)

    def _make_label(r):
        rank_str = f"#{int(r['rank'])} " if (show_rank and pd.notna(r.get("rank"))) else ""
        n_str = f" (n={int(r['n'])})" if (show_n and pd.notna(r.get("n"))) else (" (n=N/A)" if show_n else "")
        return f"{rank_str}{r['value']:.2f}{n_str}"

    long["label"] = long.apply(_make_label, axis=1)

    model_dim = hv.Dimension("model", values=model_order)
    dataset_dim = hv.Dimension("dataset", values=dataset_order)
    heatmap = hv.HeatMap(long, kdims=[dataset_dim, model_dim], vdims=["color", "value"]).opts(
        responsive=True,
        frame_height=plot_height,
        colorbar=False,
        clim=(0.0, 1.0),
        cmap="RdYlGn",
        tools=["hover"],
        xrotation=45,
        title=f"{metric_key} (mean, col-normalised)",
        fontsize={"title": 13, "labels": 11, "ticks": 9},
        labelled=[],
        toolbar="above",
    )
    labels = hv.Labels(long, kdims=[dataset_dim, model_dim], vdims=["label"]).opts(
        text_color="white",
        text_font_style="bold",
        text_font_size="9pt",
    )
    return pn.pane.HoloViews(heatmap * labels, sizing_mode="stretch_width")


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
    alt_metric_key: str | None = None,
) -> pn.viewable.Viewable:
    """Horizontal bar chart with error bars for a single dataset."""
    import holoviews as hv
    hv.extension("bokeh")

    if dataset not in loader.datasets:
        return pn.pane.Markdown(f"Dataset **{dataset}** not found.")

    mean_row = loader.metric_matrix(metric_key, "mean").loc[dataset]
    std_row = loader.metric_matrix(metric_key, "std").loc[dataset]

    df = pd.DataFrame({"model": mean_row.index, "mean": mean_row.values, "std": std_row.values})

    if alt_metric_key:
        alt_mean = loader.metric_matrix(alt_metric_key, "mean").loc[dataset]
        alt_std = loader.metric_matrix(alt_metric_key, "std").loc[dataset]
        alt_mask = df["model"].apply(_is_alt_model)
        df.loc[alt_mask, "mean"] = alt_mean.reindex(df.loc[alt_mask, "model"]).values
        df.loc[alt_mask, "std"] = alt_std.reindex(df.loc[alt_mask, "model"]).values

    df = df.dropna(subset=["mean"]).sort_values("mean", ascending=(mode == "min"))
    df["std"] = df["std"].fillna(0)

    if df.empty:
        return pn.pane.Markdown(f"No data for **{dataset}** / **{metric_key}**.")

    # Apply shortened display names
    df["model"] = df["model"].map(_shorten_model_name)

    model_order = df["model"].tolist()
    n_models = len(df)
    bar_height = max(30, min(50, 400 // max(n_models, 1)))
    plot_height = max(300, bar_height * n_models)

    model_dim = hv.Dimension("model", values=model_order)
    bars = hv.Bars(df, kdims=[model_dim], vdims=["mean"]).opts(
        responsive=True,
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
    errorbars = hv.ErrorBars(df, kdims=[model_dim], vdims=["mean", "std"]).opts(
        line_width=1.5, line_color="black"
    )
    return pn.pane.HoloViews(bars * errorbars, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _parse_dataset_filter(dataset: str) -> tuple[str, str | None]:
    """Split 'DatasetName [project_name]' into (base_name, project_name)."""
    if " [" in dataset and dataset.endswith("]"):
        base, proj = dataset.rsplit(" [", 1)
        return base, proj.rstrip("]")
    return dataset, None


def build_summary_table(
    loader: ResultsLoader,
    dataset: str | None = None,
    model: str | None = None,
    metric: str | None = None,
) -> pn.viewable.Viewable:
    df = loader.summary_df
    if df is None:
        return pn.pane.Markdown("No `summary.csv` found in report output.")
    if dataset:
        ds_base, project = _parse_dataset_filter(dataset)
        df = df[df["Dataset"] == ds_base]
        if project:
            df = df[df["Project"] == project]
    if model:
        df = df[df["Model"] == model]
    if metric:
        df = df[df["Metric"] == metric]
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

def build_stats_table(
    loader: ResultsLoader,
    dataset: str | None = None,
    model: str | None = None,
) -> pn.viewable.Viewable:
    df = loader.stats_df
    if df is None:
        return pn.pane.Markdown("No `experiment_stats.csv` found in report output.")
    if dataset:
        ds_base, project = _parse_dataset_filter(dataset)
        df = df[df["Dataset"] == ds_base]
        if project:
            df = df[df["Project"] == project]
    if model:
        df = df[df["Model"] == model]
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
            f"No HP config table found for **{dataset}** / **{model}**. "
            f"Re-run the report pipeline to generate it."
        )

    is_raw = any(c.endswith("_mean") for c in df.columns)

    if is_raw:
        # New format: numeric _mean/_std columns + seeds
        mean_cols = [c for c in df.columns if c.endswith("_mean")]
        std_cols = [c for c in df.columns if c.endswith("_std")]
        stat_cols = set(mean_cols) | set(std_cols) | {"seeds"}
        hp_cols = [c for c in df.columns if c not in stat_cols]
        for c in mean_cols + std_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(4)
        ordered_cols = hp_cols + (["seeds"] if "seeds" in df.columns else []) + mean_cols + std_cols
    else:
        # Legacy format: formatted string columns ("mean ± std (n=X)")
        # Extract seeds from the first metric column that has (n=X)
        metric_cols = [c for c in df.columns if "±" in df[c].astype(str).iloc[0] if not df.empty]
        hp_cols = [c for c in df.columns if c not in metric_cols]
        if metric_cols and not df.empty:
            df.insert(len(hp_cols), "seeds",
                df[metric_cols[0]].str.extract(r"\(n=(\d+)\)")[0].astype("Int64"))
        ordered_cols = hp_cols + (["seeds"] if "seeds" in df.columns else []) + metric_cols

    df = df[ordered_cols]

    # Pre-select only varying HP columns (nunique > 1) + seeds + metric cols
    if is_raw:
        stat_default = (["seeds"] if "seeds" in df.columns else []) + mean_cols + std_cols
        varying_hp = [c for c in hp_cols if df[c].nunique() > 1]
        default_cols = (varying_hp or hp_cols) + stat_default
    else:
        stat_default = (["seeds"] if "seeds" in df.columns else []) + metric_cols
        varying_hp = [c for c in hp_cols if df[c].nunique() > 1]
        default_cols = (varying_hp or hp_cols) + stat_default

    col_selector = pn.widgets.MultiChoice(
        name="Visible columns",
        options=ordered_cols,
        value=default_cols,
        sizing_mode="stretch_width",
    )

    rotated_header_css = """
    .tabulator .tabulator-col .tabulator-col-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-top: 20px;
        position: relative;
    }
    .tabulator .tabulator-col .tabulator-col-content .tabulator-col-title {
        writing-mode: vertical-rl;
        transform: rotate(180deg);
        white-space: nowrap;
    }
    .tabulator .tabulator-col .tabulator-col-content .tabulator-arrow {
        position: absolute;
        top: 4px;
        left: 50%;
        transform: translateX(-50%);
    }
    """

    def _table(cols):
        visible = cols if cols else ordered_cols
        return pn.widgets.Tabulator(
            df[visible],
            pagination="local",
            page_size=30,
            show_index=False,
            sortable=True,
            sizing_mode="stretch_width",
            stylesheets=[rotated_header_css],
        )

    return pn.Column(
        col_selector,
        pn.bind(_table, col_selector),
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
