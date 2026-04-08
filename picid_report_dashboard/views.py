"""
Panel component builders for the piCID dashboard.

Each function takes a ResultsLoader and widget selections, returns a Panel object.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd
import panel as pn

from .data import ResultsLoader, is_alt_model


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

def _selected_metric_frame(
    loader: ResultsLoader,
    metric_key: str,
    *,
    sort_metric_key: str | None,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
    dataset: str | None = None,
) -> pd.DataFrame:
    """Build a long-form frame of dynamic dashboard values."""
    records = loader.selected_metric_records(
        metric_key=metric_key,
        sort_metric_key=sort_metric_key,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
        dataset=dataset,
    )
    if not records:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "value",
                "std",
                "n",
                "display_metric_key",
                "requested_sort_metric_key",
                "effective_sort_metric_key",
            ]
        )
    return pd.DataFrame(records)


def _build_heatmap_frame(
    loader: ResultsLoader,
    metric_key: str,
    *,
    sort_metric_key: str | None,
    mode: str,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the heatmap frame including Average and Average rank summary columns."""
    long = _selected_metric_frame(
        loader,
        metric_key,
        sort_metric_key=sort_metric_key,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
    )
    long = long.dropna(subset=["value"]).copy()
    if long.empty:
        return long, [], []

    # Column-normalise: within each dataset, rank models 0→1
    def _col_norm(x: pd.Series) -> pd.Series:
        lo, hi = x.min(), x.max()
        if hi == lo:
            return pd.Series(0.5, index=x.index)
        return (x - lo) / (hi - lo)

    long["summary_kind"] = "dataset"
    long["color"] = long.groupby("dataset")["value"].transform(_col_norm)
    if mode == "min":
        long["color"] = 1.0 - long["color"]

    # Rank within each dataset (1 = best); computed before summary rows are added
    long["rank"] = long.groupby("dataset")["value"].rank(
        method="min", ascending=(mode == "min")
    ).astype(float)

    # Average metric-value column
    avg_val = long.groupby("model")["value"].mean()
    avg_n = long.groupby("model")["n"].mean()
    lo, hi = avg_val.min(), avg_val.max()
    if hi > lo:
        avg_color = (avg_val - lo) / (hi - lo)
        if mode == "min":
            avg_color = 1.0 - avg_color
    else:
        avg_color = pd.Series(0.5, index=avg_val.index)

    avg_rows = pd.DataFrame(
        {
            "dataset": "Average",
            "model": avg_val.index,
            "value": avg_val.values,
            "n": avg_n.values,
            "color": avg_color.reindex(avg_val.index).values,
            "rank": np.nan,
            "summary_kind": "average",
        }
    )

    # Average-rank column: mean of per-dataset ranks, not rank(avg performance)
    avg_rank = long.groupby("model")["rank"].mean()
    rank_lo, rank_hi = avg_rank.min(), avg_rank.max()
    if rank_hi > rank_lo:
        avg_rank_color = 1.0 - ((avg_rank - rank_lo) / (rank_hi - rank_lo))
    else:
        avg_rank_color = pd.Series(0.5, index=avg_rank.index)

    avg_rank_rows = pd.DataFrame(
        {
            "dataset": "Average rank",
            "model": avg_rank.index,
            "value": avg_rank.values,
            "n": np.nan,
            "color": avg_rank_color.reindex(avg_rank.index).values,
            "rank": np.nan,
            "summary_kind": "average_rank",
        }
    )

    # Sort models by mean per-dataset rank (1 = best), not by average metric value.
    # Break ties with the average-value color so the order stays deterministic.
    model_order = (
        pd.DataFrame({"avg_rank": avg_rank, "avg_color": avg_color})
        .sort_values(["avg_rank", "avg_color"], ascending=[True, False])
        .index.tolist()
    )
    long = pd.concat([long, avg_rows, avg_rank_rows], ignore_index=True)
    dataset_order = [
        dataset_name
        for dataset_name in long["dataset"].unique()
        if dataset_name not in ("Average", "Average rank")
    ] + ["Average", "Average rank"]
    return long, model_order, dataset_order


def build_heatmap(
    loader: ResultsLoader,
    metric_key: str,
    sort_metric_key: str | None = None,
    mode: str = "min",
    width: int = 900,
    height: int = 400,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
    show_n: bool = True,
    show_rank: bool = False,
) -> pn.viewable.Viewable:
    """Color-coded heatmap of mean metric values across datasets and models."""
    import holoviews as hv
    hv.extension("bokeh")

    long, model_order, dataset_order = _build_heatmap_frame(
        loader,
        metric_key,
        sort_metric_key=sort_metric_key,
        mode=mode,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
    )

    if long.empty:
        return pn.pane.Markdown(f"No data for metric **{metric_key}**.")

    # Shorten model and dataset names for readability
    long["model"] = long["model"].map(_shorten_model_name)
    long["dataset"] = long["dataset"].map(_shorten_dataset_name)
    model_order = [_shorten_model_name(model_name) for model_name in model_order]
    dataset_label_map = {
        dataset_name: _shorten_dataset_name(dataset_name) for dataset_name in dataset_order
    }
    dataset_order = [dataset_label_map[dataset_name] for dataset_name in dataset_order]

    n_models = len(model_order)
    row_height = max(30, min(60, 400 // max(n_models, 1)))
    plot_height = max(300, row_height * n_models)

    def _make_label(r):
        if r.get("summary_kind") == "average_rank":
            return f"{r['value']:.2f}"
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
        title=f"{metric_key} (selected by active sort metric)",
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
    sort_metric_key: str | None = None,
    mode: str = "min",
    width: int = 700,
    height: int = 400,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
) -> pn.viewable.Viewable:
    """Horizontal bar chart with error bars for a single dataset."""
    import holoviews as hv
    hv.extension("bokeh")

    if dataset not in loader.datasets:
        return pn.pane.Markdown(f"Dataset **{dataset}** not found.")

    df = _selected_metric_frame(
        loader,
        metric_key,
        sort_metric_key=sort_metric_key,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
        dataset=dataset,
    ).rename(columns={"value": "mean"})

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


def _effective_summary_metric(
    model: str,
    metric: str | None,
    *,
    alt_metric_key: str | None,
    use_alt_metric: bool,
) -> str | None:
    """Resolve the metric row to show in the static summary table for one model."""
    if metric is None:
        return None
    if use_alt_metric and alt_metric_key and is_alt_model(model):
        return alt_metric_key
    return metric


def build_summary_table(
    loader: ResultsLoader,
    dataset: str | None = None,
    model: str | None = None,
    metric: str | None = None,
    *,
    alt_metric_key: str | None = None,
    use_alt_metric: bool = True,
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
        expected_metric = df["Model"].map(
            lambda model_name: _effective_summary_metric(
                str(model_name),
                metric,
                alt_metric_key=alt_metric_key,
                use_alt_metric=use_alt_metric,
            )
        )
        df = df[df["Metric"] == expected_metric]
    return pn.widgets.Tabulator(
        df.reset_index(drop=True),
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

def _format_metric_cell(mean_v: float, std_v: float, count_v: float, precision: int) -> str:
    """Format one HP impact metric cell."""
    if np.isnan(mean_v):
        return "-"
    std_display = 0.0 if np.isnan(std_v) else std_v
    count_display = 0 if np.isnan(count_v) else int(count_v)
    return f"{mean_v:.{precision}f} ± {std_display:.{precision}f} (n={count_display})"


def _css_attr_value(value: str) -> str:
    """Escape a string for use in a CSS attribute selector."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _sort_metric_highlight_css(metric_name: str | None) -> str:
    """Return CSS that highlights the active sort-metric column."""
    if not metric_name:
        return ""
    escaped = _css_attr_value(metric_name)
    return f"""
    .tabulator .tabulator-col[tabulator-field="{escaped}"] {{
        background: #ffe3bf;
    }}
    .tabulator .tabulator-cell[tabulator-field="{escaped}"] {{
        background: #fff4e5;
    }}
    """


def _prioritize_metric_columns(
    metric_names: list[str],
    display_metric_key: str | None,
    sort_metric_key: str | None,
) -> list[str]:
    """Order metrics as display metric, sort metric, then all remaining metrics."""
    prioritized: list[str] = []
    for metric_name in (display_metric_key, sort_metric_key):
        if metric_name and metric_name in metric_names and metric_name not in prioritized:
            prioritized.append(metric_name)
    return prioritized + [metric for metric in metric_names if metric not in prioritized]


def _build_hp_impact_frame(
    loader: ResultsLoader,
    dataset: str,
    model: str,
    *,
    metric_key: str,
    sort_metric_key: str | None,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
    precision: int = 4,
) -> tuple[pd.DataFrame | None, dict[str, object], list[str], list[str]]:
    """Build the sorted HP impact DataFrame and selection metadata."""
    ds, selection = loader.sorted_hp_configs(
        dataset,
        model,
        metric_key=metric_key,
        sort_metric_key=sort_metric_key,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
    )
    if ds is None or ds.sizes.get("config", 0) == 0 or "metric" not in ds.coords:
        return None, selection, [], []

    hp_coord_names = [c for c in ds.coords if c not in ("config", "metric")]
    metric_names = [str(metric) for metric in ds.coords["metric"].values.tolist()]
    metric_names = _prioritize_metric_columns(
        metric_names,
        selection.get("display_metric_key") if isinstance(selection, dict) else None,
        selection.get("effective_sort_metric_key") if isinstance(selection, dict) else None,
    )

    rows = []
    for i in range(ds.sizes["config"]):
        cfg = ds.isel(config=i)
        row = {hp: cfg.coords[hp].item() for hp in hp_coord_names}
        count_values = np.asarray(cfg["count"].values, dtype=float)
        first_count = count_values[0] if count_values.size else np.nan
        row["seeds"] = 0 if np.isnan(first_count) else int(first_count)
        for metric in metric_names:
            metric_slice = cfg.sel(metric=metric)
            mean_v = float(metric_slice["mean"].values)
            std_v = float(metric_slice["std"].values)
            count_v = float(metric_slice["count"].values)
            row[metric] = _format_metric_cell(mean_v, std_v, count_v, precision)
        rows.append(row)

    df = pd.DataFrame(rows)
    ordered_cols = (
        hp_coord_names
        + metric_names
        + (["seeds"] if "seeds" in df.columns else [])
    )
    return df[ordered_cols], selection, hp_coord_names, metric_names


def build_hp_impact_table(
    loader: ResultsLoader,
    dataset: str,
    model: str,
    metric_key: str,
    sort_metric_key: str | None,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
    precision: int = 4,
) -> pn.viewable.Viewable:
    df, selection, hp_cols, metric_cols = _build_hp_impact_frame(
        loader,
        dataset,
        model,
        metric_key=metric_key,
        sort_metric_key=sort_metric_key,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
        precision=precision,
    )
    if df is None:
        return pn.pane.Markdown(
            f"No HP config table found for **{dataset}** / **{model}**. "
            f"Re-run the report pipeline to generate it."
        )

    ordered_cols = hp_cols + metric_cols + (["seeds"] if "seeds" in df.columns else [])
    df = df[ordered_cols]

    # Pre-select only varying HP columns (nunique > 1) + seeds + metric cols
    stat_default = metric_cols + (["seeds"] if "seeds" in df.columns else [])
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
    highlight_css = _sort_metric_highlight_css(
        selection.get("effective_sort_metric_key")
        if isinstance(selection, dict)
        else None
    )

    def _table(cols):
        visible = cols if cols else ordered_cols
        return pn.widgets.Tabulator(
            df[visible],
            pagination="local",
            page_size=30,
            show_index=False,
            sortable=True,
            sizing_mode="stretch_width",
            stylesheets=[rotated_header_css, highlight_css],
        )

    return pn.Column(
        col_selector,
        pn.bind(_table, col_selector),
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# Metadata helpers and tables
# ---------------------------------------------------------------------------

def _display_metadata_value(value: object) -> str:
    """Render metadata values consistently for metadata tables."""
    if value is None:
        return "—"
    try:
        missing = pd.isna(value)
    except Exception:
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return "—"
    return str(value)


_MODEL_LINK_STYLESHEET = """
.tabulator .tabulator-cell[tabulator-field="Model"] {
  color: #0d6efd;
  text-decoration: underline;
  cursor: pointer;
}
"""


def _metadata_rows(
    loader: ResultsLoader,
    dataset: str,
    model: str,
    *,
    metric_key: str,
    sort_metric_key: str | None,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
) -> dict[str, str]:
    """Build the metadata property/value mapping for one dataset/model pair."""
    if dataset not in loader.datasets or model not in loader.models:
        return {}

    selection = loader.resolve_metric_selection(
        dataset,
        model,
        metric_key=metric_key,
        sort_metric_key=sort_metric_key,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
    )

    active_sort_metric = selection["effective_sort_metric_key"]
    if selection["sort_metric_fell_back"]:
        active_sort_metric = (
            f"{active_sort_metric} "
            f"(fallback from {selection['fallback_from_sort_metric_key']})"
        )

    return {
        "Dashboard metric": _display_metadata_value(selection["display_metric_key"]),
        "Dashboard sort metric": _display_metadata_value(active_sort_metric),
        "Report sort metric": _display_metadata_value(
            loader.metadata_value("sort_metric", dataset, model)
        ),
        "Opt metric": _display_metadata_value(
            loader.metadata_value("opt_metric", dataset, model)
        ),
        "Opt mode": _display_metadata_value(loader.metadata_value("opt_mode", dataset, model)),
        "Opt value": _display_metadata_value(
            loader.metadata_value("opt_value", dataset, model)
        ),
        "Total runs": _display_metadata_value(
            loader.metadata_value("total_runs", dataset, model)
        ),
        "Configs failed (seed)": _display_metadata_value(
            loader.metadata_value("configs_failed_seed", dataset, model)
        ),
        "Configs failed (metric)": _display_metadata_value(
            loader.metadata_value("configs_failed_metric", dataset, model)
        ),
    }


def build_model_summary_table(
    loader: ResultsLoader,
    dataset: str | None,
    *,
    metric_key: str,
    sort_metric_key: str | None,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
    on_model_click: Callable[[str], None] | None = None,
) -> pn.viewable.Viewable:
    """Build a model-summary table for one dataset using metadata-panel properties."""
    if dataset is None:
        return pn.pane.Markdown("Select a dataset to view model summary.")
    if dataset not in loader.datasets:
        return pn.pane.Markdown(f"Dataset **{dataset}** not found.")

    rows = []
    for model in loader.models:
        metadata = _metadata_rows(
            loader,
            dataset,
            model,
            metric_key=metric_key,
            sort_metric_key=sort_metric_key,
            alt_metric_key=alt_metric_key,
            alt_sort_metric_key=alt_sort_metric_key,
            use_alt_metric=use_alt_metric,
        )
        if not metadata:
            continue

        row = {"Model": _shorten_model_name(model), **metadata, "_model_key": model}
        # Skip models that do not have any meaningful values for this dataset.
        if all(value == "—" for key, value in row.items() if key not in {"Model", "_model_key"}):
            continue
        rows.append(row)

    if not rows:
        return pn.pane.Markdown(f"No model summary available for **{dataset}**.")

    df = pd.DataFrame(rows).fillna("—")
    table = pn.widgets.Tabulator(
        df,
        show_index=False,
        pagination="local",
        page_size=25,
        sizing_mode="stretch_width",
        hidden_columns=["_model_key"],
        stylesheets=[_MODEL_LINK_STYLESHEET],
    )
    if on_model_click is not None:
        def _handle_model_click(event: object) -> None:
            row = getattr(event, "row", None)
            if not isinstance(row, int) or row < 0 or row >= len(df):
                return
            model_key = df.iloc[row]["_model_key"]
            if isinstance(model_key, str):
                on_model_click(model_key)

        table.on_click(_handle_model_click, column="Model")
    return table


def build_metadata_panel(
    loader: ResultsLoader,
    dataset: str,
    model: str,
    *,
    metric_key: str,
    sort_metric_key: str | None,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
) -> pn.viewable.Viewable:
    metadata = _metadata_rows(
        loader,
        dataset,
        model,
        metric_key=metric_key,
        sort_metric_key=sort_metric_key,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
    )
    if not metadata:
        return pn.pane.Markdown("Select a valid dataset and model.")

    df = pd.DataFrame(list(metadata.items()), columns=["Property", "Value"])
    return pn.widgets.Tabulator(
        df,
        show_index=False,
        sizing_mode="stretch_width",
        width=400,
    )
