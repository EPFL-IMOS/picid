"""
Panel component builders for the piCID dashboard.

Each function takes a ResultsLoader and widget selections, returns a Panel object.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import panel as pn

from .data import ResultsLoader, is_alt_model


@dataclass
class _DatasetStats:
    """Per-dataset stats used to drive LaTeX cell highlighting."""

    best_models: set[str]
    best_mean: float
    best_std: float
    second_best_model: str | None = None
    within_1std_models: set[str] = field(default_factory=set)


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
    if " · " in name:
        source, *constraints = name.split(" · ")
        compact_constraints = []
        for constraint in constraints:
            hp_name, separator, value = constraint.partition("=")
            compact_name = hp_name.rsplit(".", 1)[-1]
            compact_constraints.append(
                f"{compact_name}={value}" if separator else constraint
            )
        constraint_label = " · ".join(compact_constraints)
        source_budget = max(8, max_len - len(constraint_label) - 3)
        source_label = _shorten_dataset_name(source, max_len=source_budget)
        if len(source_label) > source_budget:
            source_label = source_label[: source_budget - 1] + "…"
        return f"{source_label} · {constraint_label}"
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


def _dataset_display_labels(
    dataset_names: Collection[str], *, shorten_names: bool = True
) -> dict[str, str]:
    """Build readable labels without merging datasets that shorten identically."""
    names = list(dataset_names)
    base_labels = {
        name: _shorten_dataset_name(name) if shorten_names else name for name in names
    }
    label_counts: dict[str, int] = {}
    for label in base_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1

    occurrences: dict[str, int] = {}
    labels: dict[str, str] = {}
    for name in names:
        base = base_labels[name]
        if label_counts[base] == 1:
            labels[name] = base
            continue
        occurrences[base] = occurrences.get(base, 0) + 1
        labels[name] = f"{base} ({occurrences[base]})"
    return labels


def _dashboard_datasets(loader: ResultsLoader) -> list[str]:
    """Return the full catalog while tolerating minimal test loaders."""
    return list(getattr(loader, "dashboard_datasets", loader.datasets))


def _expanded_dataset(loader: ResultsLoader, dataset: str | None) -> object | None:
    """Resolve a virtual dataset spec when the loader supports expansion."""
    resolver = getattr(loader, "expanded_dataset", None)
    if dataset is None or not callable(resolver):
        return None
    return resolver(dataset)


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
    enabled_datasets: Collection[str] | None = None,
    enabled_models: Collection[str] | None = None,
) -> pd.DataFrame:
    """Build a long-form frame of dynamic dashboard values."""
    records = loader.selected_metric_records(
        metric_key=metric_key,
        sort_metric_key=sort_metric_key,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
        dataset=dataset,
        enabled_datasets=enabled_datasets,
        enabled_models=enabled_models,
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


def _average_duplicate_model_rows(long: pd.DataFrame) -> pd.DataFrame:
    """Average rows that share a dataset and canonical model label."""
    group_columns = ["dataset", "model"]
    mean_columns = {"value", "std", "n"}
    aggregations = {
        column: "mean" if column in mean_columns else "first"
        for column in long.columns
        if column not in group_columns
    }
    return long.groupby(
        group_columns,
        as_index=False,
        sort=False,
        dropna=False,
    ).agg(aggregations)


def _build_heatmap_frame(
    loader: ResultsLoader,
    metric_key: str,
    *,
    sort_metric_key: str | None,
    mode: str,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
    enabled_datasets: Collection[str] | None = None,
    enabled_models: Collection[str] | None = None,
    shorten_model_names: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build summary columns after optional model-name canonicalization."""
    long = _selected_metric_frame(
        loader,
        metric_key,
        sort_metric_key=sort_metric_key,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
        enabled_datasets=enabled_datasets,
        enabled_models=enabled_models,
    )
    long = long.dropna(subset=["value"]).copy()
    if long.empty:
        return long, [], []

    if shorten_model_names:
        long["model"] = long["model"].map(_shorten_model_name)
        long = _average_duplicate_model_rows(long)

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
    long["rank"] = (
        long.groupby("dataset")["value"]
        .rank(method="min", ascending=(mode == "min"))
        .astype(float)
    )

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
    enabled_datasets: Collection[str] | None = None,
    enabled_models: Collection[str] | None = None,
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
        enabled_datasets=enabled_datasets,
        enabled_models=enabled_models,
        shorten_model_names=True,
    )

    if long.empty:
        return pn.pane.Markdown(f"No data for metric **{metric_key}**.")

    # Model names were shortened before ranking and summary aggregation.
    # Dataset labels stay unique even when project-name abbreviation collides.
    long["source_dataset"] = long["dataset"]
    dataset_label_map = _dataset_display_labels(dataset_order)
    long["dataset"] = long["dataset"].map(dataset_label_map)
    dataset_order = [dataset_label_map[dataset_name] for dataset_name in dataset_order]

    n_models = len(model_order)
    row_height = max(30, min(60, 400 // max(n_models, 1)))
    plot_height = max(300, row_height * n_models)

    def _make_label(r):
        if r.get("summary_kind") == "average_rank":
            return f"{r['value']:.2f}"
        rank_str = (
            f"#{int(r['rank'])} " if (show_rank and pd.notna(r.get("rank"))) else ""
        )
        n_str = (
            f" (n={int(r['n'])})"
            if (show_n and pd.notna(r.get("n")))
            else (" (n=N/A)" if show_n else "")
        )
        return f"{rank_str}{r['value']:.2f}{n_str}"

    long["label"] = long.apply(_make_label, axis=1)
    effective_sort_metric = sort_metric_key or metric_key
    effective_alt_sort_metric = alt_sort_metric_key or "—"

    model_dim = hv.Dimension("model", values=model_order)
    dataset_dim = hv.Dimension("dataset", values=dataset_order)
    heatmap = hv.HeatMap(
        long,
        kdims=[dataset_dim, model_dim],
        vdims=["color", "value", "source_dataset"],
    ).opts(
        responsive=True,
        frame_height=plot_height,
        colorbar=False,
        clim=(0.0, 1.0),
        cmap="RdYlGn",
        tools=["hover"],
        xrotation=45,
        title=(
            f"{metric_key} "
            f"(sort: {effective_sort_metric}; alt sort: {effective_alt_sort_metric})"
        ),
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
# LaTeX table export
# ---------------------------------------------------------------------------


def _latex_escape(s: str) -> str:
    """Escape LaTeX special characters in plain-text strings (index / column names)."""
    # Backslash must come first to avoid double-escaping.
    s = s.replace("\\", "\\textbackslash{}")
    for ch, rep in [
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\^{}"),
    ]:
        s = s.replace(ch, rep)
    return s


def _markdown_escape(s: str) -> str:
    """Escape plain text for use in a Markdown table cell."""
    return str(s).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def build_latex_table(
    loader: ResultsLoader,
    metric_key: str,
    sort_metric_key: str | None = None,
    mode: str = "min",
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
    show_n: bool = True,
    show_rank: bool = False,
    show_std: bool = True,
    highlight_best_bg: bool = True,
    underline_2nd_best: bool = True,
    highlight_within_1std: bool = True,
    precision: int = 4,
    multiplier: float = 1.0,
    shorten_names: bool = True,
    rename_map: dict[str, str] | None = None,
    enabled_datasets: Collection[str] | None = None,
    enabled_models: Collection[str] | None = None,
) -> pn.viewable.Viewable:
    """Build LaTeX and Markdown tables, optionally preserving source names."""
    long, model_order, dataset_order = _build_heatmap_frame(
        loader,
        metric_key,
        sort_metric_key=sort_metric_key,
        mode=mode,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
        enabled_datasets=enabled_datasets,
        enabled_models=enabled_models,
    )

    if long.empty:
        return pn.pane.Markdown(f"No data for metric **{metric_key}**.")

    # Apply optional shortening before any user-defined rename.
    _renames = rename_map or {}

    def _display_model(name: str) -> str:
        default = _shorten_model_name(name) if shorten_names else name
        return _renames.get(default) or default

    dataset_label_map = _dataset_display_labels(
        dataset_order, shorten_names=shorten_names
    )

    def _display_dataset(name: str) -> str:
        default = dataset_label_map[name]
        return _renames.get(default) or default

    long["model"] = long["model"].map(_display_model)
    model_order = [_display_model(m) for m in model_order]
    dataset_order_display = [_display_dataset(d) for d in dataset_order]

    # Reverse so the best model (lowest avg rank from _build_heatmap_frame)
    # ends up at the bottom of the LaTeX table.
    model_order = list(reversed(model_order))

    # Per-dataset stats for highlighting (bold/grey best, underline 2nd best, blue within-1σ)
    dataset_rows = long[long["summary_kind"] == "dataset"]
    ds_stats: dict[str, _DatasetStats] = {}
    for ds, grp in dataset_rows.groupby("dataset"):
        valid = grp.dropna(subset=["value"])
        if valid.empty:
            continue
        best_idx = valid["value"].idxmin() if mode == "min" else valid["value"].idxmax()
        best_mean = float(valid.loc[best_idx, "value"])
        best_std_raw = valid.loc[best_idx, "std"]
        best_std = 0.0 if pd.isna(best_std_raw) else float(best_std_raw)

        # All models tied at best_mean (could be 1 or many).
        tied_at_best = valid[valid["value"] == best_mean]
        best_models = {str(m) for m in tied_at_best["model"].tolist()}

        # Second best: only meaningful when the top is a single row. If multiple
        # models tie at best_mean, there is no clear "second place".
        second_best_model: str | None = None
        if len(best_models) == 1:
            not_best = valid[valid["value"] != best_mean]
            if not not_best.empty:
                second_idx = (
                    not_best["value"].idxmin()
                    if mode == "min"
                    else not_best["value"].idxmax()
                )
                second_best_model = str(not_best.loc[second_idx, "model"])

        # Within-1σ band around best (using best's std). Excludes all tied bests.
        within_models: set[str] = set()
        if best_std > 0:
            band = (valid["value"] - best_mean).abs() <= best_std
            within = valid[band & (~valid["model"].isin(best_models))]
            within_models = {str(m) for m in within["model"].tolist()}

        ds_stats[str(ds)] = _DatasetStats(
            best_models=best_models,
            best_mean=best_mean,
            best_std=best_std,
            second_best_model=second_best_model,
            within_1std_models=within_models,
        )

    # Format each cell for both export formats.
    fmt = f".{precision}f"

    def _fmt_cell(r: pd.Series, output_format: Literal["latex", "markdown"]) -> str:
        if pd.isna(r["value"]):
            return "—"
        kind = r.get("summary_kind", "dataset")
        if kind == "average_rank":
            return f"{r['value']:.2f}"

        scaled_value = float(r["value"]) * multiplier
        scaled_std = (
            float(r["std"]) * multiplier if pd.notna(r.get("std")) else r.get("std")
        )
        std_str = (
            f" ± {scaled_std:{fmt}}" if (show_std and pd.notna(scaled_std)) else ""
        )
        n_str = f" (n={int(r['n'])})" if (show_n and pd.notna(r.get("n"))) else ""
        txt = f"{scaled_value:{fmt}}{std_str}{n_str}"
        if show_rank and kind == "dataset" and pd.notna(r.get("rank")):
            rank_prefix = (
                f"\\#{int(r['rank'])} "
                if output_format == "latex"
                else f"#{int(r['rank'])} "
            )
            txt = rank_prefix + txt

        stats = ds_stats.get(str(r["dataset"]))
        if stats is None:
            return txt

        model = str(r["model"])
        is_best = model in stats.best_models
        is_second = (not is_best) and stats.second_best_model == model
        is_within = (not is_best) and (model in stats.within_1std_models)

        if output_format == "latex":
            if is_best:
                txt = f"\\textbf{{{txt}}}"
            if is_second and underline_2nd_best:
                txt = f"\\underline{{{txt}}}"
            if is_best and highlight_best_bg:
                txt = f"\\cellcolor[gray]{{0.85}}{txt}"
            elif is_within and highlight_within_1std:
                txt = f"\\cellcolor{{blue!15}}{txt}"
            return txt

        if is_best:
            txt = f"**{txt}**"
        if is_second and underline_2nd_best:
            txt = f"<u>{txt}</u>"
        if is_best and highlight_best_bg:
            txt = f'<span style="background-color:#d9d9d9">{txt}</span>'
        elif is_within and highlight_within_1std:
            txt = f'<span style="background-color:#dbeafe">{txt}</span>'
        return txt

    long["latex_cell"] = long.apply(lambda row: _fmt_cell(row, "latex"), axis=1)
    long["markdown_cell"] = long.apply(lambda row: _fmt_cell(row, "markdown"), axis=1)

    def _pivot_cells(column: str) -> pd.DataFrame:
        result = long.pivot_table(
            index="model", columns="dataset", values=column, aggfunc="first"
        )
        result = result.reindex(index=model_order, columns=dataset_order)
        result.columns = dataset_order_display
        result.index.name = "Model"
        result.columns.name = None
        return result

    pivot = _pivot_cells("latex_cell")
    markdown_pivot = _pivot_cells("markdown_cell")

    pivot.index = pd.Index(
        [_latex_escape(m) for m in pivot.index], name=pivot.index.name
    )
    pivot.columns = pd.Index([_latex_escape(d) for d in pivot.columns])
    markdown_pivot.index = pd.Index(
        [_markdown_escape(m) for m in markdown_pivot.index],
        name=markdown_pivot.index.name,
    )
    markdown_pivot.columns = pd.Index(
        [_markdown_escape(d) for d in markdown_pivot.columns]
    )

    # Build LaTeX string
    effective_sort = sort_metric_key or metric_key
    safe_metric = metric_key.replace("/", "_").replace(".", "_")
    header_comment = (
        "% Requires: \\usepackage{booktabs}\n"
        "% Requires: \\usepackage[table]{xcolor}  % for \\cellcolor\n"
        "% Requires: \\usepackage{adjustbox}      % for max-width scaling\n"
        f"% Metric: {metric_key}  |  Sort: {effective_sort}  |  Mode: {mode}\n"
    )
    if multiplier != 1.0:
        header_comment += f"% Values multiplied by {multiplier:g}\n"
    header_comment += "\n"

    # Caption: display metric + sort metric + a "display = full name" dataset legend
    # for any dataset whose displayed label differs from its full name.
    legend_entries: list[str] = []
    for full, short in zip(dataset_order, dataset_order_display, strict=False):
        if full in ("Average", "Average rank"):
            continue
        if short != full:
            legend_entries.append(f"{_latex_escape(short)} = {_latex_escape(full)}")

    caption_parts = [
        f"Results for display metric \\texttt{{{_latex_escape(metric_key)}}} "
        f"(sorted by \\texttt{{{_latex_escape(effective_sort)}}}, mode: {mode})."
    ]
    if legend_entries:
        caption_parts.append("Datasets: " + "; ".join(legend_entries) + ".")
    if multiplier != 1.0:
        caption_parts.append(f"Values multiplied by {multiplier:g}.")
    caption = " ".join(caption_parts)

    latex_body = pivot.to_latex(
        escape=False,
        caption=caption,
        label=f"tab:results_{safe_metric}",
        column_format="l" + "r" * len(pivot.columns),
    )
    # Wrap the inner tabular in adjustbox so wide tables shrink to fit \textwidth
    latex_body = latex_body.replace(
        "\\begin{tabular}",
        "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}",
        1,
    ).replace(
        "\\end{tabular}",
        "\\end{tabular}\n\\end{adjustbox}",
        1,
    )
    latex_str = header_comment + latex_body

    def _markdown_code(value: str) -> str:
        escaped = _markdown_escape(value).replace("`", "&#96;")
        return f"`{escaped}`"

    markdown_lines = [
        f"**Display metric:** {_markdown_code(metric_key)}  ",
        f"**Sort metric:** {_markdown_code(effective_sort)}  ",
        f"**Mode:** {_markdown_code(mode)}",
    ]
    if multiplier != 1.0:
        markdown_lines.append(f"**Values multiplied by:** {multiplier:g}")
    markdown_lines.append("")

    headers = ["Model", *[str(column) for column in markdown_pivot.columns]]
    markdown_lines.append("| " + " | ".join(headers) + " |")
    markdown_lines.append(
        "| " + " | ".join([":---", *(["---:"] * len(markdown_pivot.columns))]) + " |"
    )
    for model_name, row in markdown_pivot.iterrows():
        markdown_lines.append(
            "| "
            + " | ".join([str(model_name), *[str(value) for value in row.tolist()]])
            + " |"
        )

    markdown_legend = []
    for full, short in zip(dataset_order, dataset_order_display, strict=False):
        if full not in ("Average", "Average rank") and short != full:
            markdown_legend.append(f"{_markdown_code(short)} = {_markdown_code(full)}")
    if markdown_legend:
        markdown_lines.extend(["", "**Datasets:** " + "; ".join(markdown_legend)])
    markdown_str = "\n".join(markdown_lines)

    textarea = pn.widgets.TextAreaInput(
        value=latex_str,
        disabled=True,
        height=500,
        sizing_mode="stretch_width",
    )
    format_btn = pn.widgets.Button(
        name="Show Markdown",
        button_type="default",
        width=160,
    )

    def _toggle_output_format(_event: object) -> None:
        showing_latex = textarea.value == latex_str
        textarea.value = markdown_str if showing_latex else latex_str
        format_btn.name = "Show LaTeX" if showing_latex else "Show Markdown"

    format_btn.on_click(_toggle_output_format)
    copy_btn = pn.widgets.Button(
        name="Copy to clipboard",
        button_type="primary",
        width=200,
    )
    copy_btn.js_on_click(
        args={"ta": textarea},
        code="navigator.clipboard.writeText(ta.value)",
    )
    return pn.Column(
        pn.Row(format_btn, copy_btn),
        textarea,
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# Parallel coordinates overview
# ---------------------------------------------------------------------------


def _build_parallel_coordinates_frame(
    loader: ResultsLoader,
    metric_key: str,
    *,
    sort_metric_key: str | None,
    mode: str,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
    enabled_datasets: Collection[str] | None = None,
    enabled_models: Collection[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a wide overview frame for the parallel-coordinates plot."""
    long, model_order, axis_order = _build_heatmap_frame(
        loader,
        metric_key,
        sort_metric_key=sort_metric_key,
        mode=mode,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
        enabled_datasets=enabled_datasets,
        enabled_models=enabled_models,
    )
    if long.empty:
        return pd.DataFrame(columns=["model", "Model"]), [], axis_order

    wide = long.pivot_table(
        index="model", columns="dataset", values="value", aggfunc="first"
    )
    ordered_axes = [axis_name for axis_name in axis_order if axis_name in wide.columns]
    wide = wide.reindex(index=model_order).reindex(columns=ordered_axes)
    wide = wide.apply(pd.to_numeric, errors="coerce")
    wide = wide[wide.notna().any(axis=1)]
    if wide.empty:
        return pd.DataFrame(columns=["model", "Model"]), [], ordered_axes

    ordered_models = [
        model_name for model_name in model_order if model_name in wide.index
    ]
    wide = wide.loc[ordered_models].reset_index()
    wide.insert(1, "Model", wide["model"].map(_shorten_model_name))
    return wide, ordered_models, ordered_axes


def _plotly_axis_range(
    values: pd.Series | np.ndarray, *, reverse: bool = False
) -> list[float] | None:
    """Compute a Plotly axis range, expanding degenerate ranges slightly."""
    numeric = np.asarray(values, dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return None

    lo = float(finite.min())
    hi = float(finite.max())
    if lo == hi:
        delta = max(abs(lo) * 0.05, 1.0)
        lo -= delta
        hi += delta
    return [hi, lo] if reverse else [lo, hi]


def _parallel_coordinates_line_colors(models: list[str]) -> list[str]:
    """Return colors for model lines using Matplotlib's default color cycle."""
    if not models:
        return []

    from matplotlib import rcParams

    palette = list(rcParams["axes.prop_cycle"].by_key().get("color", ()))
    if not palette:
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    repeats = (len(models) + len(palette) - 1) // len(palette)
    return (palette * repeats)[: len(models)]


def _plotly_discrete_colorscale(colors: list[str]) -> list[list[float | str]]:
    """Build a discrete Plotly colorscale from a list of colors."""
    if not colors:
        return []
    if len(colors) == 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]

    step = 1.0 / len(colors)
    colorscale: list[list[float | str]] = []
    for index, color in enumerate(colors):
        start = index * step
        end = (index + 1) * step
        colorscale.append([start, color])
        colorscale.append([end, color])
    return colorscale


def build_parallel_coordinates(
    loader: ResultsLoader,
    metric_key: str,
    sort_metric_key: str | None = None,
    mode: str = "min",
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
    enabled_datasets: Collection[str] | None = None,
    enabled_models: Collection[str] | None = None,
) -> pn.viewable.Viewable:
    """Parallel-coordinates overview of per-model selected metric values."""
    import plotly.graph_objects as go

    frame, ordered_models, axis_order = _build_parallel_coordinates_frame(
        loader,
        metric_key,
        sort_metric_key=sort_metric_key,
        mode=mode,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
        enabled_datasets=enabled_datasets,
        enabled_models=enabled_models,
    )

    if frame.empty:
        return pn.pane.Markdown(f"No data for metric **{metric_key}**.")

    model_positions = np.arange(len(frame), dtype=float)
    model_range = [float(len(frame) - 1), 0.0] if len(frame) > 1 else [0.5, -0.5]
    dimensions: list[dict[str, object]] = [
        {
            "label": "Model",
            "values": model_positions.tolist(),
            "range": model_range,
            "tickvals": model_positions.tolist(),
            "ticktext": frame["Model"].tolist(),
        }
    ]

    for axis_name in axis_order:
        values = frame[axis_name].astype(float)
        axis_range = _plotly_axis_range(
            values,
            reverse=(axis_name == "Average rank")
            or (mode == "min" and axis_name != "Average rank"),
        )
        if axis_range is None:
            continue
        dimensions.append(
            {
                "label": (
                    axis_name
                    if axis_name in {"Average", "Average rank"}
                    else _shorten_dataset_name(axis_name)
                ),
                "values": values.tolist(),
                "range": axis_range,
            }
        )

    line_colors = _parallel_coordinates_line_colors(ordered_models)
    line_color_values = list(range(len(ordered_models)))

    fig = go.Figure(
        data=go.Parcoords(
            line={
                "color": line_color_values,
                "colorscale": _plotly_discrete_colorscale(line_colors),
                "cmin": -0.5,
                "cmax": max(len(line_colors) - 0.5, 0.5),
                "showscale": False,
            },
            dimensions=dimensions,
            labelfont={"size": 12},
            tickfont={"size": 10},
            labelside="bottom",
            labelangle=-45,
        )
    )
    fig.update_layout(
        title=f"{metric_key} (selected by active sort metric)",
        height=max(520, 120 + 45 * len(ordered_models)),
        margin={"l": 60, "r": 60, "t": 60, "b": 180},
    )
    return pn.pane.Plotly(
        fig,
        config={"responsive": True, "displaylogo": False},
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# Spiderweb overview
# ---------------------------------------------------------------------------


def _normalize_spiderweb_scores(values: pd.Series, mode: str) -> pd.Series:
    """Normalize one dataset spoke across models to a 0..1 spiderweb radius."""
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.dropna()
    if finite.empty:
        return pd.Series(np.nan, index=numeric.index, dtype=float)

    lo = float(finite.min())
    hi = float(finite.max())
    if hi == lo:
        return pd.Series(
            np.where(numeric.notna(), 0.5, np.nan), index=numeric.index, dtype=float
        )

    normalized = (numeric - lo) / (hi - lo)
    if mode == "min":
        normalized = 1.0 - normalized
    return normalized.astype(float)


def _build_spiderweb_frame(
    loader: ResultsLoader,
    metric_key: str,
    *,
    sort_metric_key: str | None,
    mode: str,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
    enabled_datasets: Collection[str] | None = None,
    enabled_models: Collection[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build raw and normalized dataset-by-model frames for the spiderweb view."""
    raw_long = _selected_metric_frame(
        loader,
        metric_key,
        sort_metric_key=sort_metric_key,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
        enabled_datasets=enabled_datasets,
        enabled_models=enabled_models,
    )
    if raw_long.empty:
        empty = pd.DataFrame()
        return empty, empty, [], []

    _heatmap_long, model_order, _dataset_order = _build_heatmap_frame(
        loader,
        metric_key,
        sort_metric_key=sort_metric_key,
        mode=mode,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
        enabled_datasets=enabled_datasets,
        enabled_models=enabled_models,
    )
    if not model_order:
        empty = pd.DataFrame()
        return empty, empty, [], []

    raw_matrix = raw_long.pivot_table(
        index="dataset",
        columns="model",
        values="value",
        aggfunc="first",
    )
    dataset_order = [
        dataset for dataset in loader.datasets if dataset in raw_matrix.index
    ]
    raw_matrix = raw_matrix.reindex(index=dataset_order, columns=model_order)
    normalized_matrix = raw_matrix.apply(
        lambda series: _normalize_spiderweb_scores(series, mode), axis=1
    )

    keep_mask = normalized_matrix.notna().any(axis=1)
    raw_matrix = raw_matrix.loc[keep_mask]
    normalized_matrix = normalized_matrix.loc[keep_mask]
    dataset_order = [
        dataset for dataset in dataset_order if dataset in raw_matrix.index
    ]
    return raw_matrix, normalized_matrix, model_order, dataset_order


def _spiderweb_palette(count: int) -> list[str]:
    """Return a stable palette for model polygons."""
    from bokeh.palettes import Category10, Category20

    if count <= 10:
        palette = list(Category10[10])
    else:
        palette = list(Category20[20])
    if count <= len(palette):
        return palette[:count]
    repeats = (count + len(palette) - 1) // len(palette)
    return (palette * repeats)[:count]


def _build_spiderweb_interactive(
    raw_matrix: pd.DataFrame,
    normalized_matrix: pd.DataFrame,
    model_order: list[str],
    dataset_order: list[str],
    *,
    metric_key: str,
    selected_dataset: str | None,
) -> pn.pane.HoloViews:
    """Render the existing interactive spiderweb using HoloViews/Bokeh."""
    import holoviews as hv

    hv.extension("bokeh")

    angles = np.linspace(
        np.pi / 2,
        np.pi / 2 - (2 * np.pi),
        len(dataset_order),
        endpoint=False,
    )
    closed_angles = np.append(angles, angles[0])

    ring_radii = [0.25, 0.5, 0.75, 1.0]
    ring_paths = [
        pd.DataFrame(
            {
                "x": radius * np.cos(closed_angles),
                "y": radius * np.sin(closed_angles),
            }
        )
        for radius in ring_radii
    ]
    rings = hv.Path(ring_paths, kdims=["x", "y"]).opts(
        color="#d1d5db",
        line_width=1,
        alpha=0.8,
    )

    spoke_radius = 1.02
    spokes_df = pd.DataFrame(
        {
            "x0": np.zeros(len(dataset_order), dtype=float),
            "y0": np.zeros(len(dataset_order), dtype=float),
            "x1": spoke_radius * np.cos(angles),
            "y1": spoke_radius * np.sin(angles),
            "dataset": dataset_order,
        }
    )
    selected_spokes = spokes_df[spokes_df["dataset"] == selected_dataset]
    base_spokes = spokes_df[spokes_df["dataset"] != selected_dataset]
    spokes = hv.Segments(base_spokes, kdims=["x0", "y0", "x1", "y1"]).opts(
        color="#d1d5db",
        line_width=1,
        alpha=0.9,
    )
    highlighted_spoke = hv.Segments(
        selected_spokes, kdims=["x0", "y0", "x1", "y1"]
    ).opts(
        color="#2563eb",
        line_width=2.5,
        alpha=0.95,
    )

    label_radius = 1.15
    labels_df = pd.DataFrame(
        {
            "x": label_radius * np.cos(angles),
            "y": label_radius * np.sin(angles),
            "text": [_shorten_dataset_name(dataset) for dataset in dataset_order],
            "dataset": dataset_order,
        }
    )
    base_labels = hv.Labels(
        labels_df[labels_df["dataset"] != selected_dataset], ["x", "y"], "text"
    ).opts(
        text_color="#374151",
        text_font_size="8pt",
    )
    selected_labels = hv.Labels(
        labels_df[labels_df["dataset"] == selected_dataset], ["x", "y"], "text"
    ).opts(
        text_color="#2563eb",
        text_font_size="9pt",
    )

    overlay = rings * spokes * highlighted_spoke * base_labels * selected_labels
    palette = _spiderweb_palette(len(model_order))
    for model_name, color in zip(model_order, palette, strict=False):
        model_scores = normalized_matrix[model_name]
        model_values = raw_matrix[model_name]
        path_df = pd.DataFrame(
            {
                "x": np.append(
                    model_scores.to_numpy(dtype=float) * np.cos(angles), np.nan
                ),
                "y": np.append(
                    model_scores.to_numpy(dtype=float) * np.sin(angles), np.nan
                ),
                "Model": _shorten_model_name(model_name),
            }
        )
        if len(path_df) > 1:
            path_df.iloc[-1] = {
                "x": path_df.iloc[0]["x"],
                "y": path_df.iloc[0]["y"],
                "Model": path_df.iloc[0]["Model"],
            }

        model_path = (
            hv.Path(path_df, kdims=["x", "y"], vdims=["Model"])
            .relabel(_shorten_model_name(model_name))
            .opts(
                color=color,
                line_width=2.2,
                alpha=0.65,
                muted_alpha=0.08,
                show_legend=True,
            )
        )

        point_rows = []
        for dataset_name, angle in zip(dataset_order, angles, strict=False):
            score = model_scores[dataset_name]
            raw_value = model_values[dataset_name]
            if pd.isna(score):
                continue
            point_rows.append(
                {
                    "x": float(score) * float(np.cos(angle)),
                    "y": float(score) * float(np.sin(angle)),
                    "Model": _shorten_model_name(model_name),
                    "Dataset": _shorten_dataset_name(dataset_name),
                    "Raw value": float(raw_value),
                    "Normalized score": float(score),
                }
            )
        if point_rows:
            model_points = hv.Points(
                pd.DataFrame(point_rows),
                kdims=["x", "y"],
                vdims=["Model", "Dataset", "Raw value", "Normalized score"],
            ).opts(
                color=color,
                size=6,
                alpha=0.75,
                tools=["hover"],
                show_legend=False,
            )
            overlay = overlay * model_path * model_points
        else:
            overlay = overlay * model_path

    overlay = overlay.opts(
        responsive=True,
        min_height=700,
        aspect="equal",
        xaxis=None,
        yaxis=None,
        show_frame=False,
        show_grid=False,
        padding=0.15,
        toolbar="above",
        legend_position="top_right",
        title=f"{metric_key} normalized spiderweb across datasets (configs selected by active sort metric)",
    ).redim.range(x=(-1.25, 1.25), y=(-1.25, 1.25))
    return pn.pane.HoloViews(overlay, sizing_mode="stretch_width")


def _build_spiderweb_matplotlib(
    raw_matrix: pd.DataFrame,
    normalized_matrix: pd.DataFrame,
    model_order: list[str],
    dataset_order: list[str],
    *,
    metric_key: str,
    selected_dataset: str | None,
) -> pn.pane.Matplotlib:
    """Render the spiderweb as a static Matplotlib radar chart."""
    import matplotlib.pyplot as plt

    angles = np.linspace(0.0, 2.0 * np.pi, len(dataset_order), endpoint=False)
    closed_angles = np.append(angles, angles[0])
    short_dataset_names = [_shorten_dataset_name(dataset) for dataset in dataset_order]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(angles)
    ax.set_xticklabels(short_dataset_names)
    ax.tick_params(axis="x", pad=12)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_rlabel_position(90)
    ax.yaxis.grid(True, color="#d1d5db", linewidth=1, alpha=0.8)
    ax.xaxis.grid(False)
    ax.spines["polar"].set_color("#d1d5db")
    ax.spines["polar"].set_linewidth(1)

    for dataset_name, angle in zip(dataset_order, angles, strict=False):
        is_selected = dataset_name == selected_dataset
        ax.plot(
            [angle, angle],
            [0.0, 1.02],
            color="#2563eb" if is_selected else "#d1d5db",
            linewidth=2.5 if is_selected else 1,
            alpha=0.95 if is_selected else 0.9,
            zorder=1,
        )

    for tick_label, dataset_name in zip(
        ax.get_xticklabels(), dataset_order, strict=False
    ):
        is_selected = dataset_name == selected_dataset
        tick_label.set_color("#2563eb" if is_selected else "#374151")
        tick_label.set_fontsize(9 if is_selected else 8)

    colors = _parallel_coordinates_line_colors(model_order)
    for model_name, color in zip(model_order, colors, strict=False):
        model_scores = normalized_matrix[model_name].to_numpy(dtype=float)
        closed_scores = np.append(
            model_scores,
            model_scores[0] if len(model_scores) else np.nan,
        )
        ax.plot(
            closed_angles,
            closed_scores,
            color=color,
            linewidth=2.2,
            alpha=0.8,
            label=_shorten_model_name(model_name),
            zorder=2,
        )
        ax.scatter(
            angles,
            model_scores,
            color=color,
            s=30,
            alpha=0.85,
            zorder=3,
        )

    ax.set_title(
        f"{metric_key} normalized spiderweb across datasets (configs selected by active sort metric)",
        pad=24,
    )
    if model_order:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.05, 1.02),
            borderaxespad=0.0,
            frameon=False,
        )
    fig.tight_layout()
    return pn.pane.Matplotlib(fig, sizing_mode="stretch_width")


def build_spiderweb(
    loader: ResultsLoader,
    metric_key: str,
    sort_metric_key: str | None = None,
    mode: str = "min",
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
    selected_dataset: str | None = None,
    plot_mode: Literal["interactive", "matplotlib"] = "interactive",
    enabled_datasets: Collection[str] | None = None,
    enabled_models: Collection[str] | None = None,
) -> pn.viewable.Viewable:
    """Spiderweb overview with dataset spokes and one polygon per model."""
    raw_matrix, normalized_matrix, model_order, dataset_order = _build_spiderweb_frame(
        loader,
        metric_key,
        sort_metric_key=sort_metric_key,
        mode=mode,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
        enabled_datasets=enabled_datasets,
        enabled_models=enabled_models,
    )

    if raw_matrix.empty or normalized_matrix.empty:
        return pn.pane.Markdown(f"No data for metric **{metric_key}**.")

    if plot_mode == "interactive":
        return _build_spiderweb_interactive(
            raw_matrix,
            normalized_matrix,
            model_order,
            dataset_order,
            metric_key=metric_key,
            selected_dataset=selected_dataset,
        )
    if plot_mode == "matplotlib":
        return _build_spiderweb_matplotlib(
            raw_matrix,
            normalized_matrix,
            model_order,
            dataset_order,
            metric_key=metric_key,
            selected_dataset=selected_dataset,
        )
    raise ValueError(f"Unsupported spiderweb plot_mode: {plot_mode}")


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
    enabled_models: Collection[str] | None = None,
) -> pn.viewable.Viewable:
    """Horizontal bar chart with error bars for a single dataset."""
    import holoviews as hv

    hv.extension("bokeh")

    if dataset not in _dashboard_datasets(loader):
        return pn.pane.Markdown(f"Dataset **{dataset}** not found.")

    df = _selected_metric_frame(
        loader,
        metric_key,
        sort_metric_key=sort_metric_key,
        alt_metric_key=alt_metric_key,
        alt_sort_metric_key=alt_sort_metric_key,
        use_alt_metric=use_alt_metric,
        dataset=dataset,
        enabled_models=enabled_models,
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
        labelled=["x"],  # keep value axis label, suppress category label
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
    enabled_models: Collection[str] | None = None,
    sort_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    mode: str = "min",
) -> pn.viewable.Viewable:
    expanded = _expanded_dataset(loader, dataset)
    if expanded is not None:
        records = loader.selected_metric_records(
            metric_key=metric or "",
            sort_metric_key=sort_metric_key,
            alt_metric_key=alt_metric_key,
            alt_sort_metric_key=alt_sort_metric_key,
            use_alt_metric=use_alt_metric,
            dataset=dataset,
            enabled_models=enabled_models,
        )
        rows = []
        for record in records:
            constraints = ", ".join(
                f"{name}={value}" for name, value in record["hp_constraints"].items()
            )
            rows.append(
                {
                    "Dataset": dataset,
                    "Source Dataset": record["source_dataset"],
                    "Model": record["model"],
                    "Metric": record["display_metric_key"],
                    "Mean": record["value"],
                    "Std": record["std"],
                    "Runs": record["n"],
                    "HP constraints": constraints,
                    "Matching Configs": record["matching_configs"],
                    "Selected Config": record["config_index"],
                }
            )
        dynamic = pd.DataFrame(rows)
        if not dynamic.empty:
            dynamic["Rank"] = dynamic["Mean"].rank(
                method="min",
                ascending=mode == "min",
                na_option="bottom",
            )
        return pn.widgets.Tabulator(
            dynamic,
            pagination="local",
            page_size=25,
            show_index=False,
            sizing_mode="stretch_width",
        )
    df = loader.summary_df
    if df is None:
        return pn.pane.Markdown("No `summary.csv` found in report output.")
    if dataset:
        ds_base, project = _parse_dataset_filter(dataset)
        df = df[df["Dataset"] == ds_base]
        if project:
            df = df[df["Project"] == project]
    if enabled_models is not None:
        df = df[df["Model"].isin(enabled_models)]
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
    enabled_models: Collection[str] | None = None,
    metric_key: str | None = None,
    sort_metric_key: str | None = None,
    alt_metric_key: str | None = None,
    alt_sort_metric_key: str | None = None,
    use_alt_metric: bool = True,
) -> pn.viewable.Viewable:
    expanded = _expanded_dataset(loader, dataset)
    if expanded is not None:
        records = loader.selected_metric_records(
            metric_key=metric_key or "",
            sort_metric_key=sort_metric_key,
            alt_metric_key=alt_metric_key,
            alt_sort_metric_key=alt_sort_metric_key,
            use_alt_metric=use_alt_metric,
            dataset=dataset,
            enabled_models=enabled_models,
        )
        if model is not None:
            records = [record for record in records if record["model"] == model]
        rows = []
        for record in records:
            constraints = ", ".join(
                f"{name}={value}" for name, value in record["hp_constraints"].items()
            )
            rows.append(
                {
                    "Dataset": dataset,
                    "Source Dataset": record["source_dataset"],
                    "Model": record["model"],
                    "Matching Configs": record["matching_configs"],
                    "Selected Config": record["config_index"],
                    "Runs in Selected Config": record["n"],
                    "HP constraints": constraints,
                }
            )
        return pn.widgets.Tabulator(
            pd.DataFrame(rows),
            pagination="local",
            page_size=25,
            show_index=False,
            sizing_mode="stretch_width",
        )
    df = loader.stats_df
    if df is None:
        return pn.pane.Markdown("No `experiment_stats.csv` found in report output.")
    if dataset:
        ds_base, project = _parse_dataset_filter(dataset)
        df = df[df["Dataset"] == ds_base]
        if project:
            df = df[df["Project"] == project]
    if enabled_models is not None:
        df = df[df["Model"].isin(enabled_models)]
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


def _format_metric_cell(
    mean_v: float, std_v: float, count_v: float, precision: int
) -> str:
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
        if (
            metric_name
            and metric_name in metric_names
            and metric_name not in prioritized
        ):
            prioritized.append(metric_name)
    return prioritized + [
        metric for metric in metric_names if metric not in prioritized
    ]


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
        selection.get("effective_sort_metric_key")
        if isinstance(selection, dict)
        else None,
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
        hp_coord_names + metric_names + (["seeds"] if "seeds" in df.columns else [])
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

    expanded = _expanded_dataset(loader, dataset)
    if expanded is not None:
        constraints = ", ".join(
            f"`{name}={value}`" for name, value in expanded.constraints.items()
        )
        filter_notice = pn.pane.Alert(
            f"Filtered to {constraints}. Unselected hyperparameters remain "
            f"visible and are ranked by `{selection['effective_sort_metric_key']}`.",
            alert_type="info",
        )
    else:
        filter_notice = pn.layout.Spacer(height=0)

    return pn.Column(
        filter_notice,
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
    if dataset not in _dashboard_datasets(loader) or model not in loader.models:
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

    expanded = _expanded_dataset(loader, dataset)
    expanded_metadata: dict[str, str] = {}
    if expanded is not None:
        record = loader.selected_metric_record(
            dataset,
            model,
            metric_key=metric_key,
            sort_metric_key=sort_metric_key,
            alt_metric_key=alt_metric_key,
            alt_sort_metric_key=alt_sort_metric_key,
            use_alt_metric=use_alt_metric,
        )
        expanded_metadata = {
            "Source dataset": record["source_dataset"],
            "HP constraints": ", ".join(
                f"{name}={value}" for name, value in record["hp_constraints"].items()
            ),
            "Matching configs": _display_metadata_value(record["matching_configs"]),
            "Selected config": _display_metadata_value(record["config_index"]),
            "Displayed score": _display_metadata_value(record["value"]),
            "Displayed std": _display_metadata_value(record["std"]),
            "Runs in selected config": _display_metadata_value(record["n"]),
        }

    return {
        "Dashboard metric": _display_metadata_value(selection["display_metric_key"]),
        "Dashboard sort metric": _display_metadata_value(active_sort_metric),
        "Report sort metric": _display_metadata_value(
            loader.metadata_value("sort_metric", dataset, model)
        ),
        "Opt metric": _display_metadata_value(
            loader.metadata_value("opt_metric", dataset, model)
        ),
        "Opt mode": _display_metadata_value(
            loader.metadata_value("opt_mode", dataset, model)
        ),
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
        **expanded_metadata,
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
    enabled_models: Collection[str] | None = None,
) -> pn.viewable.Viewable:
    """Build a model-summary table for one dataset using metadata-panel properties."""
    if dataset is None:
        return pn.pane.Markdown("Select a dataset to view model summary.")
    if dataset not in _dashboard_datasets(loader):
        return pn.pane.Markdown(f"Dataset **{dataset}** not found.")
    models = loader.models
    if enabled_models is not None:
        enabled = set(enabled_models)
        models = [model for model in models if model in enabled]

    rows = []
    for model in models:
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
        if all(
            value == "—"
            for key, value in row.items()
            if key not in {"Model", "_model_key"}
        ):
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
        width=400,
    )
