# picid_report/report/plots.py
"""
Paper-ready plotting helpers. Reads from all_results (output of analyze_results).

Requires matplotlib. If not installed, plot functions raise a clear error.
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Sequence, Union

if TYPE_CHECKING:
    import xarray as xr

import numpy as np
import pandas as pd

from picid_report.core.analysis import _ds_is_empty

# Canonical short display names for known model class names (last dotted component).
_MODEL_SHORT_NAMES = {
    "LSTM_Forecaster": "LSTM",
    "PatchTST_Forecaster": "PatchTST",
    "TiDE_Forecaster": "TiDE",
    "Spacetimeformer_Forecaster": "Spacetimeformer",
    "Timeseries_Transformer_Forecaster": "TSTransformer",
    "Crossformer_Forecaster": "Crossformer",
    "MLPWrapper": "MLP",
    "CNN1D_Wrapper": "CNN1D",
    "FitPredictXGBoostWrapper": "XGBoost",
    "FitPredictTabDPTWrapper": "TabDPT",
    "FitPredictTabPFNWrapper": "TabPFN",
    "StatisticalBaselineWrapper": "StatBaseline",
    "StatisticalBaselineWrapper (linear)": "StatBaseline (lin)",
    "StatisticalBaselineWrapper (exponential)": "StatBaseline (exp)",
}

# Colorblind-safe palette (Okabe-Ito) for categorical assignments.
_OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]


def _short_model_name(full_name: str) -> str:
    """Return a short display name for a model class path."""
    if full_name in _MODEL_SHORT_NAMES:
        return _MODEL_SHORT_NAMES[full_name]
    last = full_name.split(".")[-1]
    return _MODEL_SHORT_NAMES.get(last, last)


def _short_dataset_name(name: str) -> str:
    """Return a short display name for a dataset key."""
    return name.replace("_", " ").strip()


def _ensure_matplotlib():
    """Raise a clear error if matplotlib is not available."""
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from e


def plot_best_metric_bars(
    all_results: defaultdict,
    metric: Union[str, Sequence[str]] = "test/mse",
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> "object":
    """Bar chart of best metric value per model/dataset.

    Parameters
    ----------
    all_results : defaultdict
        Result of analyze_results (dataset -> model -> result dict).
    metric : str or list[str], optional
        Metric key(s) in prefix/name form, e.g. "test/mse". If a list is provided,
        a figure is produced for each metric. Default "test/mse".
    save_path : str, optional
        If set, save figure to this path.
    figsize : tuple, optional
        (width, height) in inches for the figure.

    Returns
    -------
    matplotlib figure, list of figures, or None if no data.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    metrics = [metric] if isinstance(metric, str) else list(metric)

    def _plot_single(metric_key: str):
        parts = metric_key.split("/", 1)
        prefix = parts[0] if len(parts) == 2 else ""
        metric_name = parts[1] if len(parts) == 2 else metric_key

        labels = []
        means = []
        stds = []
        counts = []
        datasets_seen = set()
        for dataset, models in all_results.items():
            for model, res in models.items():
                bp = res.get("best_performance", {}).get("metrics", {})
                if metric_name not in bp or prefix not in bp[metric_name]:
                    continue
                vals = bp[metric_name][prefix]
                std = vals.get("std")
                cnt = vals.get("cnt")
                mean = vals.get("mean")
                if mean is None or pd.isna(mean):
                    continue
                human_readable_model = model.split(".")[-1]
                datasets_seen.add(dataset)
                labels.append((human_readable_model, dataset))
                means.append(float(mean))
                std_val = float(std) if std is not None and not pd.isna(std) else 0.0
                stds.append(std_val)
                cnt_val = int(cnt) if cnt is not None and not pd.isna(cnt) else None
                counts.append(cnt_val)

        if not labels:
            return None

        multi_dataset = len(datasets_seen) > 1
        tick_labels = [f"{m}\n{d}" if multi_dataset else m for m, d in labels]
        dataset_name = next(iter(datasets_seen))

        if figsize is None:
            local_figsize = (max(6, len(labels) * 0.5), 5)
        else:
            local_figsize = figsize
        fig, ax = plt.subplots(figsize=local_figsize)
        x = range(len(labels))
        any_std = any(s > 0 for s in stds)
        bar_container = ax.bar(
            x,
            means,
            yerr=stds if any_std else None,
            capsize=4 if any_std else 0,
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_ylabel(metric_key)
        ax.set_title(f"Best {metric_key} on {dataset_name}")
        text_offset = max(max(stds, default=0), max(means) * 0.02)
        for bar, std_val, cnt_val in zip(bar_container, stds, counts):
            height = bar.get_height()
            pieces = []
            if std_val:
                pieces.append(f"σ={std_val:.3g}")
            if cnt_val is not None:
                pieces.append(f"n={cnt_val}")
            if pieces:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + std_val + text_offset,
                    "\n".join(pieces),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

    figures = []
    for metric_key in metrics:
        fig = _plot_single(metric_key)
        if fig is not None:
            figures.append(fig)

    if isinstance(metric, str):
        return figures[0] if figures else None
    return figures


def plot_hp_impact(
    all_results: defaultdict,
    model: str,
    dataset: str,
    metric: str,
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
    max_configs: int = 50,
) -> "object":
    """Plot metric vs HP configuration for one model/dataset (strip/bar).

    Uses sorted_aggregated_results; each bar is one HP configuration.

    Parameters
    ----------
    all_results : defaultdict
        Result of analyze_results.
    model : str
        Model key.
    dataset : str
        Dataset key.
    metric : str
        Metric to plot (e.g. "test/mse"); must match a _mean column in aggregated results.
    save_path : str, optional
        If set, save figure to this path.
    figsize : tuple, optional
        (width, height) in inches.
    max_configs : int, optional
        Max number of HP configs to show (avoid huge figures). Default 50.

    Returns
    -------
    matplotlib figure (or None if model/dataset/metric not found).
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if dataset not in all_results or model not in all_results[dataset]:
        return None
    res = all_results[dataset][model]
    ds = res.get("sorted_aggregated_results")
    if _ds_is_empty(ds) or metric not in ds.coords["metric"].values:
        return None

    y = ds.sel(metric=metric)["mean"].values[:max_configs]
    x_labels = [str(i) for i in range(len(y))]

    if figsize is None:
        figsize = (8, max(4, len(y) * 0.15))
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(x_labels, y)
    ax.set_xlabel(metric)
    ax.set_ylabel("HP config (rank)")
    ax.set_title(f"HP impact: {model} on {dataset} — {metric}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Publication rcParams applied by plot_summary (and usable by callers).
# ---------------------------------------------------------------------------
_PUB_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.5,
}


def plot_summary(
    ds: "xr.Dataset",
    metric_key: str,
    mode: str = "min",
    fallback_metric_keys: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
) -> "object":
    """Heatmap overview of model × dataset performance.

    Each cell shows the best-run mean for ``metric_key``; colour encodes a
    per-dataset normalised score (0 = worst, 1 = best) so that different
    dataset scales are comparable at a glance.  Models are sorted top-to-bottom
    by their mean normalised score (best model first).

    Parameters
    ----------
    ds : xr.Dataset
        Output of ``all_results_to_xarray``.
    metric_key : str
        Primary metric to display, in ``"prefix/metric_name"`` form, e.g.
        ``"test_best_rerun/mae_normalized"`` or ``"test_best_rerun/accuracy"``.
    mode : str
        ``"min"`` if lower is better (regression), ``"max"`` if higher is
        better (classification).  Controls the normalisation direction.
    fallback_metric_keys : sequence of str, optional
        Alternative metric_keys tried in order for cells where *metric_key* is
        NaN.  Use this for fit-predict models (XGBoost, TabPFN, TabDPT) that
        lack ``_best_rerun`` metrics, e.g.
        ``fallback_metric_keys=["test/mae_normalized"]``.
    save_path : str, optional
        If provided, both a ``.pdf`` (vector) and a ``.png`` (300 DPI) are
        written; the extension in *save_path* is replaced as needed.
    figsize : tuple, optional
        ``(width_inches, height_inches)``.  Defaults to journal double-column
        width (7.0") with height scaled to the number of models.
    title : str, optional
        Axes title.  Defaults to ``metric_key``.

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` when ``metric_key`` is not present in *ds*.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Guard: primary metric_key must exist
    available = list(ds.coords["metric_key"].values)
    if metric_key not in available:
        return None

    mpl.rcParams.update(_PUB_RCPARAMS)

    # --- data extraction ---
    # values shape: (n_datasets, n_models)
    values = ds["mean"].sel(metric_key=metric_key).values.copy()

    # Fill NaN cells from fallback keys (e.g. fit-predict models lack _best_rerun)
    if fallback_metric_keys:
        for fb_key in fallback_metric_keys:
            if fb_key not in available:
                continue
            nan_mask = np.isnan(values)
            if not nan_mask.any():
                break
            fb_values = ds["mean"].sel(metric_key=fb_key).values
            values[nan_mask] = fb_values[nan_mask]
    datasets = list(ds.coords["dataset"].values)
    models = list(ds.coords["model"].values)

    # --- per-dataset min-max normalisation → score in [0, 1] (1 = best) ---
    normed = np.full_like(values, np.nan)
    for i in range(len(datasets)):
        row = values[i]
        valid = row[~np.isnan(row)]
        if len(valid) == 0:
            continue
        if len(valid) == 1 or valid.max() == valid.min():
            normed[i] = np.where(np.isnan(row), np.nan, 1.0)
            continue
        vmin, vmax = valid.min(), valid.max()
        if mode == "min":
            normed[i] = (vmax - row) / (vmax - vmin)  # smaller value → score 1
        else:
            normed[i] = (row - vmin) / (vmax - vmin)  # larger value → score 1

    # --- sort models by mean score descending (best first) ---
    mean_score = np.nanmean(normed, axis=0)  # (n_models,)
    order = np.argsort(mean_score)[::-1]
    normed = normed[:, order]      # (n_datasets, n_models)
    values = values[:, order]
    sorted_short = [_short_model_name(models[i]) for i in order]
    short_ds = [_short_dataset_name(d) for d in datasets]

    # Transpose so rows = models, columns = datasets
    normed_T = normed.T   # (n_models, n_datasets)
    values_T = values.T

    n_models, n_datasets = normed_T.shape

    # --- figure sizing (journal double-column default) ---
    if figsize is None:
        cell_w = max(0.55, min(1.0, 7.0 / max(n_datasets, 1)))
        cell_h = max(0.30, min(0.55, 5.0 / max(n_models, 1)))
        fig_w = min(7.0, n_datasets * cell_w + 2.2)
        fig_h = n_models * cell_h + 1.2
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)

    # --- heatmap via imshow ---
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#eeeeee")  # NaN → light grey

    masked = np.ma.array(normed_T, mask=np.isnan(normed_T))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation="none")

    # --- cell annotations: actual metric value ---
    for ri in range(n_models):
        for ci in range(n_datasets):
            v = values_T[ri, ci]
            nv = normed_T[ri, ci]
            if np.isnan(v):
                continue
            text_color = "white" if (not np.isnan(nv) and nv > 0.55) else "#222222"
            ax.text(ci, ri, f"{v:.3g}", ha="center", va="center",
                    fontsize=5.5, color=text_color, clip_on=True)

    # --- axes labels ---
    ax.set_xticks(range(n_datasets))
    ax.set_xticklabels(short_ds, rotation=40, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(sorted_short)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, n_datasets, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_models, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    # --- colourbar ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.7)
    cbar.set_label(
        f"Score (1 = best {'low' if mode == 'min' else 'high'})", fontsize=6
    )
    cbar.ax.tick_params(labelsize=5)

    ax.set_title(title if title is not None else metric_key, pad=6)

    fig.tight_layout()

    if save_path:
        base = str(save_path).rsplit(".", 1)[0]
        fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight", format="pdf")
        fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", format="png")

    return fig
