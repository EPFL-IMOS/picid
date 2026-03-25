# picid_report/report/plots.py
"""
Paper-ready plotting helpers. Reads from all_results (output of analyze_results).

Requires matplotlib. If not installed, plot functions raise a clear error.
"""

from collections import defaultdict
from typing import Optional, Sequence, Union

import pandas as pd

from picid_report.utils import flatten_aggregated_columns


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
                labels.append(human_readable_model)
                means.append(float(mean))
                std_val = float(std) if std is not None and not pd.isna(std) else 0.0
                stds.append(std_val)
                cnt_val = int(cnt) if cnt is not None and not pd.isna(cnt) else None
                counts.append(cnt_val)

        if not labels:
            return None

        if len(datasets_seen) != 1:
            raise AssertionError(
                "plot_best_metric_bars expects results from a single dataset"
            )
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
            tick_label=labels,
        )
        ax.set_ylabel(metric_key)
        ax.set_title(f"Best {metric_key} on {dataset_name}")
        plt.xticks(rotation=45, ha="right")
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
    if "sorted_aggregated_results" not in res or res["sorted_aggregated_results"].empty:
        return None

    df = flatten_aggregated_columns(res["sorted_aggregated_results"].copy())
    mean_col = f"{metric}_mean"
    if mean_col not in df.columns:
        return None

    df = df.head(max_configs)
    y = df[mean_col].values
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
