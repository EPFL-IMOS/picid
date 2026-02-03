"""
Paper-ready plotting helpers. Reads from all_results (output of analyze_results).

Requires matplotlib. If not installed, plot functions raise a clear error.
"""

from collections import defaultdict
from typing import Optional

import pandas as pd

from lmetk_report.utils import flatten_aggregated_columns


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
    metric: str = "test/mse",
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> "object":
    """Bar chart of best metric value per model/dataset.

    Parameters
    ----------
    all_results : defaultdict
        Result of analyze_results (dataset -> model -> result dict).
    metric : str, optional
        Metric key in prefix/name form, e.g. "test/mse". Default "test/mse".
    save_path : str, optional
        If set, save figure to this path.
    figsize : tuple, optional
        (width, height) in inches for the figure.

    Returns
    -------
    matplotlib figure (or None if no data).
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    parts = metric.split("/", 1)
    prefix = parts[0] if len(parts) == 2 else ""
    metric_name = parts[1] if len(parts) == 2 else metric

    labels = []
    means = []
    for dataset, models in all_results.items():
        for model, res in models.items():
            bp = res.get("best_performance", {}).get("metrics", {})
            if metric_name not in bp or prefix not in bp[metric_name]:
                continue
            vals = bp[metric_name][prefix]
            mean = vals.get("mean")
            if mean is None or pd.isna(mean):
                continue
            labels.append(f"{model}\n{dataset}")
            means.append(float(mean))

    if not labels:
        return None

    if figsize is None:
        figsize = (max(6, len(labels) * 0.5), 5)
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(labels))
    ax.bar(x, means, tick_label=labels)
    ax.set_ylabel(metric)
    ax.set_title(f"Best {metric} by model/dataset")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


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
