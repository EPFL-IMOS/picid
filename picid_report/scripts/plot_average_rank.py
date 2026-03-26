"""
Standalone script: average rank bar plot across datasets.

Reads all report_output/<project>/tables/summary.csv files under a given output
directory, parses the ranking metric column, ranks models per dataset, then plots
one bar per model with y = average rank across datasets. Model names are shortened
for the x-axis.

Two modes:
- Task-type mode (default): Produces two plot sets — classification and regression.
  Classification: only CSVs with classification metrics (e.g. accuracy, f1); rank by
  test_best_rerun/accuracy or test/accuracy (fallback for fit_predict). Regression:
  only CSVs with regression metrics (loss, mse, mae, etc.); rank by test_best_rerun/*
  or test/*, preferring normalized over denormalized.
  Outputs: average_rank_classification.png, average_rank_regression.png (rank bar plots);
  average_accuracy_classification.png, average_score_regression.png (actual score bar plots,
  mean ± std across datasets); and under plots/classification/ and plots/regression/,
  per-dataset rank plots (*.png) and per-dataset score plots (*_scores.png).
- Single-metric mode: Pass --rank-metric <name> to use one metric for all files; one plot.

Usage:
  python -m picid_report.scripts.plot_average_rank --output-dir report_output
  python -m picid_report.scripts.plot_average_rank -o report_output --task-type both
  python -m picid_report.scripts.plot_average_rank -o report_output --rank-metric val_best_rerun/loss --out report_output/plots/average_rank.png
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Logical metrics: (logical_name, [candidates in preference order]).
# For each model we use the first candidate column that has a parseable value (so fit_predict
# with only test/* get included via test/ fallback).
CLASSIFICATION_LOGICAL_METRICS = [
    ("accuracy", ["test_best_rerun/accuracy", "test/accuracy"]),
    ("f1", ["test_best_rerun/f1", "test/f1"]),
]


# Prefer mae_normalized so regression plots rank by MAE when available.
# Include _mean variants so we match both CSV formats (e.g. test/mae_normalized vs test/mae_normalized_mean).
def _reg_cands(base: str) -> list:
    """Candidate columns for a regression metric: base and _mean variant, test_best_rerun then test."""
    return [
        f"test_best_rerun/{base}",
        f"test/{base}",
        f"test_best_rerun/{base}_mean",
        f"test/{base}_mean",
    ]


REGRESSION_LOGICAL_METRICS = [
    ("mae_normalized", _reg_cands("mae_normalized")),
    ("mae_denormalized", _reg_cands("mae_denormalized")),
    ("mse_normalized", _reg_cands("mse_normalized")),
    ("mse_denormalized", _reg_cands("mse_denormalized")),
    ("rmse_normalized", _reg_cands("rmse_normalized")),
    ("rmse_denormalized", _reg_cands("rmse_denormalized")),
    ("loss", ["test_best_rerun/loss", "test/loss"]),  # no _mean variant for loss
]

# Flat candidate lists for backward compatibility (single-column selection)
CLASSIFICATION_CANDIDATES = [
    c for _name, cands in CLASSIFICATION_LOGICAL_METRICS for c in cands
]
REGRESSION_CANDIDATES = [
    c for _name, cands in REGRESSION_LOGICAL_METRICS for c in cands
]

# Metric names (suffix or full) that indicate classification vs regression
CLASSIFICATION_METRIC_HINTS = ("accuracy", "f1", "precision", "recall")
REGRESSION_METRIC_HINTS = ("loss", "mse", "mae", "rmse", "nasa_score", "phm_score")


def _parse_value_cell(s: str):
    """Extract numeric mean from a summary table cell like '0.0657 ± 0.0000 (n=1)' or '-'."""
    out = _parse_value_cell_with_std(s)
    return out[0] if out else None


def _parse_value_cell_with_std(s: str):
    """Extract (mean, std, n) from a summary table cell like '0.0657 ± 0.0000 (n=1)' or '0.5'.
    Return (float, float|None, int|None); std and n are None if not present."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if s == "-" or s.lower() == "nan":
        return None
    # Match "mean ± std (n=N)" or "mean ± std" or "mean"
    m = re.match(
        r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:±\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?\s*(?:\(n=(\d+)\))?\s*$",
        s,
    )
    if m:
        try:
            mean = float(m.group(1))
            std = float(m.group(2)) if m.group(2) is not None else None
            n = int(m.group(3)) if m.group(3) is not None else None
            return (mean, std, n)
        except (ValueError, TypeError):
            return None
    return None


def _shorten_model_name(full_name: str) -> str:
    """Use last segment after '.'; keep trailing ' (linear)' etc. for readability."""
    if not full_name:
        return full_name
    # Keep suffix like " (linear)" or " (exponential)"
    base = full_name
    suffix = ""
    if " (" in full_name and ")" in full_name:
        idx = full_name.index(" (")
        base = full_name[:idx]
        suffix = full_name[idx:]
    parts = base.split(".")
    short = parts[-1] if parts else base
    return short + suffix


def _load_summary_csv(path: Path):
    """Load summary CSV with csv.reader (handles quoted fields); return (row0, row1, data_rows). data_rows: list of rows, row[0]=model, row[1:]=values."""
    with open(path, encoding="utf-8", newline="") as f:
        lines = list(csv.reader(f))
    if len(lines) < 4:
        return None, None, []
    # Row 0 = Dataset names, row 1 = Metric names, row 2 = "Model" header, row 3+ = data
    return lines[0], lines[1], lines[3:]


def _metrics_in_row(row1: list) -> set:
    """Set of metric names (lowercase) present in row1 (column headers)."""
    return {
        (row1[i] or "").strip().lower()
        for i in range(1, len(row1))
        if (row1[i] or "").strip()
    }


def _metric_looks_classification(metric: str) -> bool:
    m = metric.lower()
    return any(h in m for h in CLASSIFICATION_METRIC_HINTS)


def _metric_looks_regression(metric: str) -> bool:
    m = metric.lower()
    return any(h in m for h in REGRESSION_METRIC_HINTS)


def infer_task_type(path: Path) -> str | None:
    """
    Infer task type from summary CSV row 1.
    Returns "classification", "regression", or None if unreadable.
    If both classification and regression metrics appear, prefer classification.
    """
    row0, row1, _ = _load_summary_csv(path)
    if row1 is None:
        return None
    metrics = _metrics_in_row(row1)
    has_clf = any(_metric_looks_classification(m) for m in metrics)
    has_reg = any(_metric_looks_regression(m) for m in metrics)
    if has_clf:
        return "classification"
    if has_reg:
        return "regression"
    return None


def _column_has_parseable_value(col_idx: int, data_rows: list) -> bool:
    """True if at least one data row has a parseable numeric value in that column."""
    for row in data_rows:
        if col_idx < len(row):
            if _parse_value_cell(row[col_idx]) is not None:
                return True
    return False


def _dataset_key(path: Path, dataset_name: str) -> str:
    """Unique key per (file, dataset) so the same dataset name from different files is not merged when ranking."""
    run_id = (
        path.parent.parent.name if path and path.parent and path.parent.parent else ""
    )
    return f"{run_id}::{dataset_name}" if run_id else dataset_name


def pick_metric_and_collect_records_with_fallback(
    path: Path,
    logical_metrics: list[tuple[str, list[str]]],
) -> tuple[str | None, list]:
    """
    For each logical metric, try to collect (dataset, model, value) using per-model fallback:
    for each model use the first candidate column that has a parseable value (e.g. test_best_rerun/X
    then test/X). So fit_predict models with only test/* get included. Return (chosen_logical_name, records).
    Dataset in records is made unique per file (run_id::dataset_name) so ranking is per run, not merged across files.
    """
    row0, row1, data_rows = _load_summary_csv(path)
    if row0 is None or row1 is None:
        return None, []
    # (col_idx, dataset_name, metric_name_lower) for columns we care about
    col_info = []
    for i in range(1, min(len(row0), len(row1))):
        m = (row1[i] or "").strip().lower()
        if not m:
            continue
        dataset = (row0[i] or "").strip() or f"dataset_{i}"
        col_info.append((i, dataset, m))

    for logical_name, candidates in logical_metrics:
        cands_lower = [c.strip().lower() for c in candidates]
        # For this logical metric, get columns that match any candidate; group by dataset, order by candidate preference
        by_dataset = {}  # dataset -> [(col_idx, cand_index), ...] sorted by cand_index
        for col_idx, dataset, m in col_info:
            if m not in cands_lower:
                continue
            idx_in_cand = cands_lower.index(m)
            if dataset not in by_dataset:
                by_dataset[dataset] = []
            by_dataset[dataset].append((col_idx, idx_in_cand))
        if not by_dataset:
            continue
        # Sort each dataset's columns by candidate order
        for d in by_dataset:
            by_dataset[d].sort(key=lambda x: x[1])
            by_dataset[d] = [col_idx for col_idx, _ in by_dataset[d]]
        # At least one model must have a parseable value in at least one of these columns
        all_cols = {col_idx for cols in by_dataset.values() for col_idx in cols}
        if not any(
            _column_has_parseable_value(col_idx, data_rows) for col_idx in all_cols
        ):
            continue
        out = []
        for row in data_rows:
            if not row:
                continue
            model = (row[0] or "").strip()
            if not model:
                continue
            for dataset, col_idxs in by_dataset.items():
                val, std = None, None
                n_runs = None
                for col_idx in col_idxs:
                    if col_idx < len(row):
                        parsed = _parse_value_cell_with_std(row[col_idx])
                        if parsed is not None:
                            val, std = parsed[0], parsed[1]
                            n_runs = parsed[2] if len(parsed) >= 3 else None
                            break
                if val is not None:
                    out.append((_dataset_key(path, dataset), model, val, std, n_runs))
        if out:
            return logical_name, out
    return None, []


def pick_metric_and_collect_records(
    path: Path, candidates: list
) -> tuple[str | None, list]:
    """
    For a summary CSV, pick the first candidate metric that appears in row1 and has
    at least one parseable value. Return (chosen_metric, list of (dataset, model, value)).
    (Single-column mode; fit_predict models with only test/* are excluded.)
    """
    row0, row1, data_rows = _load_summary_csv(path)
    if row0 is None or row1 is None:
        return None, []
    metrics_lower = {
        i: (row1[i] or "").strip().lower() for i in range(1, min(len(row0), len(row1)))
    }
    for cand in candidates:
        cand_lower = cand.strip().lower()
        cols_to_use = []  # (col_idx, dataset_name)
        for i, m in metrics_lower.items():
            if m == cand_lower:
                dataset = (row0[i] or "").strip() or f"dataset_{i}"
                cols_to_use.append((i, dataset))
        if not cols_to_use:
            continue
        if not any(
            _column_has_parseable_value(col_idx, data_rows)
            for col_idx, _ in cols_to_use
        ):
            continue
        out = []
        for row in data_rows:
            if not row:
                continue
            model = (row[0] or "").strip()
            if not model:
                continue
            for col_idx, dataset in cols_to_use:
                if col_idx >= len(row):
                    continue
                val = _parse_value_cell(row[col_idx])
                if val is not None:
                    out.append((dataset, model, val))
        if out:
            return cand, out
    return None, []


def parse_summary_csv(path: Path, rank_metric: str):
    """
    Parse a summary.csv (pivot format: row0=Dataset, row1=Metric, row2=Model, then data).
    Returns list of (dataset, model, value) for columns where Metric == rank_metric.
    """
    with open(path, encoding="utf-8") as f:
        lines = [line.strip().split(",") for line in f]
    if len(lines) < 4:
        return []
    row0 = lines[0]  # Dataset, name, name, ...
    row1 = lines[1]  # Metric, metric, metric, ...
    # Find column indices where metric matches rank_metric (exact or normalized)
    rank_metric_norm = rank_metric.strip().lower()
    cols_to_use = []  # (col_idx, dataset_name)
    for i in range(1, min(len(row0), len(row1))):
        metric = (row1[i] or "").strip()
        if metric.lower() == rank_metric_norm:
            dataset = (row0[i] or "").strip() or f"dataset_{i}"
            cols_to_use.append((i, dataset))
    if not cols_to_use:
        return []
    out = []
    for row in lines[3:]:
        if not row:
            continue
        model = (row[0] or "").strip()
        if not model:
            continue
        for col_idx, dataset in cols_to_use:
            if col_idx >= len(row):
                continue
            val = _parse_value_cell(row[col_idx])
            if val is not None:
                out.append((dataset, model, val))
    return out


def discover_summary_csvs(output_dir: Path):
    """Find all tables/summary.csv under output_dir (any depth)."""
    return list(output_dir.rglob("tables/summary.csv"))


def collect_all_records(summary_paths: list, rank_metric: str):
    """Parse each summary CSV and return a single list of (dataset, model, value)."""
    all_records = []
    for p in summary_paths:
        try:
            records = parse_summary_csv(p, rank_metric)
            all_records.extend(records)
        except Exception:
            continue
    return all_records


def _safe_filename(s: str) -> str:
    """Make a string safe for use as a filename: replace :: and other problematic chars with _."""
    return re.sub(r"[^\w\-.]", "_", s).strip("_") or "unnamed"


def _format_metrics_label(metrics_used: list) -> str:
    """Format list of metric names (e.g. from each file) for plot title: 'mse_normalized (8), loss (2)' or 'accuracy'."""
    if not metrics_used:
        return "test metric (auto)"
    counts = Counter(metrics_used)
    if len(counts) == 1:
        return list(counts.keys())[0]
    parts = [
        f"{m} ({c})" for m, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]
    return ", ".join(parts)


def compute_average_ranks(records: list, rank_mode: str = "min"):
    """
    Rank models per dataset (1 = best), then average rank per model.
    rank_mode: 'min' = lower is better, 'max' = higher is better.
    Returns DataFrame with columns: model_full, model_short, average_rank, n_datasets.
    """
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records, columns=["dataset", "model", "value"])
    ascending = rank_mode == "min"
    df["rank"] = df.groupby("dataset")["value"].rank(
        ascending=ascending, method="average", na_option="bottom"
    )
    agg = (
        df.groupby("model")
        .agg(
            average_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            n_datasets=("dataset", "nunique"),
        )
        .reset_index()
    )
    agg = agg.rename(columns={"model": "model_full"})
    agg["model_short"] = agg["model_full"].map(_shorten_model_name)
    agg["std_rank"] = agg["std_rank"].fillna(0)
    agg = agg.sort_values("average_rank").reset_index(drop=True)
    return agg


def compute_rank_one_dataset(records: list, rank_mode: str = "min"):
    """
    Rank models for a single dataset. records: list of (model, value).
    Returns DataFrame with columns: model_full, model_short, average_rank (rank), n_datasets=1.
    """
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records, columns=["model", "value"])
    ascending = rank_mode == "min"
    df["rank"] = df["value"].rank(
        ascending=ascending, method="average", na_option="bottom"
    )
    df = df.rename(columns={"model": "model_full", "rank": "average_rank"})
    df["model_short"] = df["model_full"].map(_shorten_model_name)
    df["n_datasets"] = 1
    df = df.sort_values("average_rank").reset_index(drop=True)
    return df[["model_full", "model_short", "average_rank", "n_datasets"]]


def compute_mean_scores(records: list) -> pd.DataFrame:
    """
    Aggregate (dataset, model, value) to mean ± std per model.
    Returns DataFrame with columns: model_full, model_short, mean_value, std_value, n_datasets.
    """
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records, columns=["dataset", "model", "value"])
    agg = (
        df.groupby("model")["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={"mean": "mean_value", "std": "std_value", "count": "n_datasets"}
        )
    )
    agg["std_value"] = agg["std_value"].fillna(0)
    agg = agg.rename(columns={"model": "model_full"})
    agg["model_short"] = agg["model_full"].map(_shorten_model_name)
    return agg


def plot_average_rank(
    df: pd.DataFrame,
    save_path: Path = None,
    title: str = "Average rank across datasets",
    figsize: tuple = None,
    ylabel: str = "Average rank",
):
    """Bar plot: x = model_short, y = average_rank. Lower rank = better. Light grid and n= above bars."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Plotting requires matplotlib. pip install matplotlib") from e
    if df.empty:
        return None
    if figsize is None:
        n = len(df)
        figsize = (max(6, n * 0.45), 6)
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(df))
    heights = df["average_rank"].values
    yerr = df["std_rank"].values if "std_rank" in df.columns else None
    ax.bar(
        x,
        heights,
        tick_label=df["model_short"],
        yerr=yerr,
        capsize=4,
        error_kw={"elinewidth": 1, "ecolor": "gray"},
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="gray")
    if "n_datasets" in df.columns:
        cols = [
            c
            for c in ["model_short", "n_datasets", "average_rank", "std_rank"]
            if c in df.columns
        ]
        # print(f"\n[DEBUG] n_datasets per model:\n{df[cols].to_string()}")
        y_max = heights.max() if len(heights) else 0
        for i, (h, n) in enumerate(zip(heights, df["n_datasets"])):
            ax.text(
                i,
                h + (y_max * 0.02 if y_max else 0.1),
                f"n={int(n)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="gray",
            )
        ax.margins(y=0.06)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def _scale_to_percent_if_needed(y, yerr, metric_label: str) -> tuple:
    """
    For classification percentage metrics (accuracy, f1, precision, recall), convert
    values in 0–1 to 0–100 so the plot matches report tables. Values > 1.5 are left as-is
    (already in percent). Returns (y_scaled, yerr_scaled).
    """
    import numpy as np

    metric = (metric_label or "").strip().lower()
    if not any(m in metric for m in ("accuracy", "f1", "precision", "recall")):
        return y, yerr
    y = np.asarray(y, dtype=float)
    scale = np.where(y <= 1.5, 100.0, 1.0)
    y_scaled = y * scale
    if yerr is not None:
        yerr = np.nan_to_num(np.asarray(yerr, dtype=float), nan=0.0)
        yerr_scaled = yerr * scale
        return y_scaled, yerr_scaled
    return y_scaled, None


def plot_metric_scores(
    df: pd.DataFrame,
    save_path: Path = None,
    title: str = "Mean score across datasets",
    figsize: tuple = None,
    ylabel: str = "Score",
    higher_better: bool = True,
    show_error_bars: bool = True,
):
    """
    Bar plot: x = model_short, y = mean_value. Optional error bars from std_value.
    Light grid and n= above bars (n = runs or datasets used for the statistic).
    For accuracy/f1/precision/recall, values in 0–1 are scaled to 0–100 for display.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise ImportError("Plotting requires matplotlib. pip install matplotlib") from e
    if df.empty or "mean_value" not in df.columns:
        return None
    if figsize is None:
        n = len(df)
        figsize = (max(6, n * 0.45), 6)
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(df))
    y = df["mean_value"].values
    if show_error_bars and "std_value" in df.columns:
        yerr = np.nan_to_num(df["std_value"].values, nan=0.0)
    else:
        yerr = None
    y, yerr = _scale_to_percent_if_needed(y, yerr, ylabel)
    ax.bar(x, y, tick_label=df["model_short"], yerr=yerr, capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="gray")
    if "n_datasets" in df.columns:
        import numpy as np

        yerr_safe = np.nan_to_num(yerr, nan=0.0) if yerr is not None else None
        top = y + (yerr_safe if yerr_safe is not None else 0)
        y_plot_max = float(np.nanmax(top)) if len(top) else 0.0
        for i, (yi, ni) in enumerate(zip(y, df["n_datasets"])):
            err_i = (
                yerr_safe[i] if yerr_safe is not None and i < len(yerr_safe) else 0.0
            )
            label_y = yi + err_i + (y_plot_max * 0.02 if y_plot_max else 0.01)
            ax.text(
                i,
                label_y,
                f"n={int(ni)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="gray",
            )
        ax.margins(y=0.06)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def _score_df_one_dataset(records: list) -> pd.DataFrame:
    """Build a DataFrame for score plot. records: list of (model, value), (model, value, std), or (model, value, std, n_runs)."""
    if not records:
        return pd.DataFrame()
    r0 = records[0]
    if len(r0) >= 4:
        df = pd.DataFrame(records, columns=["model", "value", "std_value", "n_runs"])
    elif len(r0) >= 3:
        df = pd.DataFrame(records, columns=["model", "value", "std_value"])
        df["n_runs"] = None
    else:
        df = pd.DataFrame(records, columns=["model", "value"])
        df["std_value"] = 0.0
        df["n_runs"] = None
    df = df.rename(columns={"model": "model_full"})
    df["model_short"] = df["model_full"].map(_shorten_model_name)
    df["mean_value"] = df["value"]
    df["n_datasets"] = df["n_runs"].fillna(1).astype(int)  # for display: n runs per bar
    return df[["model_full", "model_short", "mean_value", "std_value", "n_datasets"]]


def _save_per_dataset_score_plots(
    records_with_metric: list,
    task_dir: Path,
    task_label: str,
    higher_better: bool,
) -> None:
    """Save one bar plot per dataset: actual scores (value) per model with std. records_with_metric: (dataset_key, model, value, std, n_runs, metric)."""
    if not records_with_metric:
        return
    by_dataset = {}
    for dataset_key, model, value, std, n_runs, metric in records_with_metric:
        key = (dataset_key, metric)
        if key not in by_dataset:
            by_dataset[key] = []
        by_dataset[key].append((model, value, std if std is not None else 0.0, n_runs))
    for (dataset_key, metric), recs in by_dataset.items():
        df = _score_df_one_dataset(recs)
        if df.empty:
            continue
        # Order bars by score: best first (higher_better => descending, else ascending)
        df = df.sort_values("mean_value", ascending=not higher_better).reset_index(
            drop=True
        )
        safe_name = _safe_filename(f"{dataset_key}_{metric}")
        out_path = task_dir / f"{safe_name}_scores.png"
        display_name = (
            dataset_key.split("::")[-1] if "::" in dataset_key else dataset_key
        )
        title = f"{metric} — {display_name} ({task_label})"
        plot_metric_scores(
            df,
            save_path=out_path,
            title=title,
            ylabel=metric,
            show_error_bars=True,
        )
        print(f"Saved plot to {out_path}")


def _save_per_dataset_rank_plots(
    records_with_metric: list,
    task_dir: Path,
    rank_mode: str,
    task_label: str,
) -> None:
    """Save one bar plot per dataset: rank by metric. records_with_metric: (dataset_key, model, value, std, n_runs, metric)."""
    if not records_with_metric:
        return
    # Group by (dataset_key, metric) - each key has one metric
    by_dataset = {}
    for dataset_key, model, value, _std, _n_runs, metric in records_with_metric:
        key = (dataset_key, metric)
        if key not in by_dataset:
            by_dataset[key] = []
        by_dataset[key].append((model, value))
    for (dataset_key, metric), recs in by_dataset.items():
        df = compute_rank_one_dataset(recs, rank_mode=rank_mode)
        if df.empty:
            continue
        safe_name = _safe_filename(f"{dataset_key}_{metric}")
        out_path = task_dir / f"{safe_name}.png"
        # Title: show dataset display name (after ::) if present
        display_name = (
            dataset_key.split("::")[-1] if "::" in dataset_key else dataset_key
        )
        title = f"Rank by {metric} — {display_name} ({task_label})"
        plot_average_rank(
            df,
            save_path=out_path,
            title=title,
            ylabel="Rank",
        )
        print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot average rank of models across datasets from existing summary CSVs."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("report_output"),
        help="Root directory under which to find **/tables/summary.csv (default: report_output)",
    )
    parser.add_argument(
        "--rank-metric",
        type=str,
        default=None,
        help="If set, single-metric mode: use this metric for all files, produce one plot. If unset, task-type mode (two plots: classification + regression).",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=("both", "classification", "regression"),
        default="both",
        help="In task-type mode: produce both plots, or only classification, or only regression. Default: both",
    )
    parser.add_argument(
        "--rank-mode",
        type=str,
        choices=("min", "max"),
        default="min",
        help="Single-metric mode only: min = lower is better, max = higher is better. Default: min",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path: in task-type mode, directory for PNGs (default: <output-dir>/plots). In single-metric mode, path to the PNG file.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="If set, also write CSV(s). In task-type mode, directory for CSVs; in single-metric mode, path to the CSV file.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        print(f"Error: output-dir is not a directory: {output_dir}", file=sys.stderr)
        sys.exit(1)

    summary_paths = discover_summary_csvs(output_dir)
    if not summary_paths:
        print(
            f"No tables/summary.csv files found under {output_dir}. Run the report pipeline first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Single-metric mode: one plot with user-specified metric
    if args.rank_metric is not None:
        records = collect_all_records(summary_paths, args.rank_metric)
        if not records:
            print(
                f"No records found for rank metric {args.rank_metric!r}. Try e.g. val_best_rerun/loss or test/loss.",
                file=sys.stderr,
            )
            sys.exit(1)
        df = compute_average_ranks(records, rank_mode=args.rank_mode)
        if df.empty:
            print("No data after computing ranks.", file=sys.stderr)
            sys.exit(1)
        out_path = (
            args.out
            if args.out is not None
            else output_dir / "plots" / "average_rank.png"
        )
        out_path = out_path.resolve()
        plot_average_rank(
            df,
            save_path=out_path,
            title=f"Average rank across datasets (metric: {args.rank_metric})",
        )
        print(f"Saved plot to {out_path}")
        if args.csv is not None:
            csv_path = Path(args.csv).resolve()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV to {csv_path}")
        return 0

    # Task-type mode: classify CSVs, produce one or two plot sets
    plots_dir = (args.out if args.out is not None else output_dir / "plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.csv.resolve() if args.csv is not None else None
    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)

    classification_paths = []
    regression_paths = []
    for p in summary_paths:
        try:
            tt = infer_task_type(p)
            if tt == "classification":
                classification_paths.append(p)
            elif tt == "regression":
                regression_paths.append(p)
        except Exception:
            continue

    any_saved = False

    if args.task_type in ("both", "classification") and classification_paths:
        all_records = []
        records_with_metric = []  # (dataset_key, model, value, metric)
        metrics_used = []
        for p in classification_paths:
            chosen, recs = pick_metric_and_collect_records_with_fallback(
                p, CLASSIFICATION_LOGICAL_METRICS
            )
            if chosen:
                metrics_used.append(chosen)
            for dk, model, val, std, n_runs in recs:
                all_records.append((dk, model, val))
                records_with_metric.append(
                    (dk, model, val, std, n_runs, chosen or "accuracy")
                )
        if all_records:
            df = compute_average_ranks(all_records, rank_mode="max")
            if not df.empty:
                out_png = plots_dir / "average_rank_classification.png"
                metric_label = _format_metrics_label(metrics_used)
                plot_average_rank(
                    df,
                    save_path=out_png,
                    title=f"Average rank across datasets (classification, ranked by: {metric_label})",
                )
                print(f"Saved plot to {out_png}")
                any_saved = True
                if csv_dir is not None:
                    csv_path = csv_dir / "average_rank_classification.csv"
                    df.to_csv(csv_path, index=False)
                    print(f"Saved CSV to {csv_path}")
                # Per-dataset rank plots in classification/
                task_dir = plots_dir / "classification"
                task_dir.mkdir(parents=True, exist_ok=True)
                _save_per_dataset_rank_plots(
                    records_with_metric,
                    task_dir,
                    rank_mode="max",
                    task_label="classification",
                )
                # Aggregate score plot (mean accuracy/f1 across datasets) and per-dataset score plots
                df_scores = compute_mean_scores(all_records)
                if not df_scores.empty:
                    df_scores = df_scores.sort_values(
                        "mean_value", ascending=False
                    ).reset_index(drop=True)
                    score_png = plots_dir / "average_accuracy_classification.png"
                    plot_metric_scores(
                        df_scores,
                        save_path=score_png,
                        title=f"Mean score across datasets (classification, {metric_label})",
                        ylabel=metric_label,
                        higher_better=True,
                        show_error_bars=True,
                    )
                    print(f"Saved plot to {score_png}")
                    _save_per_dataset_score_plots(
                        records_with_metric,
                        task_dir,
                        task_label="classification",
                        higher_better=True,
                    )

    if args.task_type in ("both", "regression") and regression_paths:
        all_records = []
        records_with_metric = []
        metrics_used = []
        for p in regression_paths:
            chosen, recs = pick_metric_and_collect_records_with_fallback(
                p, REGRESSION_LOGICAL_METRICS
            )
            if chosen:
                metrics_used.append(chosen)
            for dk, model, val, std, n_runs in recs:
                all_records.append((dk, model, val))
                records_with_metric.append(
                    (dk, model, val, std, n_runs, chosen or "loss")
                )
        if all_records:
            df = compute_average_ranks(all_records, rank_mode="min")
            if not df.empty:
                out_png = plots_dir / "average_rank_regression.png"
                metric_label = _format_metrics_label(metrics_used)
                plot_average_rank(
                    df,
                    save_path=out_png,
                    title=f"Average rank across datasets (regression, ranked by: {metric_label})",
                )
                print(f"Saved plot to {out_png}")
                any_saved = True
                if csv_dir is not None:
                    csv_path = csv_dir / "average_rank_regression.csv"
                    df.to_csv(csv_path, index=False)
                    print(f"Saved CSV to {csv_path}")
                # Per-dataset rank plots in regression/
                task_dir = plots_dir / "regression"
                task_dir.mkdir(parents=True, exist_ok=True)
                _save_per_dataset_rank_plots(
                    records_with_metric,
                    task_dir,
                    rank_mode="min",
                    task_label="regression",
                )
                # Aggregate score plot (mean MAE/loss across datasets) and per-dataset score plots
                df_scores = compute_mean_scores(all_records)
                if not df_scores.empty:
                    df_scores = df_scores.sort_values(
                        "mean_value", ascending=True
                    ).reset_index(drop=True)
                    score_png = plots_dir / "average_score_regression.png"
                    plot_metric_scores(
                        df_scores,
                        save_path=score_png,
                        title=f"Mean score across datasets (regression, {metric_label})",
                        ylabel=metric_label,
                        higher_better=False,
                        show_error_bars=True,
                    )
                    print(f"Saved plot to {score_png}")
                    _save_per_dataset_score_plots(
                        records_with_metric,
                        task_dir,
                        task_label="regression",
                        higher_better=False,
                    )

    if not any_saved:
        print(
            "No data collected for the requested task type(s). Try --rank-metric <name> for single-metric mode.",
            file=sys.stderr,
        )
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
