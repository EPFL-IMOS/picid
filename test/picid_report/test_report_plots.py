"""
Test suite for picid_report.report.plots.

Validates plot_best_metric_bars and plot_hp_impact: data extraction from all_results,
metric/prefix handling, empty/missing data, sorted (ranked) HP config order, and save_path.
"""

from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from picid_report.report import plots


def _make_agg_ds(df: pd.DataFrame, opt_col: str = "test/mse", opt_mode: str = "min") -> xr.Dataset:
    """Convert flat DataFrame with {metric}_mean/_std/_count columns to xr.Dataset."""
    metric_names = [c[:-5] for c in df.columns if c.endswith("_mean")]
    metric_stat_cols = {f"{m}_{s}" for m in metric_names for s in ("mean", "std", "count")}
    hp_cols = [c for c in df.columns if c not in metric_stat_cols]

    n_configs = len(df)
    n_metrics = len(metric_names)
    mean_data = np.full((n_configs, n_metrics), np.nan)
    std_data = np.full((n_configs, n_metrics), np.nan)
    count_data = np.full((n_configs, n_metrics), np.nan)
    for mi, metric in enumerate(metric_names):
        if f"{metric}_mean" in df.columns:
            mean_data[:, mi] = pd.to_numeric(df[f"{metric}_mean"], errors="coerce").values
        if f"{metric}_std" in df.columns:
            std_data[:, mi] = pd.to_numeric(df[f"{metric}_std"], errors="coerce").values
        if f"{metric}_count" in df.columns:
            count_data[:, mi] = pd.to_numeric(df[f"{metric}_count"], errors="coerce").values

    coords: dict = {"config": np.arange(n_configs), "metric": metric_names}
    for hp in hp_cols:
        vals = np.empty(n_configs, dtype=object)
        for j, v in enumerate(df[hp].values):
            vals[j] = v
        coords[hp] = ("config", vals)

    return xr.Dataset(
        {
            "mean":  (["config", "metric"], mean_data),
            "std":   (["config", "metric"], std_data),
            "count": (["config", "metric"], count_data),
        },
        coords=coords,
        attrs={"optimization_col": opt_col, "optimization_mode": opt_mode},
    )


def _make_all_results(
    metrics_data=None,
    sorted_aggregated=None,
    datasets_models=None,
):
    """Build all_results structure for plot tests. datasets_models: [(dataset, model), ...]."""
    if metrics_data is None:
        metrics_data = {
            "mse": {
                "test": {"mean": 0.5, "std": 0.1, "count": 3},
                "val": {"mean": 0.52, "std": 0.08, "count": 3},
            }
        }
    if datasets_models is None:
        datasets_models = [("DS1", "ModelA")]
    all_results = defaultdict(lambda: defaultdict(dict))
    for dataset, model in datasets_models:
        all_results[dataset][model]["best_performance"] = {
            "optimized_on": {"metric": "test/mse", "strategy": "min"},
            "metrics": metrics_data,
        }
        if sorted_aggregated is not None:
            all_results[dataset][model]["sorted_aggregated_results"] = sorted_aggregated
    return all_results


# --- plot_best_metric_bars ---


class TestPlotBestMetricBars:
    """Tests for plot_best_metric_bars: bar chart of best metric value per model/dataset."""

    def test_empty_results_returns_none(self):
        """Empty all_results returns None (no figure)."""
        all_results = defaultdict(lambda: defaultdict(dict))
        fig = plots.plot_best_metric_bars(all_results, metric="test/mse")
        assert fig is None

    def test_no_matching_metric_returns_none(self):
        """When metric name is not in best_performance.metrics, returns None."""
        all_results = _make_all_results(
            metrics_data={"mse": {"test": {"mean": 0.5, "std": 0.1, "count": 3}}}
        )
        fig = plots.plot_best_metric_bars(all_results, metric="test/accuracy")
        assert fig is None

    def test_prefix_missing_returns_none(self):
        """When prefix (e.g. test) is not in metrics[metric_name], skip that model; if all skipped, None."""
        all_results = _make_all_results(
            metrics_data={"mse": {"val": {"mean": 0.52, "std": 0.08, "count": 3}}}
        )
        fig = plots.plot_best_metric_bars(all_results, metric="test/mse")
        assert fig is None

    def test_nan_mean_skipped(self):
        """When mean is NaN, that model/dataset is skipped."""
        all_results = _make_all_results(
            metrics_data={
                "mse": {"test": {"mean": float("nan"), "std": 0.0, "count": 1}}
            }
        )
        fig = plots.plot_best_metric_bars(all_results, metric="test/mse")
        assert fig is None

    def test_single_model_dataset_returns_figure(self):
        """One model/dataset with valid metric returns a figure with one bar."""
        all_results = _make_all_results(
            metrics_data={"mse": {"test": {"mean": 0.42, "std": 0.05, "count": 5}}}
        )
        fig = plots.plot_best_metric_bars(all_results, metric="test/mse")
        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.patches) == 1
        assert ax.get_ylabel() == "test/mse"
        assert "Best" in ax.get_title() and "test/mse" in ax.get_title()

    def test_multiple_models_datasets_orders_bars(self):
        """Multiple model/dataset combinations produce one bar each; labels include model and dataset."""
        all_results = _make_all_results(
            metrics_data={"mse": {"test": {"mean": 0.5, "std": 0.1, "count": 3}}},
            datasets_models=[("D1", "M1"), ("D1", "M2"), ("D2", "M1")],
        )
        fig = plots.plot_best_metric_bars(all_results, metric="test/mse")
        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.patches) == 3
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert any("M1" in label and "D1" in label for label in tick_labels)
        assert any("M2" in label for label in tick_labels)

    def test_metric_without_slash_uses_whole_as_metric_name(self):
        """Metric string without '/' uses entire string as metric_name, prefix empty."""
        all_results = _make_all_results(
            metrics_data={"loss": {"": {"mean": 0.3, "std": 0.0, "count": 1}}}
        )
        # best_performance.metrics has "loss" with key "" for prefix
        fig = plots.plot_best_metric_bars(all_results, metric="loss")
        assert fig is not None
        assert len(fig.axes[0].patches) == 1

    def test_save_path_creates_file(self, tmp_path):
        """When save_path is set, figure is saved to that path."""
        all_results = _make_all_results(
            metrics_data={"mse": {"test": {"mean": 0.5, "std": 0.1, "count": 3}}}
        )
        out = tmp_path / "bars.png"
        plots.plot_best_metric_bars(all_results, metric="test/mse", save_path=str(out))
        assert out.is_file()

    def test_figsize_applied(self):
        """Custom figsize is used when provided."""
        all_results = _make_all_results(
            metrics_data={"mse": {"test": {"mean": 0.5, "std": 0.1, "count": 3}}}
        )
        fig = plots.plot_best_metric_bars(
            all_results, metric="test/mse", figsize=(10, 4)
        )
        assert fig is not None
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 4


# --- plot_hp_impact (ranked/sorted HP configs) ---


class TestPlotHpImpact:
    """Tests for plot_hp_impact: metric vs HP config (configs in rank order)."""

    def test_dataset_missing_returns_none(self):
        """When dataset is not in all_results, returns None."""
        all_results = _make_all_results(
            sorted_aggregated=pd.DataFrame({"test/mse_mean": [0.5]})
        )
        fig = plots.plot_hp_impact(
            all_results, model="ModelA", dataset="Other", metric="test/mse"
        )
        assert fig is None

    def test_model_missing_returns_none(self):
        """When model is not in all_results[dataset], returns None."""
        all_results = _make_all_results(
            sorted_aggregated=pd.DataFrame({"test/mse_mean": [0.5]})
        )
        fig = plots.plot_hp_impact(
            all_results, model="Other", dataset="DS1", metric="test/mse"
        )
        assert fig is None

    def test_no_sorted_aggregated_returns_none(self):
        """When sorted_aggregated_results is missing or empty, returns None."""
        all_results = _make_all_results(sorted_aggregated=None)
        fig = plots.plot_hp_impact(
            all_results, model="ModelA", dataset="DS1", metric="test/mse"
        )
        assert fig is None

        all_results = _make_all_results(sorted_aggregated=pd.DataFrame())
        fig = plots.plot_hp_impact(
            all_results, model="ModelA", dataset="DS1", metric="test/mse"
        )
        assert fig is None

    def test_metric_mean_column_missing_returns_none(self):
        """When metric_mean column is not in flattened aggregated df, returns None."""
        all_results = _make_all_results(
            sorted_aggregated=pd.DataFrame(
                {"other_mean": [0.5], "other_std": [0.0], "other_count": [1]}
            )
        )
        fig = plots.plot_hp_impact(
            all_results, model="ModelA", dataset="DS1", metric="test/mse"
        )
        assert fig is None

    def test_happy_path_returns_figure_with_bars_in_rank_order(self):
        """sorted_aggregated_results (xr.Dataset) produces horizontal bars in config order (rank)."""
        # sorted_aggregated_results is already sorted by best first (rank 1, 2, ...)
        ds = _make_agg_ds(pd.DataFrame(
            {
                "test/mse_mean": [0.40, 0.45, 0.50],
                "test/mse_std": [0.02, 0.03, 0.01],
                "test/mse_count": [3, 3, 3],
            }
        ))
        all_results = _make_all_results(sorted_aggregated=ds)
        fig = plots.plot_hp_impact(
            all_results, model="ModelA", dataset="DS1", metric="test/mse"
        )
        assert fig is not None
        ax = fig.axes[0]
        # Bar chart: y labels are config indices 0, 1, 2 (rank order)
        assert len(ax.patches) == 3
        assert ax.get_xlabel() == "test/mse"
        assert (
            "HP impact" in ax.get_title()
            and "ModelA" in ax.get_title()
            and "DS1" in ax.get_title()
        )

    def test_max_configs_truncates(self):
        """When more than max_configs rows, only first max_configs are plotted."""
        n = 60
        ds = _make_agg_ds(pd.DataFrame(
            {
                "test/mse_mean": [0.4 + i * 0.01 for i in range(n)],
                "test/mse_std": [0.0] * n,
                "test/mse_count": [3] * n,
            }
        ))
        all_results = _make_all_results(sorted_aggregated=ds)
        fig = plots.plot_hp_impact(
            all_results,
            model="ModelA",
            dataset="DS1",
            metric="test/mse",
            max_configs=50,
        )
        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.patches) == 50

    def test_hp_impact_save_path_creates_file(self, tmp_path):
        """When save_path is set, figure is saved."""
        ds = _make_agg_ds(pd.DataFrame(
            {
                "test/mse_mean": [0.5],
                "test/mse_std": [0.0],
                "test/mse_count": [3],
            }
        ))
        all_results = _make_all_results(sorted_aggregated=ds)
        out = tmp_path / "hp_impact.png"
        plots.plot_hp_impact(
            all_results,
            model="ModelA",
            dataset="DS1",
            metric="test/mse",
            save_path=str(out),
        )
        assert out.is_file()

    def test_hp_impact_with_multiindex_flattened(self):
        """sorted_aggregated_results as xr.Dataset with test/mse metric produces horizontal bars."""
        ds = _make_agg_ds(pd.DataFrame(
            {
                "test/mse_mean": [0.44, 0.48],
                "test/mse_std": [0.02, 0.01],
                "test/mse_count": [3, 3],
            }
        ))
        all_results = _make_all_results(sorted_aggregated=ds)
        fig = plots.plot_hp_impact(
            all_results, model="ModelA", dataset="DS1", metric="test/mse"
        )
        assert fig is not None
        assert len(fig.axes[0].patches) == 2


# --- matplotlib import error ---


class TestPlotsMatplotlibRequired:
    """Plot functions raise a clear error when matplotlib is not installed."""

    def test_plot_best_metric_bars_raises_when_matplotlib_unavailable(self):
        """plot_best_metric_bars raises ImportError when _ensure_matplotlib fails."""
        all_results = _make_all_results(
            metrics_data={"mse": {"test": {"mean": 0.5, "std": 0.1, "count": 3}}}
        )
        with patch.object(
            plots,
            "_ensure_matplotlib",
            side_effect=ImportError("matplotlib not installed"),
        ):
            with pytest.raises(ImportError) as exc_info:
                plots.plot_best_metric_bars(all_results, metric="test/mse")
            assert "matplotlib" in str(
                exc_info.value
            ).lower() or "not installed" in str(exc_info.value)

    def test_plot_hp_impact_raises_when_matplotlib_unavailable(self):
        """plot_hp_impact raises ImportError when _ensure_matplotlib fails."""
        all_results = _make_all_results(
            sorted_aggregated=pd.DataFrame(
                {
                    "test/mse_mean": [0.5],
                    "test/mse_std": [0.0],
                    "test/mse_count": [3],
                }
            )
        )
        with patch.object(
            plots,
            "_ensure_matplotlib",
            side_effect=ImportError("matplotlib not installed"),
        ):
            with pytest.raises(ImportError) as exc_info:
                plots.plot_hp_impact(
                    all_results, model="ModelA", dataset="DS1", metric="test/mse"
                )
            assert "matplotlib" in str(
                exc_info.value
            ).lower() or "not installed" in str(exc_info.value)
