"""Tests for dashboard dynamic metric selection helpers and views."""

from __future__ import annotations

import numpy as np
import pandas as pd
import panel as pn
import pytest
import xarray as xr
from matplotlib.colors import to_hex
from types import SimpleNamespace

from picid_report_dashboard.data import ResultsLoader
from picid_report_dashboard import views


PRIMARY_MODEL = "baselines.lstm.model.LSTM_Forecaster"
ALT_MODEL = "model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper"
DATASET = "Dataset A [project_a]"
PRIMARY_DISPLAY_METRIC = "test_best_rerun/mae_denormalized"
ALT_DISPLAY_METRIC = "test/mae_denormalized"
PRIMARY_OPT_METRIC = "test_best_rerun/loss"
ALT_OPT_METRIC = "test/loss"
MISSING_SORT_METRIC = "test/accuracy"
SECOND_DATASET = "Dataset B [project_b]"
THIRD_DATASET = "Dataset C [project_c]"


class FakeResultsLoader(ResultsLoader):
    """Minimal ResultsLoader subclass backed by in-memory xarray datasets."""

    def __init__(
        self, xarray_dataset: xr.Dataset, hp_map: dict[tuple[str, str], xr.Dataset]
    ):
        self.base_dir = "/tmp/fake_dashboard"
        self._xarray_dataset = xarray_dataset
        self._hp_map = hp_map
        self._hp_ds_cache = {}

    @property
    def xarray_dataset(self) -> xr.Dataset:
        return self._xarray_dataset

    def hp_impact_ds(self, dataset: str, model: str) -> xr.Dataset | None:
        cache_key = (dataset, model)
        if cache_key not in self._hp_ds_cache:
            self._hp_ds_cache[cache_key] = self._hp_map.get(cache_key)
        return self._hp_ds_cache[cache_key]


def _make_hp_dataset(
    metrics: list[str],
    means: list[list[float]],
    stds: list[list[float]],
    counts: list[list[float]],
    **hp_coords: list[object],
) -> xr.Dataset:
    coords: dict[str, object] = {
        "config": np.arange(len(means)),
        "metric": metrics,
    }
    for key, values in hp_coords.items():
        coords[key] = ("config", values)

    return xr.Dataset(
        {
            "mean": xr.DataArray(
                np.asarray(means, dtype=float), dims=["config", "metric"]
            ),
            "std": xr.DataArray(
                np.asarray(stds, dtype=float), dims=["config", "metric"]
            ),
            "count": xr.DataArray(
                np.asarray(counts, dtype=float), dims=["config", "metric"]
            ),
        },
        coords=coords,
    )


def _make_loader() -> FakeResultsLoader:
    metric_keys = [
        PRIMARY_DISPLAY_METRIC,
        ALT_DISPLAY_METRIC,
        PRIMARY_OPT_METRIC,
        ALT_OPT_METRIC,
        MISSING_SORT_METRIC,
    ]
    coords3 = {
        "dataset": [DATASET],
        "model": [PRIMARY_MODEL, ALT_MODEL],
        "metric_key": metric_keys,
    }
    coords2 = {
        "dataset": [DATASET],
        "model": [PRIMARY_MODEL, ALT_MODEL],
    }

    xarray_dataset = xr.Dataset(
        {
            "mean": xr.DataArray(
                np.asarray(
                    [
                        [
                            [10.0, np.nan, 0.2, np.nan, np.nan],
                            [np.nan, 20.0, np.nan, 0.3, np.nan],
                        ]
                    ],
                    dtype=float,
                ),
                dims=["dataset", "model", "metric_key"],
                coords=coords3,
            ),
            "std": xr.DataArray(
                np.asarray(
                    [
                        [
                            [1.0, np.nan, 0.02, np.nan, np.nan],
                            [np.nan, 1.5, np.nan, 0.03, np.nan],
                        ]
                    ],
                    dtype=float,
                ),
                dims=["dataset", "model", "metric_key"],
                coords=coords3,
            ),
            "n": xr.DataArray(
                np.asarray(
                    [
                        [
                            [5.0, np.nan, 5.0, np.nan, np.nan],
                            [np.nan, 5.0, np.nan, 5.0, np.nan],
                        ]
                    ],
                    dtype=float,
                ),
                dims=["dataset", "model", "metric_key"],
                coords=coords3,
            ),
            "sort_metric": xr.DataArray(
                np.asarray([[PRIMARY_OPT_METRIC, ALT_OPT_METRIC]], dtype=object),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_metric": xr.DataArray(
                np.asarray([[PRIMARY_OPT_METRIC, ALT_OPT_METRIC]], dtype=object),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_mode": xr.DataArray(
                np.asarray([["min", "min"]], dtype=object),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_value": xr.DataArray(
                np.asarray([[0.2, 0.3]], dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "total_runs": xr.DataArray(
                np.asarray([[10.0, 8.0]], dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "configs_failed_seed": xr.DataArray(
                np.asarray([[1.0, 0.0]], dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "configs_failed_metric": xr.DataArray(
                np.asarray([[0.0, 1.0]], dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
        }
    )

    hp_map = {
        (DATASET, PRIMARY_MODEL): _make_hp_dataset(
            [PRIMARY_DISPLAY_METRIC, PRIMARY_OPT_METRIC],
            [[10.0, 0.2], [5.0, 0.4]],
            [[1.0, 0.02], [0.5, 0.04]],
            [[5.0, 5.0], [5.0, 5.0]],
            **{
                "task_definition.seq_len": [16, 32],
                "optimization.lr": [0.001, 0.002],
            },
        ),
        (DATASET, ALT_MODEL): _make_hp_dataset(
            [ALT_DISPLAY_METRIC, ALT_OPT_METRIC],
            [[20.0, 0.3], [7.0, 0.6]],
            [[1.5, 0.03], [0.7, 0.06]],
            [[4.0, 4.0], [4.0, 4.0]],
            **{
                "task_definition.seq_len": [8, 24],
                "optimization.lr": [0.01, 0.02],
            },
        ),
    }
    return FakeResultsLoader(xarray_dataset, hp_map)


def _make_average_rank_loader() -> FakeResultsLoader:
    metric_keys = [
        PRIMARY_DISPLAY_METRIC,
        ALT_DISPLAY_METRIC,
        PRIMARY_OPT_METRIC,
        ALT_OPT_METRIC,
        MISSING_SORT_METRIC,
    ]
    coords3 = {
        "dataset": [DATASET, SECOND_DATASET],
        "model": [PRIMARY_MODEL, ALT_MODEL],
        "metric_key": metric_keys,
    }
    coords2 = {
        "dataset": [DATASET, SECOND_DATASET],
        "model": [PRIMARY_MODEL, ALT_MODEL],
    }

    xarray_dataset = xr.Dataset(
        {
            "mean": xr.DataArray(
                np.asarray(
                    [
                        [
                            [1.0, np.nan, 0.2, np.nan, np.nan],
                            [np.nan, 2.0, np.nan, 0.3, np.nan],
                        ],
                        [
                            [100.0, np.nan, 0.4, np.nan, np.nan],
                            [np.nan, 3.0, np.nan, 0.5, np.nan],
                        ],
                    ],
                    dtype=float,
                ),
                dims=["dataset", "model", "metric_key"],
                coords=coords3,
            ),
            "std": xr.DataArray(
                np.zeros((2, 2, len(metric_keys)), dtype=float),
                dims=["dataset", "model", "metric_key"],
                coords=coords3,
            ),
            "n": xr.DataArray(
                np.ones((2, 2, len(metric_keys)), dtype=float) * 5.0,
                dims=["dataset", "model", "metric_key"],
                coords=coords3,
            ),
            "sort_metric": xr.DataArray(
                np.asarray(
                    [
                        [PRIMARY_OPT_METRIC, ALT_OPT_METRIC],
                        [PRIMARY_OPT_METRIC, ALT_OPT_METRIC],
                    ],
                    dtype=object,
                ),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_metric": xr.DataArray(
                np.asarray(
                    [
                        [PRIMARY_OPT_METRIC, ALT_OPT_METRIC],
                        [PRIMARY_OPT_METRIC, ALT_OPT_METRIC],
                    ],
                    dtype=object,
                ),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_mode": xr.DataArray(
                np.asarray([["min", "min"], ["min", "min"]], dtype=object),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_value": xr.DataArray(
                np.asarray([[0.2, 0.3], [0.4, 0.5]], dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "total_runs": xr.DataArray(
                np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "configs_failed_seed": xr.DataArray(
                np.zeros((2, 2), dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "configs_failed_metric": xr.DataArray(
                np.zeros((2, 2), dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
        }
    )

    hp_map = {
        (DATASET, PRIMARY_MODEL): _make_hp_dataset(
            [PRIMARY_DISPLAY_METRIC, PRIMARY_OPT_METRIC],
            [[1.0, 0.2]],
            [[0.0, 0.0]],
            [[5.0, 5.0]],
            **{"task_definition.seq_len": [16], "optimization.lr": [0.001]},
        ),
        (DATASET, ALT_MODEL): _make_hp_dataset(
            [ALT_DISPLAY_METRIC, ALT_OPT_METRIC],
            [[2.0, 0.3]],
            [[0.0, 0.0]],
            [[5.0, 5.0]],
            **{"task_definition.seq_len": [8], "optimization.lr": [0.01]},
        ),
        (SECOND_DATASET, PRIMARY_MODEL): _make_hp_dataset(
            [PRIMARY_DISPLAY_METRIC, PRIMARY_OPT_METRIC],
            [[100.0, 0.4]],
            [[0.0, 0.0]],
            [[5.0, 5.0]],
            **{"task_definition.seq_len": [32], "optimization.lr": [0.002]},
        ),
        (SECOND_DATASET, ALT_MODEL): _make_hp_dataset(
            [ALT_DISPLAY_METRIC, ALT_OPT_METRIC],
            [[3.0, 0.5]],
            [[0.0, 0.0]],
            [[5.0, 5.0]],
            **{"task_definition.seq_len": [24], "optimization.lr": [0.02]},
        ),
    }
    return FakeResultsLoader(xarray_dataset, hp_map)


def _make_rank_order_loader() -> FakeResultsLoader:
    metric_keys = [
        PRIMARY_DISPLAY_METRIC,
        ALT_DISPLAY_METRIC,
        PRIMARY_OPT_METRIC,
        ALT_OPT_METRIC,
        MISSING_SORT_METRIC,
    ]
    coords3 = {
        "dataset": [DATASET, SECOND_DATASET, THIRD_DATASET],
        "model": [PRIMARY_MODEL, ALT_MODEL],
        "metric_key": metric_keys,
    }
    coords2 = {
        "dataset": [DATASET, SECOND_DATASET, THIRD_DATASET],
        "model": [PRIMARY_MODEL, ALT_MODEL],
    }

    xarray_dataset = xr.Dataset(
        {
            "mean": xr.DataArray(
                np.asarray(
                    [
                        [
                            [1.0, np.nan, 0.2, np.nan, np.nan],
                            [np.nan, 2.0, np.nan, 0.3, np.nan],
                        ],
                        [
                            [1.0, np.nan, 0.4, np.nan, np.nan],
                            [np.nan, 2.0, np.nan, 0.5, np.nan],
                        ],
                        [
                            [100.0, np.nan, 0.6, np.nan, np.nan],
                            [np.nan, 3.0, np.nan, 0.7, np.nan],
                        ],
                    ],
                    dtype=float,
                ),
                dims=["dataset", "model", "metric_key"],
                coords=coords3,
            ),
            "std": xr.DataArray(
                np.zeros((3, 2, len(metric_keys)), dtype=float),
                dims=["dataset", "model", "metric_key"],
                coords=coords3,
            ),
            "n": xr.DataArray(
                np.ones((3, 2, len(metric_keys)), dtype=float) * 5.0,
                dims=["dataset", "model", "metric_key"],
                coords=coords3,
            ),
            "sort_metric": xr.DataArray(
                np.asarray(
                    [
                        [PRIMARY_OPT_METRIC, ALT_OPT_METRIC],
                        [PRIMARY_OPT_METRIC, ALT_OPT_METRIC],
                        [PRIMARY_OPT_METRIC, ALT_OPT_METRIC],
                    ],
                    dtype=object,
                ),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_metric": xr.DataArray(
                np.asarray(
                    [
                        [PRIMARY_OPT_METRIC, ALT_OPT_METRIC],
                        [PRIMARY_OPT_METRIC, ALT_OPT_METRIC],
                        [PRIMARY_OPT_METRIC, ALT_OPT_METRIC],
                    ],
                    dtype=object,
                ),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_mode": xr.DataArray(
                np.asarray(
                    [["min", "min"], ["min", "min"], ["min", "min"]],
                    dtype=object,
                ),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_value": xr.DataArray(
                np.asarray([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]], dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "total_runs": xr.DataArray(
                np.ones((3, 2), dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "configs_failed_seed": xr.DataArray(
                np.zeros((3, 2), dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "configs_failed_metric": xr.DataArray(
                np.zeros((3, 2), dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
        }
    )

    hp_map = {
        (DATASET, PRIMARY_MODEL): _make_hp_dataset(
            [PRIMARY_DISPLAY_METRIC, PRIMARY_OPT_METRIC],
            [[1.0, 0.2]],
            [[0.0, 0.0]],
            [[5.0, 5.0]],
            **{"task_definition.seq_len": [16], "optimization.lr": [0.001]},
        ),
        (DATASET, ALT_MODEL): _make_hp_dataset(
            [ALT_DISPLAY_METRIC, ALT_OPT_METRIC],
            [[2.0, 0.3]],
            [[0.0, 0.0]],
            [[5.0, 5.0]],
            **{"task_definition.seq_len": [8], "optimization.lr": [0.01]},
        ),
        (SECOND_DATASET, PRIMARY_MODEL): _make_hp_dataset(
            [PRIMARY_DISPLAY_METRIC, PRIMARY_OPT_METRIC],
            [[1.0, 0.4]],
            [[0.0, 0.0]],
            [[5.0, 5.0]],
            **{"task_definition.seq_len": [32], "optimization.lr": [0.002]},
        ),
        (SECOND_DATASET, ALT_MODEL): _make_hp_dataset(
            [ALT_DISPLAY_METRIC, ALT_OPT_METRIC],
            [[2.0, 0.5]],
            [[0.0, 0.0]],
            [[5.0, 5.0]],
            **{"task_definition.seq_len": [24], "optimization.lr": [0.02]},
        ),
        (THIRD_DATASET, PRIMARY_MODEL): _make_hp_dataset(
            [PRIMARY_DISPLAY_METRIC, PRIMARY_OPT_METRIC],
            [[100.0, 0.6]],
            [[0.0, 0.0]],
            [[5.0, 5.0]],
            **{"task_definition.seq_len": [48], "optimization.lr": [0.003]},
        ),
        (THIRD_DATASET, ALT_MODEL): _make_hp_dataset(
            [ALT_DISPLAY_METRIC, ALT_OPT_METRIC],
            [[3.0, 0.7]],
            [[0.0, 0.0]],
            [[5.0, 5.0]],
            **{"task_definition.seq_len": [40], "optimization.lr": [0.03]},
        ),
    }
    return FakeResultsLoader(xarray_dataset, hp_map)


def test_selected_metric_record_uses_requested_sort_metric_not_precomputed_best_row():
    loader = _make_loader()

    record = loader.selected_metric_record(
        DATASET,
        PRIMARY_MODEL,
        metric_key=PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_DISPLAY_METRIC,
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
    )

    assert record["effective_sort_metric_key"] == PRIMARY_DISPLAY_METRIC
    assert record["config_index"] == 1
    assert record["value"] == pytest.approx(5.0)


def test_alt_sort_metric_controls_alt_model_reselection():
    loader = _make_loader()

    record = loader.selected_metric_record(
        DATASET,
        ALT_MODEL,
        metric_key=PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_DISPLAY_METRIC,
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
    )

    assert record["effective_sort_metric_key"] == ALT_DISPLAY_METRIC
    assert record["config_index"] == 1
    assert record["value"] == pytest.approx(7.0)


def test_missing_sort_metric_falls_back_to_opt_metric():
    loader = _make_loader()

    record = loader.selected_metric_record(
        DATASET,
        PRIMARY_MODEL,
        metric_key=PRIMARY_DISPLAY_METRIC,
        sort_metric_key=MISSING_SORT_METRIC,
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=MISSING_SORT_METRIC,
        use_alt_metric=True,
    )

    assert record["sort_metric_fell_back"] is True
    assert record["effective_sort_metric_key"] == PRIMARY_OPT_METRIC
    assert record["value"] == pytest.approx(10.0)


def test_missing_sort_metric_prefers_report_sort_metric_over_opt_metric():
    dataset = "Dataset D [project_d]"
    model = "baselines.crossformer_model.Crossformer_Forecaster"
    display_metric = "test/mae_denormalized"
    opt_metric = "val/loss"
    report_sort_metric = "val_best_rerun/loss"
    missing_sort_metric = "val/f1"

    coords3 = {
        "dataset": [dataset],
        "model": [model],
        "metric_key": [display_metric],
    }
    coords2 = {
        "dataset": [dataset],
        "model": [model],
    }
    xarray_dataset = xr.Dataset(
        {
            "mean": xr.DataArray(
                np.asarray([[[10.0]]], dtype=float),
                dims=["dataset", "model", "metric_key"],
                coords=coords3,
            ),
            "std": xr.DataArray(
                np.asarray([[[1.0]]], dtype=float),
                dims=["dataset", "model", "metric_key"],
                coords=coords3,
            ),
            "n": xr.DataArray(
                np.asarray([[[5.0]]], dtype=float),
                dims=["dataset", "model", "metric_key"],
                coords=coords3,
            ),
            "sort_metric": xr.DataArray(
                np.asarray([[report_sort_metric]], dtype=object),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_metric": xr.DataArray(
                np.asarray([[opt_metric]], dtype=object),
                dims=["dataset", "model"],
                coords=coords2,
            ),
        }
    )
    hp_map = {
        (dataset, model): _make_hp_dataset(
            [display_metric, opt_metric, report_sort_metric],
            [[10.0, 0.2, 0.3], [5.0, 0.4, 0.1]],
            [[1.0, 0.02, 0.03], [0.5, 0.04, 0.01]],
            [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            **{
                "task_definition.seq_len": [16, 32],
                "optimization.lr": [0.001, 0.002],
            },
        ),
    }
    loader = FakeResultsLoader(xarray_dataset, hp_map)

    record = loader.selected_metric_record(
        dataset,
        model,
        metric_key=display_metric,
        sort_metric_key=missing_sort_metric,
        use_alt_metric=False,
    )

    assert record["sort_metric_fell_back"] is True
    assert record["effective_sort_metric_key"] == report_sort_metric
    assert record["value"] == pytest.approx(5.0)


def test_selected_metric_frame_reflects_dynamic_reselection_for_overview():
    loader = _make_loader()

    frame = views._selected_metric_frame(
        loader,
        PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_DISPLAY_METRIC,
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
    )

    values = frame.set_index("model")["value"].to_dict()
    assert values[PRIMARY_MODEL] == pytest.approx(5.0)
    assert values[ALT_MODEL] == pytest.approx(7.0)


def test_build_heatmap_frame_adds_average_rank_from_dataset_ranks():
    loader = _make_average_rank_loader()

    long, _model_order, dataset_order = views._build_heatmap_frame(
        loader,
        PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_DISPLAY_METRIC,
        mode="min",
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
    )

    average_values = (
        long[long["dataset"] == "Average"].set_index("model")["value"].to_dict()
    )
    average_rank_values = (
        long[long["dataset"] == "Average rank"].set_index("model")["value"].to_dict()
    )

    assert dataset_order == [DATASET, SECOND_DATASET, "Average", "Average rank"]
    assert average_values[PRIMARY_MODEL] == pytest.approx(50.5)
    assert average_values[ALT_MODEL] == pytest.approx(2.5)
    assert average_rank_values[PRIMARY_MODEL] == pytest.approx(1.5)
    assert average_rank_values[ALT_MODEL] == pytest.approx(1.5)


def test_build_heatmap_frame_orders_models_by_average_rank_not_average_value():
    loader = _make_rank_order_loader()

    _long, model_order, _dataset_order = views._build_heatmap_frame(
        loader,
        PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_DISPLAY_METRIC,
        mode="min",
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
    )

    assert model_order == [PRIMARY_MODEL, ALT_MODEL]


def test_build_heatmap_title_includes_sort_metrics():
    import holoviews as hv

    hv.extension("bokeh")
    loader = _make_loader()

    pane = views.build_heatmap(
        loader,
        PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_OPT_METRIC,
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_OPT_METRIC,
        use_alt_metric=True,
    )

    plot = hv.renderer("bokeh").get_plot(pane.object)
    assert plot.state.title.text == (
        f"{PRIMARY_DISPLAY_METRIC} "
        f"(sort: {PRIMARY_OPT_METRIC}; alt sort: {ALT_OPT_METRIC})"
    )


def test_build_parallel_coordinates_returns_plotly_pane_with_expected_axes():
    loader = _make_average_rank_loader()

    pane = views.build_parallel_coordinates(
        loader,
        PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_DISPLAY_METRIC,
        mode="min",
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
    )

    assert isinstance(pane, pn.pane.Plotly)
    dimensions = pane.object.data[0].dimensions
    assert [dimension.label for dimension in dimensions] == [
        "Model",
        "Dataset A [project_a]",
        "Dataset B [project_b]",
        "Average",
        "Average rank",
    ]
    assert pane.object.data[0].labelside == "bottom"
    assert pane.object.data[0].labelangle == -45
    assert pane.object.data[0].line.showscale is False
    assert len(pane.object.data[0].line.color) == 2
    assert list(pane.object.data[0].line.color) == [0, 1]
    colorscale = pane.object.data[0].line.colorscale
    assert colorscale[0][1].startswith("#")
    assert colorscale[1][1].startswith("#")
    assert colorscale[0][1] != colorscale[-1][1]


def test_build_parallel_coordinates_frame_uses_alt_metric_for_alt_models():
    loader = _make_loader()

    frame, ordered_models, axis_order = views._build_parallel_coordinates_frame(
        loader,
        PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_DISPLAY_METRIC,
        mode="min",
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
    )

    assert ordered_models == [PRIMARY_MODEL, ALT_MODEL]
    assert axis_order == [DATASET, "Average", "Average rank"]
    values = frame.set_index("model")[DATASET].to_dict()
    assert values[PRIMARY_MODEL] == pytest.approx(5.0)
    assert values[ALT_MODEL] == pytest.approx(7.0)


def test_build_parallel_coordinates_frame_orders_models_by_average_rank():
    loader = _make_rank_order_loader()

    frame, ordered_models, _axis_order = views._build_parallel_coordinates_frame(
        loader,
        PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_DISPLAY_METRIC,
        mode="min",
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
    )

    assert ordered_models == [PRIMARY_MODEL, ALT_MODEL]
    assert list(frame["model"]) == [PRIMARY_MODEL, ALT_MODEL]


def test_build_spiderweb_interactive_returns_holoviews_pane():
    loader = _make_average_rank_loader()

    pane = views.build_spiderweb(
        loader,
        PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_DISPLAY_METRIC,
        mode="min",
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
        selected_dataset=DATASET,
        plot_mode="interactive",
    )

    assert isinstance(pane, pn.pane.HoloViews)
    assert not isinstance(pane, pn.pane.Plotly)


def test_build_spiderweb_interactive_uses_dataset_spokes_and_model_paths():
    loader = _make_average_rank_loader()

    pane = views.build_spiderweb(
        loader,
        PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_DISPLAY_METRIC,
        mode="min",
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
        selected_dataset=DATASET,
        plot_mode="interactive",
    )

    overlay = pane.object
    label_texts = [
        row["text"]
        for element in overlay
        if element.__class__.__name__ == "Labels"
        for _, row in element.data.iterrows()
    ]
    model_path_labels = [
        element.label
        for element in overlay
        if element.__class__.__name__ == "Path" and element.label != ""
    ]

    assert sorted(label_texts) == sorted(
        [
            views._shorten_dataset_name(DATASET),
            views._shorten_dataset_name(SECOND_DATASET),
        ]
    )
    assert model_path_labels == [
        views._shorten_model_name(ALT_MODEL),
        views._shorten_model_name(PRIMARY_MODEL),
    ]


def test_build_spiderweb_matplotlib_returns_radar_chart_with_expected_labels():
    loader = _make_average_rank_loader()

    pane = views.build_spiderweb(
        loader,
        PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_DISPLAY_METRIC,
        mode="min",
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
        selected_dataset=DATASET,
        plot_mode="matplotlib",
    )

    assert isinstance(pane, pn.pane.Matplotlib)

    figure = pane.object
    axis = figure.axes[0]
    legend = axis.get_legend()

    assert axis.name == "polar"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        views._shorten_dataset_name(DATASET),
        views._shorten_dataset_name(SECOND_DATASET),
    ]
    assert to_hex(axis.get_xticklabels()[0].get_color()) == "#2563eb"
    assert to_hex(axis.get_xticklabels()[1].get_color()) == "#374151"
    assert [text.get_text() for text in legend.get_texts()] == [
        views._shorten_model_name(ALT_MODEL),
        views._shorten_model_name(PRIMARY_MODEL),
    ]


def test_build_spiderweb_frame_orders_model_paths_by_average_rank():
    loader = _make_rank_order_loader()

    raw_matrix, normalized_matrix, model_order, dataset_order = (
        views._build_spiderweb_frame(
            loader,
            PRIMARY_DISPLAY_METRIC,
            sort_metric_key=PRIMARY_DISPLAY_METRIC,
            mode="min",
            alt_metric_key=ALT_DISPLAY_METRIC,
            alt_sort_metric_key=ALT_DISPLAY_METRIC,
            use_alt_metric=True,
        )
    )

    assert model_order == [PRIMARY_MODEL, ALT_MODEL]
    assert list(raw_matrix.columns) == [PRIMARY_MODEL, ALT_MODEL]
    assert list(normalized_matrix.columns) == [PRIMARY_MODEL, ALT_MODEL]
    assert dataset_order == [DATASET, SECOND_DATASET, THIRD_DATASET]


def test_build_spiderweb_frame_normalizes_scores_by_mode():
    loader = _make_average_rank_loader()

    _raw_min, normalized_min, _model_order, _dataset_order = (
        views._build_spiderweb_frame(
            loader,
            PRIMARY_DISPLAY_METRIC,
            sort_metric_key=PRIMARY_DISPLAY_METRIC,
            mode="min",
            alt_metric_key=ALT_DISPLAY_METRIC,
            alt_sort_metric_key=ALT_DISPLAY_METRIC,
            use_alt_metric=True,
        )
    )
    _raw_max, normalized_max, _model_order, _dataset_order = (
        views._build_spiderweb_frame(
            loader,
            PRIMARY_DISPLAY_METRIC,
            sort_metric_key=PRIMARY_DISPLAY_METRIC,
            mode="max",
            alt_metric_key=ALT_DISPLAY_METRIC,
            alt_sort_metric_key=ALT_DISPLAY_METRIC,
            use_alt_metric=True,
        )
    )

    assert normalized_min.loc[DATASET, PRIMARY_MODEL] == pytest.approx(1.0)
    assert normalized_min.loc[DATASET, ALT_MODEL] == pytest.approx(0.0)
    assert normalized_min.loc[SECOND_DATASET, PRIMARY_MODEL] == pytest.approx(0.0)
    assert normalized_min.loc[SECOND_DATASET, ALT_MODEL] == pytest.approx(1.0)

    assert normalized_max.loc[DATASET, PRIMARY_MODEL] == pytest.approx(0.0)
    assert normalized_max.loc[DATASET, ALT_MODEL] == pytest.approx(1.0)
    assert normalized_max.loc[SECOND_DATASET, PRIMARY_MODEL] == pytest.approx(1.0)
    assert normalized_max.loc[SECOND_DATASET, ALT_MODEL] == pytest.approx(0.0)


def test_build_spiderweb_frame_uses_alt_metric_for_alt_models():
    loader = _make_loader()

    raw_matrix, _normalized_matrix, model_order, dataset_order = (
        views._build_spiderweb_frame(
            loader,
            PRIMARY_DISPLAY_METRIC,
            sort_metric_key=PRIMARY_DISPLAY_METRIC,
            mode="min",
            alt_metric_key=ALT_DISPLAY_METRIC,
            alt_sort_metric_key=ALT_DISPLAY_METRIC,
            use_alt_metric=True,
        )
    )

    assert model_order == [PRIMARY_MODEL, ALT_MODEL]
    assert dataset_order == [DATASET]
    assert raw_matrix.loc[DATASET, PRIMARY_MODEL] == pytest.approx(5.0)
    assert raw_matrix.loc[DATASET, ALT_MODEL] == pytest.approx(7.0)


def test_build_hp_impact_frame_sorts_rows_by_active_sort_metric():
    loader = _make_loader()

    frame, selection, hp_cols, metric_cols = views._build_hp_impact_frame(
        loader,
        DATASET,
        PRIMARY_MODEL,
        metric_key=PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_DISPLAY_METRIC,
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
    )

    assert hp_cols == ["task_definition.seq_len", "optimization.lr"]
    assert PRIMARY_DISPLAY_METRIC in metric_cols
    assert selection["effective_sort_metric_key"] == PRIMARY_DISPLAY_METRIC
    assert list(frame["task_definition.seq_len"]) == [32, 16]
    assert frame.iloc[0][PRIMARY_DISPLAY_METRIC].startswith("5.0000")


def test_build_hp_impact_frame_places_metric_and_sort_metric_before_other_metrics():
    loader = _make_loader()

    frame, selection, hp_cols, metric_cols = views._build_hp_impact_frame(
        loader,
        DATASET,
        PRIMARY_MODEL,
        metric_key=PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_OPT_METRIC,
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_OPT_METRIC,
        use_alt_metric=True,
    )

    assert selection["display_metric_key"] == PRIMARY_DISPLAY_METRIC
    assert selection["effective_sort_metric_key"] == PRIMARY_OPT_METRIC
    assert metric_cols[:2] == [PRIMARY_DISPLAY_METRIC, PRIMARY_OPT_METRIC]
    assert list(frame.columns[:4]) == [
        hp_cols[0],
        hp_cols[1],
        PRIMARY_DISPLAY_METRIC,
        PRIMARY_OPT_METRIC,
    ]


def test_sort_metric_highlight_css_targets_active_metric_column():
    css = views._sort_metric_highlight_css(PRIMARY_DISPLAY_METRIC)

    assert PRIMARY_DISPLAY_METRIC in css
    assert "#ffe3bf" in css
    assert "#fff4e5" in css


def test_build_metadata_panel_shows_dashboard_and_report_sort_metrics():
    loader = _make_loader()

    panel = views.build_metadata_panel(
        loader,
        DATASET,
        PRIMARY_MODEL,
        metric_key=PRIMARY_DISPLAY_METRIC,
        sort_metric_key=MISSING_SORT_METRIC,
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=MISSING_SORT_METRIC,
        use_alt_metric=True,
    )

    metadata = dict(panel.value.itertuples(index=False, name=None))
    assert metadata["Dashboard metric"] == PRIMARY_DISPLAY_METRIC
    assert metadata["Dashboard sort metric"] == (
        f"{PRIMARY_OPT_METRIC} (fallback from {MISSING_SORT_METRIC})"
    )
    assert metadata["Report sort metric"] == PRIMARY_OPT_METRIC


def test_build_summary_table_uses_alt_metric_for_alt_models():
    loader = SimpleNamespace(
        summary_df=pd.DataFrame(
            [
                {
                    "Project": "project_a",
                    "Dataset": "Dataset A",
                    "Metric": PRIMARY_DISPLAY_METRIC,
                    "Model": PRIMARY_MODEL,
                    "Value": "primary row",
                },
                {
                    "Project": "project_a",
                    "Dataset": "Dataset A",
                    "Metric": PRIMARY_DISPLAY_METRIC,
                    "Model": ALT_MODEL,
                    "Value": "wrong alt row",
                },
                {
                    "Project": "project_a",
                    "Dataset": "Dataset A",
                    "Metric": ALT_DISPLAY_METRIC,
                    "Model": ALT_MODEL,
                    "Value": "alt row",
                },
            ]
        )
    )

    table = views.build_summary_table(
        loader,
        DATASET,
        metric=PRIMARY_DISPLAY_METRIC,
        alt_metric_key=ALT_DISPLAY_METRIC,
        use_alt_metric=True,
    )

    rows = table.value[["Model", "Metric", "Value"]].to_dict(orient="records")
    assert rows == [
        {
            "Model": PRIMARY_MODEL,
            "Metric": PRIMARY_DISPLAY_METRIC,
            "Value": "primary row",
        },
        {
            "Model": ALT_MODEL,
            "Metric": ALT_DISPLAY_METRIC,
            "Value": "alt row",
        },
    ]


def test_display_metadata_value_hides_numpy_and_pandas_missing_scalars():
    assert views._display_metadata_value(np.float64(np.nan)) == "—"
    assert views._display_metadata_value(pd.NA) == "—"


def test_build_model_summary_table_pivots_metadata_properties_into_columns():
    loader = _make_loader()

    panel = views.build_model_summary_table(
        loader,
        DATASET,
        metric_key=PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_OPT_METRIC,
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_OPT_METRIC,
        use_alt_metric=True,
    )

    df = panel.value
    assert list(df.columns[:4]) == [
        "Model",
        "Dashboard metric",
        "Dashboard sort metric",
        "Report sort metric",
    ]
    rows = df.set_index("Model").to_dict(orient="index")
    assert rows["LSTM"]["Dashboard metric"] == PRIMARY_DISPLAY_METRIC
    assert rows["LSTM"]["Dashboard sort metric"] == PRIMARY_OPT_METRIC
    assert rows["FitPredictTabPFN"]["Dashboard metric"] == ALT_DISPLAY_METRIC
    assert rows["FitPredictTabPFN"]["Dashboard sort metric"] == ALT_OPT_METRIC


def test_build_model_summary_table_replaces_missing_cells_with_dash(monkeypatch):
    loader = SimpleNamespace(
        datasets=[DATASET],
        models=[PRIMARY_MODEL, ALT_MODEL],
    )

    def fake_metadata_rows(loader, dataset, model, **kwargs) -> dict[str, object]:
        if model == PRIMARY_MODEL:
            return {
                "Dashboard metric": PRIMARY_DISPLAY_METRIC,
                "Opt mode": "min",
            }
        return {
            "Dashboard metric": np.float64(np.nan),
            "Opt mode": pd.NA,
        }

    monkeypatch.setattr(views, "_metadata_rows", fake_metadata_rows)

    table = views.build_model_summary_table(
        loader,
        DATASET,
        metric_key=PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_OPT_METRIC,
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_OPT_METRIC,
        use_alt_metric=True,
    )

    rows = table.value.set_index("Model").to_dict(orient="index")
    assert rows["FitPredictTabPFN"]["Dashboard metric"] == "—"
    assert rows["FitPredictTabPFN"]["Opt mode"] == "—"


def test_build_model_summary_table_clicks_model_cell_with_full_model_key():
    loader = _make_loader()
    clicked_models: list[str] = []

    table = views.build_model_summary_table(
        loader,
        DATASET,
        metric_key=PRIMARY_DISPLAY_METRIC,
        sort_metric_key=PRIMARY_OPT_METRIC,
        alt_metric_key=ALT_DISPLAY_METRIC,
        alt_sort_metric_key=ALT_OPT_METRIC,
        use_alt_metric=True,
        on_model_click=clicked_models.append,
    )

    callback = table._on_click_callbacks["Model"][0]
    callback(SimpleNamespace(row=1))

    assert clicked_models == [ALT_MODEL]


# ---------------------------------------------------------------------------
# build_latex_table — cell highlighting (best / 2nd best / within-1σ)
# ---------------------------------------------------------------------------


HL_DATASET = "Dataset H [project_h]"
HL_METRIC = "test/mae_denormalized"
HL_MODEL_BEST = "models.demo.BestForecaster"
HL_MODEL_NEAR = "models.demo.NearForecaster"
HL_MODEL_FAR = "models.demo.FarForecaster"
HL_LABEL_BEST = "Best"  # _shorten_model_name strips "Forecaster"
HL_LABEL_NEAR = "Near"
HL_LABEL_FAR = "Far"


def _make_highlight_loader(
    means: dict[str, float],
    stds: dict[str, float],
    *,
    dataset: str = HL_DATASET,
    metric_key: str = HL_METRIC,
) -> FakeResultsLoader:
    """Build a single-dataset loader with one config per model for highlight tests."""
    models = list(means.keys())
    n_models = len(models)

    coords3 = {"dataset": [dataset], "model": models, "metric_key": [metric_key]}
    coords2 = {"dataset": [dataset], "model": models}

    mean_arr = np.asarray([[[means[m]] for m in models]], dtype=float)
    std_arr = np.asarray([[[stds[m]] for m in models]], dtype=float)
    n_arr = np.full((1, n_models, 1), 5.0, dtype=float)

    xarray_dataset = xr.Dataset(
        {
            "mean": xr.DataArray(
                mean_arr, dims=["dataset", "model", "metric_key"], coords=coords3
            ),
            "std": xr.DataArray(
                std_arr, dims=["dataset", "model", "metric_key"], coords=coords3
            ),
            "n": xr.DataArray(
                n_arr, dims=["dataset", "model", "metric_key"], coords=coords3
            ),
            "sort_metric": xr.DataArray(
                np.asarray([[metric_key] * n_models], dtype=object),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_metric": xr.DataArray(
                np.asarray([[metric_key] * n_models], dtype=object),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_mode": xr.DataArray(
                np.asarray([["min"] * n_models], dtype=object),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "opt_value": xr.DataArray(
                np.asarray([[means[m] for m in models]], dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "total_runs": xr.DataArray(
                np.full((1, n_models), 5.0, dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "configs_failed_seed": xr.DataArray(
                np.zeros((1, n_models), dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
            "configs_failed_metric": xr.DataArray(
                np.zeros((1, n_models), dtype=float),
                dims=["dataset", "model"],
                coords=coords2,
            ),
        }
    )

    hp_map = {
        (dataset, m): _make_hp_dataset(
            [metric_key],
            [[means[m]]],
            [[stds[m]]],
            [[5.0]],
            **{"task_definition.seq_len": [16]},
        )
        for m in models
    }
    return FakeResultsLoader(xarray_dataset, hp_map)


def _latex_value(component) -> str:
    """Extract the LaTeX text from a build_latex_table return value (Column[btn, textarea])."""
    return component[1].value


def _latex_row(latex_str: str, model_label: str) -> str:
    """Find the rendered tabular row for *model_label* in the LaTeX output."""
    for line in latex_str.splitlines():
        if line.lstrip().startswith(model_label):
            return line
    raise AssertionError(f"row for {model_label!r} not found in LaTeX output:\n{latex_str}")


_HL_MEANS_BASIC = {HL_MODEL_BEST: 1.0, HL_MODEL_NEAR: 1.3, HL_MODEL_FAR: 2.5}
_HL_STDS_BASIC = {HL_MODEL_BEST: 0.5, HL_MODEL_NEAR: 0.1, HL_MODEL_FAR: 0.1}


def test_build_latex_table_bolds_best_and_greys_background():
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
    )
    latex_str = _latex_value(pane)
    best_row = _latex_row(latex_str, HL_LABEL_BEST)

    assert "\\textbf{" in best_row
    assert "\\cellcolor[gray]{0.85}" in best_row
    assert "\\underline{" not in best_row
    assert "\\cellcolor{blue!15}" not in best_row


def test_build_latex_table_underlines_second_best():
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
    )
    latex_str = _latex_value(pane)
    near_row = _latex_row(latex_str, HL_LABEL_NEAR)

    assert "\\underline{" in near_row
    assert "\\textbf{" not in near_row
    assert "\\cellcolor[gray]{0.85}" not in near_row


def test_build_latex_table_blue_for_within_one_std():
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
    )
    latex_str = _latex_value(pane)
    near_row = _latex_row(latex_str, HL_LABEL_NEAR)
    far_row = _latex_row(latex_str, HL_LABEL_FAR)

    # Near is within 1σ of best (|1.3 − 1.0| = 0.3 ≤ best_std = 0.5)
    assert "\\cellcolor{blue!15}" in near_row
    # Far has no decorations whatsoever
    for marker in (
        "\\textbf{",
        "\\underline{",
        "\\cellcolor[gray]{0.85}",
        "\\cellcolor{blue!15}",
    ):
        assert marker not in far_row, f"unexpected {marker!r} in Far row: {far_row}"


@pytest.mark.parametrize(
    "kwargs, missing_marker",
    [
        ({"highlight_best_bg": False}, "\\cellcolor[gray]{0.85}"),
        ({"underline_2nd_best": False}, "\\underline{"),
        ({"highlight_within_1std": False}, "\\cellcolor{blue!15}"),
    ],
)
def test_build_latex_table_toggles_off_disable_decorations_independently(
    kwargs, missing_marker
):
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
        **kwargs,
    )
    latex_str = _latex_value(pane)

    assert missing_marker not in latex_str
    # Bold of the best is unconditional and unaffected by the toggles
    assert "\\textbf{" in latex_str


def test_build_latex_table_all_highlights_off_keeps_only_bold():
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
        highlight_best_bg=False,
        underline_2nd_best=False,
        highlight_within_1std=False,
    )
    latex_str = _latex_value(pane)

    # No background colors and no underlines anywhere in the body
    body_start = latex_str.find("\\begin{tabular}")
    assert body_start != -1
    body = latex_str[body_start:]
    assert "\\cellcolor" not in body
    assert "\\underline{" not in body
    # Bold remains on the best cell
    assert "\\textbf{" in body


def test_build_latex_table_tied_best_no_second_underline():
    loader = _make_highlight_loader(
        means={HL_MODEL_BEST: 1.0, HL_MODEL_NEAR: 1.0, HL_MODEL_FAR: 2.0},
        stds={HL_MODEL_BEST: 0.5, HL_MODEL_NEAR: 0.5, HL_MODEL_FAR: 0.1},
    )
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
    )
    latex_str = _latex_value(pane)
    best_row = _latex_row(latex_str, HL_LABEL_BEST)
    near_row = _latex_row(latex_str, HL_LABEL_NEAR)
    far_row = _latex_row(latex_str, HL_LABEL_FAR)

    # Both tied bests are bolded + grey-backgrounded
    assert "\\textbf{" in best_row and "\\cellcolor[gray]{0.85}" in best_row
    assert "\\textbf{" in near_row and "\\cellcolor[gray]{0.85}" in near_row
    # No second-best is promoted when the top is tied
    assert "\\underline{" not in far_row
    body_start = latex_str.find("\\begin{tabular}")
    assert body_start != -1
    assert "\\underline{" not in latex_str[body_start:]


def test_build_latex_table_max_mode_picks_best_correctly():
    loader = _make_highlight_loader(
        means={HL_MODEL_BEST: 1.0, HL_MODEL_NEAR: 1.3, HL_MODEL_FAR: 2.5},
        stds={HL_MODEL_BEST: 0.1, HL_MODEL_NEAR: 0.1, HL_MODEL_FAR: 0.5},
    )
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="max",
        use_alt_metric=False,
    )
    latex_str = _latex_value(pane)
    best_row = _latex_row(latex_str, HL_LABEL_BEST)
    far_row = _latex_row(latex_str, HL_LABEL_FAR)

    # Far has the largest mean → it is now the best
    assert "\\textbf{" in far_row
    assert "\\cellcolor[gray]{0.85}" in far_row
    # Best has the smallest mean → no decorations
    for marker in (
        "\\textbf{",
        "\\underline{",
        "\\cellcolor[gray]{0.85}",
        "\\cellcolor{blue!15}",
    ):
        assert marker not in best_row, f"unexpected {marker!r} in Best row: {best_row}"


def test_build_latex_table_header_documents_required_packages():
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
    )
    latex_str = _latex_value(pane)

    assert "\\usepackage{booktabs}" in latex_str
    assert "\\usepackage[table]{xcolor}" in latex_str
    assert "\\usepackage{adjustbox}" in latex_str


def test_build_latex_table_orders_best_model_at_bottom():
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
    )
    latex_str = _latex_value(pane)

    lines = latex_str.splitlines()

    def _row_index(label: str) -> int:
        for idx, line in enumerate(lines):
            if line.lstrip().startswith(label):
                return idx
        raise AssertionError(f"row {label!r} not found in:\n{latex_str}")

    far_idx = _row_index(HL_LABEL_FAR)
    near_idx = _row_index(HL_LABEL_NEAR)
    best_idx = _row_index(HL_LABEL_BEST)

    # Worst → best, top to bottom
    assert far_idx < near_idx < best_idx


def _extract_caption(latex_str: str) -> str:
    """Return the contents of the \\caption{...} command, handling nested braces."""
    marker = "\\caption{"
    start = latex_str.find(marker)
    assert start != -1, "no \\caption{...} found in LaTeX output"
    i = start + len(marker)
    depth = 1
    while i < len(latex_str) and depth > 0:
        ch = latex_str[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return latex_str[start + len(marker) : i]
        i += 1
    raise AssertionError("unmatched braces in \\caption{...}")


def test_build_latex_table_caption_lists_display_and_sort_metric():
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)
    sort_metric = "val_best_rerun/loss"
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=sort_metric,
        mode="min",
        use_alt_metric=False,
    )
    caption = _extract_caption(_latex_value(pane))

    assert HL_METRIC.replace("_", "\\_") in caption
    assert sort_metric.replace("_", "\\_") in caption
    assert "sorted by" in caption
    assert "min" in caption


def test_build_latex_table_caption_lists_dataset_shortcuts_to_full_names():
    # Dataset name with a date-stamped project suffix → _shorten_dataset_name
    # strips the date, so the displayed shortcut differs from the full name.
    full_dataset = "Dataset H [16_03_2026_project_h]"
    expected_short = "Dataset H [project_h]"

    loader = _make_highlight_loader(
        _HL_MEANS_BASIC,
        _HL_STDS_BASIC,
        dataset=full_dataset,
    )
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
    )
    caption = _extract_caption(_latex_value(pane))

    assert "Datasets:" in caption
    escaped_short = expected_short.replace("_", "\\_")
    escaped_full = full_dataset.replace("_", "\\_")
    assert f"{escaped_short} = {escaped_full}" in caption


def test_build_latex_table_wraps_tabular_in_adjustbox():
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
    )
    latex_str = _latex_value(pane)

    assert "\\begin{adjustbox}{max width=\\textwidth}" in latex_str
    assert "\\end{adjustbox}" in latex_str
    # adjustbox brackets must enclose the tabular environment, not the other way around
    adj_open = latex_str.index("\\begin{adjustbox}")
    tab_open = latex_str.index("\\begin{tabular}")
    tab_close = latex_str.index("\\end{tabular}")
    adj_close = latex_str.index("\\end{adjustbox}")
    assert adj_open < tab_open < tab_close < adj_close


def test_build_latex_table_multiplier_default_is_no_op():
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)
    baseline = _latex_value(
        views.build_latex_table(
            loader,
            HL_METRIC,
            sort_metric_key=HL_METRIC,
            mode="min",
            use_alt_metric=False,
        )
    )
    explicit = _latex_value(
        views.build_latex_table(
            loader,
            HL_METRIC,
            sort_metric_key=HL_METRIC,
            mode="min",
            use_alt_metric=False,
            multiplier=1.0,
        )
    )
    assert explicit == baseline


def test_build_latex_table_multiplier_scales_mean_and_std():
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
        multiplier=100.0,
    )
    latex_str = _latex_value(pane)
    best_row = _latex_row(latex_str, HL_LABEL_BEST)
    near_row = _latex_row(latex_str, HL_LABEL_NEAR)

    # mean 1.0 → 100.0000, std 0.5 → 50.0000
    assert "100.0000" in best_row and "50.0000" in best_row
    # mean 1.3 → 130.0000, std 0.1 → 10.0000
    assert "130.0000" in near_row and "10.0000" in near_row
    # raw values must be gone
    assert "1.0000" not in best_row
    assert "1.3000" not in near_row


def test_build_latex_table_multiplier_skips_average_rank():
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)
    pane = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
        multiplier=100.0,
    )
    latex_str = _latex_value(pane)
    best_row = _latex_row(latex_str, HL_LABEL_BEST)
    far_row = _latex_row(latex_str, HL_LABEL_FAR)

    # Per-dataset rank uses ":.2f"; with one dataset and three models, ranks are
    # 1.00 / 2.00 / 3.00. A wrongly-multiplied rank would render as 100.00 / 300.00.
    assert "1.00" in best_row
    assert "3.00" in far_row
    assert "300.00" not in far_row


def test_build_latex_table_multiplier_in_header_and_caption():
    loader = _make_highlight_loader(_HL_MEANS_BASIC, _HL_STDS_BASIC)

    pane_default = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
    )
    latex_default = _latex_value(pane_default)
    assert "Values multiplied by" not in latex_default

    pane_scaled = views.build_latex_table(
        loader,
        HL_METRIC,
        sort_metric_key=HL_METRIC,
        mode="min",
        use_alt_metric=False,
        multiplier=100.0,
    )
    latex_scaled = _latex_value(pane_scaled)
    assert "% Values multiplied by 100" in latex_scaled
    caption = _extract_caption(latex_scaled)
    assert "Values multiplied by 100." in caption
