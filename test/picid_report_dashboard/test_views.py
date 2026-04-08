"""Tests for dashboard dynamic metric selection helpers and views."""

from __future__ import annotations

import numpy as np
import pandas as pd
import panel as pn
import pytest
import xarray as xr
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

    def __init__(self, xarray_dataset: xr.Dataset, hp_map: dict[tuple[str, str], xr.Dataset]):
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
            "mean": xr.DataArray(np.asarray(means, dtype=float), dims=["config", "metric"]),
            "std": xr.DataArray(np.asarray(stds, dtype=float), dims=["config", "metric"]),
            "count": xr.DataArray(np.asarray(counts, dtype=float), dims=["config", "metric"]),
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
