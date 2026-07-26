"""Tests for dashboard state persistence helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import panel as pn
import xarray as xr

from panel.io.location import Location

from picid_report_dashboard import app as dashboard_app
from picid_report_dashboard.data import ResultsLoader as DataResultsLoader


def _loader_factory(loader_map):
    class FakeResultsLoader:
        def __init__(self, base_dir: str) -> None:
            if base_dir not in loader_map:
                raise FileNotFoundError(base_dir)
            config = loader_map[base_dir]
            self.base_dir = base_dir
            self.metric_keys = list(config["metric_keys"])
            self.datasets = list(config["datasets"])
            self.models = list(config["models"])
            self.xarray_dataset = object()
            self.summary_df = pd.DataFrame()
            self.stats_df = pd.DataFrame()
            self.applied_model_aliases: list[tuple[str, str]] = []
            self._hp_map = dict(config.get("hp_map", {}))

        def hp_impact_ds(self, dataset: str, model: str):
            return self._hp_map.get((dataset, model))

        def dataset_has_display_metric(
            self,
            dataset: str,
            *,
            metric_key: str,
            alt_metric_key: str | None = None,
            use_alt_metric: bool = True,
            enabled_models: list[str] | None = None,
        ) -> bool:
            models = self.models
            if enabled_models is not None:
                enabled = set(enabled_models)
                models = [model_name for model_name in models if model_name in enabled]

            if not self._hp_map:
                return dataset in self.datasets and bool(models)
            for model_name in models:
                display_metric_key = (
                    alt_metric_key
                    if (
                        use_alt_metric
                        and alt_metric_key
                        and dashboard_app.is_alt_model(model_name)
                    )
                    else metric_key
                )
                hp_ds = self.hp_impact_ds(dataset, model_name)
                if hp_ds is None or "metric" not in hp_ds.coords:
                    continue
                metrics = {
                    str(metric) for metric in hp_ds.coords["metric"].values.tolist()
                }
                if display_metric_key not in metrics:
                    continue
                mean_values = np.asarray(
                    hp_ds.sel(metric=display_metric_key)["mean"].values,
                    dtype=float,
                )
                if np.isfinite(mean_values).any():
                    return True
            return False

        def datasets_with_display_metric(
            self,
            *,
            metric_key: str,
            alt_metric_key: str | None = None,
            use_alt_metric: bool = True,
            enabled_models: list[str] | None = None,
        ) -> list[str]:
            return [
                dataset_name
                for dataset_name in self.datasets
                if self.dataset_has_display_metric(
                    dataset_name,
                    metric_key=metric_key,
                    alt_metric_key=alt_metric_key,
                    use_alt_metric=use_alt_metric,
                    enabled_models=enabled_models,
                )
            ]

    return FakeResultsLoader


def _make_dashboard(monkeypatch, loader_map, report_dir: str = "report_a"):
    monkeypatch.setattr(dashboard_app, "ResultsLoader", _loader_factory(loader_map))
    return dashboard_app.PiCIDDashboard(report_dir=report_dir)


def _make_hp_dataset(metrics: list[str]) -> xr.Dataset:
    return xr.Dataset(
        {
            "mean": xr.DataArray(
                np.zeros((1, len(metrics)), dtype=float),
                dims=["config", "metric"],
            ),
            "std": xr.DataArray(
                np.zeros((1, len(metrics)), dtype=float),
                dims=["config", "metric"],
            ),
            "count": xr.DataArray(
                np.ones((1, len(metrics)), dtype=float),
                dims=["config", "metric"],
            ),
        },
        coords={"config": [0], "metric": metrics},
    )


def test_serialize_state_contains_expected_keys(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                    "custom_metric",
                ],
                "datasets": ["dataset_a", "dataset_b"],
                "models": ["model_a", "model_b"],
            }
        },
    )

    dash.metric_key = "custom_metric"
    dash.sort_metric_key = "custom_metric"
    dash.mode = "max"
    dash.use_alt_metric = False
    dash.alt_metric_key = "test/mae_denormalized"
    dash.alt_sort_metric_key = "test/mae_denormalized"
    dash.show_n = False
    dash.show_rank = True
    dash.spiderweb_plot_mode = "interactive"
    dash.selected_dataset = "dataset_b"
    dash.selected_model = "model_b"
    dash.active_tab = 3

    state = dash._serialize_state()

    assert set(state) == {"version", *dashboard_app._PERSISTED_STATE_PARAMS}
    assert state["version"] == dashboard_app._PERSISTED_STATE_VERSION
    assert state["metric_key"] == "custom_metric"
    assert state["sort_metric_key"] == "custom_metric"
    assert state["alt_metric_overridden"] is True
    assert state["alt_sort_metric_overridden"] is True
    assert state["spiderweb_plot_mode"] == "interactive"
    assert state["selected_model"] == "model_b"
    assert state["active_tab"] == 3


def test_named_view_roundtrip_preserves_name_date_and_state(monkeypatch):
    loader_map = {
        "report_a": {
            "metric_keys": [
                dashboard_app._DEFAULT_METRIC,
                "test/mae_denormalized",
            ],
            "datasets": ["dataset_a"],
            "models": ["model_a", "model_b"],
        }
    }
    dash = _make_dashboard(monkeypatch, loader_map)
    dash.show_rank = True
    dash.selected_model = "model_b"

    dash._save_named_view(
        "Prognostics overview",
        saved_at="2026-07-26T12:34:56Z",
    )

    restored = _make_dashboard(monkeypatch, loader_map)
    views = restored._parse_named_views_json(dash._serialize_named_views_json())
    assert views is not None
    assert views[0]["name"] == "Prognostics overview"
    assert views[0]["saved_at"] == "2026-07-26T12:34:56Z"
    assert (
        restored._format_saved_view_date(views[0]["saved_at"]) == "2026-07-26 12:34 UTC"
    )

    restored._named_views = views
    assert restored._recall_named_view("Prognostics overview") is True
    assert restored.show_rank is True
    assert restored.selected_model == "model_b"


def test_saving_same_named_view_replaces_and_delete_removes_it(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )
    dash.show_rank = False
    dash._save_named_view("Overview", saved_at="2026-07-25T10:00:00Z")
    dash.show_rank = True

    dash._save_named_view("Overview", saved_at="2026-07-26T10:00:00Z")

    assert len(dash._named_views) == 1
    assert dash._named_views[0]["saved_at"] == "2026-07-26T10:00:00Z"
    assert dash._named_views[0]["state"]["show_rank"] is True
    assert dash._delete_named_view("Overview") is True
    assert dash._named_views == []
    assert dash._delete_named_view("Overview") is False


def test_enabled_models_filter_focus_selector_and_restore(monkeypatch):
    loader_map = {
        "report_a": {
            "metric_keys": [
                dashboard_app._DEFAULT_METRIC,
                "test/mae_denormalized",
            ],
            "datasets": ["dataset_a"],
            "models": ["model_a", "model_b", "model_c"],
        }
    }
    dash = _make_dashboard(monkeypatch, loader_map)

    assert dash.enabled_models == ["model_a", "model_b", "model_c"]
    dash.enabled_models = ["model_b"]
    assert list(dash.param["selected_model"].objects) == [
        dashboard_app._ALL_MODELS_OPTION,
        "model_b",
    ]
    dash.selected_model = "model_b"

    restored = _make_dashboard(monkeypatch, loader_map)
    restored._apply_saved_state(dash._serialize_state())

    assert restored.enabled_models == ["model_b"]
    assert restored.selected_model == "model_b"
    assert list(restored.param["selected_model"].objects) == [
        dashboard_app._ALL_MODELS_OPTION,
        "model_b",
    ]


def test_dashboard_initializes_alt_selectors_from_derived_metric(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    assert dash.alt_metric_key == "test/mae_denormalized"
    assert dash.alt_sort_metric_key == "test/mae_denormalized"
    assert dash.alt_metric_overridden is False
    assert dash.alt_sort_metric_overridden is False


def test_metric_availability_stylesheet_marks_unavailable_options_gray():
    css = dashboard_app._metric_availability_stylesheet({"val/mse_denormalized_mean"})

    assert 'option[value="val/mse_denormalized_mean"]' in css
    assert dashboard_app._UNAVAILABLE_METRIC_COLOR in css


def test_unavailable_metric_values_follow_active_model_family(monkeypatch):
    dataset = "dataset_a"
    primary_model = "model_a"
    alt_model = "model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper"
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                    "val/mse_denormalized",
                    "val/mse_denormalized_mean",
                ],
                "datasets": [dataset],
                "models": [primary_model, alt_model],
                "hp_map": {
                    (dataset, primary_model): _make_hp_dataset(
                        ["val/mse_denormalized", "val/mse_denormalized_mean"]
                    ),
                    (dataset, alt_model): _make_hp_dataset(["val/mse_denormalized"]),
                },
            }
        },
    )

    dash.selected_dataset = dataset
    dash.selected_model = alt_model

    alt_unavailable = dash._unavailable_metric_values(
        alt=True, selector_name="alt_metric_key"
    )
    primary_unavailable = dash._unavailable_metric_values(
        alt=False, selector_name="metric_key"
    )

    assert "val/mse_denormalized_mean" in alt_unavailable
    assert "val/mse_denormalized" not in alt_unavailable
    assert "val/mse_denormalized_mean" not in primary_unavailable


def test_sync_selectors_preserves_valid_values_and_falls_back(monkeypatch):
    loader_map = {
        "report_a": {
            "metric_keys": [
                dashboard_app._DEFAULT_METRIC,
                "test/mae_denormalized",
                "custom_metric",
                "custom_alt",
                "sort_metric",
                "sort_alt",
            ],
            "datasets": ["dataset_a", "dataset_b"],
            "models": ["model_a", "model_b"],
        },
        "report_preserve": {
            "metric_keys": [
                dashboard_app._DEFAULT_METRIC,
                "test/mae_denormalized",
                "custom_metric",
                "custom_alt",
                "sort_metric",
                "sort_alt",
            ],
            "datasets": ["dataset_a", "dataset_b", "dataset_c"],
            "models": ["model_a", "model_b", "model_c"],
        },
        "report_fallback": {
            "metric_keys": [
                dashboard_app._DEFAULT_METRIC,
                "test/mae_denormalized",
            ],
            "datasets": ["fallback_dataset"],
            "models": ["fallback_model"],
        },
    }
    dash = _make_dashboard(monkeypatch, loader_map)

    dash.metric_key = "custom_metric"
    dash.sort_metric_key = "sort_metric"
    dash.alt_metric_key = "custom_alt"
    dash.alt_sort_metric_key = "sort_alt"
    dash.selected_dataset = "dataset_b"
    dash.selected_model = "model_b"

    dash._loader = dashboard_app.ResultsLoader("report_preserve")
    dash._sync_selectors()

    assert dash.metric_key == "custom_metric"
    assert dash.sort_metric_key == "sort_metric"
    assert dash.alt_metric_key == "custom_alt"
    assert dash.alt_sort_metric_key == "sort_alt"
    assert dash.selected_dataset == "dataset_b"
    assert dash.selected_model == "model_b"

    dash.report_dir = "report_fallback"

    assert dash.metric_key == dashboard_app._DEFAULT_METRIC
    assert dash.sort_metric_key == dashboard_app._DEFAULT_METRIC
    assert dash.alt_metric_key == "test/mae_denormalized"
    assert dash.alt_sort_metric_key == "test/mae_denormalized"
    assert dash.selected_dataset == "fallback_dataset"
    assert dash.selected_model == dashboard_app._ALL_MODELS_OPTION


def test_metric_change_reselects_dataset_to_same_base_valid_dataset(monkeypatch):
    invalid_dataset = (
        "MultiSource_concepts_N-CMAPSS [16_03_2026_concepts_n_cmapss_prognostics_]"
    )
    same_base_valid_dataset = (
        "MultiSource_concepts_N-CMAPSS "
        "[16_03_2026_concepts_n_cmapss_multi_diagnostics_]"
    )
    other_valid_dataset = "HSF15 [16_03_2026_hsf15_pump_diagnostics_]"
    model = "model_a"
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/f1",
                    "val/f1",
                ],
                "datasets": [
                    invalid_dataset,
                    same_base_valid_dataset,
                    other_valid_dataset,
                ],
                "models": [model],
                "hp_map": {
                    (invalid_dataset, model): _make_hp_dataset(
                        [dashboard_app._DEFAULT_METRIC]
                    ),
                    (same_base_valid_dataset, model): _make_hp_dataset(
                        ["test/f1", "val/f1"]
                    ),
                    (other_valid_dataset, model): _make_hp_dataset(["test/f1"]),
                },
            }
        },
    )

    dash.selected_dataset = invalid_dataset
    dash.metric_key = "test/f1"

    assert dash.selected_dataset == same_base_valid_dataset
    assert list(dash.param["selected_dataset"].objects) == [
        same_base_valid_dataset,
        other_valid_dataset,
    ]
    unavailable = dash._unavailable_metric_values(
        alt=False, selector_name="sort_metric_key"
    )
    assert "val/f1" not in unavailable
    assert dashboard_app._DEFAULT_METRIC in unavailable


def test_metric_change_keeps_current_dataset_when_it_remains_valid(monkeypatch):
    dataset_a = "dataset_a"
    dataset_b = "dataset_b"
    model = "model_a"
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [dashboard_app._DEFAULT_METRIC, "test/f1"],
                "datasets": [dataset_a, dataset_b],
                "models": [model],
                "hp_map": {
                    (dataset_a, model): _make_hp_dataset(
                        [dashboard_app._DEFAULT_METRIC, "test/f1"]
                    ),
                    (dataset_b, model): _make_hp_dataset(["test/f1"]),
                },
            }
        },
    )

    dash.selected_dataset = dataset_a
    dash.metric_key = "test/f1"

    assert dash.selected_dataset == dataset_a


def test_metric_change_falls_back_to_first_valid_dataset_without_same_base_match(
    monkeypatch,
):
    invalid_dataset = "dataset_z [project_prog]"
    first_valid_dataset = "dataset_a [project_diag]"
    second_valid_dataset = "dataset_b [project_diag]"
    model = "model_a"
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [dashboard_app._DEFAULT_METRIC, "test/f1"],
                "datasets": [
                    invalid_dataset,
                    first_valid_dataset,
                    second_valid_dataset,
                ],
                "models": [model],
                "hp_map": {
                    (invalid_dataset, model): _make_hp_dataset(
                        [dashboard_app._DEFAULT_METRIC]
                    ),
                    (first_valid_dataset, model): _make_hp_dataset(["test/f1"]),
                    (second_valid_dataset, model): _make_hp_dataset(["test/f1"]),
                },
            }
        },
    )

    dash.selected_dataset = invalid_dataset
    dash.metric_key = "test/f1"

    assert dash.selected_dataset == first_valid_dataset


def test_sort_metric_change_derives_alt_sort_metric(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                    "custom_metric",
                    "custom_alt",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    dash.sort_metric_key = "custom_metric"
    dash.alt_sort_metric_key = "custom_alt"
    dash.sort_metric_key = dashboard_app._DEFAULT_METRIC

    assert dash.alt_sort_metric_key == "test/mae_denormalized"


def test_val_best_rerun_metrics_derive_val_alternatives(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                    "val_best_rerun/loss",
                    "val/loss",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    dash.metric_key = "val_best_rerun/loss"
    dash.sort_metric_key = "val_best_rerun/loss"

    assert dash.alt_metric_key == "val/loss"
    assert dash.alt_sort_metric_key == "val/loss"


def test_plain_metrics_keep_alt_selectors_in_sync(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                    "val_best_rerun/loss",
                    "val/loss",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    dash.metric_key = "val/loss"
    dash.sort_metric_key = "val/loss"

    assert dash.alt_metric_key == "val/loss"
    assert dash.alt_sort_metric_key == "val/loss"

    dash.metric_key = "test/mae_denormalized"
    dash.sort_metric_key = "test/mae_denormalized"

    assert dash.alt_metric_key == "test/mae_denormalized"
    assert dash.alt_sort_metric_key == "test/mae_denormalized"


def test_apply_saved_state_restores_report_dir_before_selectors(monkeypatch):
    loader_map = {
        "report_a": {
            "metric_keys": [
                dashboard_app._DEFAULT_METRIC,
                "test/mae_denormalized",
            ],
            "datasets": ["dataset_a"],
            "models": ["model_a"],
        },
        "report_b": {
            "metric_keys": [
                dashboard_app._DEFAULT_METRIC,
                "test/mae_denormalized",
                "target_metric",
                "target_alt_metric",
                "target_sort_metric",
                "target_alt_sort_metric",
            ],
            "datasets": ["target_dataset"],
            "models": ["target_model"],
        },
    }
    dash = _make_dashboard(monkeypatch, loader_map)

    dash._apply_saved_state(
        {
            "report_dir": "report_b",
            "metric_key": "target_metric",
            "sort_metric_key": "target_sort_metric",
            "mode": "max",
            "use_alt_metric": False,
            "alt_metric_key": "target_alt_metric",
            "alt_sort_metric_key": "target_alt_sort_metric",
            "alt_metric_overridden": True,
            "alt_sort_metric_overridden": True,
            "show_n": False,
            "show_rank": True,
            "spiderweb_plot_mode": "interactive",
            "selected_dataset": "target_dataset",
            "selected_model": "target_model",
            "active_tab": 2,
        }
    )

    assert dash.report_dir == "report_b"
    assert dash.metric_key == "target_metric"
    assert dash.sort_metric_key == "target_sort_metric"
    assert dash.mode == "max"
    assert dash.use_alt_metric is False
    assert dash.alt_metric_key == "target_alt_metric"
    assert dash.alt_sort_metric_key == "target_alt_sort_metric"
    assert dash.alt_metric_overridden is True
    assert dash.alt_sort_metric_overridden is True
    assert dash.show_n is False
    assert dash.show_rank is True
    assert dash.spiderweb_plot_mode == "interactive"
    assert dash.selected_dataset == "target_dataset"
    assert dash.selected_model == "target_model"
    assert dash.active_tab == 2


def test_apply_saved_state_ignores_stale_alt_metrics_without_override_flags(
    monkeypatch,
):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                    "val_best_rerun/mae_denormalized_mean",
                    "val/mae_denormalized",
                    "val/mae_denormalized_mean",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    dash._apply_saved_state(
        {
            "metric_key": "val_best_rerun/mae_denormalized_mean",
            "sort_metric_key": "val_best_rerun/mae_denormalized_mean",
            "alt_metric_key": "val/mae_denormalized",
            "alt_sort_metric_key": "val/mae_denormalized",
        }
    )

    assert dash.metric_key == "val_best_rerun/mae_denormalized_mean"
    assert dash.sort_metric_key == "val_best_rerun/mae_denormalized_mean"
    assert dash.alt_metric_key == "val/mae_denormalized_mean"
    assert dash.alt_sort_metric_key == "val/mae_denormalized_mean"
    assert dash.alt_metric_overridden is False
    assert dash.alt_sort_metric_overridden is False


def test_apply_saved_state_restores_explicit_alt_metric_overrides(monkeypatch):
    loader_map = {
        "report_a": {
            "metric_keys": [
                dashboard_app._DEFAULT_METRIC,
                "test/mae_denormalized",
                "custom_metric",
                "custom_alt",
                "sort_metric",
                "sort_alt",
            ],
            "datasets": ["dataset_a"],
            "models": ["model_a"],
        }
    }
    dash = _make_dashboard(monkeypatch, loader_map)
    dash.metric_key = "custom_metric"
    dash.sort_metric_key = "sort_metric"
    dash.alt_metric_key = "custom_alt"
    dash.alt_sort_metric_key = "sort_alt"
    dash.spiderweb_plot_mode = "interactive"

    state = dash._serialize_state()

    restored = _make_dashboard(monkeypatch, loader_map)
    restored._apply_saved_state(state)

    assert restored.metric_key == "custom_metric"
    assert restored.sort_metric_key == "sort_metric"
    assert restored.alt_metric_key == "custom_alt"
    assert restored.alt_sort_metric_key == "sort_alt"
    assert restored.spiderweb_plot_mode == "interactive"
    assert restored.alt_metric_overridden is True
    assert restored.alt_sort_metric_overridden is True


def test_apply_saved_state_reselects_invalid_dataset_to_same_base_valid_dataset(
    monkeypatch,
):
    invalid_dataset = (
        "MultiSource_concepts_N-CMAPSS [16_03_2026_concepts_n_cmapss_prognostics_]"
    )
    same_base_valid_dataset = (
        "MultiSource_concepts_N-CMAPSS "
        "[16_03_2026_concepts_n_cmapss_multi_diagnostics_]"
    )
    other_valid_dataset = "HSF15 [16_03_2026_hsf15_pump_diagnostics_]"
    model = "model_a"
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [dashboard_app._DEFAULT_METRIC, "test/f1"],
                "datasets": [
                    invalid_dataset,
                    same_base_valid_dataset,
                    other_valid_dataset,
                ],
                "models": [model],
                "hp_map": {
                    (invalid_dataset, model): _make_hp_dataset(
                        [dashboard_app._DEFAULT_METRIC]
                    ),
                    (same_base_valid_dataset, model): _make_hp_dataset(["test/f1"]),
                    (other_valid_dataset, model): _make_hp_dataset(["test/f1"]),
                },
            }
        },
    )
    dash.selected_dataset = other_valid_dataset

    dash._apply_saved_state(
        {
            "metric_key": "test/f1",
            "sort_metric_key": "test/f1",
            "selected_dataset": invalid_dataset,
        }
    )

    assert dash.selected_dataset == same_base_valid_dataset


def test_use_alt_metric_change_reselects_dataset_for_alt_display_metric(monkeypatch):
    dataset_primary = "dataset_primary [project_prog]"
    dataset_alt = "dataset_alt [project_diag]"
    alt_model = "model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper"
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    "test/custom_primary",
                    "test/custom_alt",
                ],
                "datasets": [dataset_primary, dataset_alt],
                "models": [alt_model],
                "hp_map": {
                    (dataset_primary, alt_model): _make_hp_dataset(
                        ["test/custom_primary"]
                    ),
                    (dataset_alt, alt_model): _make_hp_dataset(["test/custom_alt"]),
                },
            }
        },
    )

    dash.use_alt_metric = False
    dash.selected_dataset = dataset_primary
    dash.alt_metric_key = "test/custom_alt"
    dash.use_alt_metric = True

    assert dash.selected_dataset == dataset_alt


def test_parse_saved_state_migrates_v2_active_tab_indices(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    migrated = dash._parse_saved_state('{"version": 2, "active_tab": 4}')
    unmigrated = dash._parse_saved_state('{"version": 2, "active_tab": 0}')

    assert migrated == {"active_tab": 6}
    assert unmigrated == {"active_tab": 0}


def test_parse_saved_state_migrates_v4_active_tab_indices(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    migrated = dash._parse_saved_state('{"version": 4, "active_tab": 4}')
    untouched = dash._parse_saved_state('{"version": 4, "active_tab": 1}')

    assert migrated == {"active_tab": 5}
    assert untouched == {"active_tab": 1}


def test_parse_saved_state_accepts_v5_payload(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    parsed = dash._parse_saved_state('{"version": 5, "active_tab": 7}')

    assert parsed == {"active_tab": 7}


def test_latex_multiplier_persistence_roundtrip(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    dash.latex_multiplier = 100.0
    dash.latex_shorten_names = False
    state = dash._serialize_state()
    assert state["latex_multiplier"] == 100.0
    assert state["latex_shorten_names"] is False

    restored = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )
    parsed = restored._parse_saved_state(dash._serialize_state_json())
    assert parsed is not None
    restored._apply_saved_state(parsed)
    assert restored.latex_multiplier == 100.0
    assert restored.latex_shorten_names is False


def test_parse_saved_state_v6_payload_omits_latex_multiplier(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    # A v6 payload predates latex_multiplier — it must still parse, and applying
    # it must leave the multiplier at its default (1.0).
    parsed = dash._parse_saved_state('{"version": 6, "active_tab": 7}')
    assert parsed == {"active_tab": 7}

    dash.latex_multiplier = 1.0
    dash._apply_saved_state(parsed)
    assert dash.latex_multiplier == 1.0


def test_parse_saved_state_rejects_bool_latex_multiplier(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    parsed = dash._parse_saved_state('{"version": 7, "latex_multiplier": true}')
    assert "latex_multiplier" not in parsed


def test_url_tab_override_takes_precedence_over_saved_state(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )
    location = Location()
    location.search = "?tab=4"
    monkeypatch.setattr(pn.state, "_location", location, raising=False)
    location.sync(dash, {"active_tab": "tab"})

    assert dash.active_tab == 4

    dash._apply_saved_state({"active_tab": 1})

    assert dash.active_tab == 4


def test_panel_can_render_with_state_bridge(monkeypatch):
    for builder_name in (
        "build_heatmap",
        "build_parallel_coordinates",
        "build_spiderweb",
        "build_model_summary_table",
        "build_bar_chart",
        "build_metadata_panel",
        "build_hp_impact_table",
        "build_summary_table",
        "build_stats_table",
        "build_latex_table",
    ):
        monkeypatch.setattr(
            dashboard_app, builder_name, lambda *args, **kwargs: pn.pane.Markdown("ok")
        )

    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    template = dash.panel()

    doc = template.server_doc(title="Dashboard Test")

    assert doc.title == "piCID Experiment Dashboard"
    assert doc.roots


def test_panel_includes_parallel_coordinates_tab_after_heatmap(monkeypatch):
    for builder_name in (
        "build_heatmap",
        "build_parallel_coordinates",
        "build_spiderweb",
        "build_model_summary_table",
        "build_bar_chart",
        "build_metadata_panel",
        "build_hp_impact_table",
        "build_summary_table",
        "build_stats_table",
        "build_latex_table",
    ):
        monkeypatch.setattr(
            dashboard_app, builder_name, lambda *args, **kwargs: pn.pane.Markdown("ok")
        )

    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )

    template = dash.panel()
    tabs = template.main[0]

    assert tabs._names[:4] == [
        "Heatmap",
        "Parallel Coordinates",
        "Spiderweb",
        "Bar Chart",
    ]


def test_show_hp_impact_for_model_selects_model_and_tab(monkeypatch):
    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a", "model_b"],
            }
        },
    )

    dash.selected_model = dashboard_app._ALL_MODELS_OPTION
    dash.active_tab = 0

    dash._show_hp_impact_for_model("model_b")

    assert dash.selected_model == "model_b"
    assert dash.active_tab == dashboard_app._HP_IMPACT_TAB_INDEX


def test_parallel_coordinates_view_includes_verification_model_summary(monkeypatch):
    parallel_call: dict[str, object] = {}
    model_summary_call: dict[str, object] = {}

    def _fake_parallel_coordinates(*args, **kwargs):
        parallel_call["args"] = args
        parallel_call["kwargs"] = kwargs
        return pn.pane.Markdown("parallel")

    def _fake_model_summary_table(*args, **kwargs):
        model_summary_call["args"] = args
        model_summary_call["kwargs"] = kwargs
        return pn.pane.Markdown("model summary")

    monkeypatch.setattr(
        dashboard_app, "build_parallel_coordinates", _fake_parallel_coordinates
    )
    monkeypatch.setattr(
        dashboard_app, "build_model_summary_table", _fake_model_summary_table
    )

    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )
    dash.selected_dataset = "dataset_a"
    dash.metric_key = dashboard_app._DEFAULT_METRIC
    dash.sort_metric_key = dashboard_app._DEFAULT_METRIC
    dash.alt_metric_key = "test/mae_denormalized"
    dash.alt_sort_metric_key = "test/mae_denormalized"

    view = dash._parallel_coordinates_view()

    assert isinstance(view, pn.Column)
    assert view.objects[0].object == "parallel"
    assert view.objects[2].object == "### View Configuration Summary for dataset_a"
    assert view.objects[3].object == "model summary"
    assert parallel_call["args"][1] == dashboard_app._DEFAULT_METRIC
    assert parallel_call["kwargs"] == {
        "sort_metric_key": dashboard_app._DEFAULT_METRIC,
        "mode": "min",
        "alt_metric_key": "test/mae_denormalized",
        "alt_sort_metric_key": "test/mae_denormalized",
        "use_alt_metric": True,
        "enabled_datasets": ["dataset_a"],
        "enabled_models": ["model_a"],
    }
    assert model_summary_call["kwargs"]["metric_key"] == dashboard_app._DEFAULT_METRIC
    assert (
        model_summary_call["kwargs"]["sort_metric_key"] == dashboard_app._DEFAULT_METRIC
    )
    assert model_summary_call["kwargs"]["alt_metric_key"] == "test/mae_denormalized"
    assert (
        model_summary_call["kwargs"]["alt_sort_metric_key"] == "test/mae_denormalized"
    )
    assert model_summary_call["kwargs"]["enabled_models"] == ["model_a"]


def test_spiderweb_view_includes_verification_model_summary(monkeypatch):
    spider_call: dict[str, object] = {}
    model_summary_call: dict[str, object] = {}

    def _fake_spiderweb(*args, **kwargs):
        spider_call["args"] = args
        spider_call["kwargs"] = kwargs
        return pn.pane.Markdown("spider")

    def _fake_model_summary_table(*args, **kwargs):
        model_summary_call["args"] = args
        model_summary_call["kwargs"] = kwargs
        return pn.pane.Markdown("model summary")

    monkeypatch.setattr(dashboard_app, "build_spiderweb", _fake_spiderweb)
    monkeypatch.setattr(
        dashboard_app, "build_model_summary_table", _fake_model_summary_table
    )

    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )
    dash.selected_dataset = "dataset_a"
    dash.metric_key = dashboard_app._DEFAULT_METRIC
    dash.sort_metric_key = dashboard_app._DEFAULT_METRIC
    dash.alt_metric_key = "test/mae_denormalized"
    dash.alt_sort_metric_key = "test/mae_denormalized"
    dash.spiderweb_plot_mode = "interactive"

    view = dash._spiderweb_view()

    assert isinstance(view, pn.Column)
    assert view.objects[0].object == "spider"
    assert view.objects[2].object == "### View Configuration Summary for dataset_a"
    assert view.objects[3].object == "model summary"
    assert spider_call["args"][1] == dashboard_app._DEFAULT_METRIC
    assert spider_call["kwargs"] == {
        "sort_metric_key": dashboard_app._DEFAULT_METRIC,
        "mode": "min",
        "alt_metric_key": "test/mae_denormalized",
        "alt_sort_metric_key": "test/mae_denormalized",
        "use_alt_metric": True,
        "selected_dataset": "dataset_a",
        "plot_mode": "interactive",
        "enabled_datasets": ["dataset_a"],
        "enabled_models": ["model_a"],
    }
    assert model_summary_call["kwargs"]["metric_key"] == dashboard_app._DEFAULT_METRIC
    assert (
        model_summary_call["kwargs"]["sort_metric_key"] == dashboard_app._DEFAULT_METRIC
    )
    assert model_summary_call["kwargs"]["alt_metric_key"] == "test/mae_denormalized"
    assert (
        model_summary_call["kwargs"]["alt_sort_metric_key"] == "test/mae_denormalized"
    )
    assert model_summary_call["kwargs"]["enabled_models"] == ["model_a"]


def test_spiderweb_view_reacts_to_plot_mode_changes():
    assert (
        "spiderweb_plot_mode"
        in dashboard_app.PiCIDDashboard._spiderweb_view._dinfo["dependencies"]
    )
    assert (
        "spiderweb_plot_mode"
        not in dashboard_app.PiCIDDashboard._parallel_coordinates_view._dinfo[
            "dependencies"
        ]
    )


def test_hp_impact_view_includes_title_for_selected_dataset_and_model(monkeypatch):
    monkeypatch.setattr(
        dashboard_app,
        "build_metadata_panel",
        lambda *args, **kwargs: pn.pane.Markdown("meta"),
    )
    monkeypatch.setattr(
        dashboard_app,
        "build_hp_impact_table",
        lambda *args, **kwargs: pn.pane.Markdown("table"),
    )

    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )
    dash.selected_dataset = "dataset_a"
    dash.selected_model = "model_a"

    view = dash._hp_impact_view()

    assert isinstance(view, pn.Column)
    assert view.objects[0].object == "### HP Impact for model_a on dataset_a"


def test_summary_view_includes_verification_model_summary(monkeypatch):
    summary_call: dict[str, object] = {}
    model_summary_call: dict[str, object] = {}

    def _fake_summary_table(*args, **kwargs):
        summary_call["args"] = args
        summary_call["kwargs"] = kwargs
        return pn.pane.Markdown("summary")

    def _fake_model_summary_table(*args, **kwargs):
        model_summary_call["args"] = args
        model_summary_call["kwargs"] = kwargs
        return pn.pane.Markdown("model summary")

    monkeypatch.setattr(dashboard_app, "build_summary_table", _fake_summary_table)
    monkeypatch.setattr(
        dashboard_app, "build_model_summary_table", _fake_model_summary_table
    )

    dash = _make_dashboard(
        monkeypatch,
        {
            "report_a": {
                "metric_keys": [
                    dashboard_app._DEFAULT_METRIC,
                    "test/mae_denormalized",
                ],
                "datasets": ["dataset_a"],
                "models": ["model_a"],
            }
        },
    )
    dash.selected_dataset = "dataset_a"
    dash.metric_key = dashboard_app._DEFAULT_METRIC
    dash.sort_metric_key = dashboard_app._DEFAULT_METRIC
    dash.alt_metric_key = "test/mae_denormalized"
    dash.alt_sort_metric_key = "test/mae_denormalized"

    view = dash._summary_view()

    assert isinstance(view, pn.Column)
    assert (
        view.objects[0].object
        == f"### Summary Table for {dashboard_app._DEFAULT_METRIC} on dataset_a"
    )
    assert view.objects[1].object == "summary"
    assert view.objects[3].object == "### View Configuration Summary for dataset_a"
    assert view.objects[4].object == "model summary"
    assert summary_call["args"][1:] == (
        "dataset_a",
        None,
        dashboard_app._DEFAULT_METRIC,
    )
    assert summary_call["kwargs"] == {
        "alt_metric_key": "test/mae_denormalized",
        "use_alt_metric": True,
        "enabled_models": ["model_a"],
        "sort_metric_key": dashboard_app._DEFAULT_METRIC,
        "alt_sort_metric_key": "test/mae_denormalized",
        "mode": "min",
    }
    assert model_summary_call["kwargs"]["metric_key"] == dashboard_app._DEFAULT_METRIC
    assert (
        model_summary_call["kwargs"]["sort_metric_key"] == dashboard_app._DEFAULT_METRIC
    )
    assert model_summary_call["kwargs"]["alt_metric_key"] == "test/mae_denormalized"
    assert (
        model_summary_call["kwargs"]["alt_sort_metric_key"] == "test/mae_denormalized"
    )
    assert model_summary_call["kwargs"]["enabled_models"] == ["model_a"]


class _HpStateResultsLoader(DataResultsLoader):
    DATASET = "Dataset A [project]"
    MODEL = "model_a"

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self._expanded_datasets = {}
        self._hp_ds_cache = {}
        self._applied_model_aliases = []
        coords = {
            "dataset": [self.DATASET],
            "model": [self.MODEL],
            "metric_key": [dashboard_app._DEFAULT_METRIC],
        }
        metadata_coords = {
            "dataset": [self.DATASET],
            "model": [self.MODEL],
        }
        self._xarray_dataset = xr.Dataset(
            {
                "mean": xr.DataArray(
                    [[[1.0]]],
                    dims=("dataset", "model", "metric_key"),
                    coords=coords,
                ),
                "std": xr.DataArray(
                    [[[0.1]]],
                    dims=("dataset", "model", "metric_key"),
                    coords=coords,
                ),
                "n": xr.DataArray(
                    [[[5.0]]],
                    dims=("dataset", "model", "metric_key"),
                    coords=coords,
                ),
                "sort_metric": xr.DataArray(
                    [[dashboard_app._DEFAULT_METRIC]],
                    dims=("dataset", "model"),
                    coords=metadata_coords,
                ),
                "opt_metric": xr.DataArray(
                    [[dashboard_app._DEFAULT_METRIC]],
                    dims=("dataset", "model"),
                    coords=metadata_coords,
                ),
                "opt_mode": xr.DataArray(
                    [["min"]],
                    dims=("dataset", "model"),
                    coords=metadata_coords,
                ),
            }
        )
        self._hp_map = {
            (self.DATASET, self.MODEL): xr.Dataset(
                {
                    "mean": xr.DataArray(
                        [[2.0], [1.0]],
                        dims=("config", "metric"),
                    ),
                    "std": xr.DataArray(
                        [[0.2], [0.1]],
                        dims=("config", "metric"),
                    ),
                    "count": xr.DataArray(
                        [[5.0], [5.0]],
                        dims=("config", "metric"),
                    ),
                },
                coords={
                    "config": [0, 1],
                    "metric": [dashboard_app._DEFAULT_METRIC],
                    "task_definition.seq_len": ("config", [10, 50]),
                },
            )
        }

    @property
    def xarray_dataset(self) -> xr.Dataset:
        return self._xarray_dataset

    def hp_impact_ds(self, dataset: str, model: str) -> xr.Dataset | None:
        return self._hp_map.get((self.source_dataset(dataset), model))


def test_hp_comparison_state_roundtrip_restores_virtual_dataset(monkeypatch) -> None:
    monkeypatch.setattr(dashboard_app, "ResultsLoader", _HpStateResultsLoader)
    dash = dashboard_app.PiCIDDashboard(report_dir="report_a")
    labels = dash._apply_hp_comparison(
        _HpStateResultsLoader.DATASET,
        ["task_definition.seq_len"],
    )
    state = dash._parse_saved_state(dash._serialize_state_json())

    restored = dashboard_app.PiCIDDashboard(report_dir="report_a")
    assert state is not None
    restored._apply_saved_state(state)

    assert restored.hp_comparison_source == _HpStateResultsLoader.DATASET
    assert restored.hp_comparison_dimensions == ["task_definition.seq_len"]
    assert restored.enabled_datasets == labels
    assert restored.selected_dataset == labels[0]
