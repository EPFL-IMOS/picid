"""Tests for dashboard state persistence helpers."""

from __future__ import annotations

import pandas as pd
import panel as pn

from panel.io.location import Location

from picid_report_dashboard import app as dashboard_app


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

    return FakeResultsLoader


def _make_dashboard(monkeypatch, loader_map, report_dir: str = "report_a"):
    monkeypatch.setattr(dashboard_app, "ResultsLoader", _loader_factory(loader_map))
    return dashboard_app.PiCIDDashboard(report_dir=report_dir)


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
    dash.mode = "max"
    dash.use_alt_metric = False
    dash.alt_metric_key = "test/mae_denormalized"
    dash.show_n = False
    dash.show_rank = True
    dash.selected_dataset = "dataset_b"
    dash.selected_model = "model_b"
    dash.active_tab = 3

    state = dash._serialize_state()

    assert set(state) == {"version", *dashboard_app._PERSISTED_STATE_PARAMS}
    assert state["version"] == dashboard_app._PERSISTED_STATE_VERSION
    assert state["metric_key"] == "custom_metric"
    assert state["selected_model"] == "model_b"
    assert state["active_tab"] == 3


def test_sync_selectors_preserves_valid_values_and_falls_back(monkeypatch):
    loader_map = {
        "report_a": {
            "metric_keys": [
                dashboard_app._DEFAULT_METRIC,
                "test/mae_denormalized",
                "custom_metric",
                "custom_alt",
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
    dash.alt_metric_key = "custom_alt"
    dash.selected_dataset = "dataset_b"
    dash.selected_model = "model_b"

    dash._loader = dashboard_app.ResultsLoader("report_preserve")
    dash._sync_selectors()

    assert dash.metric_key == "custom_metric"
    assert dash.alt_metric_key == "custom_alt"
    assert dash.selected_dataset == "dataset_b"
    assert dash.selected_model == "model_b"

    dash._loader = dashboard_app.ResultsLoader("report_fallback")
    dash._sync_selectors()

    assert dash.metric_key == dashboard_app._DEFAULT_METRIC
    assert dash.alt_metric_key == "test/mae_denormalized"
    assert dash.selected_dataset == "fallback_dataset"
    assert dash.selected_model == dashboard_app._ALL_MODELS_OPTION


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
            "mode": "max",
            "use_alt_metric": False,
            "alt_metric_key": "target_alt_metric",
            "show_n": False,
            "show_rank": True,
            "selected_dataset": "target_dataset",
            "selected_model": "target_model",
            "active_tab": 2,
        }
    )

    assert dash.report_dir == "report_b"
    assert dash.metric_key == "target_metric"
    assert dash.mode == "max"
    assert dash.use_alt_metric is False
    assert dash.alt_metric_key == "target_alt_metric"
    assert dash.show_n is False
    assert dash.show_rank is True
    assert dash.selected_dataset == "target_dataset"
    assert dash.selected_model == "target_model"
    assert dash.active_tab == 2


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
        "build_bar_chart",
        "build_metadata_panel",
        "build_hp_impact_table",
        "build_summary_table",
        "build_stats_table",
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
