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
    dash.sort_metric_key = "custom_metric"
    dash.mode = "max"
    dash.use_alt_metric = False
    dash.alt_metric_key = "test/mae_denormalized"
    dash.alt_sort_metric_key = "test/mae_denormalized"
    dash.show_n = False
    dash.show_rank = True
    dash.selected_dataset = "dataset_b"
    dash.selected_model = "model_b"
    dash.active_tab = 3

    state = dash._serialize_state()

    assert set(state) == {"version", *dashboard_app._PERSISTED_STATE_PARAMS}
    assert state["version"] == dashboard_app._PERSISTED_STATE_VERSION
    assert state["metric_key"] == "custom_metric"
    assert state["sort_metric_key"] == "custom_metric"
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

    dash._loader = dashboard_app.ResultsLoader("report_fallback")
    dash._sync_selectors()

    assert dash.metric_key == dashboard_app._DEFAULT_METRIC
    assert dash.sort_metric_key == dashboard_app._DEFAULT_METRIC
    assert dash.alt_metric_key == "test/mae_denormalized"
    assert dash.alt_sort_metric_key == "test/mae_denormalized"
    assert dash.selected_dataset == "fallback_dataset"
    assert dash.selected_model == dashboard_app._ALL_MODELS_OPTION


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
            "show_n": False,
            "show_rank": True,
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
        "build_model_summary_table",
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
    assert dash.active_tab == 2


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
    }
    assert model_summary_call["kwargs"]["metric_key"] == dashboard_app._DEFAULT_METRIC
    assert (
        model_summary_call["kwargs"]["sort_metric_key"] == dashboard_app._DEFAULT_METRIC
    )
    assert model_summary_call["kwargs"]["alt_metric_key"] == "test/mae_denormalized"
    assert (
        model_summary_call["kwargs"]["alt_sort_metric_key"] == "test/mae_denormalized"
    )
