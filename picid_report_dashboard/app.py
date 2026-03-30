"""
piCID Results Dashboard

Entry point for Panel serve:

    panel serve picid_report_dashboard/app.py --show --args --report-dir report_output

Or run directly:

    python picid_report_dashboard/app.py --report-dir report_output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import panel as pn
import param

if __name__ == "__main__" or not __name__.startswith("picid_report_dashboard"):
    import pathlib

    _pkg = pathlib.Path(__file__).parent
    sys.path.insert(0, str(_pkg.parent))

from picid_report_dashboard.data import ResultsLoader
from picid_report_dashboard.views import (
    build_bar_chart,
    build_heatmap,
    build_hp_impact_table,
    build_metadata_panel,
    build_stats_table,
    build_summary_table,
)

pn.extension("tabulator", sizing_mode="stretch_width")

# ---------------------------------------------------------------------------
# Defaults — can be overridden via CLI args or URL query params
# ---------------------------------------------------------------------------

_DEFAULT_REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report_output")
_DEFAULT_METRIC = "test_best_rerun/mae_denormalized"
_PERSISTED_STATE_VERSION = 1
_STORAGE_PREFIX = "picid_dashboard_state"
_ALL_MODELS_OPTION = "— All models —"
_MAIN_TAB_TITLES = (
    "Heatmap",
    "Bar Chart",
    "HP Impact",
    "Summary Table",
    "Experiment Stats",
)
_PERSISTED_STATE_PARAMS = (
    "report_dir",
    "metric_key",
    "mode",
    "use_alt_metric",
    "alt_metric_key",
    "show_n",
    "show_rank",
    "selected_dataset",
    "selected_model",
    "active_tab",
)
_UNSET = object()


# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------


class PiCIDDashboard(param.Parameterized):
    """Reactive Panel dashboard backed by ResultsLoader."""

    report_dir = param.String(
        default=_DEFAULT_REPORT_DIR, doc="Path to report_output directory"
    )
    metric_key = param.Selector(
        default=_DEFAULT_METRIC, objects=[], doc="Metric to visualise"
    )
    mode = param.Selector(
        default="min", objects=["min", "max"], doc="Lower or higher is better"
    )
    use_alt_metric = param.Boolean(
        default=True, doc="Use separate metric for XGBoost/TabPFN/TabDPT"
    )
    alt_metric_key = param.Selector(
        default=_DEFAULT_METRIC,
        objects=[],
        doc="Alternate metric for fit-predict models",
    )
    show_n = param.Boolean(default=True, doc="Show n= run count in heatmap cells")
    show_rank = param.Boolean(
        default=False, doc="Show rank within dataset in heatmap cells"
    )
    selected_dataset = param.Selector(
        default=None, objects=[], doc="Dataset for bar chart / HP impact"
    )
    selected_model = param.Selector(
        default=None, objects=[], doc="Model for HP impact / metadata"
    )
    active_tab = param.Integer(default=0, doc="Active main tab index (synced to URL)")

    def __init__(self, report_dir: str = _DEFAULT_REPORT_DIR, **params):
        super().__init__(report_dir=report_dir, **params)
        self._loader: ResultsLoader | None = None
        self._load_error: str | None = None
        self._restore_in_progress = False
        self._state_store: pn.widgets.TextInput | None = None
        self._load_data()
        self._sync_selectors()
        self.param.watch(self._sync_state_store, list(_PERSISTED_STATE_PARAMS))
        self._sync_state_store()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        try:
            self._loader = ResultsLoader(self.report_dir)
            _ = self._loader.xarray_dataset  # trigger xarray load
            _ = self._loader.summary_df  # pre-warm CSV cache
            _ = self._loader.stats_df  # pre-warm CSV cache
            self._load_error = None
        except Exception as exc:
            self._loader = None
            self._load_error = str(exc)

    @param.depends("report_dir", watch=True)
    def _on_report_dir_change(self) -> None:
        self._load_data()
        self._sync_selectors()
        self.param.trigger("metric_key")  # force view refresh

    def _derive_alt_metric(self, metric: str) -> str | None:
        """If metric is under test_best_rerun/*, return the test/* equivalent if it exists."""
        if not metric.startswith("test_best_rerun/"):
            return None
        candidate = "test/" + metric[len("test_best_rerun/") :]
        available = self.param["alt_metric_key"].objects
        return candidate if candidate in available else None

    @param.depends("metric_key", watch=True)
    def _on_metric_key_change(self) -> None:
        derived = self._derive_alt_metric(self.metric_key)
        if derived:
            self.alt_metric_key = derived

    def _set_selector_value(
        self,
        name: str,
        options: list[Any],
        *,
        preferred: Any = _UNSET,
        fallback: Any = _UNSET,
    ) -> None:
        self.param[name].objects = options
        if preferred is not _UNSET and preferred in options:
            setattr(self, name, preferred)
            return
        if fallback is not _UNSET and fallback in options:
            setattr(self, name, fallback)
            return
        setattr(self, name, options[0] if options else None)

    def _sync_selectors(self) -> None:
        if self._loader is None:
            return
        loader = self._loader
        current_metric = self.metric_key
        current_alt_metric = self.alt_metric_key
        current_dataset = self.selected_dataset
        current_model = self.selected_model

        metrics = loader.metric_keys or [_DEFAULT_METRIC]
        self._set_selector_value(
            "metric_key",
            metrics,
            preferred=current_metric,
            fallback=_DEFAULT_METRIC,
        )
        derived = self._derive_alt_metric(self.metric_key)
        self._set_selector_value(
            "alt_metric_key",
            metrics,
            preferred=current_alt_metric,
            fallback=derived if derived is not None else _DEFAULT_METRIC,
        )

        datasets = loader.datasets or []
        self._set_selector_value(
            "selected_dataset",
            datasets,
            preferred=current_dataset,
        )

        models = loader.models or []
        self._set_selector_value(
            "selected_model",
            [_ALL_MODELS_OPTION, *models],
            preferred=current_model,
            fallback=_ALL_MODELS_OPTION,
        )

    def _serialize_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"version": _PERSISTED_STATE_VERSION}
        for name in _PERSISTED_STATE_PARAMS:
            value = getattr(self, name)
            if isinstance(value, bool) or value is None or isinstance(value, int):
                state[name] = value
            else:
                state[name] = str(value)
        return state

    def _serialize_state_json(self) -> str:
        return json.dumps(self._serialize_state(), sort_keys=True)

    def _parse_saved_state(self, state_json: str) -> dict[str, Any] | None:
        if not state_json:
            return None
        try:
            payload = json.loads(state_json)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        version = payload.get("version", _PERSISTED_STATE_VERSION)
        if version != _PERSISTED_STATE_VERSION:
            return None

        parsed: dict[str, Any] = {}
        string_fields = (
            "report_dir",
            "metric_key",
            "mode",
            "alt_metric_key",
            "selected_dataset",
            "selected_model",
        )
        bool_fields = ("use_alt_metric", "show_n", "show_rank")
        for field in string_fields:
            value = payload.get(field)
            if isinstance(value, str):
                parsed[field] = value
        for field in bool_fields:
            value = payload.get(field)
            if isinstance(value, bool):
                parsed[field] = value
        active_tab = payload.get("active_tab")
        if isinstance(active_tab, bool):
            active_tab = None
        elif isinstance(active_tab, str) and active_tab.isdigit():
            active_tab = int(active_tab)
        if isinstance(active_tab, int) and 0 <= active_tab < len(_MAIN_TAB_TITLES):
            parsed["active_tab"] = active_tab
        return parsed

    def _has_url_tab_override(self) -> bool:
        location = pn.state.location
        return location is not None and "tab" in location.query_params

    def _apply_selector_state(self, name: str, value: Any) -> None:
        if value in self.param[name].objects:
            setattr(self, name, value)

    def _apply_saved_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self._restore_in_progress = True
        try:
            report_dir = state.get("report_dir")
            if isinstance(report_dir, str) and report_dir != self.report_dir:
                self.report_dir = report_dir

            self._apply_selector_state("mode", state.get("mode"))

            for name in ("use_alt_metric", "show_n", "show_rank"):
                if name in state:
                    setattr(self, name, state[name])

            if self._loader is not None:
                for name in (
                    "metric_key",
                    "alt_metric_key",
                    "selected_dataset",
                    "selected_model",
                ):
                    self._apply_selector_state(name, state.get(name))

            if not self._has_url_tab_override() and "active_tab" in state:
                self.active_tab = state["active_tab"]
        finally:
            self._restore_in_progress = False
            self._sync_state_store()

    def _sync_state_store(self, *_events: Any) -> None:
        if self._restore_in_progress or self._state_store is None:
            return
        self._state_store.value = self._serialize_state_json()

    # ------------------------------------------------------------------
    # View methods (reactive)
    # ------------------------------------------------------------------

    def _active_alt_metric(self) -> str | None:
        return self.alt_metric_key if self.use_alt_metric else None

    @param.depends(
        "metric_key", "mode", "use_alt_metric", "alt_metric_key", "show_n", "show_rank"
    )
    def _heatmap_view(self):
        if self._load_error:
            return pn.pane.Alert(
                f"**Error loading data:** {self._load_error}", alert_type="danger"
            )
        return build_heatmap(
            self._loader,
            self.metric_key,
            self.mode,
            alt_metric_key=self._active_alt_metric(),
            show_n=self.show_n,
            show_rank=self.show_rank,
        )

    @param.depends(
        "metric_key", "mode", "selected_dataset", "use_alt_metric", "alt_metric_key"
    )
    def _bar_chart_view(self):
        if self._load_error or self.selected_dataset is None:
            return pn.pane.Markdown("Select a dataset.")
        return build_bar_chart(
            self._loader,
            self.metric_key,
            self.selected_dataset,
            self.mode,
            alt_metric_key=self._active_alt_metric(),
        )

    def _active_model(self) -> str | None:
        """Return None when the 'All models' sentinel is selected."""
        m = self.selected_model
        return None if (m is None or m == _ALL_MODELS_OPTION) else m

    @param.depends("selected_dataset", "selected_model")
    def _hp_impact_view(self):
        model = self._active_model()
        if self._load_error or self.selected_dataset is None or model is None:
            return pn.pane.Markdown("Select a dataset and model.")
        return pn.Column(
            build_metadata_panel(self._loader, self.selected_dataset, model),
            pn.layout.Divider(),
            build_hp_impact_table(self._loader, self.selected_dataset, model),
        )

    @param.depends("report_dir", "selected_dataset", "selected_model", "metric_key")
    def _summary_view(self):
        if self._load_error:
            return pn.pane.Alert(self._load_error, alert_type="danger")
        return build_summary_table(
            self._loader, self.selected_dataset, self._active_model(), self.metric_key
        )

    @param.depends("report_dir", "selected_dataset", "selected_model")
    def _stats_view(self):
        if self._load_error:
            return pn.pane.Alert(self._load_error, alert_type="danger")
        return build_stats_table(
            self._loader, self.selected_dataset, self._active_model()
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def panel(self) -> pn.viewable.Viewable:
        # --- Sidebar controls ---
        dir_input = pn.widgets.TextInput.from_param(
            self.param.report_dir, name="Report Output Directory", width=280
        )
        metric_sel = pn.widgets.Select.from_param(
            self.param.metric_key, name="Metric", width=280
        )
        mode_sel = pn.widgets.RadioButtonGroup.from_param(
            self.param.mode, name="Optimise", width=140
        )
        alt_toggle = pn.widgets.Toggle.from_param(
            self.param.use_alt_metric,
            name="Alt metric for XGBoost / TabPFN / TabDPT",
            width=280,
        )
        alt_metric_sel = pn.widgets.Select.from_param(
            self.param.alt_metric_key, name="Alt metric", width=280
        )
        alt_metric_panel = pn.bind(
            lambda use: alt_metric_sel if use else pn.layout.Spacer(height=0),
            self.param.use_alt_metric,
        )
        dataset_sel = pn.widgets.Select.from_param(
            self.param.selected_dataset, name="Dataset", width=280
        )
        model_sel = pn.widgets.Select.from_param(
            self.param.selected_model, name="Model", width=280
        )
        state_store = pn.widgets.TextInput(
            value=self._serialize_state_json(),
            width=0,
            height=0,
            margin=0,
            styles={"display": "none"},
        )
        state_store.jscallback(
            value=f"""
            const key = `{_STORAGE_PREFIX}:${{window.location.pathname}}`;
            try {{
                window.localStorage.setItem(key, cb_obj.value);
            }} catch (err) {{
                console.debug("Unable to persist dashboard state.", err);
            }}
            """
        )
        self._state_store = state_store
        self._sync_state_store()

        # On-load restore: a hidden IntInput whose value is bumped to 1 after the
        # WebSocket is established (via pn.state.onload). The jscallback reads
        # localStorage and pushes the JSON to restore_store, which Python then
        # applies via _apply_saved_state.
        restore_store = pn.widgets.TextInput(value="", visible=False)
        load_trigger = pn.widgets.IntInput(value=0, visible=False)
        load_trigger.jscallback(
            args={"restore": restore_store},
            value=f"""
            const key = `{_STORAGE_PREFIX}:${{window.location.pathname}}`;
            let saved = null;
            try {{
                saved = window.localStorage.getItem(key);
            }} catch (err) {{
                console.debug("Unable to read dashboard state from localStorage.", err);
            }}
            if (saved) {{
                restore.value = saved;
            }}
            """
        )

        def _on_restore(event: param.parameterized.Event) -> None:
            if not event.new:
                return
            state = self._parse_saved_state(event.new)
            if state:
                self._apply_saved_state(state)
            restore_store.value = ""

        restore_store.param.watch(_on_restore, "value")

        sidebar = pn.Column(
            pn.pane.Markdown("## piCID Results", styles={"font-size": "1.1em"}),
            pn.layout.Divider(),
            "**Data source**",
            dir_input,
            pn.layout.Divider(),
            "**Visualisation**",
            metric_sel,
            "**Direction**",
            mode_sel,
            alt_toggle,
            alt_metric_panel,
            pn.widgets.Toggle.from_param(
                self.param.show_n, name="Show n= in heatmap", width=280
            ),
            pn.widgets.Toggle.from_param(
                self.param.show_rank, name="Show rank in heatmap", width=280
            ),
            pn.layout.Divider(),
            "**Focus**",
            dataset_sel,
            model_sel,
            # Hidden persistence widgets — kept in sidebar so they don't
            # create empty cards in the main content area.
            state_store,
            load_trigger,
            restore_store,
            width=300,
        )

        # --- Main area tabs ---
        tabs = pn.Tabs(
            (_MAIN_TAB_TITLES[0], pn.panel(self._heatmap_view)),
            (_MAIN_TAB_TITLES[1], pn.panel(self._bar_chart_view)),
            (_MAIN_TAB_TITLES[2], pn.panel(self._hp_impact_view)),
            (_MAIN_TAB_TITLES[3], pn.panel(self._summary_view)),
            (_MAIN_TAB_TITLES[4], pn.panel(self._stats_view)),
            dynamic=True,
            active=self.active_tab,
        )

        # Keep tabs.active ↔ self.active_tab in sync (bidirectional)
        tabs.param.watch(lambda e: setattr(self, "active_tab", e.new), "active")
        self.param.watch(lambda e: setattr(tabs, "active", e.new), "active_tab")

        # URL sync — active tab index only
        if pn.state.location is not None:
            pn.state.location.sync(self, {"active_tab": "tab"})

        # Trigger the localStorage restore after the browser WebSocket is ready
        pn.state.onload(lambda: setattr(load_trigger, "value", 1))

        return pn.template.FastListTemplate(
            title="piCID Experiment Dashboard",
            sidebar=[sidebar],
            main=[tabs],
            accent="#0072B2",
        )


# ---------------------------------------------------------------------------
# Standalone entry points
# ---------------------------------------------------------------------------


def get_args() -> str:
    """Parse --report-dir from argv (works for both panel serve and direct run)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report-dir", default=_DEFAULT_REPORT_DIR)
    # panel serve passes unknown args — ignore them
    args, _ = parser.parse_known_args()
    return args.report_dir


def create_app(report_dir: str | None = None) -> pn.viewable.Viewable:
    """Factory used by `panel serve` (module-level callable)."""
    directory = report_dir or get_args()
    dash = PiCIDDashboard(report_dir=directory)
    return dash.panel().servable()


# For `panel serve app.py` — execute at module level so .servable() fires.
app = create_app()


if __name__ == "__main__":
    report_dir = get_args()
    dash = PiCIDDashboard(report_dir=report_dir)
    dash.panel().show(title="piCID Experiment Dashboard")
