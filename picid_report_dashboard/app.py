"""
piCID Results Dashboard

Entry point for Panel serve:

    panel serve picid_report_dashboard/app.py --show --args --report-dir report_output

Or run directly:

    python picid_report_dashboard/app.py --report-dir report_output
"""

from __future__ import annotations

import argparse
import os
import sys

import panel as pn
import param

if __name__ == "__main__" or not __name__.startswith("picid_report_dashboard"):
    import importlib.util, pathlib
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

_DEFAULT_REPORT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "report_output"
)
_DEFAULT_METRIC = "test_best_rerun/mae_denormalized"


# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------

class PiCIDDashboard(param.Parameterized):
    """Reactive Panel dashboard backed by ResultsLoader."""

    report_dir = param.String(default=_DEFAULT_REPORT_DIR, doc="Path to report_output directory")
    metric_key = param.Selector(default=_DEFAULT_METRIC, objects=[], doc="Metric to visualise")
    mode = param.Selector(default="min", objects=["min", "max"], doc="Lower or higher is better")
    use_alt_metric = param.Boolean(default=True, doc="Use separate metric for XGBoost/TabPFN/TabDPT")
    alt_metric_key = param.Selector(default=_DEFAULT_METRIC, objects=[], doc="Alternate metric for fit-predict models")
    show_n = param.Boolean(default=True, doc="Show n= run count in heatmap cells")
    show_rank = param.Boolean(default=False, doc="Show rank within dataset in heatmap cells")
    selected_dataset = param.Selector(default=None, objects=[], doc="Dataset for bar chart / HP impact")
    selected_model = param.Selector(default=None, objects=[], doc="Model for HP impact / metadata")
    active_tab = param.Integer(default=0, doc="Active main tab index (synced to URL)")

    def __init__(self, report_dir: str = _DEFAULT_REPORT_DIR, **params):
        super().__init__(report_dir=report_dir, **params)
        self._loader: ResultsLoader | None = None
        self._load_data()
        self._sync_selectors()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        try:
            self._loader = ResultsLoader(self.report_dir)
            _ = self._loader.xarray_dataset  # trigger xarray load
            _ = self._loader.summary_df      # pre-warm CSV cache
            _ = self._loader.stats_df        # pre-warm CSV cache
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
        candidate = "test/" + metric[len("test_best_rerun/"):]
        available = self.param["alt_metric_key"].objects
        return candidate if candidate in available else None

    @param.depends("metric_key", watch=True)
    def _on_metric_key_change(self) -> None:
        derived = self._derive_alt_metric(self.metric_key)
        if derived:
            self.alt_metric_key = derived

    def _sync_selectors(self) -> None:
        if self._loader is None:
            return
        loader = self._loader
        metrics = loader.metric_keys or [_DEFAULT_METRIC]
        self.param["metric_key"].objects = metrics
        self.param["alt_metric_key"].objects = metrics
        if _DEFAULT_METRIC in metrics:
            self.metric_key = _DEFAULT_METRIC
        elif metrics:
            self.metric_key = metrics[0]
        # Set initial alt metric derived from the default
        derived = self._derive_alt_metric(self.metric_key)
        if derived:
            self.alt_metric_key = derived

        datasets = loader.datasets or []
        self.param["selected_dataset"].objects = datasets
        if datasets:
            self.selected_dataset = datasets[0]

        models = loader.models or []
        self.param["selected_model"].objects = ["— All models —"] + models
        self.selected_model = "— All models —"

    # ------------------------------------------------------------------
    # View methods (reactive)
    # ------------------------------------------------------------------

    def _active_alt_metric(self) -> str | None:
        return self.alt_metric_key if self.use_alt_metric else None

    @param.depends("metric_key", "mode", "use_alt_metric", "alt_metric_key", "show_n", "show_rank")
    def _heatmap_view(self):
        if self._load_error:
            return pn.pane.Alert(f"**Error loading data:** {self._load_error}", alert_type="danger")
        return build_heatmap(self._loader, self.metric_key, self.mode, alt_metric_key=self._active_alt_metric(), show_n=self.show_n, show_rank=self.show_rank)

    @param.depends("metric_key", "mode", "selected_dataset", "use_alt_metric", "alt_metric_key")
    def _bar_chart_view(self):
        if self._load_error or self.selected_dataset is None:
            return pn.pane.Markdown("Select a dataset.")
        return build_bar_chart(self._loader, self.metric_key, self.selected_dataset, self.mode, alt_metric_key=self._active_alt_metric())

    def _active_model(self) -> str | None:
        """Return None when the 'All models' sentinel is selected."""
        m = self.selected_model
        return None if (m is None or m == "— All models —") else m

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
        return build_summary_table(self._loader, self.selected_dataset, self._active_model(), self.metric_key)

    @param.depends("report_dir", "selected_dataset", "selected_model")
    def _stats_view(self):
        if self._load_error:
            return pn.pane.Alert(self._load_error, alert_type="danger")
        return build_stats_table(self._loader, self.selected_dataset, self._active_model())

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
            self.param.use_alt_metric, name="Alt metric for XGBoost / TabPFN / TabDPT", width=280
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
            pn.widgets.Toggle.from_param(self.param.show_n, name="Show n= in heatmap", width=280),
            pn.widgets.Toggle.from_param(self.param.show_rank, name="Show rank in heatmap", width=280),
            pn.layout.Divider(),
            "**Focus**",
            dataset_sel,
            model_sel,
            width=300,
        )

        # --- Main area tabs ---
        tabs = pn.Tabs(
            ("Heatmap", pn.panel(self._heatmap_view)),
            ("Bar Chart", pn.panel(self._bar_chart_view)),
            ("HP Impact", pn.panel(self._hp_impact_view)),
            ("Summary Table", pn.panel(self._summary_view)),
            ("Experiment Stats", pn.panel(self._stats_view)),
            dynamic=True,
            active=self.active_tab,
        )

        # Keep tabs.active ↔ self.active_tab in sync (bidirectional)
        tabs.param.watch(lambda e: setattr(self, "active_tab", e.new), "active")
        self.param.watch(lambda e: setattr(tabs, "active", e.new), "active_tab")

        # URL sync — all sidebar params + active tab
        if pn.state.location is not None:
            pn.state.location.sync(self, {
                "metric_key": "metric",
                "mode": "mode",
                "use_alt_metric": "alt",
                "alt_metric_key": "alt_metric",
                "show_n": "show_n",
                "show_rank": "show_rank",
                "selected_dataset": "dataset",
                "selected_model": "model",
                "active_tab": "tab",
            })

        # localStorage persistence:
        #   • On first load (no URL params): read saved search string → redirect so
        #     Panel's location.sync picks up the params on the next server request.
        #   • After load: intercept history.pushState / replaceState (used by Panel
        #     to update URL params) and persist the search string to localStorage.
        _ls_js = pn.pane.HTML("""
<script>
(function () {
    var KEY = 'picid_dashboard_state';

    // Restore: if URL has no meaningful params, reload with saved state so that
    // Panel's server-side location.sync reads them from the fresh request.
    var search = window.location.search;
    if (!search || search === '?') {
        var saved = localStorage.getItem(KEY);
        if (saved && saved.length > 1) {
            window.location.replace(window.location.pathname + saved);
            return;
        }
    }

    // Save: intercept the history API calls Panel uses to update URL params.
    function save() {
        var s = window.location.search;
        if (s && s !== '?') { localStorage.setItem(KEY, s); }
    }
    var _push = history.pushState.bind(history);
    var _replace = history.replaceState.bind(history);
    history.pushState    = function () { _push.apply(this, arguments);    save(); };
    history.replaceState = function () { _replace.apply(this, arguments); save(); };
    window.addEventListener('popstate', save);
    setInterval(save, 3000);  // fallback poll
})();
</script>
""", width=0, height=0, margin=0)

        return pn.template.FastListTemplate(
            title="piCID Experiment Dashboard",
            sidebar=[sidebar],
            main=[_ls_js, tabs],
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
