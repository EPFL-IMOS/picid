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

pn.extension("tabulator", sizing_mode="stretch_width", template="fast")

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
    selected_dataset = param.Selector(default=None, objects=[], doc="Dataset for bar chart / HP impact")
    selected_model = param.Selector(default=None, objects=[], doc="Model for HP impact / metadata")

    def __init__(self, report_dir: str = _DEFAULT_REPORT_DIR, **params):
        super().__init__(report_dir=report_dir, **params)
        self._loader: ResultsLoader | None = None
        self._summary_widget = None   # cached widget
        self._stats_widget = None     # cached widget
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
        self._summary_widget = None
        self._stats_widget = None
        self._load_data()
        self._sync_selectors()
        self.param.trigger("metric_key")  # force view refresh

    def _sync_selectors(self) -> None:
        if self._loader is None:
            return
        loader = self._loader
        metrics = loader.metric_keys or [_DEFAULT_METRIC]
        self.param["metric_key"].objects = metrics
        if _DEFAULT_METRIC in metrics:
            self.metric_key = _DEFAULT_METRIC
        elif metrics:
            self.metric_key = metrics[0]

        datasets = loader.datasets or []
        self.param["selected_dataset"].objects = datasets
        if datasets:
            self.selected_dataset = datasets[0]

        models = loader.models or []
        self.param["selected_model"].objects = models
        if models:
            self.selected_model = models[0]

    # ------------------------------------------------------------------
    # View methods (reactive)
    # ------------------------------------------------------------------

    @param.depends("metric_key", "mode")
    def _heatmap_view(self):
        if self._load_error:
            return pn.pane.Alert(f"**Error loading data:** {self._load_error}", alert_type="danger")
        return build_heatmap(self._loader, self.metric_key, self.mode)

    @param.depends("metric_key", "mode", "selected_dataset")
    def _bar_chart_view(self):
        if self._load_error or self.selected_dataset is None:
            return pn.pane.Markdown("Select a dataset.")
        return build_bar_chart(self._loader, self.metric_key, self.selected_dataset, self.mode)

    @param.depends("selected_dataset", "selected_model")
    def _hp_impact_view(self):
        if self._load_error or self.selected_dataset is None or self.selected_model is None:
            return pn.pane.Markdown("Select a dataset and model.")
        return pn.Column(
            build_metadata_panel(self._loader, self.selected_dataset, self.selected_model),
            pn.layout.Divider(),
            build_hp_impact_table(self._loader, self.selected_dataset, self.selected_model),
        )

    @param.depends("report_dir")
    def _summary_view(self):
        if self._load_error:
            return pn.pane.Alert(self._load_error, alert_type="danger")
        if self._summary_widget is None:
            self._summary_widget = build_summary_table(self._loader)
        return self._summary_widget

    @param.depends("report_dir")
    def _stats_view(self):
        if self._load_error:
            return pn.pane.Alert(self._load_error, alert_type="danger")
        if self._stats_widget is None:
            self._stats_widget = build_stats_table(self._loader)
        return self._stats_widget

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
        )

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
    return dash.panel()


# For `panel serve app.py` — Panel looks for a callable named `app` or
# a module-level Panel object.
app = create_app


if __name__ == "__main__":
    report_dir = get_args()
    dash = PiCIDDashboard(report_dir=report_dir)
    dash.panel().show(title="piCID Experiment Dashboard")
