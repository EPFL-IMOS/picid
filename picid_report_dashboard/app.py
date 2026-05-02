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
import logging
import os
import sys
from typing import Any

import pandas as pd
import panel as pn
import param

if __name__ == "__main__" or not __name__.startswith("picid_report_dashboard"):
    import pathlib

    _pkg = pathlib.Path(__file__).parent
    sys.path.insert(0, str(_pkg.parent))

from picid_report_dashboard.data import ResultsLoader, is_alt_model
from picid_report_dashboard.views import (
    _shorten_dataset_name,
    _shorten_model_name,
    build_bar_chart,
    build_heatmap,
    build_hp_impact_table,
    build_latex_table,
    build_metadata_panel,
    build_model_summary_table,
    build_parallel_coordinates,
    build_spiderweb,
    build_stats_table,
    build_summary_table,
)


class _DropBokehStaleRefFilter(logging.Filter):
    """Suppress Bokeh's 'Dropping a patch...' churn under --autoreload.

    Bokeh logs these at WARNING when a session message references a model that
    has already been replaced — common for reactive views that fully rebuild
    on parameter changes. The library itself notes the message is harmless.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Dropping a patch because it contains" not in record.getMessage()


for _name in ("bokeh", "bokeh.server.session", "bokeh.document"):
    logging.getLogger(_name).addFilter(_DropBokehStaleRefFilter())


pn.extension("tabulator", "plotly", sizing_mode="stretch_width")

# ---------------------------------------------------------------------------
# Defaults — can be overridden via CLI args or URL query params
# ---------------------------------------------------------------------------

_DEFAULT_REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report_output")
_DEFAULT_METRIC = "test_best_rerun/mae_denormalized"
_PERSISTED_STATE_VERSION = 7
_STORAGE_PREFIX = "picid_dashboard_state"
_ALL_MODELS_OPTION = "— All models —"
_MAIN_TAB_TITLES = (
    "Heatmap",
    "Parallel Coordinates",
    "Spiderweb",
    "Bar Chart",
    "HP Impact",
    "Summary Table",
    "Experiment Stats",
    "LaTeX Table",
)
_HP_IMPACT_TAB_INDEX = _MAIN_TAB_TITLES.index("HP Impact")
_PERSISTED_STATE_PARAMS = (
    "report_dir",
    "metric_key",
    "sort_metric_key",
    "mode",
    "use_alt_metric",
    "alt_metric_key",
    "alt_sort_metric_key",
    "alt_metric_overridden",
    "alt_sort_metric_overridden",
    "show_n",
    "show_rank",
    "spiderweb_plot_mode",
    "selected_dataset",
    "selected_model",
    "active_tab",
    "latex_precision",
    "latex_multiplier",
    "latex_show_std",
    "latex_show_n",
    "latex_highlight_best_bg",
    "latex_underline_2nd_best",
    "latex_highlight_within_1std",
    "latex_rename_json",
)
_UNSET = object()
_ACTIVE_BUTTON_STYLESHEET = """
:host(.solid) .bk-btn.bk-btn-default.bk-active {
  background-color: var(--primary-color);
  color: var(--button-primary-text-color);
  border-color: var(--primary-color);
  box-shadow: inset 0px 3px 5px rgb(0 0 0 / 25%);
}
"""
_UNAVAILABLE_METRIC_COLOR = "#9ca3af"


def _selected_primary_button(widget: pn.widgets.Widget) -> pn.widgets.Widget:
    """Keep default grey buttons but render the active state with the primary color."""
    if hasattr(widget, "button_type"):
        widget.button_type = "default"
    if hasattr(widget, "stylesheets"):
        widget.stylesheets = [*widget.stylesheets, _ACTIVE_BUTTON_STYLESHEET]
    return widget


def _css_attr_value(value: str) -> str:
    """Escape a string for use in a CSS attribute selector."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _dataset_base_name(dataset: object) -> str | None:
    """Return the dataset label without its optional ``[project]`` suffix."""
    if dataset is None:
        return None
    dataset_str = str(dataset)
    if " [" in dataset_str and dataset_str.endswith("]"):
        return dataset_str.rsplit(" [", 1)[0]
    return dataset_str


def _metric_availability_stylesheet(unavailable_values: set[str]) -> str:
    """Return CSS that grays out unavailable Select options."""
    if not unavailable_values:
        return ""
    selectors: list[str] = []
    for metric_name in sorted(unavailable_values):
        escaped = _css_attr_value(metric_name)
        selectors.extend(
            [
                f':host select option[value="{escaped}"]',
                f':host .bk-input option[value="{escaped}"]',
                f':host option[value="{escaped}"]',
            ]
        )
    joined_selectors = ",\n".join(selectors)
    return f"""
{joined_selectors} {{
  color: {_UNAVAILABLE_METRIC_COLOR} !important;
}}
"""


def _apply_metric_availability_style(
    widget: pn.widgets.Select,
    *,
    unavailable_values: set[str],
    active_value: str | None,
    base_stylesheets: list[str],
) -> None:
    """Apply per-option and selected-value styling for unavailable metrics."""
    stylesheet = _metric_availability_stylesheet(unavailable_values)
    widget.stylesheets = (
        [*base_stylesheets, stylesheet] if stylesheet else list(base_stylesheets)
    )
    current_styles = dict(getattr(widget, "styles", {}) or {})
    if active_value is not None and active_value in unavailable_values:
        current_styles["color"] = _UNAVAILABLE_METRIC_COLOR
    else:
        current_styles.pop("color", None)
    widget.styles = current_styles


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
    sort_metric_key = param.Selector(
        default=_DEFAULT_METRIC, objects=[], doc="Metric used to select configs"
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
    alt_sort_metric_key = param.Selector(
        default=_DEFAULT_METRIC,
        objects=[],
        doc="Alternate sort metric for fit-predict models",
    )
    alt_metric_overridden = param.Boolean(
        default=False, doc="Whether the alt metric differs from the derived default"
    )
    alt_sort_metric_overridden = param.Boolean(
        default=False,
        doc="Whether the alt sort metric differs from the derived default",
    )
    show_n = param.Boolean(default=True, doc="Show n= run count in heatmap cells")
    show_rank = param.Boolean(
        default=False, doc="Show rank within dataset in heatmap cells"
    )
    spiderweb_plot_mode = param.Selector(
        default="matplotlib",
        objects=["matplotlib", "interactive"],
        doc="Renderer for the spiderweb tab",
    )
    selected_dataset = param.Selector(
        default=None, objects=[], doc="Dataset for bar chart / HP impact"
    )
    selected_model = param.Selector(
        default=None, objects=[], doc="Model for HP impact / metadata"
    )
    active_tab = param.Integer(default=0, doc="Active main tab index (synced to URL)")
    latex_precision = param.Integer(
        default=4, bounds=(1, 8), doc="Decimal precision for LaTeX table cells"
    )
    latex_multiplier = param.Number(
        default=1.0,
        doc="Scalar applied to all metric values and stds in the LaTeX table",
    )
    latex_show_std = param.Boolean(default=True, doc="Show ± std in LaTeX table cells")
    latex_show_n = param.Boolean(default=True, doc="Show (n=…) count in LaTeX table cells")
    latex_highlight_best_bg = param.Boolean(
        default=True, doc="Light-grey background on the best cell per dataset"
    )
    latex_underline_2nd_best = param.Boolean(
        default=True, doc="Underline the second-best cell per dataset"
    )
    latex_highlight_within_1std = param.Boolean(
        default=True,
        doc="Light-blue background on cells within 1σ of the best (using best's std)",
    )
    latex_rename_json = param.String(
        default="{}", doc="JSON map of shortened name → display name for LaTeX table"
    )

    def __init__(self, report_dir: str = _DEFAULT_REPORT_DIR, **params):
        super().__init__(report_dir=report_dir, **params)
        self._loader: ResultsLoader | None = None
        self._load_error: str | None = None
        self._alias_warnings: list[tuple[str, str]] = []
        self._restore_in_progress = False
        self._state_store: pn.widgets.TextInput | None = None
        self._sync_selectors_in_progress = False
        self._suspend_alt_metric_tracking = False
        self._suspend_alt_sort_metric_tracking = False
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
            self._alias_warnings = self._loader.applied_model_aliases
        except Exception as exc:
            self._loader = None
            self._load_error = str(exc)
            self._alias_warnings = []

    @param.depends("report_dir", watch=True)
    def _on_report_dir_change(self) -> None:
        self._load_data()
        self._sync_selectors()
        self.param.trigger(
            "metric_key",
            "sort_metric_key",
            "selected_dataset",
            "selected_model",
        )

    def _derive_alt_metric(
        self,
        metric: str,
        selector_name: str,
        *,
        available: list[str] | None = None,
    ) -> str | None:
        """Map a primary metric choice to the alt-selector value to keep in sync."""
        available_metrics = (
            list(available)
            if available is not None
            else list(self.param[selector_name].objects)
        )
        for source_prefix, target_prefix in (
            ("test_best_rerun/", "test/"),
            ("val_best_rerun/", "val/"),
        ):
            if metric.startswith(source_prefix):
                candidate = target_prefix + metric[len(source_prefix) :]
                return candidate if candidate in available_metrics else None
        return metric if metric in available_metrics else None

    def _set_alt_metric_value(self, value: str, *, overridden: bool) -> None:
        """Assign the alt metric while suppressing manual-override tracking."""
        self._suspend_alt_metric_tracking = True
        try:
            self.alt_metric_key = value
            self.alt_metric_overridden = overridden
        finally:
            self._suspend_alt_metric_tracking = False

    def _set_alt_sort_metric_value(self, value: str, *, overridden: bool) -> None:
        """Assign the alt sort metric while suppressing manual-override tracking."""
        self._suspend_alt_sort_metric_tracking = True
        try:
            self.alt_sort_metric_key = value
            self.alt_sort_metric_overridden = overridden
        finally:
            self._suspend_alt_sort_metric_tracking = False

    @param.depends("metric_key", watch=True)
    def _on_metric_key_change(self) -> None:
        if self._sync_selectors_in_progress:
            return
        derived = self._derive_alt_metric(self.metric_key, "alt_metric_key")
        if derived:
            self._set_alt_metric_value(derived, overridden=False)
        if self._restore_in_progress:
            return
        self._sync_selectors()

    @param.depends("sort_metric_key", watch=True)
    def _on_sort_metric_key_change(self) -> None:
        if self._sync_selectors_in_progress:
            return
        derived = self._derive_alt_metric(self.sort_metric_key, "alt_sort_metric_key")
        if derived:
            self._set_alt_sort_metric_value(derived, overridden=False)

    @param.depends("alt_metric_key", watch=True)
    def _on_alt_metric_key_change(self) -> None:
        if (
            self._sync_selectors_in_progress
            or self._suspend_alt_metric_tracking
            or self._restore_in_progress
        ):
            return
        derived = self._derive_alt_metric(self.metric_key, "alt_metric_key")
        self.alt_metric_overridden = (
            self.alt_metric_key != derived
            if derived is not None
            else self.alt_metric_key not in (None, self.metric_key)
        )
        self._sync_selectors()

    @param.depends("alt_sort_metric_key", watch=True)
    def _on_alt_sort_metric_key_change(self) -> None:
        if (
            self._sync_selectors_in_progress
            or self._suspend_alt_sort_metric_tracking
            or self._restore_in_progress
        ):
            return
        derived = self._derive_alt_metric(self.sort_metric_key, "alt_sort_metric_key")
        self.alt_sort_metric_overridden = (
            self.alt_sort_metric_key != derived
            if derived is not None
            else self.alt_sort_metric_key not in (None, self.sort_metric_key)
        )

    @param.depends("use_alt_metric", watch=True)
    def _on_use_alt_metric_change(self) -> None:
        if self._sync_selectors_in_progress or self._restore_in_progress:
            return
        self._sync_selectors()

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

    def _preferred_dataset_for_display_metric(
        self,
        current_dataset: object,
        valid_datasets: list[str],
    ) -> str | None:
        """Resolve the best dataset to keep when display-metric validity changes."""
        if current_dataset in valid_datasets:
            return str(current_dataset)
        current_base = _dataset_base_name(current_dataset)
        if current_base is not None:
            for dataset_name in valid_datasets:
                if _dataset_base_name(dataset_name) == current_base:
                    return dataset_name
        return None

    def _sync_selectors(self, *, preferred_dataset: Any = _UNSET) -> None:
        if self._loader is None or self._sync_selectors_in_progress:
            return
        self._sync_selectors_in_progress = True
        try:
            loader = self._loader
            current_metric = self.metric_key
            current_sort_metric = self.sort_metric_key
            current_alt_metric = self.alt_metric_key
            current_alt_sort_metric = self.alt_sort_metric_key
            current_alt_metric_overridden = self.alt_metric_overridden
            current_alt_sort_metric_overridden = self.alt_sort_metric_overridden
            current_dataset = (
                preferred_dataset
                if preferred_dataset is not _UNSET
                else self.selected_dataset
            )
            current_model = self.selected_model

            metrics = loader.metric_keys or [_DEFAULT_METRIC]
            self._set_selector_value(
                "metric_key",
                metrics,
                preferred=current_metric,
                fallback=_DEFAULT_METRIC,
            )
            self._set_selector_value(
                "sort_metric_key",
                metrics,
                preferred=current_sort_metric,
                fallback=self.metric_key,
            )
            derived = self._derive_alt_metric(
                self.metric_key, "alt_metric_key", available=metrics
            )
            keep_alt_metric_override = bool(
                current_alt_metric_overridden and current_alt_metric in metrics
            )
            self._set_selector_value(
                "alt_metric_key",
                metrics,
                preferred=current_alt_metric if keep_alt_metric_override else _UNSET,
                fallback=derived if derived is not None else _DEFAULT_METRIC,
            )
            self.alt_metric_overridden = (
                keep_alt_metric_override and self.alt_metric_key == current_alt_metric
            )
            derived_sort = self._derive_alt_metric(
                self.sort_metric_key,
                "alt_sort_metric_key",
                available=metrics,
            )
            keep_alt_sort_override = bool(
                current_alt_sort_metric_overridden
                and current_alt_sort_metric in metrics
            )
            self._set_selector_value(
                "alt_sort_metric_key",
                metrics,
                preferred=current_alt_sort_metric if keep_alt_sort_override else _UNSET,
                fallback=derived_sort
                if derived_sort is not None
                else self.sort_metric_key,
            )
            self.alt_sort_metric_overridden = (
                keep_alt_sort_override
                and self.alt_sort_metric_key == current_alt_sort_metric
            )

            valid_datasets = loader.datasets_with_display_metric(
                metric_key=self.metric_key,
                alt_metric_key=self._active_alt_metric(),
                use_alt_metric=self.use_alt_metric,
            )
            preferred_valid_dataset = self._preferred_dataset_for_display_metric(
                current_dataset,
                valid_datasets,
            )
            self._set_selector_value(
                "selected_dataset",
                valid_datasets,
                preferred=preferred_valid_dataset
                if preferred_valid_dataset is not None
                else _UNSET,
            )

            models = loader.models or []
            self._set_selector_value(
                "selected_model",
                [_ALL_MODELS_OPTION, *models],
                preferred=current_model,
                fallback=_ALL_MODELS_OPTION,
            )
        finally:
            self._sync_selectors_in_progress = False

    def _availability_scope_models(self, *, alt: bool) -> list[str]:
        """Return the models whose HP metric availability should drive one selector family."""
        if self._loader is None:
            return []
        active_model = self._active_model()
        if active_model is not None and is_alt_model(active_model) == alt:
            return [active_model]
        return [model for model in self._loader.models if is_alt_model(model) == alt]

    def _available_hp_metrics_for_family(self, *, alt: bool) -> set[str]:
        """Return union of HP metric names for the active dataset and model family."""
        if self._loader is None or self.selected_dataset is None:
            return set()
        available: set[str] = set()
        for model in self._availability_scope_models(alt=alt):
            hp_ds = self._loader.hp_impact_ds(self.selected_dataset, model)
            if hp_ds is None or "metric" not in hp_ds.coords:
                continue
            available.update(
                str(metric) for metric in hp_ds.coords["metric"].values.tolist()
            )
        return available

    def _unavailable_metric_values(self, *, alt: bool, selector_name: str) -> set[str]:
        """Return selector values unavailable in the current HP-config scope."""
        if self._loader is None or self.selected_dataset is None:
            return set()
        available = self._available_hp_metrics_for_family(alt=alt)
        if not available:
            return set()
        return {
            str(metric)
            for metric in self.param[selector_name].objects
            if str(metric) not in available
        }

    def _serialize_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"version": _PERSISTED_STATE_VERSION}
        for name in _PERSISTED_STATE_PARAMS:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or value is None
                or isinstance(value, (int, float))
            ):
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
        if version not in {2, 3, 4, 5, 6, _PERSISTED_STATE_VERSION}:
            return None

        parsed: dict[str, Any] = {}
        string_fields = (
            "report_dir",
            "metric_key",
            "sort_metric_key",
            "mode",
            "spiderweb_plot_mode",
            "alt_metric_key",
            "alt_sort_metric_key",
            "selected_dataset",
            "selected_model",
            "latex_rename_json",
        )
        bool_fields = (
            "use_alt_metric",
            "alt_metric_overridden",
            "alt_sort_metric_overridden",
            "show_n",
            "show_rank",
            "latex_show_std",
            "latex_show_n",
            "latex_highlight_best_bg",
            "latex_underline_2nd_best",
            "latex_highlight_within_1std",
        )
        for field in string_fields:
            value = payload.get(field)
            if isinstance(value, str):
                parsed[field] = value
        for field in bool_fields:
            value = payload.get(field)
            if isinstance(value, bool):
                parsed[field] = value
        latex_precision = payload.get("latex_precision")
        if isinstance(latex_precision, int) and not isinstance(latex_precision, bool):
            parsed["latex_precision"] = max(1, min(8, latex_precision))
        latex_multiplier = payload.get("latex_multiplier")
        if (
            isinstance(latex_multiplier, (int, float))
            and not isinstance(latex_multiplier, bool)
        ):
            parsed["latex_multiplier"] = float(latex_multiplier)
        active_tab = payload.get("active_tab")
        if isinstance(active_tab, bool):
            active_tab = None
        elif isinstance(active_tab, str) and active_tab.isdigit():
            active_tab = int(active_tab)
        if isinstance(active_tab, int):
            if version == 2 and active_tab >= 1:
                active_tab += 2
            elif version in {3, 4} and active_tab >= 2:
                active_tab += 1
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
            self._apply_selector_state(
                "spiderweb_plot_mode", state.get("spiderweb_plot_mode")
            )

            for name in ("use_alt_metric", "show_n", "show_rank"):
                if name in state:
                    setattr(self, name, state[name])

            if self._loader is not None:
                for name in ("metric_key", "sort_metric_key"):
                    self._apply_selector_state(name, state.get(name))

                if state.get("alt_metric_overridden"):
                    alt_metric = state.get("alt_metric_key")
                    if alt_metric in self.param["alt_metric_key"].objects:
                        self._set_alt_metric_value(alt_metric, overridden=True)
                else:
                    self.alt_metric_overridden = False

                if state.get("alt_sort_metric_overridden"):
                    alt_sort_metric = state.get("alt_sort_metric_key")
                    if alt_sort_metric in self.param["alt_sort_metric_key"].objects:
                        self._set_alt_sort_metric_value(
                            alt_sort_metric, overridden=True
                        )
                else:
                    self.alt_sort_metric_overridden = False

                self._sync_selectors(
                    preferred_dataset=state.get("selected_dataset", _UNSET)
                )

                for name in ("selected_dataset", "selected_model"):
                    self._apply_selector_state(name, state.get(name))

            if not self._has_url_tab_override() and "active_tab" in state:
                self.active_tab = state["active_tab"]

            if "latex_precision" in state:
                self.latex_precision = state["latex_precision"]
            if "latex_multiplier" in state:
                self.latex_multiplier = state["latex_multiplier"]
            for name in (
                "latex_show_std",
                "latex_show_n",
                "latex_highlight_best_bg",
                "latex_underline_2nd_best",
                "latex_highlight_within_1std",
            ):
                if name in state:
                    setattr(self, name, state[name])
            if "latex_rename_json" in state:
                self.latex_rename_json = state["latex_rename_json"]
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

    def _active_alt_sort_metric(self) -> str | None:
        return self.alt_sort_metric_key if self.use_alt_metric else None

    @param.depends(
        "metric_key",
        "sort_metric_key",
        "mode",
        "selected_dataset",
        "use_alt_metric",
        "alt_metric_key",
        "alt_sort_metric_key",
        "show_n",
        "show_rank",
    )
    def _heatmap_view(self):
        if self._load_error:
            return pn.pane.Alert(
                f"**Error loading data:** {self._load_error}", alert_type="danger"
            )
        return pn.Column(
            build_heatmap(
                self._loader,
                self.metric_key,
                sort_metric_key=self.sort_metric_key,
                mode=self.mode,
                alt_metric_key=self._active_alt_metric(),
                alt_sort_metric_key=self._active_alt_sort_metric(),
                use_alt_metric=self.use_alt_metric,
                show_n=self.show_n,
                show_rank=self.show_rank,
            ),
            pn.layout.Divider(),
            pn.pane.Markdown(
                f"### View Configuration Summary for {self.selected_dataset}"
                if self.selected_dataset is not None
                else "### View Configuration Summary"
            ),
            build_model_summary_table(
                self._loader,
                self.selected_dataset,
                metric_key=self.metric_key,
                sort_metric_key=self.sort_metric_key,
                alt_metric_key=self._active_alt_metric(),
                alt_sort_metric_key=self._active_alt_sort_metric(),
                use_alt_metric=self.use_alt_metric,
                on_model_click=self._show_hp_impact_for_model,
            ),
            sizing_mode="stretch_width",
        )

    @param.depends(
        "metric_key",
        "sort_metric_key",
        "mode",
        "selected_dataset",
        "use_alt_metric",
        "alt_metric_key",
        "alt_sort_metric_key",
    )
    def _bar_chart_view(self):
        if self._load_error or self.selected_dataset is None:
            return pn.pane.Markdown("Select a dataset.")
        return build_bar_chart(
            self._loader,
            self.metric_key,
            self.selected_dataset,
            mode=self.mode,
            sort_metric_key=self.sort_metric_key,
            alt_metric_key=self._active_alt_metric(),
            alt_sort_metric_key=self._active_alt_sort_metric(),
            use_alt_metric=self.use_alt_metric,
        )

    def _active_model(self) -> str | None:
        """Return None when the 'All models' sentinel is selected."""
        m = self.selected_model
        return None if (m is None or m == _ALL_MODELS_OPTION) else m

    def _show_hp_impact_for_model(self, model: str) -> None:
        """Select a model and switch to the HP Impact tab."""
        if model in self.param["selected_model"].objects:
            self.selected_model = model
        self.active_tab = _HP_IMPACT_TAB_INDEX

    @param.depends(
        "metric_key",
        "sort_metric_key",
        "mode",
        "selected_dataset",
        "use_alt_metric",
        "alt_metric_key",
        "alt_sort_metric_key",
    )
    def _parallel_coordinates_view(self):
        if self._load_error:
            return pn.pane.Alert(
                f"**Error loading data:** {self._load_error}", alert_type="danger"
            )
        return pn.Column(
            build_parallel_coordinates(
                self._loader,
                self.metric_key,
                sort_metric_key=self.sort_metric_key,
                mode=self.mode,
                alt_metric_key=self._active_alt_metric(),
                alt_sort_metric_key=self._active_alt_sort_metric(),
                use_alt_metric=self.use_alt_metric,
            ),
            pn.layout.Divider(),
            pn.pane.Markdown(
                f"### View Configuration Summary for {self.selected_dataset}"
                if self.selected_dataset is not None
                else "### View Configuration Summary"
            ),
            build_model_summary_table(
                self._loader,
                self.selected_dataset,
                metric_key=self.metric_key,
                sort_metric_key=self.sort_metric_key,
                alt_metric_key=self._active_alt_metric(),
                alt_sort_metric_key=self._active_alt_sort_metric(),
                use_alt_metric=self.use_alt_metric,
                on_model_click=self._show_hp_impact_for_model,
            ),
            sizing_mode="stretch_width",
        )

    @param.depends(
        "metric_key",
        "sort_metric_key",
        "mode",
        "selected_dataset",
        "spiderweb_plot_mode",
        "use_alt_metric",
        "alt_metric_key",
        "alt_sort_metric_key",
    )
    def _spiderweb_view(self):
        if self._load_error:
            return pn.pane.Alert(
                f"**Error loading data:** {self._load_error}", alert_type="danger"
            )
        return pn.Column(
            build_spiderweb(
                self._loader,
                self.metric_key,
                sort_metric_key=self.sort_metric_key,
                mode=self.mode,
                alt_metric_key=self._active_alt_metric(),
                alt_sort_metric_key=self._active_alt_sort_metric(),
                use_alt_metric=self.use_alt_metric,
                selected_dataset=self.selected_dataset,
                plot_mode=self.spiderweb_plot_mode,
            ),
            pn.layout.Divider(),
            pn.pane.Markdown(
                f"### View Configuration Summary for {self.selected_dataset}"
                if self.selected_dataset is not None
                else "### View Configuration Summary"
            ),
            build_model_summary_table(
                self._loader,
                self.selected_dataset,
                metric_key=self.metric_key,
                sort_metric_key=self.sort_metric_key,
                alt_metric_key=self._active_alt_metric(),
                alt_sort_metric_key=self._active_alt_sort_metric(),
                use_alt_metric=self.use_alt_metric,
                on_model_click=self._show_hp_impact_for_model,
            ),
            sizing_mode="stretch_width",
        )

    @param.depends(
        "selected_dataset",
        "selected_model",
        "metric_key",
        "sort_metric_key",
        "use_alt_metric",
        "alt_metric_key",
        "alt_sort_metric_key",
    )
    def _hp_impact_view(self):
        model = self._active_model()
        if self._load_error or self.selected_dataset is None or model is None:
            return pn.pane.Markdown("Select a dataset and model.")
        return pn.Column(
            pn.pane.Markdown(f"### HP Impact for {model} on {self.selected_dataset}"),
            build_metadata_panel(
                self._loader,
                self.selected_dataset,
                model,
                metric_key=self.metric_key,
                sort_metric_key=self.sort_metric_key,
                alt_metric_key=self._active_alt_metric(),
                alt_sort_metric_key=self._active_alt_sort_metric(),
                use_alt_metric=self.use_alt_metric,
            ),
            pn.layout.Divider(),
            build_hp_impact_table(
                self._loader,
                self.selected_dataset,
                model,
                metric_key=self.metric_key,
                sort_metric_key=self.sort_metric_key,
                alt_metric_key=self._active_alt_metric(),
                alt_sort_metric_key=self._active_alt_sort_metric(),
                use_alt_metric=self.use_alt_metric,
            ),
        )

    @param.depends(
        "report_dir",
        "selected_dataset",
        "selected_model",
        "metric_key",
        "sort_metric_key",
        "use_alt_metric",
        "alt_metric_key",
        "alt_sort_metric_key",
    )
    def _summary_view(self):
        if self._load_error:
            return pn.pane.Alert(self._load_error, alert_type="danger")
        return pn.Column(
            pn.pane.Markdown(
                (
                    f"### Summary Table for {self.metric_key} on {self.selected_dataset}"
                    if self.selected_dataset is not None
                    else f"### Summary Table for {self.metric_key}"
                )
            ),
            build_summary_table(
                self._loader,
                self.selected_dataset,
                self._active_model(),
                self.metric_key,
                alt_metric_key=self._active_alt_metric(),
                use_alt_metric=self.use_alt_metric,
            ),
            pn.layout.Divider(),
            pn.pane.Markdown(
                f"### View Configuration Summary for {self.selected_dataset}"
                if self.selected_dataset is not None
                else "### View Configuration Summary"
            ),
            build_model_summary_table(
                self._loader,
                self.selected_dataset,
                metric_key=self.metric_key,
                sort_metric_key=self.sort_metric_key,
                alt_metric_key=self._active_alt_metric(),
                alt_sort_metric_key=self._active_alt_sort_metric(),
                use_alt_metric=self.use_alt_metric,
                on_model_click=self._show_hp_impact_for_model,
            ),
            sizing_mode="stretch_width",
        )

    @param.depends("report_dir", "selected_dataset", "selected_model")
    def _stats_view(self):
        if self._load_error:
            return pn.pane.Alert(self._load_error, alert_type="danger")
        return build_stats_table(
            self._loader, self.selected_dataset, self._active_model()
        )

    def _latex_configurator_panel(self) -> pn.viewable.Viewable:
        """Precision input + rename table. Built once per tab activation; not itself reactive."""
        loader = self._loader
        if loader is None:
            return pn.pane.Markdown("No data loaded.")

        try:
            rename_map: dict[str, str] = json.loads(self.latex_rename_json or "{}")
        except json.JSONDecodeError:
            rename_map = {}

        short_models = [_shorten_model_name(m) for m in loader.models]
        short_datasets = [_shorten_dataset_name(d) for d in loader.datasets]
        all_names = short_models + short_datasets
        types = ["Model"] * len(short_models) + ["Dataset"] * len(short_datasets)

        df = pd.DataFrame(
            {
                "type": types,
                "original": all_names,
                "display_name": [rename_map.get(n, "") for n in all_names],
            }
        )

        tabulator = pn.widgets.Tabulator(
            df,
            editors={"type": None, "original": None, "display_name": "input"},
            titles={
                "type": "Type",
                "original": "Default name",
                "display_name": "Display name (empty = keep default)",
            },
            show_index=False,
            sizing_mode="stretch_width",
            height=min(400, 45 + 35 * max(len(all_names), 1)),
        )

        def _on_edit(_event: Any) -> None:
            new_map = {
                row["original"]: row["display_name"]
                for _, row in tabulator.value.iterrows()
                if str(row["display_name"]).strip()
            }
            self.latex_rename_json = json.dumps(new_map, ensure_ascii=False)

        tabulator.on_edit(_on_edit)

        precision_input = pn.widgets.IntInput(
            name="Decimal precision",
            value=self.latex_precision,
            start=1,
            end=8,
            step=1,
            width=120,
        )
        precision_input.param.watch(
            lambda e: setattr(self, "latex_precision", e.new), "value"
        )

        multiplier_input = pn.widgets.FloatInput(
            name="Multiplier",
            value=self.latex_multiplier,
            step=1.0,
            width=120,
        )
        multiplier_input.param.watch(
            lambda e: setattr(self, "latex_multiplier", e.new), "value"
        )

        show_std_toggle = _selected_primary_button(
            pn.widgets.Toggle(
                name="Show ± std", value=self.latex_show_std, width=120
            )
        )
        show_std_toggle.param.watch(
            lambda e: setattr(self, "latex_show_std", e.new), "value"
        )

        show_n_toggle = _selected_primary_button(
            pn.widgets.Toggle(
                name="Show (n=…)", value=self.latex_show_n, width=120
            )
        )
        show_n_toggle.param.watch(
            lambda e: setattr(self, "latex_show_n", e.new), "value"
        )

        best_bg_toggle = _selected_primary_button(
            pn.widgets.Toggle(
                name="Best cell grey",
                value=self.latex_highlight_best_bg,
                width=140,
            )
        )
        best_bg_toggle.param.watch(
            lambda e: setattr(self, "latex_highlight_best_bg", e.new), "value"
        )

        second_underline_toggle = _selected_primary_button(
            pn.widgets.Toggle(
                name="Underline 2nd best",
                value=self.latex_underline_2nd_best,
                width=160,
            )
        )
        second_underline_toggle.param.watch(
            lambda e: setattr(self, "latex_underline_2nd_best", e.new), "value"
        )

        within_blue_toggle = _selected_primary_button(
            pn.widgets.Toggle(
                name="Within 1σ blue",
                value=self.latex_highlight_within_1std,
                width=140,
            )
        )
        within_blue_toggle.param.watch(
            lambda e: setattr(self, "latex_highlight_within_1std", e.new), "value"
        )

        return pn.Column(
            pn.pane.Markdown("### Configuration"),
            pn.Row(
                pn.Column("**Precision**", precision_input),
                pn.Column("**Scale**", multiplier_input),
                pn.Column("**Columns**", pn.Row(show_std_toggle, show_n_toggle)),
                pn.Column(
                    "**Highlights**",
                    pn.Row(
                        best_bg_toggle,
                        second_underline_toggle,
                        within_blue_toggle,
                    ),
                ),
            ),
            pn.layout.Divider(),
            pn.pane.Markdown(
                "**Rename models / datasets** — leave *Display name* empty to keep the default"
            ),
            tabulator,
            pn.layout.Divider(),
            sizing_mode="stretch_width",
        )

    @param.depends(
        "metric_key",
        "sort_metric_key",
        "mode",
        "use_alt_metric",
        "alt_metric_key",
        "alt_sort_metric_key",
        "latex_show_n",
        "latex_show_std",
        "latex_highlight_best_bg",
        "latex_underline_2nd_best",
        "latex_highlight_within_1std",
        "latex_precision",
        "latex_multiplier",
        "latex_rename_json",
    )
    def _latex_output_view(self) -> pn.viewable.Viewable:
        if self._load_error:
            return pn.pane.Alert(
                f"**Error loading data:** {self._load_error}", alert_type="danger"
            )
        try:
            rename_map: dict[str, str] = json.loads(self.latex_rename_json or "{}")
        except json.JSONDecodeError:
            rename_map = {}
        return build_latex_table(
            self._loader,
            self.metric_key,
            sort_metric_key=self.sort_metric_key,
            mode=self.mode,
            alt_metric_key=self._active_alt_metric(),
            alt_sort_metric_key=self._active_alt_sort_metric(),
            use_alt_metric=self.use_alt_metric,
            show_n=self.latex_show_n,
            show_std=self.latex_show_std,
            highlight_best_bg=self.latex_highlight_best_bg,
            underline_2nd_best=self.latex_underline_2nd_best,
            highlight_within_1std=self.latex_highlight_within_1std,
            precision=self.latex_precision,
            multiplier=self.latex_multiplier,
            rename_map=rename_map,
        )

    @param.depends(
        "report_dir",
        "metric_key",
        "sort_metric_key",
        "mode",
        "use_alt_metric",
        "alt_metric_key",
        "alt_sort_metric_key",
        "latex_show_n",
        "latex_show_std",
        "latex_highlight_best_bg",
        "latex_underline_2nd_best",
        "latex_highlight_within_1std",
        "latex_precision",
        "latex_multiplier",
        "latex_rename_json",
    )
    def _latex_table_view(self) -> pn.viewable.Viewable:
        if self._load_error:
            return pn.pane.Alert(
                f"**Error loading data:** {self._load_error}", alert_type="danger"
            )
        return pn.Column(
            self._latex_configurator_panel(),
            pn.panel(self._latex_output_view),
            sizing_mode="stretch_width",
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
        sort_metric_sel = pn.widgets.Select.from_param(
            self.param.sort_metric_key, name="Sort metric", width=280
        )
        mode_sel = _selected_primary_button(
            pn.widgets.RadioButtonGroup.from_param(
                self.param.mode, name="Optimise", width=140
            )
        )
        alt_toggle = _selected_primary_button(
            pn.widgets.Toggle.from_param(
                self.param.use_alt_metric,
                name="Alt metric for XGBoost / TabPFN / TabDPT",
                width=280,
            )
        )
        alt_metric_sel = pn.widgets.Select.from_param(
            self.param.alt_metric_key, name="Alt metric", width=280
        )
        alt_sort_metric_sel = pn.widgets.Select.from_param(
            self.param.alt_sort_metric_key, name="Alt sort metric", width=280
        )
        metric_sel_base_stylesheets = list(metric_sel.stylesheets)
        sort_metric_sel_base_stylesheets = list(sort_metric_sel.stylesheets)
        alt_metric_sel_base_stylesheets = list(alt_metric_sel.stylesheets)
        alt_sort_metric_sel_base_stylesheets = list(alt_sort_metric_sel.stylesheets)

        def _refresh_metric_selector_styles(*_events: Any) -> None:
            primary_unavailable = self._unavailable_metric_values(
                alt=False, selector_name="metric_key"
            )
            alt_unavailable = self._unavailable_metric_values(
                alt=True, selector_name="alt_metric_key"
            )
            _apply_metric_availability_style(
                metric_sel,
                unavailable_values=primary_unavailable,
                active_value=self.metric_key,
                base_stylesheets=metric_sel_base_stylesheets,
            )
            _apply_metric_availability_style(
                sort_metric_sel,
                unavailable_values=primary_unavailable,
                active_value=self.sort_metric_key,
                base_stylesheets=sort_metric_sel_base_stylesheets,
            )
            _apply_metric_availability_style(
                alt_metric_sel,
                unavailable_values=alt_unavailable,
                active_value=self.alt_metric_key,
                base_stylesheets=alt_metric_sel_base_stylesheets,
            )
            _apply_metric_availability_style(
                alt_sort_metric_sel,
                unavailable_values=alt_unavailable,
                active_value=self.alt_sort_metric_key,
                base_stylesheets=alt_sort_metric_sel_base_stylesheets,
            )

        self.param.watch(
            _refresh_metric_selector_styles,
            [
                "report_dir",
                "selected_dataset",
                "selected_model",
                "metric_key",
                "sort_metric_key",
                "alt_metric_key",
                "alt_sort_metric_key",
            ],
        )
        _refresh_metric_selector_styles()
        alt_metric_panel = pn.bind(
            lambda use: alt_metric_sel if use else pn.layout.Spacer(height=0),
            self.param.use_alt_metric,
        )
        alt_sort_metric_panel = pn.bind(
            lambda use: alt_sort_metric_sel if use else pn.layout.Spacer(height=0),
            self.param.use_alt_metric,
        )
        dataset_sel = pn.widgets.Select.from_param(
            self.param.selected_dataset, name="Dataset", width=280
        )
        model_sel = pn.widgets.Select.from_param(
            self.param.selected_model, name="Model", width=280
        )
        spiderweb_plot_sel = pn.widgets.Select.from_param(
            self.param.spiderweb_plot_mode, name="Spiderweb Plot", width=280
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
            """,
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
            sort_metric_sel,
            "**Direction**",
            mode_sel,
            alt_toggle,
            alt_metric_panel,
            alt_sort_metric_panel,
            _selected_primary_button(
                pn.widgets.Toggle.from_param(
                    self.param.show_n, name="Show n= in heatmap", width=280
                )
            ),
            _selected_primary_button(
                pn.widgets.Toggle.from_param(
                    self.param.show_rank, name="Show rank in heatmap", width=280
                )
            ),
            spiderweb_plot_sel,
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
            (_MAIN_TAB_TITLES[1], pn.panel(self._parallel_coordinates_view)),
            (_MAIN_TAB_TITLES[2], pn.panel(self._spiderweb_view)),
            (_MAIN_TAB_TITLES[3], pn.panel(self._bar_chart_view)),
            (_MAIN_TAB_TITLES[4], pn.panel(self._hp_impact_view)),
            (_MAIN_TAB_TITLES[5], pn.panel(self._summary_view)),
            (_MAIN_TAB_TITLES[6], pn.panel(self._stats_view)),
            (_MAIN_TAB_TITLES[7], pn.panel(self._latex_table_view)),
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

        template = pn.template.FastListTemplate(
            title="piCID Experiment Dashboard",
            sidebar=[sidebar],
            main=[tabs],
            accent="#0072B2",
        )

        if self._alias_warnings:
            rows = "\n".join(
                f"| `{old}` | `{new}` |" for old, new in self._alias_warnings
            )
            modal_content = pn.pane.Markdown(
                "## ⚠️ Model Name Aliases Applied\n\n"
                "Both old and new module paths were found simultaneously in the loaded data. "
                "Runs under the **new name** were merged into the **old name** "
                "(old data takes priority on conflict).\n\n"
                "| Canonical name (old, kept) | Merged away (new) |\n"
                "|---|---|\n" + rows,
                width=700,
            )
            template.modal.append(pn.Column(modal_content, width=720))
            pn.state.onload(template.open_modal)

        return template


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
