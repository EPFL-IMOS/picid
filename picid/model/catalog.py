"""Canonical inventory of public model configs and their capabilities."""

from typing import Literal, TypedDict


TaskName = Literal["regression", "classification", "forecasting"]
ExecutionStyle = Literal[
    "fit_predict_wrapper",
    "feed_forward_wrapper",
    "forecasting_wrapper",
    "lightning_forecaster",
    "sklearn_estimator",
    "legacy_forecasting",
]


class ModelCatalogEntry(TypedDict):
    """Serializable description of one public model config."""

    config_name: str
    class_path: str
    family: str
    execution_style: ExecutionStyle
    tasks: tuple[TaskName, ...]
    notes: str


LEGACY_MODEL_CONFIG_NAMES = frozenset({"persistence", "similar_period"})


MODEL_CATALOG: tuple[ModelCatalogEntry, ...] = (
    {
        "config_name": "carte_fit_predict",
        "class_path": "picid.model.estimators.carte.wrapper.FitPredictCarteWrapper",
        "family": "carte",
        "execution_style": "fit_predict_wrapper",
        "tasks": ("regression", "classification"),
        "notes": "CARTE-based tabular wrapper with text-aware preprocessing.",
    },
    {
        "config_name": "catboost",
        "class_path": "catboost.CatBoostRegressor",
        "family": "catboost",
        "execution_style": "sklearn_estimator",
        "tasks": ("regression",),
        "notes": "Direct CatBoost regressor target; forecasting is not part of this config.",
    },
    {
        "config_name": "cnn_1d",
        "class_path": "picid.model.estimators.cnn1d.wrapper.CNN1D_Wrapper",
        "family": "cnn1d",
        "execution_style": "feed_forward_wrapper",
        "tasks": ("regression", "classification"),
        "notes": "1D CNN encoder wrapped for supervised PHM tasks.",
    },
    {
        "config_name": "crossformer",
        "class_path": "picid.model.forecasters.crossformer_model.Crossformer_Forecaster",
        "family": "crossformer",
        "execution_style": "lightning_forecaster",
        "tasks": ("regression", "classification", "forecasting"),
        "notes": "Transformer forecaster with decoder-style task support.",
    },
    {
        "config_name": "drift",
        "class_path": "picid.model.estimators.drift.wrapper.DriftModelWrapper",
        "family": "drift",
        "execution_style": "forecasting_wrapper",
        "tasks": ("forecasting",),
        "notes": "Simple trend-following baseline for forecasting only.",
    },
    {
        "config_name": "exponential_regression",
        "class_path": "picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper",
        "family": "linear_regression",
        "execution_style": "forecasting_wrapper",
        "tasks": ("forecasting",),
        "notes": "Statistical baseline alias with exponential growth mode.",
    },
    {
        "config_name": "isolation_forest_fit_predict",
        "class_path": "picid.model.estimators.isolation_forest.wrapper.FitPredictIsolationForestWrapper",
        "family": "isolation_forest",
        "execution_style": "fit_predict_wrapper",
        "tasks": ("classification",),
        "notes": "Anomaly detection mapped into the classification family.",
    },
    {
        "config_name": "linear_forecaster",
        "class_path": "picid.model.forecasters.linear_model.linear_model.Linear_Forecaster",
        "family": "linear_forecaster",
        "execution_style": "lightning_forecaster",
        "tasks": ("forecasting",),
        "notes": "Minimal linear forecaster for time-series prediction.",
    },
    {
        "config_name": "linear_regression",
        "class_path": "picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper",
        "family": "linear_regression",
        "execution_style": "feed_forward_wrapper",
        "tasks": ("regression", "classification"),
        "notes": "Flattened statistical baseline for supervised tasks.",
    },
    {
        "config_name": "lstm",
        "class_path": "picid.model.forecasters.lstm_model.LSTM_Forecaster",
        "family": "lstm",
        "execution_style": "lightning_forecaster",
        "tasks": ("regression", "classification", "forecasting"),
        "notes": "LSTM forecaster/prognostics model; also used for state forecasting.",
    },
    {
        "config_name": "mean",
        "class_path": "picid.model.estimators.window_average.wrapper.WindowAverageWrapper",
        "family": "window_average",
        "execution_style": "forecasting_wrapper",
        "tasks": ("forecasting",),
        "notes": "Window-average alias; canonical family is window_average.",
    },
    {
        "config_name": "mlp",
        "class_path": "picid.model.estimators.mlp.wrapper.MLPWrapper",
        "family": "mlp",
        "execution_style": "feed_forward_wrapper",
        "tasks": ("regression", "classification"),
        "notes": "Fully connected baseline for supervised PHM tasks.",
    },
    {
        "config_name": "naive",
        "class_path": "picid.model.estimators.window_average.wrapper.WindowAverageWrapper",
        "family": "window_average",
        "execution_style": "forecasting_wrapper",
        "tasks": ("forecasting",),
        "notes": "Last-value alias; canonical family is window_average.",
    },
    {
        "config_name": "patchtst",
        "class_path": "picid.model.forecasters.patchtst_model.PatchTST_Forecaster",
        "family": "patchtst",
        "execution_style": "lightning_forecaster",
        "tasks": ("regression", "classification", "forecasting"),
        "notes": "PatchTST forecaster with regression/classification support.",
    },
    {
        "config_name": "polynomial_regression",
        "class_path": "picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper",
        "family": "linear_regression",
        "execution_style": "forecasting_wrapper",
        "tasks": ("forecasting",),
        "notes": "Statistical baseline alias with polynomial growth mode.",
    },
    {
        "config_name": "ses",
        "class_path": "picid.model.estimators.ses.wrapper.SESModelWrapper",
        "family": "ses",
        "execution_style": "forecasting_wrapper",
        "tasks": ("forecasting",),
        "notes": "Simple exponential smoothing baseline for forecasting only.",
    },
    {
        "config_name": "stf",
        "class_path": "picid.model.forecasters.spacetimeformer_model.Spacetimeformer_Forecaster",
        "family": "spacetimeformer",
        "execution_style": "lightning_forecaster",
        "tasks": ("regression", "classification", "forecasting"),
        "notes": "Spacetimeformer supports forecasting plus decoder-style tasks.",
    },
    {
        "config_name": "tabdpt_fit_predict",
        "class_path": "picid.model.estimators.tabdpt.wrapper.FitPredictTabDPTWrapper",
        "family": "tabdpt",
        "execution_style": "fit_predict_wrapper",
        "tasks": ("regression", "classification"),
        "notes": "TabDPT tabular wrapper for regression and classification.",
    },
    {
        "config_name": "tabpfn_fit_predict",
        "class_path": "picid.model.estimators.tabpfn.wrapper.FitPredictTabPFNWrapper",
        "family": "tabpfn",
        "execution_style": "fit_predict_wrapper",
        "tasks": ("regression", "classification"),
        "notes": "TabPFN tabular wrapper for regression and classification.",
    },
    {
        "config_name": "tide",
        "class_path": "picid.model.forecasters.tide_model.TiDE_Forecaster",
        "family": "tide",
        "execution_style": "lightning_forecaster",
        "tasks": ("regression", "classification", "forecasting"),
        "notes": "TiDE forecaster; also supports state forecasting in code paths.",
    },
    {
        "config_name": "timeseries_transformer",
        "class_path": "picid.model.forecasters.timeseries_transformer_model.Timeseries_Transformer_Forecaster",
        "family": "timeseries_transformer",
        "execution_style": "lightning_forecaster",
        "tasks": ("regression", "classification", "forecasting"),
        "notes": "Vanilla Transformer forecaster with regression/classification support.",
    },
    {
        "config_name": "xgboost_fit_predict",
        "class_path": "picid.model.estimators.xgboost.wrapper.FitPredictXGBoostWrapper",
        "family": "xgboost",
        "execution_style": "fit_predict_wrapper",
        "tasks": ("regression", "classification"),
        "notes": "Gradient-boosted sklearn baseline for tabular tasks.",
    },
)

_TABLE_HEADER = (
    "| Config | Canonical Class | Family | Execution Style | Regression | "
    "Classification | Forecasting | Notes |"
)
_TABLE_SEPARATOR = "| --- | --- | --- | --- | --- | --- | --- | --- |"


def _yes_no(entry: ModelCatalogEntry, task: TaskName) -> str:
    """
    Return a markdown-friendly yes/no flag for one task capability.

    Parameters
    ----------
    entry : ModelCatalogEntry
        Catalog row to inspect.
    task : TaskName
        Capability name to check.

    Returns
    -------
    str
        ``"yes"`` when the task is supported, otherwise ``"no"``.
    """

    return "yes" if task in entry["tasks"] else "no"


def render_model_capabilities_table() -> str:
    """
    Render the markdown table used in the contributor docs page.

    Returns
    -------
    str
        Markdown table generated from ``MODEL_CATALOG``.
    """

    rows = [_TABLE_HEADER, _TABLE_SEPARATOR]
    for entry in MODEL_CATALOG:
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{entry['config_name']}`",
                    f"`{entry['class_path']}`",
                    f"`{entry['family']}`",
                    f"`{entry['execution_style']}`",
                    _yes_no(entry, "regression"),
                    _yes_no(entry, "classification"),
                    _yes_no(entry, "forecasting"),
                    entry["notes"],
                ]
            )
            + " |"
        )
    return "\n".join(rows)


__all__ = [
    "ExecutionStyle",
    "LEGACY_MODEL_CONFIG_NAMES",
    "MODEL_CATALOG",
    "ModelCatalogEntry",
    "TaskName",
    "render_model_capabilities_table",
]
