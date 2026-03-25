"""
Sort/rank metrics configuration.

Defines which metric to use for sorting/ranking/selecting the best model.
This is separate from display metrics (which can show many metrics).

Resolution order in get_sort_metric: (1) SORT_METRIC_OVERRIDES[(dataset, model)],
(2) SORT_METRIC_BY_TASK_TYPE[task_type], (3) SORT_METRIC_BY_DATASET_CATEGORY[category],
(4) DEFAULT_SORT_METRIC.
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Global default: use validation metric for all (when task type/category not found)
DEFAULT_SORT_METRIC: str = "val_best_rerun/loss"

# Task-type defaults: sort by validation (val_best_rerun/loss) for all, for now
SORT_METRIC_BY_TASK_TYPE: Dict[str, str] = {
    "regression": "val_best_rerun/loss",
    "rul": "val_best_rerun/loss",
    "classification": "val_best_rerun/loss",
    "fault_classification": "val_best_rerun/loss",
    "forecasting": "val_best_rerun/loss",
}

# Dataset category defaults (fallback): validation metric for all
SORT_METRIC_BY_DATASET_CATEGORY: Dict[str, str] = {
    "prognostics": "val_best_rerun/loss",
    "diagnostics": "val_best_rerun/loss",
}

# Specific overrides: key is (dataset: str, model: str) -> metric name. Only specify when different from defaults.
SORT_METRIC_OVERRIDES: Dict[tuple, str] = {
    # Example: ("nb14", "baselines.lstm_model.LSTM_Forecaster"): "test/mae",
    # Add only exceptions
}


DEFAULT_MODELS_WITHOUT_RERUN = [
    "model.wrappers.fit_predict_xgboost_wrapper.FitPredictXGBoostWrapper",
    "model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper",
    "model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper",
]

DEFAULT_MODELS_WITHOUT_RERUN_LOSS_REPLACEMENT_KEY = "mse_denormalized_mean"


def get_sort_metric(
    dataset: str,
    model: str,
    task_type: Optional[str] = None,
    dataset_category: Optional[str] = None,
    fallback_to_optimization: bool = True,
) -> Optional[str]:
    """
    Resolve sort/rank metric with fallback hierarchy:
    1. Specific override (dataset, model)
    2. Task type default
    3. Dataset category default
    4. Global default (DEFAULT_SORT_METRIC = "val_best_rerun/loss")

    Note: The fallback_to_optimization parameter is kept for backward compatibility
    but is no longer used - we always return DEFAULT_SORT_METRIC if no match is found.

    Parameters
    ----------
    dataset : str
        Dataset name
    model : str
        Model name/identifier
    task_type : Optional[str]
        Task type (e.g., "regression", "classification")
    dataset_category : Optional[str]
        Dataset category (e.g., "prognostics", "diagnostics")
    fallback_to_optimization : bool
        If True, returns None to signal using optimization metric

    Returns
    -------
    Optional[str]
        Metric name (e.g., "val_best_rerun/loss") or None to use optimization metric
    """
    # 1. Check specific override
    key = (dataset, model)
    out = DEFAULT_SORT_METRIC

    if key in SORT_METRIC_OVERRIDES:
        out = SORT_METRIC_OVERRIDES[key]
        logger.debug(
            "get_sort_metric(%s, %s, task_type=%s, category=%s) -> override %s",
            dataset,
            model,
            task_type,
            dataset_category,
            out,
        )
    # 2. Check task type default
    elif task_type and task_type in SORT_METRIC_BY_TASK_TYPE:
        out = SORT_METRIC_BY_TASK_TYPE[task_type]
        logger.debug(
            "get_sort_metric(%s, %s, ...) -> task_type %s -> %s",
            dataset,
            model,
            task_type,
            out,
        )
    # 3. Check dataset category default
    elif dataset_category and dataset_category in SORT_METRIC_BY_DATASET_CATEGORY:
        out = SORT_METRIC_BY_DATASET_CATEGORY[dataset_category]
        logger.debug(
            "get_sort_metric(%s, %s, ...) -> category %s -> %s",
            dataset,
            model,
            dataset_category,
            out,
        )
    else:
        # 4. Use global default (val_best_rerun/loss) when inference fails
        logger.debug(
            "get_sort_metric(%s, %s, ...) -> default %s",
            dataset,
            model,
            out,
        )

    if model in DEFAULT_MODELS_WITHOUT_RERUN:
        out = out.replace("_best_rerun", "")
        out = out.replace("loss", DEFAULT_MODELS_WITHOUT_RERUN_LOSS_REPLACEMENT_KEY)

    return out


def infer_task_type_from_dataset(dataset: str) -> Optional[str]:
    """
    Infer task type from dataset name (heuristic).

    This is a simple heuristic - can be improved with actual dataset metadata.

    Parameters
    ----------
    dataset : str
        Dataset name

    Returns
    -------
    Optional[str]
        Inferred task type or None
    """
    dataset_lower = dataset.lower()

    # Classification datasets (heuristic)
    if any(
        keyword in dataset_lower
        for keyword in ["mzvav", "hsf15", "fault", "diagnostic"]
    ):
        return "classification"

    # Regression/RUL datasets (heuristic)
    if any(
        keyword in dataset_lower
        for keyword in [
            "nb14",
            "unibo",
            "pronostia",
            "xjtu",
            "cmapss",
            "rul",
            "unibo21",
        ]
    ):
        return "regression"

    return None


def infer_dataset_category_from_name(dataset: str) -> Optional[str]:
    """
    Infer dataset category from dataset name (heuristic).

    Parameters
    ----------
    dataset : str
        Dataset name

    Returns
    -------
    Optional[str]
        "prognostics" or "diagnostics" or None
    """
    dataset_lower = dataset.lower()

    # Diagnostics datasets (heuristic)
    if any(
        keyword in dataset_lower
        for keyword in ["mzvav", "hsf15", "fault", "diagnostic"]
    ):
        return "diagnostics"

    # Prognostics datasets (heuristic)
    if any(
        keyword in dataset_lower
        for keyword in [
            "nb14",
            "unibo",
            "pronostia",
            "xjtu",
            "cmapss",
            "phme",
            "unibo21",
        ]
    ):
        return "prognostics"

    return None
