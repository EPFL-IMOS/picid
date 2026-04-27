"""Canonical task-family definitions shared across model code."""

# x_c: past contextual features, y_c: contextual (past) targets
# x_t: future contextual features, y_t: target labels to predict

REGRESSION_TASKS = ["regression", "rul", "ahrul", "soc"]
CLASSIFICATION_TASKS = [
    "classification",
    "health_states",
    "concepts",
    "fault_classification",
    "anomaly_detection",
]
FORECASTING_TASKS = ["forecasting"]
STATE_FORECASTING_TASKS = ["state_forecasting"]

ALL_TASKS = [
    *REGRESSION_TASKS,
    *CLASSIFICATION_TASKS,
    *FORECASTING_TASKS,
    *STATE_FORECASTING_TASKS,
]

__all__ = [
    "REGRESSION_TASKS",
    "CLASSIFICATION_TASKS",
    "FORECASTING_TASKS",
    "STATE_FORECASTING_TASKS",
    "ALL_TASKS",
]
