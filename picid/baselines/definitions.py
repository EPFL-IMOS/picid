"""Legacy compatibility re-export for canonical model task definitions."""

from picid.model.definitions import (
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    FORECASTING_TASKS,
    REGRESSION_TASKS,
    STATE_FORECASTING_TASKS,
)

__all__ = [
    "REGRESSION_TASKS",
    "CLASSIFICATION_TASKS",
    "FORECASTING_TASKS",
    "STATE_FORECASTING_TASKS",
    "ALL_TASKS",
]
