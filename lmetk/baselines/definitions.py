# Define task categories for baseline models
# x_c: past contextual features, y_c: contextual (past) targets
# x_t: future contextual features, y_t: target labels to predict

REGRESSION_TASKS = ["regression", "rul", "ahrul", "soc"]  # regression from x_t -> y_t
CLASSIFICATION_TASKS = [
    "classification",
    "health_state",
    "fault_classification",
]  # binary/multi-class classification x_t -> y_t
FORECASTING_TASKS = ["forecasting"]  # contextual forecasting x_c, y_c, x_t -> y_t
STATE_FORECASTING_TASKS = ["state_forecasting"]  # non-contextual forecasting y_c -> y_t

ALL_TASKS = [
    *REGRESSION_TASKS,
    *CLASSIFICATION_TASKS,
    *FORECASTING_TASKS,
    *STATE_FORECASTING_TASKS,
]
