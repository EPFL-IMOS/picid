# Metrics and MetricManager

`MetricManager` orchestrates metric objects per task.

Metric families include:

- regression (`MAE`, `RMSE`, etc.)
- classification metrics
- PHM-specific metrics (for example RUL scores)

Metrics are computed from evaluator-updated buffers and returned as logging-ready dicts.
