# Task Formats: RUL, Forecasting, Fit-Predict

PICID provides multiple dataset contracts depending on model regime.

## RULContextBatchDataset

- Uses windowed context features.
- Forces `pred_len = 0` for aligned end-of-window labels.
- Returns flattened keys (`features`, `rul`, optional `unit_id`).

## ContextBatchDataset (forecasting-like)

- Returns nested `context` and `target` blocks.
- Uses `_seq_x` and `_seq_y` key conventions.
- Supports non-zero prediction horizon.

## FitPredictTaskDataset

- Returns task-wise arrays for fit-predict wrappers.
- Intended for models with explicit `fit(X, y)` / `predict(X)`.

These contracts should be treated as stable extension points for dataset consumers.
