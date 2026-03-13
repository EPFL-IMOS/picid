# Forecasting Fit-Predict Tutorial

Fit-predict wrappers operate task-wise (full arrays per task):

1. Dataset formats context/target arrays per task.
2. Wrapper calls `fit(X, y)` on train task slices.
3. Wrapper calls `predict(X)` on validation/test slices.

See:

- [Fit-predict vs feed-forward](../modules/modeling/fitpredict-vs-feedforward.md)
- [Task-centric dataset formats](../modules/data/datasets/task-formats-rul-forecast-fitpredict.md)
