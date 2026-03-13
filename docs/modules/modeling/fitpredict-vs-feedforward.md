# Fit-Predict vs Feed-Forward

## Fit-Predict

- task-wise arrays
- explicit `fit(X, y)` / `predict(X)` lifecycle
- used by wrappers like XGBoost/TabPFN/TabDPT

## Feed-Forward

- minibatch windowed tensors
- trained/evaluated through Lightning step hooks
- used by LSTM/CNN/PatchTST-style wrappers
