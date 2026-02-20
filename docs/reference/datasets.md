# Datasets

In this project we follow the **PyTorch** notion of a *Dataset*: an abstract class that defines how data is accessed and transformed for training and evaluation. Datasets are **task-centric** — they format data (from [datasources](datasources.md) and [data objects](dataobject.md)) into the shapes and keys expected by specific model types (e.g. sliding windows for RUL, fit-predict tables for AutoGluon).

> **Terminology:** A **datasource** is *data-centric* (raw files, I/O). A **dataset** is *task-centric* (formatting for model inputs).

All dataset classes below live in **`picid.data.datasets`**. The following sections describe the main dataset types and their batch structures.

## ContextBatchDataset

`ContextBatchDataset` is designed for time-series and sequence tasks where both **target** (e.g. RUL or forecast target) and **context** (e.g. sensor features) are needed. It wraps two `SlidingWindowBatchDataset` instances—one for the target and one for the context—with shared windowing (sequence length, label length, prediction length, stride). The *keys* inside `target` and `context` depend on the `data_dict` passed at construction: they are the modality name plus the suffixes `_seq_x` (input window) and `_seq_y` (output window). For example, if the context key is `"features"` and the target key is `"rul"`, you get `context.features_seq_x`, `context.features_seq_y`, `target.rul_seq_x`, `target.rul_seq_y`.

### Batch contents

Each batch is an `AttributeDict` with this structure:

```
{
	'batch_idx': [indices],
	'target': {
		'<target_key>_seq_x': Tensor (B, seq_len, C),   # target input window
		'<target_key>_seq_y': Tensor (B, label_len+pred_len, C)  # target output window
	},
	'context': {
		'<context_key>_seq_x': Tensor (B, seq_len, C),  # context input window
		'<context_key>_seq_y': Tensor (B, label_len+pred_len, C)
	}
}
```

- **`batch_idx`:** List of indices for the batch.
- **`target`:** Input/output windows for the target modality. Typical keys: `rul_seq_x`, `rul_seq_y` (RUL) or `target_seq_x`, `target_seq_y` (forecasting). Shapes: `_seq_x` is `(B, seq_len, C)`, `_seq_y` is `(B, label_len+pred_len, C)`.
- **`context`:** Input/output windows for the context modality. Typical key: `features_seq_x`, `features_seq_y`. Same shape convention as `target`.

So for RUL, the model typically reads `batch.context.features_seq_x` as input and `batch.target.rul_seq_x[:, -1, :]` (or the last step) as the scalar RUL target. For forecasting, `context.features_seq_x` is the history and `target.target_seq_y` is the future horizon to predict.


## FitPredictDataset

`FitPredictTaskDataset` is designed for Sklearn-like fit-predict tasks involving multiple tasks, each represented by 3D arrays of context features and targets. 
Handles multi-task datasets with context (`X`) and target (`y`) arrays, each shaped as (n_tasks, n_samples, n_features/targets). For example, in multi-step forecasting, each task could correspond to predicting future values for a separate time horizon.

### Batch Contents

Each batch from `FitPredictTaskDataset` is returned as an `AttributeDict` with the following structure:

```
{
	'task_idx': <index of the task>,
	'task_num': <total number of tasks>,
	'task_desc': <description string>,
	'target': <target array for the task>,
	'context': <context array for the task>
}
```

- `task_idx`: Index of the current task in the batch.
- `task_num`: Total number of tasks in the dataset.
- `task_desc`: Description string for the task (e.g., "Task 1 of 5").
- `target`: Target values for the task, shape (n_samples, n_targets).
- `context`: Context features for the task, shape (n_samples, n_features).

This structure allows models to access both the context and target data for each task, supporting multi-task learning and fit-predict workflows.


## ConceptRULDataset

`ConceptRULDataset` is designed for Remaining Useful Life (RUL) prediction tasks, where each sample represents a time window of features and associated concepts. It supports windowed access to time series data, with options for stride and backward/forward filling to handle sequence boundaries.

- Handles time series data for RUL prediction, with windowing and stride.
- Supports additional descriptors, concepts, health states, and timestamps.
- Concatenates descriptors to features if present.

This class is useful for predictive maintenance and prognostics, where the goal is to estimate the remaining life of an asset based on sensor data and domain concepts.

### Batch Contents

Each batch from `ConceptRULDataset` is returned as an `AttributeDict` with the following structure:

```
{
	'features': <windowed feature array, shape (n_features, window_size)>,
	'rul': <remaining useful life value for the window>,
	'concepts': <concept vector for the window>
}
```

- `features`: Windowed feature data, transposed to (n_features, window_size).
- `rul`: Remaining useful life value for the current window.
- `concepts`: Concept vector associated with the current window.

This structure enables models to learn from both raw features and high-level concepts for each time window, supporting robust RUL prediction.

## RULContextBatchDataset

`RULContextBatchDataset` is a specialized version of `ContextBatchDataset` designed specifically for **Remaining Useful Life (RUL) prediction**. It structures time-series data into fixed-length sequences where a window of operational features is used to predict the RUL value at the very end of that same window.

-   **Inherits from `ContextBatchDataset`**: It uses the same underlying mechanism for handling target and context features.
-   **Sliding Window Processing**: It moves a sliding window of a defined length (`seq_len`) over the time-series to generate samples.
-   **RUL-Specific Target**: The core feature is that the model is trained to predict the RUL at the final time step of the input sequence. This is achieved by internally forcing the prediction length (`pred_len`) to 0, which aligns the target window perfectly with the input window.

***

### Batch Contents

Since it inherits from `ContextBatchDataset`, each batch is returned as an `AttributeDict` with a similar structure. For RUL, the operational **features** are typically treated as the `context` and the **RUL values** as the `target`.

`RULContextBatchDataset` **overrides** `__getitem__` and returns a **flattened** batch (no nested `target`/`context`). You get top-level keys:

- **`batch_idx`:** Tensor of shape `(B, 1)` (batch indices).
- **`features`:** Tensor of shape `(B, seq_len, C)` — the context input (sensor window). This is the primary model input.
- **`rul`:** Tensor of shape `(B, C)` (typically `(B, 1)`) — the RUL value at the **last** time step of each window. The dataset already selects `target['rul_seq_x'][:, -1, :]` for you.
- **`unit_id`:** (optional) Present if the dataset was constructed with `get_unit_id=True`; used for per-unit evaluation.

So for RUL you do **not** need to read `target.rul_seq_x`; use `batch['rul']` and `batch['features']` directly.

***

### The Role of `pred_len=0`

Setting **`pred_len=0`** is the key mechanism that configures the dataset for this specific type of RUL task.

The `pred_len` (prediction length) parameter is designed for forecasting tasks. It tells the dataset how many steps into the **future**, beyond the end of the input sequence, the target value lies.

-   **Forecasting (`pred_len > 0`)**: If your input sequence ends at time `t`, setting `pred_len=5` means your target is the value at time `t+5`. You are predicting 5 steps *after* your input data ends.
-   **RUL Prediction (`pred_len = 0`)**: By setting `pred_len=0`, you specify that there is **no future horizon**. The target you want to predict exists at the exact same time step where your input sequence ends (time `t`).

This is why it's the correct setting for this task: it ensures the target RUL value is perfectly aligned with the end of the feature sequence you are feeding into the model.

---

## See also

- [Datasets inner workings](datasets_inner_workings.md) — Full data-loading architecture: `HydraConcatDataset`, `SlidingWindowBatchDataset`, sequencers (Dense/Ragged), and collation.
- [Data objects & containers](dataobject.md) — How datasource output becomes `SplitDatasetContainer` and feeds into these datasets.
- [Datasources](datasources.md) — Available data loaders by domain.
