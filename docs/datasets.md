# Datasets

In this project, we keep the notation of a dataset from Pytorch.
A PyTorch Dataset is an abstract class that provides a way to represent and load data for machine learning tasks. It allows you to define how data is accessed and transformed, enabling efficient iteration and preprocessing for training models.

> Therefore, in this project, a datasource is data-centric, a dataset is therefore task-centric.

We provide several datasets for different tasks

## ContextBatchDataset

`ContextBatchDataset` is a dataset class designed for time series and sequence forecasting tasks where both target and context features are needed. It wraps two `SlidingWindowBatchDataset` instances—one for the target and one for the context—using shared time features and windowing parameters (sequence length, label length, prediction length, stride).


### Batch Contents

Each batch from `ContextBatchDataset` is returned as an `AttributeDict` with the following structure:

```
{
	'batch_idx': [indices],
	'target': {
		'x': <target input sequence>,
		'y': <target output sequence>,
		'x_mark': <target input time features>,
		'y_mark': <target output time features>
	},
	'context': {
		'x': <context input sequence>,
		'y': <context output sequence>,
		'x_mark': <context input time features>,
		'y_mark': <context output time features>
	}
}
```

- `batch_idx`: List of indices for the batch.
- `target`: Contains input/output sequences and time features for the target.
- `context`: Contains input/output sequences and time features for the context.


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

-   `context['features_seq_x']`: The primary input for the model. This is a batch of `B` feature sequences, each with shape `(seq_len, C)`, where `C` is the number of features.
-   `target['rul_seq_x']`: The sequence of RUL values corresponding to the input. This sequence aligned with the `context['features_seq_x']`.
-   **Final RUL Target**: 💡 To get the single scalar RUL value to predict for each item in the batch, you would typically select the **very last element** from the `target['rul_seq_x']` tensor (e.g., `target['rul_seq_x'][:, -1, :]`). This represents the RUL at the end of the `seq_len` window.

The output batch is: 
```
{
	'batch_idx': [indices],
	'rul': target['rul_seq_x'][:, -1, :], shape (B, 1)>
	'features': context['features_seq_x'], shape (B, seq_len, C)
}
```

***

### The Role of `pred_len=0`

Setting **`pred_len=0`** is the key mechanism that configures the dataset for this specific type of RUL task.

The `pred_len` (prediction length) parameter is designed for forecasting tasks. It tells the dataset how many steps into the **future**, beyond the end of the input sequence, the target value lies.

-   **Forecasting (`pred_len > 0`)**: If your input sequence ends at time `t`, setting `pred_len=5` means your target is the value at time `t+5`. You are predicting 5 steps *after* your input data ends.
-   **RUL Prediction (`pred_len = 0`)**: By setting `pred_len=0`, you specify that there is **no future horizon**. The target you want to predict exists at the exact same time step where your input sequence ends (time `t`).

This is why it's the correct setting for this task: it ensures the target RUL value is perfectly aligned with the end of the feature sequence you are feeding into the model. ⚙️

