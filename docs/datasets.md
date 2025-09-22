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