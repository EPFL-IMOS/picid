# Evaluators

This document describes the evaluator subsystem in `lmetk.evaluator`: the abstract base, scaling utilities, buffer, hooks, metric management, and all concrete evaluators. For each component we explain **purpose**, **architecture**, **attributes**, **methods** (signatures, inputs, outputs), and **logic** in explicit form.

---

## Table of contents

1. [Overview and architecture](#overview-and-architecture)
2. [Task types](#task-types)
3. [AbstractEvaluator (base)](#abstractevaluator-base)
4. [Scaling wrappers](#scaling-wrappers)
5. [PredictionBuffer](#predictionbuffer)
6. [MetricManager](#metricmanager)
7. [Evaluation hooks](#evaluation-hooks)
8. [DefaultEvaluator](#defaultevaluator)
9. [ClassificationEvaluator](#classificationevaluator)
10. [ForecastingEvaluator](#forecastingevaluator)
11. [MultiUnitEvaluator](#multiunitevaluator)
12. [ReconstructionEvaluator](#reconstructionevaluator)
13. [Utils](#utils)
14. [Configuration examples](#configuration-examples)
15. [Summary table](#summary-table)

---

## Overview and architecture

**Location:** `lmetk/evaluator/`

**Role:** Evaluators consume model outputs (predictions and targets), optionally apply inverse scaling via the scaling wrapper, update internal metrics via a `MetricManager`, accumulate data in a `PredictionBuffer` for saving and plotting, and invoke pluggable **hooks** for side effects (saving to disk, logging plots). At the end of an evaluation phase, they return a dictionary of metric values.

**Architecture (post-refactor):**

- **AbstractEvaluator** (ABC) — selects scaling strategy (univariate vs multivariate), defines `update`, `compute`, `reset`.
- **DefaultEvaluator** — orchestrates a `PredictionBuffer`, `MetricManager`, and pluggable `BaseEvalHook` instances. Supports dual reporting (normalized + denormalized) when inverse scaling is enabled.
- **ClassificationEvaluator** — thin wrapper over `DefaultEvaluator` with `task_type="classification"` and required `num_classes`.
- **ForecastingEvaluator** — extends `DefaultEvaluator` with slicing by `effective_pred_len` and optional `target_dim_position` for univariate forecasting.
- **MultiUnitEvaluator** — extends `DefaultEvaluator`; maintains per-unit metrics via separate `MetricManager` instances per unit; requires `unit_id` in `model_out`.
- **ReconstructionEvaluator** — extends `DefaultEvaluator` with `task_type="regression"`; reconstruction plots and saving are handled by hooks.

**Data flow (typical):**

1. Training/validation loop calls `evaluator.update(model_out)` for each batch. `model_out` contains at least `"predictions"` and `"targets"`; `MultiUnitEvaluator` also requires `"unit_id"`.
2. `DefaultEvaluator.update` validates input, prepares batch data (scaling, task branching), accumulates into the buffer (if `collect_predictions`), updates the `MetricManager`, and runs `on_update_end` on each hook.
3. At end of epoch/split, the loop calls `evaluator.compute(mode, epoch, step)`, which computes metrics via `MetricManager`, runs `on_compute_end` on each hook (e.g. save predictions, log plots), and returns a `Dict[str, float]`.

**Shape conventions:**

- **Regression / forecasting:** predictions and targets are 3D: `(N, T, 1)`. `N` = batch size, `T` = time steps (often 1 for point-wise RUL).
- **Classification:** predictions `(N, T, C)` (logits), targets `(N, T, 1)` (integer class indices).
- **Single-target:** targets always have last dimension 1 in the evaluator contract.

---

## Task types

Task types are defined in `lmetk.baselines.definitions` and used by evaluators to branch logic (scaling, metric factory):

| Constant | Values |
|----------|--------|
| `REGRESSION_TASKS` | `["regression", "rul", "ahrul", "soc"]` |
| `CLASSIFICATION_TASKS` | `["classification", "health_states", "concepts", "fault_classification"]` |
| `FORECASTING_TASKS` | `["forecasting"]` |
| `STATE_FORECASTING_TASKS` | `["state_forecasting"]` |
| `ALL_TASKS` | Union of all above |

- **DefaultEvaluator** accepts any `task_type` in `ALL_TASKS`.
- **ClassificationEvaluator** fixes `task_type="classification"`.
- **ForecastingEvaluator** fixes `task_type="forecasting"` and slices predictions/targets by length and optionally by target dimension.
- **MultiUnitEvaluator** accepts any `task_type` passed to `DefaultEvaluator`; typical usage is regression (RUL) with per-unit metrics.
- **ReconstructionEvaluator** fixes `task_type="regression"`.

---

## AbstractEvaluator (base)

**Module:** `lmetk.evaluator.base`

**Purpose:** Abstract base for all evaluators. It selects the scaling strategy (univariate vs multivariate) and declares the interface: `update`, `compute`, `reset`.

### Constructor

```python
__init__(
    self,
    paths: Optional[Dict[str, str]] = None,
    inverse_transform: Optional[InverseTransformMixin] = None,
    apply_inverse_scaling: bool = False,
    task_mode: Optional[Any] = None,
    **kwargs,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `paths` | `Optional[Dict[str, str]]` | Path configuration (e.g. output dirs); stored as `self.paths`. May be a dict or an object with attributes like `eval_details`, `plot_dir`. |
| `inverse_transform` | `Optional[InverseTransformMixin]` | Scaler used for inverse transform; passed to the scaling wrapper. |
| `apply_inverse_scaling` | `bool` | Whether to apply inverse scaling when requested by the wrapper. |
| `task_mode` | `Optional[Any]` | Passed to the scaling wrapper; e.g. `"univariate"`, `"multivariate"`, `"forecasting"`. |
| `**kwargs` | | Additional arguments absorbed (no restriction). |

**Attributes set:**

- `self.scaling_wrapper` — If `task_mode == "multivariate"`, instance of `MultivariateTimeseriesScalingWrapper`; else instance of `ScalingWrapper`. Both are initialized with `inverse_transform`, `apply_inverse_scaling`, and `task_mode`.
- `self.task_mode` — The given `task_mode`.
- `self.paths` — The given `paths`.

### Abstract methods

- **`update(self, model_out: dict) -> None`** — Update internal state (buffer, metrics) with one batch. No return value.
- **`compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]`** — Compute and return metrics; may trigger hooks (saving, plotting).
- **`reset(self) -> None`** — Reset all accumulated state for reuse in another run/epoch.

### Other

- **`__repr__(self)`** — Returns a string representation using only “simple” attributes (int, float, str, bool, None).

---

## Scaling wrappers

**Module:** `lmetk.evaluator.scaling_wrapper`

Two wrappers delegate inverse scaling to an `InverseTransformMixin` instance and handle shape flattening/reshaping so the scaler sees 2D data.

### ScalingWrapper (univariate / standard)

**Purpose:** Used when `task_mode != "multivariate"`. Applies inverse transform per array while preserving 3D `(B, T, C)` or 2D shapes.

**Constructor:**

```python
__init__(
    self,
    inverse_transform: Optional[InverseTransformMixin] = None,
    apply_inverse_scaling: bool = False,
    task_mode: bool = False,
)
```

**Attributes:** `inverse_transform`, `apply_inverse`, `task_mode`.

**Method: `inverse_transform_if_needed(predictions, targets, **kwargs)`**

- **Inputs:**
  - `predictions`, `targets`: `torch.Tensor` or `np.ndarray`. Converted to numpy if needed.
  - `kwargs`: Optional `metadata` (e.g. `{"unit_id": array}`) passed to the scaler’s `inverse_transform`.
- **Behavior:**
  - If `inverse_transform` is None or `apply_inverse` is False, returns `(predictions, targets)` unchanged.
  - For **3D** `(B, T, C)`:
    - If `task_mode != "forecasting"`, asserts `T == 1`.
    - If `task_mode == "forecasting"`, asserts `C == 1`.
    - Flattens to `(B*T, C)` via `rearrange("b t c -> (b t) c")`, calls `inverse_transform` on `NamedTransformInput(predictions=...)` and `NamedTransformInput(targets=...)`, then reshapes back to original shape.
  - For **2D**, calls `inverse_transform` without reshaping.
- **Output:** `(predictions_transformed, targets_transformed)` same shape as input.

### MultivariateTimeseriesScalingWrapper

**Purpose:** Used when `task_mode == "multivariate"`. Constructor asserts `task_mode == "multivariate"`. Flattens 3D to 2D, applies inverse transform, then unflattens back to 3D. Intended for multivariate time series.

**Constructor:** Same signature as `ScalingWrapper`; asserts `task_mode == "multivariate"`.

**Method: `inverse_transform_if_needed(predictions, targets)`** — Does not accept `metadata` in the current implementation. Converts to numpy, flattens `(B, T, C)` to `(B*T, C)`, applies `inverse_transform`, then reshapes back to `(B, T, C)`.

---

## PredictionBuffer

**Module:** `lmetk.evaluator.buffer`

**Purpose:** Handles raw data accumulation for saving and plotting. Stores predictions, targets, optional normalized variants, and unit IDs.

### Constructor

```python
__init__(self) -> None
```

**Attributes:**

- `self.data: Dict[str, List[np.ndarray]]` — Keys: `"preds"`, `"targets"`, `"norm_preds"`, `"norm_targets"`, `"unit_ids"`. Each value is a list of numpy arrays appended per batch.

### Properties (backward compatibility)

- `self.preds` — `self.data["preds"]`
- `self.targets` — `self.data["targets"]`
- `self.unit_ids` — `self.data["unit_ids"]`

### Methods

- **`accumulate(self, batch: Dict[str, Any], unit_id: Optional[np.ndarray] = None) -> None`**  
  Appends `batch["preds"]` and `batch["targets"]` to internal lists. If `batch.get("is_dual")` is true, also appends `batch["norm_preds"]` and `batch["norm_targets"]`. If `unit_id` is provided, appends it to `unit_ids`.

- **`get_all(self) -> Dict[str, np.ndarray]`**  
  Concatenates all buffered lists along axis 0 and returns a dict. Returns `{}` if buffer is empty.

- **`clear(self) -> None`**  
  Clears all lists for a new epoch.

---

## MetricManager

**Module:** `lmetk.metrics.manager`

**Purpose:** Orchestrates multiple metric instances. Handles dual-scaling (normalized/denormalized) and task-specific aggregation. Used by evaluators to avoid direct metric management.

### Constructor

```python
__init__(
    self,
    metric_names: List[str],
    task_type: str,
    paths: Optional[Dict[str, str]] = None,
    num_classes: Optional[int] = None,
    is_dual: bool = False,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `metric_names` | `List[str]` | Non-empty list of metric names (e.g. `["mse", "mae"]`). Stored lowercased. |
| `task_type` | `str` | `"regression"` or `"classification"` (or other `ALL_TASKS` values). |
| `paths` | `Optional[Dict[str, str]]` | Required for file-based metrics (e.g. Railway metrics). |
| `num_classes` | `Optional[int]` | Required for classification metrics. |
| `is_dual` | `bool` | If True, tracks both normalized and denormalized metrics. |

### Methods

- **`update(predictions, targets, norm_preds=None, norm_targets=None) -> None`**  
  Updates all primary metrics with `predictions` and `targets`. If `is_dual` and `norm_preds`/`norm_targets` are provided, updates normalized metrics with them.

- **`compute() -> Dict[str, float]`**  
  For **classification**: returns `{name: metric.compute()}` (no suffix).  
  For **regression/PHM**: primary metrics use `_denormalized` suffix when `is_dual`; otherwise `_normalized`. If dual, normalized metrics use `_normalized` suffix.

- **`reset() -> None`**  
  Resets all metric states.

**Available metrics (via MetricFactory):**

- **Regression:** `mae`, `mse`, `rmse`, `mape`, `mspe`, `rse`, `corr`, `mae_railway`, `mse_railway`, `nasa_score`, `mpe`, `phm_score`
- **Classification:** `accuracy`, `precision`, `recall`, `f1`, `auroc`

---

## Evaluation hooks

**Module:** `lmetk.evaluator.hooks`

**Purpose:** Pluggable lifecycle hooks for side effects during evaluation (saving, plotting). Hooks are invoked by `DefaultEvaluator` after each batch (`on_update_end`) and after metrics are computed (`on_compute_end`).

### BaseEvalHook

**Module:** `lmetk.evaluator.hooks.base`

Abstract base for all evaluator hooks.

**Methods:**

- **`on_update_end(self, batch: Dict[str, Any], evaluator: Any) -> None`**  
  Called after each batch is processed. Optional override for streaming or partial accumulation. Default implementation is no-op.

- **`on_compute_end(self, results: Dict[str, float], evaluator: Any, mode: str, epoch: int, step: int) -> None`**  
  Abstract. Called after all metrics are computed. Primary entry point for saving and plotting.

### SavePredictionsHook

**Module:** `lmetk.evaluator.hooks.save_predictions`

**Purpose:** Handles NetCDF saving with standardized dimension naming. Resolves dimension size conflicts (e.g. classification: preds `(N, T, C)` vs targets `(N, T, 1)`).

**Constructor:**

```python
__init__(self, dims: Optional[List[str]] = None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dims` | `Optional[List[str]]` | Dimension names for 3D tensors. Required; typically `["sample", "time", "feature"]`. Must have exactly 3 elements. |

**Behavior:** In `on_compute_end`, runs only if `evaluator.save_predictions` is True and the buffer has data. Formats data via `_format_xarray`, writes to `evaluator.paths.eval_details / mode / "predictions.nc"`. Maps internal `norm_preds`/`norm_targets` to `preds_normalized`/`targets_normalized` in the output. Handles `unit_ids` if present.

### ReconstructionPlotHook

**Module:** `lmetk.evaluator.hooks.reconstruction_plot`

**Purpose:** Generates signal reconstruction comparison plots (e.g. for autoencoders). Plots target vs prediction for up to 10 evenly spaced samples.

**Behavior:** In `on_compute_end`, runs only if `evaluator.plot_reconstructions` is True, `evaluator.remote_logger` is set, and the buffer has data. Uses `data["preds"]` and `data["targets"]`; plots first channel for each sample. Logs via `evaluator.log_plot(fig, "reconstructions_overview", mode, epoch, step)`.

### UnitTrendPlotHook

**Module:** `lmetk.evaluator.hooks.unit_trend_plot`

**Purpose:** Plots degradation trends split by unit ID. Intended for multi-unit RUL evaluation.

**Constructor:**

```python
__init__(
    self,
    log_every_n_epochs: int = 10,
    enable_subsampling: bool = True,
    subsample_threshold: int = 2000,
    subsample_factor: int = 10,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `log_every_n_epochs` | `int` | Epoch period for logging plots (e.g. only when `epoch % log_every_n_epochs == 0`). |
| `enable_subsampling` | `bool` | Whether to subsample long series for plotting. |
| `subsample_threshold` | `int` | Length above which subsampling is applied. |
| `subsample_factor` | `int` | Subsampling step (plot every Nth point). |

**Behavior:** In `on_compute_end`, skips if `mode == "train"` or (for `mode == "val"`) if `epoch % log_every_n_epochs != 0`. Requires `unit_ids` in buffer data. For each unique unit, plots pred vs target (with optional subsampling) and logs via `evaluator.log_plot(fig, f"unit_{unit_key}_trend", mode, epoch, step)`.

---

## DefaultEvaluator

**Module:** `lmetk.evaluator.default`

**Purpose:** Orchestrates data flow between a `PredictionBuffer`, a `MetricManager`, and pluggable hooks. Supports regression, classification, and forecasting with optional dual reporting (normalized + denormalized) when inverse scaling is enabled.

### Constructor

```python
__init__(
    self,
    metric_names: List[str],
    task_type: str = "regression",
    save_predictions: bool = False,
    collect_predictions: Optional[bool] = None,
    num_classes: Optional[int] = None,
    paths: Optional[Any] = None,
    remote_logger: Optional[Any] = None,
    hooks: Optional[List[BaseEvalHook]] = None,
    metric_manager: Optional[MetricManager] = None,
    buffer: Optional[PredictionBuffer] = None,
    **kwargs: Any,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `metric_names` | `List[str]` | Non-empty list of metric names. Raises `ValueError` if empty. |
| `task_type` | `str` | One of `ALL_TASKS` (lowercased). |
| `save_predictions` | `bool` | Whether to write predictions to disk (via `SavePredictionsHook`; hook must be registered). |
| `collect_predictions` | `Optional[bool]` | If `None`, auto-enabled when `save_predictions` or any hook is present. If explicitly True/False, that choice is respected. |
| `num_classes` | `Optional[int]` | Required for classification tasks. |
| `paths` | `Optional[Any]` | Path config; should have `eval_details` (and optionally `plot_dir`) for hooks. |
| `remote_logger` | `Optional[Any]` | Logger for plots (e.g. `experiment.log(...)`). |
| `hooks` | `Optional[List[BaseEvalHook]]` | Hooks for saving, plotting, etc. |
| `metric_manager` | `Optional[MetricManager]` | Injected manager; if None, created from `metric_names`, `task_type`, etc. |
| `buffer` | `Optional[PredictionBuffer]` | Injected buffer; if None, new instance. |
| `**kwargs` | | Forwarded to `AbstractEvaluator` (e.g. `inverse_transform`, `apply_inverse_scaling`, `task_mode`). |

**Attributes set:**

- `self.task_type`, `self.save_predictions`, `self.remote_logger`, `self.num_classes`, `self.hooks`, `self.collect_predictions`, `self.metric_manager`, `self.buffer`.

### Methods

- **`add_hook(self, hook: BaseEvalHook) -> None`**  
  Registers a new hook. If `collect_predictions` was False, enables it automatically.

- **`update(self, model_out: Dict[str, Any]) -> None`**  
  1. Validates via `_validate_input` (returns early if empty or invalid).
  2. Prepares batch via `_prepare_batch_data` (scaling, task branching).
  3. If `collect_predictions`, calls `buffer.accumulate(batch, model_out.get("unit_id"))`.
  4. Calls `metric_manager.update(...)` with preds, targets, and optional norm_preds/norm_targets.
  5. Calls `hook.on_update_end(batch, self)` for each hook.

- **`compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]`**  
  Computes metrics via `metric_manager.compute()`, runs `hook.on_compute_end(...)` for each hook, returns the result dict.

- **`reset(self) -> None`**  
  Clears buffer and resets metric manager.

- **`log_plot(self, fig, title, mode, epoch, step) -> None`**  
  If `remote_logger` is set, logs the figure as `plots/{mode}/{title}`.

### Internal methods

- **`_validate_input(model_out)`** — Returns False if predictions/targets missing or batch empty; asserts targets shape `(N, T, 1)`.
- **`_prepare_batch_data(model_out)`** — Returns dict with `preds`, `targets`, optional `norm_preds`, `norm_targets`, `is_dual`. For classification, passes raw preds/targets (int cast); for regression/forecasting, applies `scaling_wrapper.inverse_transform_if_needed` with `metadata={"unit_id": model_out.get("unit_id")}`.

---

## ClassificationEvaluator

**Module:** `lmetk.evaluator.classification`

**Purpose:** Specialized evaluator for classification tasks. Thin wrapper over `DefaultEvaluator` with `task_type="classification"` and required `num_classes`.

### Constructor

```python
__init__(self, num_classes: int, metric_names: List[str], **kwargs: Any)
```

**Validation:** `num_classes` must be a positive integer; otherwise raises `ValueError`.

**Behavior:** Calls `super().__init__(metric_names=..., task_type="classification", num_classes=num_classes, **kwargs)`.

**Note:** Metric keys from `MetricManager` for classification have no `_normalized`/`_denormalized` suffixes. Inverse scaling is not used for classification.

---

## ForecastingEvaluator

**Module:** `lmetk.evaluator.forecasting`

**Purpose:** Forecasting evaluator with horizon slicing. Extends `DefaultEvaluator` by slicing predictions and targets to an optional effective prediction length and, for univariate mode, to an optional target dimension before calling the base `update`.

### Constructor

```python
__init__(
    self,
    target_dim_position: Optional[int],
    metric_names: List[str],
    effective_pred_len: Optional[int] = None,
    model_pred_len: Optional[int] = None,
    **kwargs: Any,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_dim_position` | `Optional[int]` | For univariate: index of the target dimension; used to slice `predictions[:, :, f:f+1]` and same for targets. |
| `effective_pred_len` | `Optional[int]` | If set, predictions and targets are sliced to the last `effective_pred_len` steps: `[:, -effective_pred_len:, :]`. |
| `model_pred_len` | `Optional[int]` | Required. Must be ≥ `effective_pred_len` if that is set. |
| `**kwargs` | | May contain `task_mode` (default `"univariate"`), `model_seq_len`, `model_label_len` for compatibility. |

**Attributes:** `model_pred_len`, `effective_pred_len`, `target_dim_position`, `task_mode`, `model_seq_len`, `model_label_len`.

### Override: `update(self, model_out: Dict[str, Any])`

1. Extracts `predictions` and `targets` from `model_out`.
2. If both present and shapes match, applies slicing:
   - If `effective_pred_len` is not None: `[:, -effective_pred_len:, :]`.
   - If `task_mode == "univariate"` and `target_dim_position` is not None: slice to `[:, :, f:f+1]` (3D with one feature).
3. Writes back into `model_out` and calls `super().update(model_out)`.

---

## MultiUnitEvaluator

**Module:** `lmetk.evaluator.multiunit`

**Purpose:** Evaluator that calculates metrics individually per unit ID. Extends `DefaultEvaluator` with per-unit `MetricManager` instances and optional per-unit metric logging.

### Constructor

```python
__init__(
    self,
    metric_names: List[str],
    log_per_unit_metrics: bool = True,
    **kwargs: Any,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `log_per_unit_metrics` | `bool` | If True, the returned dict includes per-unit entries like `mse_{unit_id}` in addition to aggregated `mse_mean`. |

**Attributes:** `log_per_unit_metrics`, `unit_managers: Dict[Any, MetricManager]` (created per unit on first update).

### Override: `update(self, model_out: Dict[str, Any])`

**Input:** `model_out` must contain `"predictions"`, `"targets"`, and `"unit_id"`. `unit_id` can be 1D (e.g. shape `(N,)`) or 2D (e.g. `(N, 2)` for composite keys).

**Logic:**

1. Calls `super().update(model_out)` (validates, prepares batch, updates global metrics and buffer).
2. Infers unique units and indices via `_get_unit_iteration_tools(unit_id)`.
3. For each unit, ensures a `MetricManager` exists in `unit_managers`, then updates it with the corresponding slice of `batch["preds"]`, `batch["targets"]`, and optional normalized variants.

### Override: `compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]`

1. Calls `super().compute(...)` for global metrics.
2. For each unit in `unit_managers`, computes unit metrics and, if `log_per_unit_metrics`, adds `{key}_{unit_id}` to the result.
3. For each metric name, aggregates values across units and adds `{key}_mean`.

### Override: `reset(self) -> None`

Calls `super().reset()` and clears `unit_managers`.

### Helper: `_get_unit_iteration_tools(self, u_ids: np.ndarray)`

Returns `(unique_units, get_indices, get_key)` for 1D or 2D unit identifiers. `get_key` maps a unit to an int (1D) or tuple (2D) for dict keys.

---

## ReconstructionEvaluator

**Module:** `lmetk.evaluator.reconstruction`

**Purpose:** Specialized evaluator for reconstruction tasks (e.g. autoencoders). Thin wrapper over `DefaultEvaluator` with `task_type="regression"`. Reconstruction plots and prediction saving are handled by hooks (e.g. `ReconstructionPlotHook`, `SavePredictionsHook`).

### Constructor

```python
__init__(self, metric_names: List[str], **kwargs: Any)
```

**Behavior:** Calls `super().__init__(metric_names=..., task_type="regression", **kwargs)`.

**Usage:** To enable reconstruction plots, register `ReconstructionPlotHook` and set `plot_reconstructions=True` on the evaluator. The hook checks `evaluator.plot_reconstructions` and `evaluator.remote_logger` before plotting. When using YAML configs, pass `plot_reconstructions: True` and ensure the evaluator stores this attribute (or set `evaluator.plot_reconstructions = True` after construction). Ensure `collect_predictions` is enabled—this happens automatically when hooks are present.

---

## Utils

**Module:** `lmetk.evaluator.utils`

**Purpose:** Standalone scalar metric functions for ad-hoc or external use. Not integrated into the evaluator class hierarchy.

**Functions:** Each takes `pred` and `true` (array-like) and returns a scalar (or tuple):

| Function | Description |
|----------|-------------|
| `RSE(pred, true)` | Relative squared error: `sqrt(sum((true-pred)^2)) / sqrt(sum((true - mean(true))^2))` |
| `CORR(pred, true)` | Correlation-like statistic over the last axis |
| `MAE(pred, true)` | Mean absolute error |
| `MSE(pred, true)` | Mean squared error |
| `RMSE(pred, true)` | Root mean squared error |
| `MAPE(pred, true)` | Mean absolute percentage error |
| `MSPE(pred, true)` | Mean squared percentage error |
| `metric(pred, true)` | Returns `(mae, mse, rmse, mape, mspe)` |

These operate on raw arrays and do not handle inverse scaling or the 3D `(N, T, C)` contract. The main regression/classification metrics used inside evaluators come from `lmetk.metrics` (MetricFactory, MetricManager).

---

## Configuration examples

The evaluators are typically instantiated via Hydra/OmegaConf from YAML configs under `configs/evaluator/`. Key patterns:

### Default regression (RUL)

```yaml
_evaluator_defaults: &evaluator_defaults
  _target_: lmetk.evaluator.default.DefaultEvaluator
  metric_names: ["mae", "mse", "rmse"]
  apply_inverse_scaling: False
  inverse_transform_name: null
  save_predictions: False
  task_mode: ${datasource.task_mode}
  paths: ${paths}

train: *evaluator_defaults
val: *evaluator_defaults
test:
  <<: *evaluator_defaults
  save_predictions: True
```

### Classification

```yaml
_target_: lmetk.evaluator.classification.ClassificationEvaluator
metric_names: ["f1", "accuracy", "precision", "recall"]
num_classes: ${task_definition.num_classes}
task_mode: ${datasource.task_mode}
paths: ${paths}
hooks:
  - _target_: lmetk.evaluator.hooks.save_predictions.SavePredictionsHook
    dims: ["sample", "time", "feature"]
```

### Multi-unit with unit trend plots

```yaml
_target_: lmetk.evaluator.multiunit.MultiUnitEvaluator
metric_names: ["mape", "phm_score", "mae", "mse", "rmse"]
paths: ${paths}
hooks:
  - _target_: lmetk.evaluator.hooks.unit_trend_plot.UnitTrendPlotHook
    enable_subsampling: True
    subsample_threshold: 1000
    subsample_factor: 10
  - _target_: lmetk.evaluator.hooks.save_predictions.SavePredictionsHook
    dims: ["sample", "time", "feature"]
```

### Reconstruction with plots

```yaml
_target_: lmetk.evaluator.reconstruction.ReconstructionEvaluator
metric_names: ["mae", "mse", "rmse"]
plot_reconstructions: True
paths: ${paths}
hooks:
  - _target_: lmetk.evaluator.hooks.reconstruction_plot.ReconstructionPlotHook
  - _target_: lmetk.evaluator.hooks.save_predictions.SavePredictionsHook
    dims: ["sample", "time", "feature"]
```

**Note:** `paths` is typically resolved to an object or dict with `eval_details` (for SavePredictionsHook output) and optionally `plot_dir`. `remote_logger` is often injected at runtime for plot logging.

---

## Summary table

| Evaluator | Task types | Inverse scaling | Per-unit metrics | Plots | Prediction save |
|-----------|------------|-----------------|------------------|-------|------------------|
| **AbstractEvaluator** | — | Via wrapper | No | No | No |
| **DefaultEvaluator** | All | Yes (non-classification) | No | Via hooks | Via SavePredictionsHook |
| **ClassificationEvaluator** | Classification | No | No | Via hooks | Via SavePredictionsHook |
| **ForecastingEvaluator** | Forecasting | Yes | No | Via hooks | Via SavePredictionsHook |
| **MultiUnitEvaluator** | Any (typically regression) | Yes | Yes | Via UnitTrendPlotHook | Via SavePredictionsHook |
| **ReconstructionEvaluator** | Regression | Yes | No | Via ReconstructionPlotHook | Via SavePredictionsHook |

All concrete evaluators implement `update(model_out)`, `compute(mode, epoch, step)`, and `reset()`, with `compute` returning a dictionary of metric names to float values. Saving and plotting are delegated to hooks, which access the evaluator’s buffer, paths, and remote logger.
