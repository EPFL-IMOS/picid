# Evaluators

This document describes the evaluator subsystem in `lmetk.evaluator`: the abstract base, scaling utilities, and all concrete evaluators (default, classification, forecasting, multi-unit, reconstruction). For each component we explain **purpose**, **attributes**, **methods** (signatures, inputs, outputs), and **logic** in as explicit a form as possible.

---

## Table of contents

1. [Overview and architecture](#overview-and-architecture)
2. [Task types](#task-types)
3. [AbstractEvaluator (base)](#abstractevaluator-base)
4. [Scaling wrappers](#scaling-wrappers)
5. [DefaultEvaluator](#defaultevaluator)
6. [ClassificationEvaluator](#classificationevaluator)
7. [ForecastingEvaluator](#forecastingevaluator)
8. [MultiUnitEvaluator](#multiunitevaluator)
9. [ReconstructionEvaluator](#reconstructionevaluator)
10. [Utils](#utils)

---

## Overview and architecture

**Location:** `lmetk/evaluator/`

**Role:** Evaluators consume model outputs (predictions and targets), optionally apply inverse scaling, update internal metrics, and at the end of an evaluation phase return a dictionary of metric values. Some evaluators also save predictions to disk (e.g. netCDF) and/or log plots to a remote logger (e.g. WandB, MLflow).

**Inheritance:**

- **AbstractEvaluator** (ABC) — defines `update`, `compute`, `reset` and sets up the scaling wrapper and paths.
- **DefaultEvaluator** (extends AbstractEvaluator) — generic regression/classification/forecasting evaluator with optional dual reporting (normalized + denormalized) and optional prediction saving.
- **ClassificationEvaluator** — thin wrapper over DefaultEvaluator with `task_type="classification"`, custom save format (logits), and metric keys without `_normalized`/`_denormalized` suffixes.
- **ForecastingEvaluator** — extends DefaultEvaluator with slicing by `effective_pred_len` and optional `target_dim_position` for univariate forecasting.
- **MultiUnitEvaluator** — extends DefaultEvaluator; **regression-only**; maintains per-unit metrics and optional pred-vs-target plots per unit; requires `unit_id` in `model_out`.
- **ReconstructionEvaluator** — extends DefaultEvaluator; adds optional reconstruction plots and local/remote plot logging.

**Data flow (typical):**

1. **Training/validation loop** calls `evaluator.update(model_out)` for each batch. `model_out` contains at least `"predictions"` and `"targets"`; MultiUnit also requires `"unit_id"`.
2. At end of epoch/split, the loop calls `evaluator.compute(mode, epoch, step)`, which:
   - Prepares data for saving (overridable),
   - Saves predictions to file if configured,
   - Computes and returns a `Dict[str, float]` of metric names to values.

**Shape conventions (throughout the doc):**

- **Regression / forecasting:** predictions and targets are 3D: `(N, T, 1)` (single target). `N` = batch size, `T` = time steps (often 1 for point-wise RUL).
- **Classification:** predictions `(N, T, C)` (logits), targets `(N, T, 1)` (integer class indices).
- **Single-target:** targets always have last dimension 1 in the evaluator contract.

---

## Task types

Task types are defined in `lmetk.baselines.definitions` and used by evaluators to branch logic (e.g. scaling vs no scaling, metric factory):

| Constant | Values |
|----------|--------|
| `REGRESSION_TASKS` | `["regression", "rul", "ahrul", "soc"]` |
| `CLASSIFICATION_TASKS` | `["classification", "health_states", "concepts", "fault_classification"]` |
| `FORECASTING_TASKS` | `["forecasting"]` |
| `STATE_FORECASTING_TASKS` | `["state_forecasting"]` |
| `ALL_TASKS` | Union of all above |

- **DefaultEvaluator** accepts any `task_type` in `ALL_TASKS`.
- **MultiUnitEvaluator** accepts only `task_type` in `REGRESSION_TASKS` (assertion in `__init__`).
- **ClassificationEvaluator** fixes `task_type="classification"`.
- **ForecastingEvaluator** does not override `task_type` in the base; it is typically used with forecasting tasks and slices predictions/targets by length and optionally by target dimension.

---

## AbstractEvaluator (base)

**Module:** `lmetk.evaluator.base`

**Purpose:** Abstract base for all evaluators. It sets up the scaling wrapper (univariate or multivariate) and paths, and declares the interface: `update`, `compute`, `reset`.

### Constructor

```text
__init__(
    self,
    paths: dict[str, str],
    inverse_transform: Optional[InverseTransformMixin] = None,
    apply_inverse_scaling: bool = False,
    task_mode: Optional[Any] = False,
    **kwargs,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `paths` | `dict[str, str]` | Path configuration (e.g. output dirs); stored as `self.paths`. |
| `inverse_transform` | `Optional[InverseTransformMixin]` | Scaler used for inverse transform; passed to the scaling wrapper. |
| `apply_inverse_scaling` | `bool` | Whether to apply inverse scaling when requested by the wrapper. |
| `task_mode` | `Optional[Any]` | Passed to the scaling wrapper; e.g. `"univariate"`, `"multivariate"`, `"forecasting"`. |
| `**kwargs` | | Only `inverse_transform_name` is allowed; any other key raises `ValueError`. |

**Attributes set:**

- `self.scaling_wrapper` — If `task_mode == "multivariate"`, instance of `MultivariateTimeseriesScalingWrapper`; else instance of `ScalingWrapper`. Both are initialized with `inverse_transform`, `apply_inverse_scaling`, and `task_mode`.
- `self.task_mode` — The given `task_mode`.
- `self.paths` — The given `paths`.

### Abstract methods

- **`update(self, model_out: dict)`** — Update internal state (e.g. metrics, collected predictions) with one batch. No return value.
- **`compute(self, ...)`** — Compute and return metrics (and possibly trigger saving/logging). Return type is implementation-defined (typically `Dict[str, float]`).
- **`reset(self)`** — Reset all accumulated state (metrics, collected lists) so the evaluator can be reused for another run/epoch.

### Other

- **`__repr__(self)`** — Returns a string representation using only “simple” attributes (int, float, str, bool, None).

---

## Scaling wrappers

**Module:** `lmetk.evaluator.scaling_wrapper`

Two wrappers delegate inverse scaling to an `InverseTransformMixin` instance and handle shape flattening/reshaping so the scaler sees 2D data.

### ScalingWrapper (univariate / standard)

**Purpose:** Used when `task_mode != "multivariate"`. Applies inverse transform per array while preserving 3D `(B, T, C)` or 2D shapes.

**Constructor:**

```text
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
  - `kwargs`: Optional `metadata` (e.g. `unit_id`) passed to the scaler’s `inverse_transform`.
- **Behavior:**
  - If `inverse_transform` is None or `apply_inverse` is False, returns `(predictions, targets)` unchanged.
  - For **3D** `(B, T, C)`:
    - If `task_mode != "forecasting"`, asserts `T == 1`.
    - If `task_mode == "forecasting"`, asserts `C == 1`.
    - Flattens to `(B*T, C)` via `rearrange("b t c -> (b t) c")`, calls `inverse_transform` on `NamedTransformInput(predictions=...)` and `NamedTransformInput(targets=...)`, then reshapes back to original shape.
  - For **2D**, calls `inverse_transform` without reshaping.
- **Output:** `(predictions_transformed, targets_transformed)` same shape as input (after possible tensor→numpy conversion).

### MultivariateTimeseriesScalingWrapper

**Purpose:** Used when `task_mode == "multivariate"`. Constructor asserts `task_mode == "multivariate"`. Flattens 3D to 2D, applies inverse transform, then unflattens back to 3D. Intended for multivariate time series (e.g. multiple channels).

**Constructor:** Same signature as `ScalingWrapper`; asserts `task_mode == "multivariate"`.

**Method: `inverse_transform_if_needed(predictions, targets)`** — No `metadata` in the current implementation. Converts to numpy, flattens `(B, T, C)` to `(B*T, C)`, applies `inverse_transform`, then reshapes back to `(B, T, C)`.

---

## DefaultEvaluator

**Module:** `lmetk.evaluator.default`

**Purpose:** Single evaluator for regression, classification, and forecasting. It supports:

- **Dual reporting:** When inverse scaling is enabled, it keeps two sets of metrics: one on **normalized** (model) space and one on **denormalized** (inverse-scaled) space, and reports both with `_normalized` / `_denormalized` suffixes (except in ClassificationEvaluator, which strips them).
- **Optional prediction saving:** Can concatenate collected predictions/targets and write them to a netCDF file.
- **Optional remote logging:** Can pass a `remote_logger` for logging plots (e.g. via `log_plot`).

**Task types:** Any in `ALL_TASKS`. Classification tasks do **not** use inverse scaling (assertion: `inverse_transform` must be None for classification).

### Constructor

```text
__init__(
    self,
    metric_names: List[str],
    task_type: str = "regression",
    num_classes: Optional[int] = None,
    inverse_transform: Optional[InverseTransformMixin] = None,
    apply_inverse_scaling: bool = False,
    save_predictions: bool = False,
    paths: Optional[str] = None,
    remote_logger: Optional[Any] = None,
    collect_predictions: bool = False,
    **kwargs,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `metric_names` | `List[str]` | Non-empty list of metric names (e.g. `["mse", "mae"]`). Stored lowercased. Raises `ValueError` if empty. |
| `task_type` | `str` | One of `ALL_TASKS` (lowercased). Determines metric factory (classification vs regression) and update logic. |
| `num_classes` | `Optional[int]` | Required for classification tasks; used when creating classification metrics. |
| `inverse_transform` | `Optional[InverseTransformMixin]` | Scaler for inverse transform; must be None for classification. |
| `apply_inverse_scaling` | `bool` | Whether to apply inverse scaling and thus enable dual (normalized/denormalized) reporting. |
| `save_predictions` | `bool` | Whether to write predictions to netCDF at compute time. |
| `paths` | `Optional[str]` or path-like | Path config; may have `.eval_details` for output directory. |
| `remote_logger` | `Optional[Any]` | Logger for plots (e.g. `experiment.log(...)`). |
| `collect_predictions` | `bool` | Whether to accumulate predictions/targets in memory for saving and (if scaling) for normalized series. Effective collection is `save_predictions or collect_predictions`. |
| `**kwargs` | | Forwarded to `AbstractEvaluator`; only `inverse_transform_name` allowed there. |

**Attributes set:**

- `self.inverse_transform` — Same as the constructor argument.
- `self.task_type` — Lowercased `task_type`.
- `self.metrics` — Dict mapping each metric name to a metric instance (e.g. from `MetricFactory.create_metric` or `create_classification_metric`). These are the **primary** metrics (denormalized when scaling is on).
- `self.paths`, `self.save_predictions`, `self.out_path`, `self.remote_logger`, `self.collect_predictions`, `self.metric_names`, `self.num_classes`.
- If `collect_predictions`: `self.preds`, `self.targets` (lists of arrays). If `apply_inverse_scaling`: also `self.scaled_preds`, `self.scaled_targets`; and `self.normalized_metrics` (same keys as `self.metrics`) for the normalized view.

### Method: `reset(self)`

- Clears `preds`, `targets`, `scaled_preds`, `scaled_targets` if they exist.
- Calls `metric.reset()` on every metric in `self.metrics` and, if present, `self.normalized_metrics`.

### Method: `update(self, model_out: dict)`

**Input:** `model_out` must contain:

- `"predictions"`: array shape `(N, T, C)` with `C=1` for regression/forecasting, or `C=num_classes` for classification.
- `"targets"`: array shape `(N, T, 1)`.

**Logic:**

1. **Empty batch:** If `predictions.shape[0] == 0`, returns immediately (no update, no assert).
2. **Target shape:** Asserts `targets.ndim == 3` and `targets.shape[2] == 1`.
3. **Store normalized (if scaling on):** If `apply_inverse_scaling`, copies predictions/targets as normalized series; if `collect_predictions`, appends copies to `preds`/`targets`.
4. **Store for save (no scaling):** If not applying inverse and `collect_predictions`, appends predictions/targets to `preds`/`targets`.
5. **Inverse scaling / task branch:**
   - **Classification:** Asserts `inverse_transform is None`, `predictions.shape[2] == num_classes`; casts targets to int.
   - **Regression / forecasting:** Asserts predictions shape `(N, T, 1)`; calls `scaling_wrapper.inverse_transform_if_needed(predictions, targets)` and uses the returned arrays for the rest of the step.
6. **Store denormalized (if scaling on and collect):** Appends to `scaled_preds` / `scaled_targets`.
7. **Update primary metrics:** All metrics in `self.metrics` are updated with the (possibly denormalized) predictions and targets.
8. **Update normalized metrics:** If dual reporting is on, all metrics in `self.normalized_metrics` are updated with the normalized copies.

**Output:** None.

### Method: `_prepare_predictions_for_save(self) -> Dict[str, Any]`

- Returns empty dict if `save_predictions` is False or `collect_predictions` is False.
- Otherwise concatenates along axis 0:
  - If scaling on: primary save uses `scaled_preds`/`scaled_targets`, and adds `preds_normalized`/`targets_normalized` from `preds`/`targets`.
  - If scaling off: primary save uses `preds`/`targets` only.
- For classification with multiple logit dimensions, predictions are converted to class indices via `argmax(axis=-1, keepdims=True)` for the save dict.
- Dimension names are `["sample"] + ["dim1", ...]`. Keys: `"preds"`, `"targets"`, and optionally `"preds_normalized"`, `"targets_normalized"`.

**Output:** Dictionary suitable for building an `xarray.Dataset` (e.g. `(dims, array)` per key).

### Method: `_save_predictions_to_file(self, dict_to_save: Dict[str, Any], mode: str)`

- No-op if `dict_to_save` is empty.
- Sets `self.out_path` from `paths.eval_details / mode` if not already set; creates directory; writes `xr.Dataset(dict_to_save).to_netcdf(out_path / "predictions.nc")`.

### Method: `_compute_metrics(self) -> Dict[str, float]`

- Computes each metric in `self.metrics`; key in result: `"{name}_denormalized"`. If a metric raises `ValueError` with "No data available", the value is set to `float("nan")`.
- If dual reporting: computes each metric in `self.normalized_metrics`; key: `"{name}_normalized"` (same NaN fallback).
- If **no** dual reporting (scaling off), renames all keys from `_denormalized` to `_normalized` so the single set of metrics is exposed under the normalized naming.

**Output:** `Dict[str, float]` with keys like `mse_denormalized`, `mse_normalized`, etc. (or only `*_normalized` when scaling is off).

### Method: `compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]`

1. Calls `_prepare_predictions_for_save()`.
2. Calls `_save_predictions_to_file(dict_to_save, mode)`.
3. Returns `_compute_metrics()`.

**Output:** Same as `_compute_metrics()`.

### Method: `log_plot(self, fig: plt.Figure, title: str, mode: str, epoch: int, step: int)`

- If `self.remote_logger` is set, logs the figure as `plots/{mode}/{title}` (implementation may vary slightly; see source).

---

## ClassificationEvaluator

**Module:** `lmetk.evaluator.classification`

**Purpose:** Evaluator for classification tasks. It is a thin wrapper around DefaultEvaluator with `task_type="classification"`, fixed save format (logits + labels), and metric keys without `_normalized`/`_denormalized` suffixes (since scaling is not used for classification).

### Constructor

```text
__init__(
    self,
    num_classes: int,
    metric_names: List[str],
    inverse_transform: Optional[InverseTransformMixin] = None,
    apply_inverse_scaling: bool = False,
    save_predictions: bool = False,
    paths: Optional[str] = None,
    **kwargs,
)
```

- **Validation:** `num_classes` must be a positive integer; otherwise raises `ValueError`.
- Calls `super().__init__(..., task_type="classification", num_classes=num_classes, ...)`.

### Override: `_prepare_predictions_for_save(self) -> Dict[str, Any]`

- Same conditions as base (save and collect enabled). Uses `self.preds` and `self.targets` (logits and labels).
- Does **not** argmax logits; saves full logits. Dimension names: `["sample", "time", "class_logits"]` for predictions and `["sample", "time", "class_label"]` for targets.
- Keys: `"preds_logits"`, `"targets_labels"`.

### Override: `_compute_metrics(self) -> Dict[str, float]`

- Calls `super()._compute_metrics()` to get the dict with `_normalized`/`_denormalized` keys.
- Builds a new dict: for each key, strips `_mean` (if present), then strips `_normalized` or `_denormalized` to get a base name (e.g. `accuracy`). Values are stored under the base name; if both normalized and denormalized exist for the same base name, asserts equality (classification does not use scaling, so they should match). Returns this simplified dict.

**Output:** `Dict[str, float]` with keys like `accuracy`, `f1`, etc. (no suffix).

---

## ForecastingEvaluator

**Module:** `lmetk.evaluator.forecasting`

**Purpose:** Evaluator for forecasting tasks. It extends DefaultEvaluator by slicing predictions and targets to an optional effective prediction length and, for univariate mode, to an optional target dimension before calling the base `update`.

### Constructor

```text
__init__(
    self,
    target_dim_position: Optional[int],
    metric_names: List[str],
    model_seq_len: int,
    model_label_len: int,
    effective_pred_len: Optional[int] = None,
    model_pred_len: Optional[int] = None,
    inverse_transform: Optional[InverseTransformMixin] = None,
    apply_inverse_scaling: bool = False,
    save_predictions: bool = False,
    paths: Optional[str] = None,
    **kwargs,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_dim_position` | `Optional[int]` | For univariate: index of the target dimension in the last axis; used to slice `predictions[:, :, f_dim]` and same for targets. |
| `model_seq_len` | `int` | Stored; not used in this class for slicing. |
| `model_label_len` | `int` | Stored; not used in this class for slicing. |
| `effective_pred_len` | `Optional[int]` | If set, predictions and targets are sliced to the last `effective_pred_len` steps: `[:, -effective_pred_len:, :]`. |
| `model_pred_len` | `int` | Required. Must be ≥ `effective_pred_len` if that is set; otherwise raises `ValueError`. |
| `**kwargs` | | May contain `task_mode`; default `"univariate"`. |

**Attributes:** `model_label_len`, `model_pred_len`, `model_seq_len`, `effective_pred_len`, `target_dim_position`, `task_mode`.

### Override: `update(self, model_out: dict)`

1. Reads `predictions` and `targets` from `model_out`.
2. If `effective_pred_len` is not None, slices both to `[:, -effective_pred_len:, :]`.
3. Asserts `predictions.shape == targets.shape`.
4. If `task_mode == "univariate"` and `target_dim_position` is not None, slices to `predictions[:, :, f_dim]` and `targets[:, :, f_dim]` (2D); these are then written back into `model_out["predictions"]` and `model_out["targets"]`.
5. Calls `super().update(model_out)`.

So the base sees either 3D `(N, T, 1)` or 2D `(N, T)` depending on the univariate slice; downstream metrics and scaling wrapper must support the resulting shape.

---

## MultiUnitEvaluator

**Module:** `lmetk.evaluator.multiunit`

**Purpose:** Regression-only evaluator that maintains **per-unit** metrics (e.g. one MSE per unit). It requires a `unit_id` in every `model_out`, splits predictions/targets by unit, and updates separate metric instances per unit. It can optionally log pred-vs-target curves per unit to the remote logger and save unit IDs together with predictions.

### Constructor

```text
__init__(
    self,
    metric_names: List[str],
    inverse_transform: Optional[InverseTransformMixin] = None,
    apply_inverse_scaling: bool = False,
    save_predictions: bool = False,
    paths: Optional[str] = None,
    remote_logger: Optional[Any] = None,
    collect_predictions: bool = False,
    log_image_every_n_epochs: bool = 10,
    log_image: bool = True,
    enable_plot_subsampling: bool = True,
    plot_subsample_threshold: int = 2000,
    plot_subsample_factor: int = 10,
    log_per_unit_metrics: bool = True,
    **kwargs,
)
```

- Calls `super().__init__(...)` (DefaultEvaluator). **Assertion:** `self.task_type` must be in `REGRESSION_TASKS` (e.g. regression, rul); otherwise raises `AssertionError` ("Unsupported task_type"). So MultiUnitEvaluator must be used with regression task types only.

**Additional parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `log_image_every_n_epochs` | `int` | Epoch period for logging pred-vs-target plots. |
| `log_image` | `bool` | Whether to log pred-vs-target curves at all. |
| `enable_plot_subsampling` | `bool` | Whether to subsample long series for plotting. |
| `plot_subsample_threshold` | `int` | Length above which subsampling is applied. |
| `plot_subsample_factor` | `int` | Subsampling step (e.g. plot every Nth point). |
| `log_per_unit_metrics` | `bool` | If True, the returned metric dict includes per-unit entries like `mse_{unit_id}_denormalized`. |

**Attributes (in addition to base):** `log_image_every_n_epochs`, `log_fit_vs_target_curves` (= `log_image`), `enable_plot_subsampling`, `plot_subsample_threshold`, `plot_subsample_factor`, `log_per_unit_metrics`. If `collect_predictions`: `unit_ids` (list of arrays). **Per-unit storage:** `unit_metrics` and, if scaling on, `normalized_unit_metrics` — both are `defaultdict(_create_metric_dict)` mapping unit key (int or tuple of ints) to a dict of metric name → metric instance.

### Method: `_create_metric_dict(self) -> Dict[str, AbstractMetric]`

- Returns `{name: MetricFactory.create_metric(name, self.paths) for name in self.metric_names}`. Used to create a fresh set of metrics per unit.

### Override: `reset(self)`

- Calls `super().reset()`, clears `unit_ids` if present, clears `unit_metrics` and `normalized_unit_metrics`.

### Override: `update(self, model_out: dict)`

**Input:** `model_out` must contain `"predictions"`, `"targets"`, and `"unit_id"`. `unit_id` can be 1D (e.g. shape `(N,)`) or 2D (e.g. shape `(N, 2)` for composite keys). Metadata for scaling: `metadata = {"unit_id": model_out["unit_id"]}` if present.

**Logic:**

1. If inverse scaling is on, keeps normalized copies of predictions/targets.
2. If collect: appends predictions, targets, and unit_ids to `preds`, `targets`, `unit_ids`.
3. Applies inverse scaling via `scaling_wrapper.inverse_transform_if_needed(..., metadata=metadata)` (predictions/targets in regression shape `(N, T, 1)`).
4. If collect and scaling on: appends to `scaled_preds`/`scaled_targets`.
5. **Per-unit update:** Infers unique units from `unit_ids` (1D or 2D). For each unit, indexes `predictions[indices]` and `targets[indices]`, then updates `unit_metrics[unit_key]` with the denormalized pair and, if dual reporting, `normalized_unit_metrics[unit_key]` with the normalized pair.

**Output:** None.

### Override: `_prepare_predictions_for_save(self) -> Dict[str, Any]`

- Calls `super()._prepare_predictions_for_save()` and, if non-empty, concatenates `self.unit_ids` and adds an `xr.DataArray` with key `"unit_ids"` and appropriate dims (`["sample"]` + `unit_dim1`, ...).

### Override: `_compute_metrics(self) -> Dict[str, float]`

- If no units were ever updated, returns `{f"{name}_mean": 0.0 for name in self.metric_names}`.
- Otherwise:
  - For each unit and each metric: computes value; if `log_per_unit_metrics`, stores `f"{name}_{unit_id}_denormalized"`; appends value to a list for `f"{name}_denormalized"`.
  - If dual reporting: same for normalized metrics and `f"{name}_{unit_id}_normalized"` / `f"{name}_normalized"`.
  - For each collected list, sets `f"{name_or_key}_mean"` to `np.mean(values)`.

**Output:** Dict with keys like `mse_denormalized_mean`, `mse_normalized_mean`, and optionally `mse_1_denormalized`, etc.

### Override: `compute(self, mode: str = "val", epoch: int = 0, step: int = 0) -> Dict[str, float]`

- If `log_fit_vs_target_curves` and (mode != "train" or (mode == "val" and epoch % log_image_every_n_epochs == 0)), calls `_log_pred_vs_target_curves(mode, step)`.
- Then returns `super().compute(mode, epoch, step)`.

### Helper: `_log_pred_vs_target_curves(self, mode: str, step: int)`

- Concatenates `preds`, `targets`, `unit_ids`. For each unique unit, plots prediction vs target (with optional subsampling), builds a figure, and calls `log_plot(...)` then closes the figure.

### Helper: `_inverse_transform_per_unit(self, predictions, targets, unit_ids)`

- Applies inverse transform per unit when unit_ids are 1D or 2D; fills output arrays by unit; used for per-unit scaling when the scaler is unit-aware. Not used in the main `update` path above (which uses the wrapper’s single call with metadata).

---

## ReconstructionEvaluator

**Module:** `lmetk.evaluator.reconstruction`

**Purpose:** Evaluator for reconstruction tasks (e.g. autoencoders). Extends DefaultEvaluator with optional **reconstruction plots**: side-by-side targets vs predictions for a subset of samples, and optional local file save plus remote logging.

### Constructor

```text
__init__(
    self,
    metric_names: List[str],
    inverse_transform: Optional[InverseTransformMixin] = None,
    apply_inverse_scaling: bool = False,
    save_predictions: bool = False,
    plot_reconstructions: bool = False,
    paths: Optional[str] = None,
    **kwargs,
)
```

- **Note:** `kwargs` may contain `collect_predictions`; it is popped. Effective collection is `collect_predictions = (popped value) or plot_reconstructions` so that enabling plots forces collection.
- Uses matplotlib backend `"Agg"`.
- **Attributes:** `plot_reconstructions`, `plots_out_path` (None until first compute).

### Method: `_plot_reconstructions(self, preds, targets, mode, epoch, step) -> Optional[plt.Figure]`

- If no `remote_logger`, returns None (and logs a warning once at step 0).
- Selects up to 10 evenly spaced sample indices. Builds a 5×2 subplot figure; for each index and each dimension (up to 10 dims), plots target (dashed) and prediction (solid). Handles 2D arrays by expanding to `(N, T, 1)`.
- **Returns:** The matplotlib Figure, or None.

### Override: `compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]`

1. Sets `plots_out_path` from `paths.plot_dir / mode` if not set and creates the directory.
2. If `plot_reconstructions`: chooses data from `scaled_preds`/`scaled_targets` when inverse scaling is on, else `preds`/`targets`; concatenates; calls `_plot_reconstructions`; if a figure is returned, logs it via `log_plot(..., "reconstructions_overview", ...)`, saves to `plots_out_path / f"reconstructions_overview_{epoch}_{step}.png"`, and closes the figure.
3. Returns `super().compute(mode, epoch=epoch, step=step)`.

So reconstruction metrics and prediction saving behave like DefaultEvaluator; the only addition is the optional reconstruction plot and its file/remote logging.

---

## Utils

**Module:** `lmetk.evaluator.utils`

**Purpose:** Standalone scalar metric functions (not integrated into the evaluator class hierarchy). Useful for ad-hoc or external use.

**Functions:** Each takes `pred` and `true` (array-like) and returns a scalar (or in one case a tuple):

- **`RSE(pred, true)`** — Relative squared error: `sqrt(sum((true-pred)^2)) / sqrt(sum((true - mean(true))^2))`.
- **`CORR(pred, true)`** — Correlation-like statistic over the last axis.
- **`MAE(pred, true)`** — Mean absolute error.
- **`MSE(pred, true)`** — Mean squared error.
- **`RMSE(pred, true)`** — Root mean squared error: `sqrt(MSE(pred, true))`.
- **`MAPE(pred, true)`** — Mean absolute percentage error.
- **`MSPE(pred, true)`** — Mean squared percentage error.
- **`metric(pred, true)`** — Returns `(mae, mse, rmse, mape, mspe)`.

These operate on raw arrays; they do not handle inverse scaling or the 3D (N, T, C) contract of the evaluators. The main regression metrics used inside evaluators come from `lmetk.metrics` (e.g. MetricFactory), not from this utils module.

---

## Summary table

| Evaluator | Task types | Inverse scaling | Per-unit metrics | Plots | Prediction save |
|-----------|------------|-----------------|------------------|-------|------------------|
| **AbstractEvaluator** | — | Via wrapper | No | No | No |
| **DefaultEvaluator** | All | Yes (non-classification) | No | Via `log_plot` | Optional netCDF |
| **ClassificationEvaluator** | Classification | No | No | Via base | Optional (logits) |
| **ForecastingEvaluator** | Any (typically forecasting) | Yes | No | Via base | Optional |
| **MultiUnitEvaluator** | Regression only | Yes | Yes | Pred vs target per unit | Optional (+ unit_ids) |
| **ReconstructionEvaluator** | Any | Yes | No | Reconstruction overview | Optional |

All concrete evaluators ultimately implement `update(model_out)`, `compute(mode, epoch, step)`, and `reset()`, with `compute` returning a dictionary of metric names to float values (and optionally writing files and logging figures).
