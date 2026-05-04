# Transforms

Transforms preprocess your data before it reaches the model.
Each transform is wrapped in a `DataTransform` object that tells the pipeline *what* to transform, *how* to fit it, and *where* to write the result.

```python
from picid.transforms.base import DataTransform
```

---

## `DataTransform`

`DataTransform` is the single wrapper you always use, regardless of which underlying transform you pick.

```python
DataTransform(
    transform_name: str,   # unique name for this transform
    transform,             # the transform instance
    metadata: dict,        # configuration: what to apply, where to fit, where to write
)
```

| Parameter | Type | Description |
|---|---|---|
| `transform_name` | `str` | A unique identifier for this transform in the pipeline. Evaluators reference it by this name when undoing scaling (see [Inverse transforms](#inverse-transforms)). |
| `transform` | `BaseTransform` instance | The transform to apply (e.g. `MinMaxScalerSklearn()`). |
| `metadata` | `dict` | Configuration dict. See below. |

---

### The `metadata` dict

| Key | Type | Required | Description |
|---|---|---|---|
| `apply_to` | `str` or `list[str]` | Yes | Which key(s) in the data container to read from (e.g. `"features"`, `"rul"`, `["features", "rul"]`). |
| `fit_on` | `str` | No | Which split to fit the transform on: `"train"`, `"val"`, or `"test"`. Required for fittable transforms (scalers, etc.). Omit for stateless transforms. |
| `assign_to` | `str` or `list[str]` | No | Which key(s) to write the result to. Defaults to `apply_to`. Use this to write to a different key than the one you read from. |

**`apply_to` as a list** — when you pass a list, the transform receives and returns each key independently in order:

```python
# Scale both features and the target, each independently
DataTransform("scaler", MinMaxScalerSklearn(), {"apply_to": "features", "fit_on": "train"})
DataTransform("scaler_target", MinMaxScalerSklearn(), {"apply_to": "rul", "fit_on": "train"})
```

**`assign_to` to write to a different key:**

```python
# Read "features", write result to "scaled_features"
DataTransform("scaler", MinMaxScalerSklearn(),
              {"apply_to": "features", "assign_to": "scaled_features", "fit_on": "train"})
```

---

### Ordering and fitting

Transforms are applied in list order.
Fitting happens exactly once, on the split named by `fit_on`, before the transform is applied to any other split.
This means a scaler fitted on `"train"` uses the training statistics when it later transforms `"val"` and `"test"` — the correct behavior to avoid data leakage.

```python
transforms = [
    DataTransform("scaler_features", MinMaxScalerSklearn(),
                  {"apply_to": "features", "fit_on": "train"}),
    DataTransform("scaler_targets", MinMaxScalerSklearn(),
                  {"apply_to": "rul", "fit_on": "train"}),
]
```

Always put scalers before transforms that depend on scaled values.

---

## Available transforms

Import all standard transforms from `picid.transforms.base_transforms`.

### Scalers

| Class | Import | Inverse | Description |
|---|---|---|---|
| `MinMaxScalerSklearn` | `picid.transforms.base_transforms.scaler` | Yes | Scales each feature to [0, 1] using sklearn's MinMaxScaler. Fits on training data. |
| `StandardScalerSklearn` | `picid.transforms.base_transforms.scaler` | Yes | Zero-mean, unit-variance standardization using sklearn's StandardScaler. |
| `ConstantScaler` | `picid.transforms.base_transforms.scaler` | Yes | Multiplies all values by a fixed constant. Stateless — no fitting needed. |

```python
from picid.transforms.base_transforms.scaler import (
    MinMaxScalerSklearn,
    StandardScalerSklearn,
    ConstantScaler,
)

# MinMax [0, 1]
DataTransform("scaler", MinMaxScalerSklearn(), {"apply_to": "features", "fit_on": "train"})

# Z-score
DataTransform("std_scaler", StandardScalerSklearn(), {"apply_to": "features", "fit_on": "train"})

# Multiply by a constant (no fitting)
DataTransform("const", ConstantScaler(factor=0.01), {"apply_to": "rul"})
```

---

### Imputation

Fills NaN values in the data.

```python
from picid.transforms.base_transforms.imputation_methods import ImputationTransform
```

| Strategy | Description |
|---|---|
| `"zero"` | Replace NaNs with 0. Stateless. |
| `"mean"` | Replace with per-channel training mean. Requires `fit_on`. |
| `"locf"` | Last Observation Carried Forward — causal, no future leakage. |
| `"linear"` | Linear interpolation. Non-causal. |
| `"stochastic"` | LOCF + Gaussian noise. Requires `fit_on`. |
| `"spectral"` | Causal spectral extrapolation via sinusoidal modelling. |
| `"copy_past"` | Blockwise copy of past observations. |

```python
# Single strategy for all channels
DataTransform("impute", ImputationTransform(strategy="locf"), {"apply_to": "features"})

# Different strategy per channel
DataTransform("impute", ImputationTransform(strategy=["mean", "linear", "zero"]),
              {"apply_to": "features", "fit_on": "train"})
```

---

### Concatenation

Merge multiple keys along a dimension.

```python
from picid.transforms.base_transforms.concatenate import ConcatenateTransform

# Concatenate features and time_features into a single "features" key
DataTransform("concat", ConcatenateTransform(dim=1),
              {"apply_to": ["features", "time_features"], "assign_to": "features"})
```

| Parameter | Description |
|---|---|
| `dim` | Dimension to concatenate along. `1` for feature axis, `2` for time axis. |

---

### Subsampling and aggregation

```python
from picid.transforms.base_transforms.subsample import (
    SubsampleTransform,
    WindowedAggregationTransform,
)

# Keep every 4th sample
DataTransform("sub", SubsampleTransform(step=4), {"apply_to": "features"})

# Sliding-window mean
DataTransform("win_mean", WindowedAggregationTransform(window_size=10, step=5, agg="mean"),
              {"apply_to": "features"})
```

`WindowedAggregationTransform` aggregations: `"mean"`, `"sum"`, `"min"`, `"max"`, `"median"`, `"std"`, `"first"`, `"last"`.
Set `window_size="full"` to collapse an entire axis into one value.

---

### Signal statistics

Compute hand-crafted features from raw time-series.

```python
from picid.transforms.base_transforms.spectral import SpectralStatsTransform
from picid.transforms.base_transforms.time_statistics import TimeStatsTransform

# Frequency-domain statistics
DataTransform("fft_stats",
              SpectralStatsTransform(stats_to_compute=["mean", "variance", "spectral_entropy"]),
              {"apply_to": "features"})

# Time-domain statistics
DataTransform("time_stats",
              TimeStatsTransform(stats_to_compute=["mean", "standard_deviation", "kurtosis"]),
              {"apply_to": "features"})
```

**`SpectralStatsTransform` available stats:** `mean`, `maximum`, `minimum`, `root_mean_square`, `peak_to_peak_value`, `variance`, `skewness`, `kurtosis`, `abs_energy`, `peak_factor`, `change_coefficient`, `clearance_factor`, `spectral_entropy`, `shannon_entropy`, `permutation_entropy`.

**`TimeStatsTransform` available stats:** `mean`, `maximum`, `minimum`, `root_mean_square`, `abs_avg`, `peak_to_peak_value`, `standard_deviation`, `skewness`, `kurtosis`, `variance`, `peak_factor`, `change_coefficient`, `clearance_factor`, `abs_energy`, `hankel_svd`.

Both output shape `(1, n_features × n_stats)`.

---

### Padding

```python
from picid.transforms.base_transforms.padding2length import PadToLength

# Pad the time axis to at least 32 steps with zeros
DataTransform("pad", PadToLength(target_length=32, axis=0, pad_value=0.0),
              {"apply_to": "features"})
```

---

### Reshaping

```python
from picid.transforms.base_transforms.reshaping import ReshapeTransform

# Uses einops pattern notation
DataTransform("reshape", ReshapeTransform(pattern="b t c -> (b t) c"), {"apply_to": "features"})
```

---

### Corruption (for robustness testing)

Injects artificial missing data — useful for evaluating robustness.

```python
from picid.transforms.base_transforms.mcar_corruption import MCARCorruptorTransform

# 10% of samples missing as random blocks
DataTransform("corrupt",
              MCARCorruptorTransform(ratios=[0.1] * 10, mode="block",
                                     block_params={"min_size": 5, "max_size": 20},
                                     seed=42),
              {"apply_to": "features"})
```

| Parameter | Description |
|---|---|
| `ratios` | List of missing-data fractions, one per channel. |
| `mode` | `"point"` (random individual samples) or `"block"` (contiguous outages). |
| `block_params` | Dict with `min_size` and `max_size` for block mode. |
| `seed` | Random seed for reproducibility. |

---

## Inverse transforms

Some evaluators undo scaling before computing metrics.
For example, `PerUnitEvaluatorConfig(inverse_transform_name="scaler_targets")` will call the inverse of the transform named `"scaler_targets"` on the predictions before computing MAE.

**Requirements:**

1. The transform must support inverse (marked with `InverseTransformMixin`): `MinMaxScalerSklearn`, `StandardScalerSklearn`, and `ConstantScaler` all do.
2. The `transform_name` you set in `DataTransform` must match the `inverse_transform_name` in the evaluator config exactly.

```python
# 1. Define the transform with a memorable name
scaler_target = DataTransform(
    transform_name="scaler_targets",           # ← this name
    transform=MinMaxScalerSklearn(),
    metadata={"apply_to": "rul", "fit_on": "train"},
)

# 2. Reference it in the evaluator
from picid.interface.schemas.evaluators import PerUnitEvaluatorConfig

evaluators = {
    "test": PerUnitEvaluatorConfig(
        inverse_transform_name="scaler_targets",  # ← must match
        apply_inverse_scaling=True,
    )
}
```

If `apply_inverse_scaling=True` but `inverse_transform_name` is not set, the pipeline automatically selects the last transform in your list that handles the target key and supports inverse. It is safer to name it explicitly.

---

## Common patterns

### RUL with scaled features and target

```python
from picid.transforms.base import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn

transforms = [
    DataTransform("scaler_features", MinMaxScalerSklearn(),
                  {"apply_to": "features", "fit_on": "train"}),
    DataTransform("scaler_targets", MinMaxScalerSklearn(),
                  {"apply_to": "rul", "fit_on": "train"}),
]
```

### Impute then scale

```python
from picid.transforms.base_transforms.imputation_methods import ImputationTransform

transforms = [
    DataTransform("impute", ImputationTransform(strategy="locf"),
                  {"apply_to": "features"}),
    DataTransform("scaler", MinMaxScalerSklearn(),
                  {"apply_to": "features", "fit_on": "train"}),
]
```

### Extract statistical features

```python
from picid.transforms.base_transforms.time_statistics import TimeStatsTransform

transforms = [
    # Extract hand-crafted features — output replaces raw signals
    DataTransform("stats",
                  TimeStatsTransform(stats_to_compute=["mean", "standard_deviation", "kurtosis"]),
                  {"apply_to": "features"}),
    # Scale the extracted features
    DataTransform("scaler", MinMaxScalerSklearn(),
                  {"apply_to": "features", "fit_on": "train"}),
]
```

### No transforms

If your data is already preprocessed, pass `transforms=None` to `train()` and make sure the datasource config defines its own transforms, or pass a pre-processed `ProcessedDatasource`.
