# Configuration Schemas

Every component of a `train()` call has a corresponding Pydantic schema.
Schemas are ordinary Python dataclasses with type validation.
You create an instance, optionally override fields, and pass the instance to `train()`.

> **Under the hood:** Each schema serializes to a dict via `model_dump(by_alias=True)`.
> The `model_class` field (aliased to `_target_`) tells Hydra which Python class to instantiate.
> You never need to touch this field directly.

---

## Task definitions

A task definition describes how raw time-series data is windowed and what the model should predict.

### `BaseTaskDefinition` — common fields

| Field | Type | Default | Description |
|---|---|---|---|
| `seq_len` | `int` | `16` | Input window length (number of time steps fed to the model). |
| `label_len` | `int` | `0` | Overlap between the input window and the prediction horizon (used by some Transformer models). |
| `pred_len` | `int` | `1` | Number of future time steps to predict. Set to `0` for prognostics. |
| `stride` | `int` | `1` | Sliding-window step size during evaluation. |
| `stride_train` | `int` | `1` | Sliding-window step size during training. |
| `subset_ratio` | `float` | `1.0` | Fraction of training windows to keep (useful for fast prototyping). |
| `padding_left_flag` | `bool` | `False` | Pad the left edge of the series so every sample has a full window. |
| `target_metric` | `str` | `"val/loss"` | Metric used for model checkpointing. |
| `target_metric_mode` | `str` | `"min"` | Whether to minimize or maximize `target_metric`. |

---

### `Prognostic`

For **RUL** (Remaining Useful Life) and **AHRUL** (Adapted Horizon RUL) tasks.

```python
from picid.interface.schemas.task_definition import Prognostic

task = Prognostic(task_type="rul")           # basic RUL
task = Prognostic(task_type="ahrul")         # adapted-horizon RUL
task = Prognostic(task_type="rul", seq_len=32, stride=2)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `task_type` | `"rul"` or `"ahrul"` | required | The prediction target key in the data dictionary. |
| `pred_len` | `int` | `0` | Fixed to 0 for prognostics (no future horizon). |

All `BaseTaskDefinition` fields are also available.

---

### `Forecasting`

For **multi-step time-series forecasting** tasks.

```python
from picid.interface.schemas.task_definition import Forecasting

task = Forecasting(seq_len=96, pred_len=24)
task = Forecasting(seq_len=336, label_len=48, pred_len=96)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `task_type` | `str` | `"forecasting"` (frozen) | Fixed identifier; do not change. |
| `pred_offset` | `int` | `0` | Time offset before the prediction horizon starts. |
| `input_tensors` | `list[str]` | `["features", "time_features", "target"]` | Tensors loaded from the dataset. |

All `BaseTaskDefinition` fields are also available.

---

## Trainer

`TrainerConfig` wraps the PyTorch Lightning `Trainer`.
Pass it as `training_config=` to `train()`.
When omitted, the project's default trainer YAML is used.

```python
from picid.interface.schemas import TrainerConfig

cfg = TrainerConfig(max_epochs=50, accelerator="gpu", devices=[0])
interface.train(..., training_config=cfg)
```

Alternatively, use raw Hydra overrides for one-off changes:

```python
interface.train(..., overrides=["trainer.max_epochs=50", "trainer.accelerator=gpu"])
```

| Field | Type | Default | Description |
|---|---|---|---|
| `max_epochs` | `int` | `10` | Maximum training epochs. |
| `min_epochs` | `int` | `1` | Minimum training epochs. |
| `accelerator` | `"cpu"` or `"gpu"` | `"cpu"` | Hardware accelerator. |
| `devices` | `list[int]` | `[0]` | Device indices to use. |
| `check_val_every_n_epoch` | `int` | `0` | How often to run validation. `0` means after every epoch. |
| `deterministic` | `bool` | `True` | Enable deterministic mode for reproducibility. |
| `inference_mode` | `bool` | `True` | Use `torch.inference_mode()` during evaluation. |

---

## Models

All model configs inherit from `AbsModelConfig`.
Import them from `picid.interface.schemas.model`.

---

### `LSTMConfig`

```python
from picid.interface.schemas.model import LSTMConfig

model = LSTMConfig()                          # defaults
model = LSTMConfig(n_layers=8, hidden_dim=64)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `hidden_dim` | `int` | `32` | Hidden state size per LSTM layer. |
| `n_layers` | `int` | `2` | Number of stacked LSTM layers. |
| `d_x` | `int \| None` | `None` | Input feature dimension (inferred from data when `None`). |
| `d_yt` | `int \| None` | `None` | Target dimension (inferred when `None`). |
| `d_yc` | `int \| None` | `None` | Context dimension (inferred when `None`). |

---

### `MLPConfig`

```python
from picid.interface.schemas.model import MLPConfig

model = MLPConfig(input_channels=10, num_targets=1)
model = MLPConfig(input_channels=10, num_targets=1, hidden_dim=128)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `input_channels` | `int` | required | Number of input features. |
| `num_targets` | `int` | required | Number of outputs. |
| `hidden_dim` | `int` | `64` | Hidden layer width. |

---

### `CNN1DConfig`

```python
from picid.interface.schemas.model import CNN1DConfig

model = CNN1DConfig(
    input_channels=10,
    seq_len=16,
    kernels=5,
    strides=1,
    dilations=1,
    latent_dim=128,
    dropout_prob=0.1,
    output_channels=64,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `input_channels` | `int` | required | Number of input feature channels. |
| `seq_len` | `int` | required | Length of the input window. |
| `kernels` | `int` | required | Convolutional kernel size. |
| `strides` | `int` | required | Convolutional stride. |
| `dilations` | `int` | required | Dilation factor. |
| `latent_dim` | `int` | required | Encoder output dimension. |
| `dropout_prob` | `float` | required | Dropout probability (0.1 – 1.0). |
| `output_channels` | `int` | required | Number of output channels. |

---

### `CrossformerConfig`

Transformer-based model for multivariate forecasting.

```python
from picid.interface.schemas.model import CrossformerConfig

model = CrossformerConfig(dropout=0.1)
model = CrossformerConfig(dropout=0.2, d_model=256, n_heads=8, e_layers=4)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `dropout` | `float` | required | Dropout rate (0.1 – 1.0). |
| `d_model` | `int` | `128` | Model embedding dimension. |
| `d_ff` | `int` | `128` | Feed-forward layer dimension. |
| `n_heads` | `int` | `4` | Number of attention heads. |
| `e_layers` | `int` | `3` | Number of encoder layers. |
| `seg_len` | `int` | `6` | Segment length for the cross-dimension attention. |
| `win_size` | `int` | `2` | Window size for hierarchical attention. |
| `factor` | `int` | `10` | Attention factor. |
| `use_revin` | `bool` | `False` | Enable reversible instance normalization. |
| `use_seasonal_decomp` | `bool` | `False` | Enable seasonal decomposition. |
| `d_x`, `d_yt`, `d_yc`, `ts_in`, `ts_out` | `int \| None` | `None` | Dimensions inferred from data when `None`. |

---

### `LinearForecasterConfig`

Linear model for forecasting tasks.

```python
from picid.interface.schemas.model import LinearForecasterConfig

model = LinearForecasterConfig(linear_window=12)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `linear_window` | `int` | `0` | Size of the linear look-back window. |
| `linear_shared_weights` | `bool` | `False` | Share weights across feature dimensions. |
| `use_revin` | `bool` | `False` | Enable reversible instance normalization. |
| `use_seasonal_decomp` | `bool` | `False` | Enable seasonal decomposition. |
| `context_points` | `int \| None` | `None` | Context length (inferred when `None`). |

---

### `LinearRegressionConfig`

Statistical linear-regression baseline.

```python
from picid.interface.schemas.model import LinearRegressionConfig

model = LinearRegressionConfig(
    pred_len=1, label_len=0, seq_len=16,
    input_channels=10, num_targets=1,
    model_type="linear",
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `pred_len` | `int` | required | Prediction horizon. |
| `label_len` | `int` | required | Label overlap length. |
| `seq_len` | `int` | required | Input window length. |
| `input_channels` | `int` | required | Number of input features. |
| `num_targets` | `int` | required | Number of outputs. |
| `model_type` | `str` | required | Regression model variant (e.g. `"linear"`). |

---

### `NaiveConfig` and `MeanConfig`

Non-learning baselines.

```python
from picid.interface.schemas.model import NaiveConfig, MeanConfig

model = NaiveConfig(pred_len=1, label_len=0, seq_len=16, features_mode="M")
model = MeanConfig(pred_len=1, label_len=0, seq_len=16,
                   window_size_to_average=4, features_mode="M")
```

| Field | Type | Default | Description |
|---|---|---|---|
| `pred_len` | `int` | required | Prediction horizon. |
| `label_len` | `int` | required | Label overlap. |
| `seq_len` | `int` | required | Input window length. |
| `window_size_to_average` | `int` | `1` (`NaiveConfig`) / required (`MeanConfig`) | Number of past steps to average. |
| `features_mode` | `str` | required | Feature mode string (e.g. `"M"`, `"S"`). |

---

## Evaluators

Evaluators compute metrics at the end of each training/validation/test epoch.
Pass them as a dict keyed by split name:

```python
from picid.interface.schemas.evaluators import RulEvaluatorConfig

evaluators = {s: RulEvaluatorConfig() for s in ["train", "val", "test"]}
interface.train(..., evaluators=evaluators)
```

All evaluator configs share these base fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `metric_names` | `list[str]` | varies | Metrics to compute. |
| `save_predictions` | `bool` | `False` | Persist model predictions to disk. |
| `apply_inverse_scaling` | `bool` | `False` | Undo scaling transforms before computing metrics. |
| `hooks` | `list[AbsHookConfig]` | varies | Post-evaluation hooks (plots, persistence). |

---

### `DefaultEvaluatorConfig`

General-purpose regression evaluator.

```python
from picid.interface.schemas.evaluators import DefaultEvaluatorConfig

ev = DefaultEvaluatorConfig()
ev = DefaultEvaluatorConfig(metric_names=["mae", "mse"])
```

Default metrics: `mae`, `mse`, `rmse`.

---

### `RulEvaluatorConfig`

Extends `DefaultEvaluatorConfig` with the NASA score, tailored for prognostics.

```python
from picid.interface.schemas.evaluators import RulEvaluatorConfig

ev = RulEvaluatorConfig()
```

Default metrics: `mae`, `mse`, `rmse`, `nasa_score`.

---

### `PerUnitEvaluatorConfig`

Evaluates per individual unit (machine / component).
Useful when the test set contains multiple distinct machines.

```python
from picid.interface.schemas.evaluators import PerUnitEvaluatorConfig

ev = PerUnitEvaluatorConfig()
ev = PerUnitEvaluatorConfig(log_image=True, inverse_transform_name="scaler_targets")
```

| Field | Type | Default | Description |
|---|---|---|---|
| `log_image` | `bool` | `False` | Log per-unit prediction plots to the logger. |
| `inverse_transform_name` | `str \| None` | `None` | Name of the transform to undo before plotting. |

Default metrics: `phm_score`, `mae`, `mse`, `rmse`.
Default hooks: `SavePredictionsHookConfig`, `UnitTrendPlotHookConfig`.

---

### `ClassificationEvaluatorConfig`

For classification tasks.

```python
from picid.interface.schemas.evaluators import ClassificationEvaluatorConfig

ev = ClassificationEvaluatorConfig()
```

Default metrics: `f1`, `accuracy`, `precision`, `recall`, `auroc`.

---

### `ForecastingEvaluatorConfig`

For multi-step forecasting tasks. Automatically handles inverse scaling and aligns prediction windows.

```python
from picid.interface.schemas.evaluators import ForecastingEvaluatorConfig

ev = ForecastingEvaluatorConfig()
ev = ForecastingEvaluatorConfig(log_image=True, inverse_transform_name="scaler_features")
```

| Field | Type | Default | Description |
|---|---|---|---|
| `log_image` | `bool` | `False` | Log forecast plots to the logger. |
| `inverse_transform_name` | `str \| None` | `None` | Name of the transform to undo before computing metrics. |
| `apply_inverse_scaling` | `bool` | `True` | Apply inverse scaling by default. |
| `target_dim_position` | `int \| None` | `None` | Which feature dimension is the forecast target. |

Default metrics: `mae`, `mse`.

---

## Loggers

Loggers record metrics and artifacts during training.
Pass a list to `train()` via `loggers=`.

### `CsvLogger`

Writes metrics to a CSV file. No external service required.

```python
from picid.interface.schemas.loggers import CsvLogger

logger = CsvLogger(name="my_run")
logger = CsvLogger(name="my_run", version="v1")
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Run name used as the log directory. |
| `version` | `str \| None` | `None` | Optional version sub-directory. |
| `prefix` | `str` | `""` | Prefix prepended to all logged metric names. |

---

### `WandbLogger`

Logs to [Weights & Biases](https://wandb.ai/).

```python
from picid.interface.schemas.loggers import WandbLogger

logger = WandbLogger(name="my_run", entity="my_team", project="phm_experiments")
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Run display name. |
| `entity` | `str` | required | W&B team or user name. |
| `project` | `str` | `"best_runs"` | W&B project name. |
| `group` | `str` | `""` | Optional group for run organization. |
| `tags` | `list[str]` | `[]` | Tags attached to the run. |
| `offline` | `bool` | `False` | Log locally without uploading. |
| `log_model` | `bool` | `False` | Upload model checkpoints to W&B. |
