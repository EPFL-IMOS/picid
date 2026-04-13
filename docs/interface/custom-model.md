# Custom Models

The interface lets you plug in any PyTorch `nn.Module` through two adapters:

- **`ModelWrapper`** — adds optional pre/post-processing to any module.
- **`CustomModelTrainer`** — connects the wrapped model to the training pipeline.

```python
from picid.interface.model import CustomModelTrainer
from picid.interface.model import ModelWrapper
```

---

## Overview

The training pipeline expects a backbone that follows a specific contract: its `forward()` must accept a batch dict and return a predictions/targets dict.
`CustomModelTrainer` provides this adapter so you can use any model that takes a plain tensor as input.

The typical setup is:

```
your nn.Module
      ↓  (wrapped by)
ModelWrapper          ← adds permute / normalize / reshape if needed
      ↓  (wrapped by)
CustomModelTrainer    ← implements the batch contract, passed to train()
```

---

## `ModelWrapper`

`ModelWrapper` is an `nn.Module` that delegates its `forward()` to an inner model, with hooks for optional pre-processing (before the forward pass) and post-processing (after).

```python
from picid.interface.model.wrapper import ModelWrapper
import torch.nn as nn

backbone = MyModel(...)   # any nn.Module

wrapped = ModelWrapper(
    model=backbone,
    pre_process_function=lambda x: x.permute(0, 2, 1),  # (B, T, C) → (B, C, T)
    post_process_function=None,   # identity by default
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | The model to wrap. |
| `pre_process_function` | `Callable \| None` | `None` (identity) | Applied to the input tensor before the forward pass. |
| `post_process_function` | `Callable \| None` | `None` (identity) | Applied to the model output after the forward pass. |

**Attribute delegation:** Any attribute not found on `ModelWrapper` is forwarded to the inner model, so `wrapped.some_attribute` works the same as `backbone.some_attribute`.

You can skip `ModelWrapper` if your model does not need any input reshaping.

---

## `CustomModelTrainer`

`CustomModelTrainer` extends `AbstractFeedForwardTrainingWrapper` and implements the batch contract that the training pipeline expects.
It extracts `"features"` and the target tensor from the incoming batch dict, runs the inner model, and returns a `{"predictions": ..., "targets": ...}` dict.

```python
from picid.interface.model import CustomModelTrainer

trainer_model = CustomModelTrainer(
    task_type="rul",    # must match the task_type in your Prognostic/Forecasting config
    model=wrapped,      # ModelWrapper or bare nn.Module
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task_type` | `str` | required | The key used to retrieve targets from the batch (e.g. `"rul"`, `"ahrul"`, `"target"`). Must match the `task_type` of your `BaseTaskDefinition`. |
| `model` | `nn.Module` | required | The backbone (plain or wrapped). |
| `pre_process_function` | `Callable \| None` | `default_pre_process` | Extracts `["features", task_type]` from the batch dict. Override only if you need a different extraction strategy. |

### What `task_type` must be

`task_type` must be one of the values recognized by the pipeline:

- For RUL / regression tasks: any value in `REGRESSION_TASKS` (e.g. `"rul"`, `"ahrul"`, `"target"`).
- For classification tasks: any value in `CLASSIFICATION_TASKS`.

It must also match the `task_type` field of the `BaseTaskDefinition` you pass to `train()`.

---

## Passing a custom model to `train()`

Pass the `CustomModelTrainer` instance directly as the `model=` argument.
You **must** also provide a `task_definition` (it cannot be a raw dict when using a custom model).

```python
interface.train(
    run_name="custom_model_run",
    model=trainer_model,            # CustomModelTrainer instance
    task_definition=task,           # required; must be a BaseTaskDefinition
    datasource=processed,
    evaluators={s: RulEvaluatorConfig() for s in ["train", "val", "test"]},
    loggers=[CsvLogger(name="csv")],
    overrides=["trainer.max_epochs=20"],
)
```

---

## Full example: MLP for RUL

```python
import torch.nn as nn
from picid.interface import EntryInterface
from picid.interface.model import CustomModelTrainer
from picid.interface.model.wrapper import ModelWrapper
from picid.interface.schemas.task_definition import Prognostic
from picid.interface.schemas.evaluators import RulEvaluatorConfig
from picid.interface.schemas.loggers import CsvLogger


# 1. Define your PyTorch model
class SimpleMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)


# 2. Define the task
task = Prognostic(task_type="rul", seq_len=16)

# 3. Instantiate your model
#    in_features = seq_len * n_features (after flatten)
backbone = SimpleMLP(in_features=16 * 10, out_features=1)

# 4. (Optional) Wrap with ModelWrapper if you need input reshaping
wrapped = ModelWrapper(
    model=backbone,
    pre_process_function=lambda x: x,   # already (B, T, C) — no reshape needed
)

# 5. Wrap in CustomModelTrainer
custom_model = CustomModelTrainer(task_type=task.task_type, model=wrapped)

# 6. Train
interface = EntryInterface()
interface.train(
    run_name="mlp_rul",
    model=custom_model,
    task_definition=task,
    datasource=processed,   # ProcessedDatasource from process_datasource()
    evaluators={s: RulEvaluatorConfig() for s in ["train", "val", "test"]},
    loggers=[CsvLogger(name="mlp_rul")],
    overrides=["trainer.max_epochs=30"],
)
```

---

## Full example: 1-D CNN for RUL

```python
from picid.model.methods.cnn_1d import EncoderModel
from picid.interface.model.wrapper import ModelWrapper
from picid.interface.model import CustomModelTrainer
from picid.interface.schemas.task_definition import Prognostic

task = Prognostic(task_type="rul", seq_len=16)

backbone = EncoderModel(
    config={
        "input_channels": 10,
        "latent_dim": 128,
        "dropout_prob": 0.2,
        "output_channels": [16, 32, 64],
        "kernels": [5, 5, 5],
        "strides": [1, 1, 1],
        "dilations": [1, 2, 4],
        "input_seq_len": task.seq_len,
    },
    task_type=task.task_type,
    num_classes=1,
)

# Conv layers expect (B, C, T); the batch arrives as (B, T, C) → permute
wrapped = ModelWrapper(
    model=backbone,
    pre_process_function=lambda x: x.permute(0, 2, 1),
)

custom_model = CustomModelTrainer(task_type=task.task_type, model=wrapped)
```

---

## Choosing between a built-in model and `CustomModelTrainer`

| Situation | Recommendation |
|---|---|
| One of the registered architectures (LSTM, MLP, CNN1D, …) fits your needs | Use `LSTMConfig`, `MLPConfig`, etc. Less code, automatic hyperparameter wiring. |
| You have your own `nn.Module` and it takes a plain tensor input | Use `ModelWrapper` + `CustomModelTrainer`. |
| Your model needs a completely custom forward pass (e.g. multi-input, ensemble) | Implement `AbstractFeedForwardTrainingWrapper` directly instead of using `CustomModelTrainer`. |
