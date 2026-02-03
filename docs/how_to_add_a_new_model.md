# Adding a New Model

This guide describes how to add a new model to the TabPHM pipeline so it can be run via Hydra experiments. The steps apply to both **fit-predict** and **batchwise (feed-forward)** models.

---

## 1. Choose the wrapper base class

All model wrappers live in **`lmetk.model.wrappers`** and inherit from one of:

| Base class | Use when |
|------------|----------|
| **`AbstractFitPredictWrapper`** | Model has a scikit-learn-style `fit(X, y)` and `predict(X)`; the pipeline will call these once per task (e.g. per unit or per horizon). |
| **`AbstractFeedForwardWrapper`** | Model is a callable that takes a batch dict and returns predictions; no training step inside the wrapper (e.g. pre-trained or external training). |
| **`AbstractFeedForwardTrainingWrapper`** | Model is a PyTorch module trained with the pipeline (e.g. Lightning); implements forward and training/validation logic. |

Import from `lmetk.model.wrappers.base`:

```python
from lmetk.model.wrappers.base import AbstractFitPredictWrapper  # or AbstractFeedForwardWrapper, AbstractFeedForwardTrainingWrapper
```

---

## 2. Implement the wrapper

Create a new file under **`lmetk/model/wrappers/`**, e.g. `my_model_wrapper.py`.

- **Fit-predict:** Implement `fit(X, y)` and `predict(X)` (and optionally `serialize_model` / `load_model` if the pipeline should checkpoint the backbone). Ensure `predict` returns a 2D tensor (or array convertible to tensor). Respect the `allows_multi_target` property if applicable.
- **Feed-forward:** Implement the interface expected by the pipeline (e.g. `forward(batch)` returning a dict with at least `"predictions"` and optionally `"targets"`). For training wrappers, implement `_training_step`, `_validation_step`, `_test_step` as required by the pipeline’s Lightning module.

Use **`lmetk.baselines.definitions`** for task-type checks (e.g. `REGRESSION_TASKS`, `CLASSIFICATION_TASKS`, `FORECASTING_TASKS`) so your wrapper only accepts supported task types and the evaluator receives consistent outputs.

---

## 3. Create or extend config files

Experiments are composed from **defaults** in `configs/`. You typically need:

### 3.1 Model config

Defines the model/wrapper and its constructor arguments. Often under **`configs/model/`** or referenced via a model config group.

Example pattern (path and keys may vary by experiment layout):

```yaml
# configs/model/my_model.yaml (or similar)
_target_: lmetk.model.wrappers.my_model_wrapper.MyModelWrapper
# ... constructor arguments (backbone, yield_strategy, etc.)
```

### 3.2 Model_config (task + model wiring)

Defines which model and datamodule/dataset setup to use for a given task. Often under **`configs/model_configs/`** (or `configs/model_config/`), e.g. **`configs/model_configs/forecasting/my_model_fit_predict.yaml`**.

Example pattern:

```yaml
# configs/model_configs/forecasting/my_model_fit_predict.yaml
defaults:
  - ../datamodule/...   # or the appropriate datamodule default
  - ../evaluator/...

model:
  _target_: lmetk.model.wrappers.my_model_wrapper.MyModelWrapper
  # ...

# task_definition, evaluator, etc. as needed
```

### 3.3 Experiment config

Ties a datasource/task to a model_config and optional transforms. Under **`configs/experiment/`**, e.g. **`configs/experiment/<datasource>/<task>/my_model_fit_predict.yaml`**.

Example (see existing experiments for exact keys):

```yaml
# @package _global_
defaults:
  - <datasource>/<task>/base           # e.g. railway_traction/forecasting/base
  - /model_configs/<task>/my_model_fit_predict
  # optional: override /transforms: ...
```

The **base** default (e.g. `railway_traction/forecasting/base`) usually sets datasource, paths, and task; the **model_config** default pulls in the model and datamodule/evaluator for that model type.

---

## 4. Register or expose the wrapper

Ensure the new wrapper can be instantiated by Hydra:

- **`_target_`** in YAML must point to the full class path, e.g. `lmetk.model.wrappers.my_model_wrapper.MyModelWrapper`.
- If the project uses an explicit registry or `__init__.py` exports, add your wrapper there so it is importable from `lmetk.model.wrappers`.

No change to the pipeline core is required if your wrapper satisfies the same interface as existing ones. The pipeline and evaluators expect **model output** (the dict your wrapper returns or that the Lightning module produces) to contain at least **`"predictions"`** and **`"targets"`**; for per-unit metrics, **`"unit_id"`** is also required. See [Evaluators](evaluators/index.md) for the full contract.

---

## 5. Run an experiment

From the project root:

```bash
source .venv/bin/activate
python lmetk/run.py paths=<your_paths> experiment=<datasource>/<task>/my_model_fit_predict
```

Example (existing): `experiment=railway_traction/forecasting/tabdpt_fit_predict`. Use `debug=default` for a faster run during development.

---

## 6. Example wrappers to copy from

| Wrapper | Base class | Use case |
|---------|------------|----------|
| `fit_predict_autogluon_wrapper` | `AbstractFitPredictWrapper` | Tabular fit-predict (AutoGluon) |
| `fit_predict_tabdpt_wrapper` | `AbstractFitPredictWrapper` | TabDPT fit-predict |
| `fit_predict_tabpfn_wrapper` | `AbstractFitPredictWrapper` | TabPFN fit-predict |
| `fit_predict_xgboost_wrapper` | `AbstractFitPredictWrapper` | XGBoost fit-predict |
| `batchcontext_tabpfn_wrapper` | `AbstractFeedForwardWrapper` | TabPFN on windowed batches |
| `mlp_wrapper` | `AbstractFeedForwardTrainingWrapper` | MLP trained with Lightning |
| `cnn1d_wrapper` | `AbstractFeedForwardTrainingWrapper` | 1D CNN trained with Lightning |

These live in **`lmetk/model/wrappers/`**; the pipeline and configs in **`configs/experiment/`** and **`configs/model_configs/`** show how they are wired.

---

## See also

- [Model types](model_types.md) — Fit-predict vs batchwise vs full-batch; data shapes and limitations.
- [Evaluators](evaluators/index.md) — What the pipeline expects in `model_out` (e.g. `predictions`, `targets`, `unit_id` for multi-unit).
- [OmegaConf resolvers](how_to_use_configs.md) — Using resolvers in model or experiment configs.
