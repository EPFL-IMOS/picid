# Adding a New Model

This guide describes how to add a new model to the PICID pipeline so it can be
run via Hydra experiments. The repo now uses a family-first layout:

- `picid.model.forecasters` for forecasting families
- `picid.model.estimators` for non-forecaster families
- `picid.model.adapters.base` for shared adapter contracts

---

## 1. Choose the extension point

Start by deciding where the family belongs:

- **Forecasting family:** create or extend `picid/model/forecasters/<family>/`
- **Non-forecaster family:** create or extend `picid/model/estimators/<family>/`

If your model needs one of the shared adapter contracts, inherit from a base
class in `picid.model.adapters.base`:

| Base class | Use when |
|------------|----------|
| **`AbstractFitPredictWrapper`** | Model has a scikit-learn-style `fit(X, y)` and `predict(X)`; the pipeline will call these once per task (e.g. per unit or per horizon). |
| **`AbstractFeedForwardWrapper`** | Model is a callable that takes a batch dict and returns predictions; no training step inside the wrapper (e.g. pre-trained or external training). |
| **`AbstractFeedForwardTrainingWrapper`** | Model is a PyTorch module trained with the pipeline (e.g. Lightning); implements forward and training/validation logic. |

Import from `picid.model.adapters.base`:

```python
from picid.model.adapters.base import AbstractFitPredictWrapper  # or AbstractFeedForwardWrapper, AbstractFeedForwardTrainingWrapper
```

---

## 2. Implement the family entry point

For non-forecaster families, colocate the public entry points with the family:

- `picid/model/estimators/<family>/wrapper.py`
- `picid/model/estimators/<family>/model.py` when a local backbone exists

For forecasting families, follow the existing structure in
`picid/model/forecasters/<family>/`.

- **Fit-predict:** Implement `fit(X, y)` and `predict(X)` (and optionally `serialize_model` / `load_model` if the pipeline should checkpoint the backbone). Ensure `predict` returns a 2D tensor (or array convertible to tensor). Respect the `allows_multi_target` property if applicable.
- **Feed-forward:** Implement the interface expected by the pipeline (e.g. `forward(batch)` returning a dict with at least `"predictions"` and optionally `"targets"`). For training wrappers, implement `_training_step`, `_validation_step`, `_test_step` as required by the pipeline’s Lightning module.

Use **`picid.model.definitions`** for task-type checks (e.g. `REGRESSION_TASKS`, `CLASSIFICATION_TASKS`, `FORECASTING_TASKS`) so your wrapper only accepts supported task types and the evaluator receives consistent outputs.

---

## 3. Create or extend config files

Experiments are composed from **defaults** in `configs/`. You typically need:

### 3.1 Model config

Defines the model/wrapper and its constructor arguments. Often under **`configs/model/`** or referenced via a model config group.

Example pattern (path and keys may vary by experiment layout):

```yaml
# configs/model/my_model.yaml (or similar)
_target_: picid.model.estimators.my_family.wrapper.MyModelWrapper
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
  _target_: picid.model.estimators.my_family.wrapper.MyModelWrapper
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

- **`_target_`** in YAML must point to the full canonical class path, e.g.
  `picid.model.estimators.my_family.wrapper.MyModelWrapper`.
- If the project uses an explicit registry or `__init__.py` exports, add your
  family entry point there so it is importable from the canonical namespace.

No change to the pipeline core is required if your wrapper satisfies the same interface as existing ones. The pipeline and evaluators expect **model output** (the dict your wrapper returns or that the Lightning module produces) to contain at least **`"predictions"`** and **`"targets"`**; for per-unit metrics, **`"unit_id"`** is also required. See [Evaluators](../reference/evaluators.md) for the full contract.

---

## 5. Run an experiment

From the project root:

```bash
source .venv/bin/activate
python picid/run.py paths=<your_paths> experiment=<datasource>/<task>/my_model_fit_predict
```

Example (existing): `experiment=railway_traction/forecasting/tabdpt_fit_predict`. Use `debug=default` for a faster run during development.

---

## 6. Example wrappers to copy from

| Wrapper | Base class | Use case |
|---------|------------|----------|
| `estimators/tabdpt/wrapper.py` | `AbstractFitPredictWrapper` | TabDPT fit-predict |
| `estimators/tabpfn/wrapper.py` | `AbstractFitPredictWrapper` | TabPFN fit-predict |
| `estimators/xgboost/wrapper.py` | `AbstractFitPredictWrapper` | XGBoost fit-predict |
| `estimators/mlp/wrapper.py` | `AbstractFeedForwardTrainingWrapper` | MLP trained with Lightning |
| `estimators/cnn1d/wrapper.py` | `AbstractFeedForwardTrainingWrapper` | 1D CNN trained with Lightning |
| `forecasters/patchtst_model/` | forecaster base classes | Forecasting family layout example |

These families live in `picid/model/estimators/` or `picid/model/forecasters/`;
the pipeline and configs in **`configs/experiment/`** and
**`configs/model_configs/`** show how they are wired.

---

## See also

- [Model types](../reference/model_types.md) — Fit-predict vs batchwise vs full-batch; data shapes and limitations.
- [Evaluators](../reference/evaluators.md) — What the pipeline expects in `model_out` (e.g. `predictions`, `targets`, `unit_id` for multi-unit).
- [OmegaConf resolvers](omegaconf_resolvers.md) — Using resolvers in model or experiment configs.

[← Guides index](index.md) | [Back to documentation index](../index.md)
