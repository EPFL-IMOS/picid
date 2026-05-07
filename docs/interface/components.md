# Interface Components

Quick-reference for every object you can pass to `EntryInterface.train()`.

---

## Required components

| Component | Python type(s) | Purpose |
|---|---|---|
| `model` | `AbsModelConfig` subclass · `str` · `CustomModelTrainer` | Architecture and hyperparameters |
| `task_definition` | `Prognostic` · `Forecasting` · `tuple[str, str]` | Window size, stride, task type |
| `datasource` | `str` · `tuple[str, str]` · `ProcessedDatasource` · loader instance | Where the data comes from |
| `evaluators` | `dict[str, AbsEvalConfig]` · `str` · `None` | Metrics computed per split |

---

## Optional components

| Component | Python type(s) | Default | Purpose |
|---|---|---|---|
| `transforms` | `list[DataTransform]` | `None` | Preprocessing steps (scaling, imputation, …) |
| `training_config` | `TrainerConfig` | framework default | Epochs, accelerator, gradient clipping, … |
| `loggers` | `list` | `None` | CSV, WandB, or other Lightning loggers |
| `callbacks` | `list` | `None` | Custom Lightning callbacks |
| `overrides` | `list[str]` | `None` | Raw Hydra override strings |
| `seed` | `int` | `None` | Global random seed |

---

## Model configs

Built-in model config classes, importable from `picid.interface.schemas.model`:

| Class | Architecture |
|---|---|
| `LSTMConfig` | LSTM |
| `MLPConfig` | Multi-layer perceptron |
| `CNNConfig` | 1-D CNN |
| `PatchTSTConfig` | PatchTST Transformer |
| `TiDEConfig` | TiDE MLP-Mixer |
| `CrossformerConfig` | Crossformer Transformer |
| `STFConfig` | Spacetimeformer |
| `TimeLLMConfig` | TimeLLM (LLM-based) |
| `XGBoostConfig` | XGBoost fit-predict |
| `TabPFNConfig` | TabPFN foundation model |
| `TabDPTConfig` | TabDPT foundation model |
| `CARTEConfig` | CARTE foundation model |

Pass a string (e.g. `"lstm"`) to load the default config, or instantiate the class to override hyperparameters.

---

## Task definitions

Importable from `picid.interface.schemas.task_definition`:

| Class | Task type | Key fields |
|---|---|---|
| `Prognostic` | RUL estimation | `task_type`, `seq_len`, `pred_len`, `stride` |
| `Forecasting` | Multi-step ahead prediction | `task_type`, `seq_len`, `pred_len`, `label_len`, `stride` |

---

## Evaluator configs

Importable from `picid.interface.schemas.evaluators`:

| Class | Suitable for |
|---|---|
| `RulEvaluatorConfig` | Prognostics (RUL) |
| `ForecastingEvaluatorConfig` | Forecasting |
| `ClassificationEvaluatorConfig` | Fault classification / diagnostics |

Pass a `dict` mapping split names to configs: `{"train": RulEvaluatorConfig(), "val": ..., "test": ...}`.

---

## Further reading

- [Introduction](intro.md) — narrative walkthrough and execution flow
- [Schemas](schemas.md) — full field reference for every config class
- [Transforms](transforms.md) — preprocessing pipeline and `DataTransform`
- [Custom Datasources](datasources.md) — `CustomSingleSourceLoader`, `CustomMultiSourceLoader`
- [Custom Models](custom-model.md) — `CustomModelTrainer`, `ModelWrapper`
- [Examples](examples.md) — copy-paste complete experiments
