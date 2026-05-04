# Introduction

## What the interface is

`picid` uses [Hydra](https://hydra.cc/) configuration files internally to assemble experiments.
Hydra is powerful, but learning its YAML conventions is a barrier when you just want to run a quick experiment or plug in your own data.
The interface is a helper class for building custom pipelines in the project.

The interface class is `EntryInterface`, implemented in `picid/interface/interface.py`, and it manages the training pipeline.

The `picid.interface` package also exposes lightweight datasource helpers (`CustomSingleSourceLoader`, `CustomMultiSourceLoader`) without importing the full training stack. `EntryInterface` is loaded lazily, so datasource-only workflows and tests can run without pulling plotting/evaluator extras at import time.

`EntryInterface` is a single Python class that accepts ordinary Python objects — Pydantic configs, NumPy arrays, PyTorch modules — and handles the Hydra plumbing for you.
You describe an experiment in Python; the interface translates it into the internal format and runs it.

```python
from picid.interface import EntryInterface
from picid.interface.schemas.model import LSTMConfig
from picid.interface.schemas.task_definition import Prognostic
from picid.interface.schemas.evaluators import RulEvaluatorConfig
from picid.interface.schemas.loggers import CsvLogger
from picid.transforms.base import DataTransform
from picid.transforms.base_transforms.scaler import MinMaxScalerSklearn
```

To check the available datasources, use the static function `get_available_datasources()`, which returns a list of strings.

```python
interface = EntryInterface()
datasource_names = EntryInterface.get_available_datasources()

interface.train(
    run_name="my_first_run",
    model=LSTMConfig(n_layers=4),
    task_definition=Prognostic(task_type="rul"),
    datasource="phme20",
    transforms=[
        DataTransform("scaler", MinMaxScalerSklearn(),
                      metadata={"apply_to": "features", "fit_on": "train"}),
    ],
    evaluators={s: RulEvaluatorConfig() for s in ["train", "val", "test"]},
    loggers=[CsvLogger(name="csv")],
)
```

---

## The four building blocks

Every call to `train()` requires four things:

| Building block | What it is | Covered in |
|---|---|---|
| **Task definition** | Window size, stride, task type (RUL, forecasting, …) | [Schemas — Task definitions](schemas.md#task-definitions) |
| **Model** | Which architecture to use and its hyperparameters | [Schemas — Models](schemas.md#models) or [Custom models](custom-model.md) |
| **Datasource** | Where the data comes from | [Custom datasources](datasources.md) |
| **Evaluators** | Which metrics to compute on each split | [Schemas — Evaluators](schemas.md#evaluators) |

Transforms, loggers, callbacks, and trainer settings are optional.

---

## `EntryInterface` methods

### `train()`

The main entry point. Runs the full pipeline: config composition → data preprocessing → training → testing.

```python
interface.train(
    run_name: str,                            # unique name for this run
    model,                                    # AbsModelConfig | CustomModelTrainer | str
    task_definition,                          # BaseTaskDefinition | tuple[str, str]
    datasource,                               # str | tuple | ProcessedDatasource
    training_config: TrainerConfig = None,    # override trainer defaults
    transforms: list[DataTransform] = None,   # preprocessing steps
    evaluators = None,                        # dict[split, AbsEvalConfig] | str
    callbacks = None,                         # Lightning callbacks
    loggers: list = None,                     # Lightning / picid loggers
    overrides: list[str] = None,              # raw Hydra override strings
    seed: int = None,                         # random seed
    enable_progress_bar: bool = True,
    debug: bool = False,                      # skip actual training
)
```

**`model`** accepts:

- An `AbsModelConfig` subclass (e.g. `LSTMConfig(n_layers=8)`) — uses the built-in model and its YAML backbone.
- A string (e.g. `"lstm"`) — loads the default config for that model.
- A `CustomModelTrainer` instance — uses your own PyTorch module. See [Custom models](custom-model.md).

**`datasource`** accepts:

- A string (e.g. `"phme20"`) — uses a built-in registered datasource.
- A `tuple[str, str]` (e.g. `("phme20", "raw")`) — loads a complete pre-configured experiment file.
- A `ProcessedDatasource` — data you have already preprocessed via `process_datasource()`.

**`evaluators`** accepts:

- A `dict` mapping split names to `AbsEvalConfig` instances, e.g. `{"train": RulEvaluatorConfig(), "val": RulEvaluatorConfig(), "test": RulEvaluatorConfig()}`.
- A string naming a built-in evaluator config file.
- `None` — no evaluation.

**`overrides`** lets you pass raw Hydra strings to tweak any config value without creating a new schema object, e.g. `overrides=["trainer.max_epochs=50", "trainer.accelerator=gpu"]`.

---

### `process_datasource()`

Applies a list of transforms to a datasource and returns a `ProcessedDatasource` object ready to pass to `train()`.

```python
processed = interface.process_datasource(
    datasource,             # str | AbstractDataSourceLoader
    transforms,             # list[DataTransform]
)
```

Use this when you want to inspect or cache the preprocessed data before training, when you want to re-use the same preprocessing result across multiple `train()` calls, or when you are using a custom DataSource (see [custom datasources](datasources.md)).

---

### `get_datasource()` and `get_available_datasources()`

```python
# Get a datasource object by name
ds = interface.get_datasource("phme20")

# List all available built-in datasource names
names = EntryInterface.get_available_datasources()
```

---

## Execution flow

When you call `train()`, the interface:

1. **Composes the config** — converts every Python object into a Hydra `DictConfig` and calls `hydra.compose()`.
2. **Loads and preprocesses data** — if you passed a raw datasource, it calls `process_datasource()` automatically.
3. **Builds datasets and the datamodule** — uses the composed config to instantiate task-specific datasets.
4. **Sets up loggers and evaluators** — including inverse-transform resolution if your evaluator requests it.
5. **Creates the Lightning module** — wraps the backbone in the appropriate Lightning wrapper based on its type.
6. **Runs `trainer.fit()` then `trainer.test()`** — returns the test results.

---

## When to use built-in vs custom components

**Use a built-in datasource** (`datasource="phme20"`) when the dataset is already registered in `picid/config/datasource/`.
Run `EntryInterface.get_available_datasources()` to see the full list.

**Use a custom datasource** (`CustomSingleSourceLoader`, `CustomMultiSourceLoader`) when you have your own NumPy arrays or DataFrames.
See [Custom datasources](datasources.md).

**Use a built-in model** (`LSTMConfig`, `MLPConfig`, …) when one of the registered architectures fits your needs.
See [Schemas — Models](schemas.md#models).

**Use `CustomModelTrainer`** when you have your own PyTorch `nn.Module`.
See [Custom models](custom-model.md).
Internally, `picid` uses Hydra config files to build an experiment. The `train()` function bridges the Python interface and Hydra composition by converting Python inputs into config fragments that are used to create and run the final experiment config.

Most model config files expose a `model_class` parameter that points to the associated class within the module.

## MultiSourceLoader constructor modes

`MultiSourceLoader` supports two equivalent source wiring styles:

- **Config-driven mode**: pass per-source Hydra configs (dict/DictConfig) and let the loader instantiate children.
- **Instance-driven mode**: pass instantiated child loaders directly in `source_list`.

For source-level splitters, both forms are accepted:

- an instantiated `BySourceSplitter`
- a Hydra config for `BySourceSplitter`
