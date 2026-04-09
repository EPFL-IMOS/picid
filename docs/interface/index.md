# Interface Guide

The `picid` interface is a high-level Python API that lets you run experiments without writing or editing any Hydra configuration file.
Everything you would normally express in YAML — models, task parameters, datasources, evaluators — is expressed as typed Python objects instead.

---

## Sections

### [Introduction](intro.md)
Start here. Explains what the interface does, how it fits into the project, and gives you the mental model you need before reading anything else.

### [Transforms](transforms.md)
How to configure data preprocessing steps. Covers `DataTransform`, the `metadata` dict (`apply_to`, `fit_on`, `assign_to`), all available transform classes (scalers, imputation, statistical features, …), and how to wire inverse transforms to evaluators.

### [Configuration Schemas](schemas.md)
Reference for every Pydantic config class: task definitions (`Prognostic`, `Forecasting`), model configs (`LSTMConfig`, `MLPConfig`, …), evaluator configs, trainer options, and loggers. Each section lists every field with its type, default, and description.

### [Custom Datasources](datasources.md)
How to bring your own data — a NumPy array, a pandas DataFrame, or a set of multiple sources — and pass it to the training pipeline. Covers `CustomSingleSourceLoader`, `CustomMultiSourceLoader`, splitting strategies, and the `process_datasource()` workflow.

### [Custom Models](custom-model.md)
How to wrap any PyTorch `nn.Module` and use it with the interface. Covers `ModelWrapper` (pre/post-processing hooks) and `CustomModelTrainer` (the adapter required by `train()`).

### [Examples](examples.md)
Five complete, copy-paste-ready examples that progress from the simplest case (built-in data + built-in model) to fully custom data and models.
