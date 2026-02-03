# TabPHM

**TabPHM** is the Prognostics, Health Management (PHM) and Forecasting Benchmark Arena. It provides a unified framework for loading datasets, preprocessing with configurable transforms, training and evaluating models (including fit-predict and batchwise regimes), and running experiments locally or on a cluster.

---

## Where to start

- **Running your first experiment:** [Setup](setup.md) → install, set paths, then run `python lmetk/run.py paths=... experiment=...`.
- **Understanding the data flow:** [Data objects & containers](dataobject.md) (how data is structured) → [Datasources](datasources.md) (where data comes from) → [Datasets](datasets.md) (how it is turned into model inputs).
- **Adding or configuring a model:** [Model types](model_types.md) (fit-predict vs batchwise) → [Adding a new model](how_to_add_a_new_model.md).

---

## Overview

The library is built around a few core concepts:

- **Datasources** — Data-centric loaders that read raw files (HDF5, CSV, etc.) and produce standardized, split-aware data.
- **Data objects & containers** — Type-safe structures (`SplitDatasetContainer`, `DatasetContainer`) that carry features, targets, and metadata through the pipeline with dot-notation access and validation.
- **Datasets** — Task-centric PyTorch-style datasets (`ContextBatchDataset`, `RULContextBatchDataset`, `FitPredictDataset`, etc.) that format data for specific model inputs (sliding windows, fit-predict tables).
- **Transforms** — Configuration-driven preprocessing (scaling, feature extraction, domain-specific steps) with fit-on-train and multi-source strategies.
- **Evaluators** — Pluggable evaluation (regression, classification, forecasting, multi-unit, reconstruction) with metrics, inverse scaling, buffers, and hooks for saving predictions and plots.
- **Pipeline** — Orchestrates datamodules, models, optimizers, and evaluators for training and testing, with Hydra-based configuration.

---

## Documentation map

| Section | Description |
|--------|-------------|
| [Setup](setup.md) | Installation, local paths, and how to run an experiment. |
| [Data objects & containers](dataobject.md) | Core data structures: `BaseDataObject`, `DatasetContainer`, `SplitDatasetContainer`, and the flow from datasource to container. |
| [Datasets](datasets.md) | Task-centric datasets: `ContextBatchDataset`, `FitPredictDataset`, `ConceptRULDataset`, `RULContextBatchDataset`, and their batch shapes. |
| [Datasets inner workings](datasets_inner_workings.md) | Data loading architecture: `HydraConcatDataset`, sequencers (Dense/Ragged), sliding windows, and collation. |
| [Datasources](datasources.md) | Available datasources (PHM, battery, bearing, railway, ETTh, etc.) and their domains and tasks. |
| [Transforms](transforms.md) | Transform system overview with links to Quick start, User guide, and Developer guide. |
| [Evaluators](evaluators/index.md) | Evaluator subsystem: base, default, classification, forecasting, multi-unit, reconstruction, hooks, and configuration. |
| [Model types](model_types.md) | Supported model types (fit-predict, batchwise, full-batch) and current limitations. |
| [Adding a new model](how_to_add_a_new_model.md) | Step-by-step guide to adding a new model and configs. |
| [OmegaConf resolvers](how_to_use_configs.md) | Custom resolvers (`flat`, `uuid`, `quot`, `infer_data_dim`, etc.) for configs. |
| [RunAI](runai.md) | Running experiments on a Kubernetes/RunAI cluster with the provided container and scripts. |
| [Dataset viewer](viewer.md) | Interactive dataset viewer (Panel app). |
| [API Reference](api.md) | API documentation for core modules (e.g. `lmetk.data.data_objects`). |

---

## Quick start

1. **Install** — Follow [Setup](setup.md) (install `uv`, clone with submodules, `uv sync`, `uv pip install -e ".[dev]"`).
2. **Paths** — Create a path config in `configs/paths` (e.g. copy `default.yaml`) and set your data/experiment roots.
3. **Run** — From the project root:
   ```bash
   source .venv/bin/activate
   python lmetk/run.py paths=<your_paths> experiment=<experiment_key>
   ```
   Example: `experiment=railway_traction/autogluon_railway_fit_predict`. Use `debug=default` for faster runs during development.

For cluster runs, see [RunAI](runai.md).
