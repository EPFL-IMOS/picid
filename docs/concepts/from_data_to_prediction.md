# From data to prediction: how one PHM run works

This page walks through the data flow of a single PHM experiment in PICID. We use the **Unibo battery RUL** setup as a running example (datasource `unibo21_bosello`, task `prognostics/rul`, transforms such as `battery/unibo/raw`, `battery/unibo/combined`, or ablation variants).

---

## 1. Datasource loads raw data into containers

A **datasource** is the entry point for data: it reads raw files (HDF5, CSV, etc.) and produces a standardized, split-aware container. For Unibo RUL, the datasource is configured via `configs/datasource/unibo21_bosello.yaml` and yields train/val/test splits with features, targets, and metadata.

→ See [Reference: Datasources](../reference/datasources.md) for the full list and options.

---

## 2. Data objects hold splits and metadata

The output of the datasource is held in **data objects** (e.g. `SplitDatasetContainer`, `DatasetContainer`). These type-safe structures carry features, targets, and metadata with dot-notation access and validation. The pipeline and datasets consume these containers.

→ See [Reference: Data objects & containers](../reference/dataobject.md).

---

## 3. Datasets build model inputs

**Datasets** are task-centric (e.g. RUL context, fit-predict tables). They take the container output and produce PyTorch-style batches: e.g. sliding windows for sequence models or tabular windows for fit-predict models. For Unibo prognostics, the config uses an RUL multi-unit dataset with `get_unit_id: true` so unit identifiers are available in the batch.

→ See [Reference: Datasets](../reference/datasets.md) and [Datasets inner workings](../reference/datasets_inner_workings.md).

---

## 4. Transforms preprocess (fit-on-train, keys)

**Transforms** are configuration-driven preprocessing steps: scaling, feature extraction, domain-specific logic. They support fit-on-train (e.g. scaler fitted on train only), input/output keys, and multi-source strategies. The experiment chooses a transform *variant* via config (e.g. `override /transforms: battery/unibo/raw` vs `battery/unibo/combined` or ablation missing-values).

→ See [Reference: Transforms](../reference/transforms.md) and the transform [Quick start](../transforms/quick_start.md) and [Guide](../transforms/guide.md).

---

## 5. Model and training

The **model** receives batches from the dataloader (shapes defined by the task and dataset). Training is driven by the pipeline (Lightning, fit-predict loops, etc.) and the chosen optimizer and callbacks (e.g. early stopping, checkpointing), all configured via YAML.

→ See [Reference: Model types](../reference/model_types.md) and [Guides: Adding a new model](../guides/how_to_add_a_new_model.md).

---

## 6. Evaluators and metrics

**Evaluators** compute metrics (e.g. PHM score, MAE, RMSE for RUL), apply inverse scaling when needed, and can run hooks (e.g. save predictions, unit trend plots). For Unibo prognostics, the per-unit evaluator is used with `inverse_transform_name: scaler_target`.

→ See [Reference: Evaluators](../reference/evaluators.md).

---

## 7. Pipeline ties it together (Hydra, config)

The **pipeline** orchestrates datamodules, models, optimizers, and evaluators. Configuration is Hydra-based: the **experiment** key (e.g. `unibo/prognostics/raw/cnn_1d`) composes datasource, task_definition, dataset, transforms, model, and evaluator via `defaults` and `override` in YAML. Running `python picid/run.py paths=<paths> experiment=unibo/prognostics/raw/cnn_1d` runs this full flow.

→ See [Guides: Config and overrides](../guides/configs_and_overrides.md) and [Examples: Unibo RUL](../examples/unibo_rul.md).

---

[← Concepts index](index.md) | [Back to documentation index](../index.md)
