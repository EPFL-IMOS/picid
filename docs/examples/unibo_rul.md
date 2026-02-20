# Unibo battery RUL: from data to prediction

This example walks through running a **Remaining Useful Life (RUL)** experiment on the Unibo battery–style setup, and shows how to switch preprocessing (transforms) and variants via config only.

---

## What you’ll run

From the project root (with environment activated and paths configured):

```bash
python picid/run.py paths=<your_paths> experiment=unibo/prognostics/raw/cnn_1d
```

For faster runs during development, use the debug config:

```bash
python picid/run.py paths=<your_paths> debug=default experiment=unibo/prognostics/raw/cnn_1d
```

---

## Config layout

The experiment key **`unibo/prognostics/raw/cnn_1d`** resolves to:

`configs/experiment/unibo/prognostics/raw/cnn_1d.yaml`

That file composes:

1. **Base (domain + task):** `unibo/prognostics/base` — sets datasource, evaluator, task_definition, dataset options.
2. **Model:** `/model_configs/prognostics/cnn_1d`.
3. **Transforms (override):** `override /transforms: battery/unibo/raw`.

### Base configs

- **`configs/experiment/unibo/base.yaml`** — Top-level Unibo defaults (datasource `unibo21_bosello`, evaluator, task_definition). The prognostics base overrides task to `prognostics/rul` and evaluator to `per_unit`.
- **`configs/experiment/unibo/prognostics/base.yaml`** — Datasource `unibo21_bosello`, evaluator `per_unit`, task_definition `prognostics/rul`, inverse scaling for RUL, callbacks (early stopping, checkpoint), and `dataset.dataset_cfg.get_unit_id: true` so unit IDs are in the batch.

### Experiment file (raw CNN 1D)

```yaml
# configs/experiment/unibo/prognostics/raw/cnn_1d.yaml

#@package _global_

defaults:
  - unibo/prognostics/base
  - /model_configs/prognostics/cnn_1d
  - override /transforms: battery/unibo/raw
```

### Datasource and task

- **Datasource:** `configs/datasource/unibo21_bosello.yaml` — Unibo21 loader, task_mode SOC, data_dir from paths.
- **Task definition:** `configs/task_definition/prognostics/rul.yaml` — seq_len, label_len, pred_len, stride, task_type rul, padding_left_flag, model input tensors (features, rul), target_metric, etc.

---

## Variants: same model, different preprocessing

You can keep the same model and only change the **transform** override:

| Experiment | Transforms override | Description |
|------------|---------------------|-------------|
| `unibo/prognostics/raw/cnn_1d` | `battery/unibo/raw` | Raw preprocessing. |
| `unibo/prognostics/combined/cnn_1d` | `battery/unibo/combined` | Combined preprocessing pipeline. |
| `unibo/prognostics/ablation/missing_values/cnn_1d` | `battery/unibo/ablation_missing_values_combined` | Ablation with missing-values handling. |

Example — run with combined transforms:

```bash
python picid/run.py paths=<your_paths> experiment=unibo/prognostics/combined/cnn_1d
```

Example — run ablation (missing values):

```bash
python picid/run.py paths=<your_paths> experiment=unibo/prognostics/ablation/missing_values/cnn_1d
```

What changes is only the `override /transforms: ...` in the experiment YAML; the rest (datasource, task, model, evaluator) stays the same.

---

## Next steps

- Try another model in the same Unibo tree (e.g. `unibo/prognostics/raw/lstm` if defined).
- Try another domain (e.g. PHM20 or railway) using the same pattern: `experiment=<domain>/<task>/<variant>/<model>`.
- Read [Config and overrides](../guides/configs_and_overrides.md) and [From data to prediction](../concepts/from_data_to_prediction.md) to see how the pipeline is wired.

---

[← Examples index](index.md) | [Back to documentation index](../index.md)
