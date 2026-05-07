# Config and overrides

Experiments in PICID are composed from **config groups** via Hydra. You choose an **experiment** key that resolves to a YAML file; that file uses `defaults` and `override` to plug in a datasource, task definition, dataset, transforms, model, and evaluator.

---

## Experiment as a composition

The run entry point is:

```bash
python picid/run.py paths=<paths> experiment=<experiment_key>
```

The **experiment key** is a path-like name that selects one config under `configs/experiment/`. For example:

- `unibo/prognostics/raw/cnn_1d` — Unibo battery RUL, raw transforms, CNN 1D model.
- `unibo/prognostics/combined/cnn_1d` — Same domain and model, different preprocessing (combined transforms).
- `unibo/prognostics/ablation/missing_values/cnn_1d` — Ablation with missing-values transforms.

So the pattern is **`<domain>/<task>/<variant>/<model>`** (or with extra nesting for ablations).

---

## Defaults and override

Each experiment YAML typically has a `defaults` list that:

1. Pulls in a **base** config (e.g. `unibo/prognostics/base`) for that domain and task.
2. Pulls in the **model** config (e.g. `/model_configs/prognostics/cnn_1d`).
3. **Overrides** one or more config groups, most commonly **transforms**:

```yaml
#@package _global_

defaults:
  - unibo/prognostics/base
  - /model_configs/prognostics/cnn_1d
  - override /transforms: battery/unibo/raw
```

- **`override /transforms: battery/unibo/raw`** — Replaces the default transform group with the `battery/unibo/raw` transform config. Changing this line (e.g. to `battery/unibo/combined` or `battery/unibo/ablation_missing_values_combined`) switches preprocessing without changing code.
- **`override /datasource: ...`** — Picks the datasource (e.g. `unibo21_bosello`).
- **`override /task_definition: ...`** — Picks the task (e.g. `prognostics/rul`).
- **`override /evaluator: ...`** — Picks the evaluator (e.g. `per_unit` for RUL).

So one experiment file is enough to define “same model, different pipeline” by only changing the overrides.

---

## Minimal example

One experiment file, one override:

```yaml
# configs/experiment/unibo/prognostics/raw/cnn_1d.yaml

#@package _global_

defaults:
  - unibo/prognostics/base
  - /model_configs/prognostics/cnn_1d
  - override /transforms: battery/unibo/raw
```

The base (`unibo/prognostics/base`) already sets datasource, evaluator, task_definition, and dataset options. This file only adds the model and the transform variant.

---

## Common CLI overrides

- **`paths=<name>`** — Selects the path config from `configs/paths/<name>.yaml` (data dir, outputs, etc.).
- **`debug=default`** — Uses the debug config group (e.g. smaller subset, fewer epochs) for faster runs during development.

Example:

```bash
python picid/run.py paths=rt_local debug=default experiment=unibo/prognostics/raw/cnn_1d
```

---

## Config composition (overview)

Rough flow:

1. **Experiment** YAML is loaded (e.g. `unibo/prognostics/raw/cnn_1d`).
2. Its **defaults** compose:
   - **Datasource** (e.g. `unibo21_bosello`) → where data comes from.
   - **Task definition** (e.g. `prognostics/rul`) → seq_len, stride, task_type, model input requirements.
   - **Dataset** (from base or model) → which dataset class and params (e.g. RUL multi-unit, `get_unit_id: true`).
   - **Transforms** (override) → which preprocessing pipeline (raw, combined, ablation, etc.).
   - **Model** (from model_configs) → architecture and training config.
   - **Evaluator** (e.g. `per_unit`) → metrics and hooks.
3. The pipeline instantiates datamodule, model, evaluator from this composed config.

---

## See also

- [OmegaConf resolvers](omegaconf_resolvers.md) — Custom resolvers for config values.
- [Example: Unibo RUL](../examples/unibo_rul.md) — Full walkthrough with config snippets and variants.

[← Guides index](index.md) | [Back to documentation index](../index.md)
