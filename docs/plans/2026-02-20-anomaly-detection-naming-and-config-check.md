# Anomaly Detection Naming and Config Check Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** After merging the detection-task branch, verify new configs work end-to-end and standardise naming to `anomaly_detection` everywhere (config paths, task_type, task_mode, and code).

**Architecture:** (1) Audit and validate the new detection experiment and dataset/evaluator wiring. (2) Rename all config directory segments and references from `detection` to `anomaly_detection`. (3) Unify task_type/task_mode to `anomaly_detection` in configs and code. (4) Simplify wrapper and docs to use a single canonical name.

**Tech Stack:** Hydra configs, YAML, Python (picid run, evaluator, datasets, model wrappers).

---

## Phase 1: Check New Configs (Validation)

### Task 1: List and verify detection-related config files

**Files:**
- No creates; read-only audit.

**Step 1: List all detection/anomaly_detection config paths**

Run:

```bash
cd /Users/leone/projects/TabPHM
rg -l 'detection|anomaly_detection' configs/ --type-add 'config:yaml' -t config 2>/dev/null || grep -r -l 'detection\|anomaly_detection' configs/
```

Expected: See `configs/dataset/detection/*`, `configs/task_definition/detection/*`, `configs/experiment/airbus_helicopter/base.yaml`, `configs/experiment/airbus_helicopter/detection/*`, `configs/model_configs/detection/*`, `configs/datasource/airbus_helicopter.yaml`, `configs/transforms/airbus_helicopter/*`.

**Step 2: Confirm dataset and task_definition wiring**

- `configs/experiment/airbus_helicopter/base.yaml` uses `override /task_definition: detection/airbus_helicopter`.
- `configs/experiment/airbus_helicopter/detection/isolation_forest.yaml` adds `override /model_configs/detection/isolation_forest` and `override /transforms: airbus_helicopter/statistics_fit_predict`.
- Ensure the experiment’s defaults include (or inherit) a dataset that uses the detection task (e.g. dataset config that references `task_definition.task_type` and is used for airbus_helicopter). Check `configs/experiment/airbus_helicopter/base.yaml` and any dataset default list for the airbus_helicopter experiment.

**Step 3: Run the detection experiment (dry or minimal)**

Run:

```bash
cd /Users/leone/projects/TabPHM
uv run python picid/run.py experiment=airbus_helicopter/detection/isolation_forest paths=rt_local debug=default
```

If data paths or dependencies fail, use a mock/small data path or skip data loading and only verify config composition (e.g. that Hydra composes without error up to the point where data would be loaded). Document any missing dataset default for airbus_helicopter (e.g. override `/dataset` to use `detection/anomaly_detection_dataset` or `detection/anomaly_detection_multiunit_dataset` if needed).

**Step 4: Commit (optional checkpoint)**

```bash
git add docs/plans/2026-02-20-anomaly-detection-naming-and-config-check.md
git commit -m "docs: add plan for anomaly_detection naming and config check"
```

---

### Task 2: Confirm dataset default for airbus_helicopter experiment

**Files:**
- Read: `configs/experiment/airbus_helicopter/base.yaml`, `configs/experiment/airbus_helicopter/detection/isolation_forest.yaml`
- Read: `picid/run.py` (or config defaults) to see how `dataset` is chosen when not specified.

**Step 1: Find where dataset is set for experiments**

Run:

```bash
cd /Users/leone/projects/TabPHM
grep -n "dataset" configs/experiment/airbus_helicopter/*.yaml configs/experiment/airbus_helicopter/detection/*.yaml
grep -n "dataset" configs/datamodule/*.yaml | head -20
```

**Step 2: If dataset is not overridden for airbus_helicopter**

Check main run config or datamodule defaults for how `dataset` is selected (e.g. by task_definition or datasource). If detection experiments expect a specific dataset config (e.g. `dataset/detection/anomaly_detection_dataset.yaml` or multiunit variant), add an explicit default in `configs/experiment/airbus_helicopter/base.yaml` or in the isolation_forest experiment so the correct dataset is used. Document the chosen dataset key in the plan (e.g. `override /dataset: detection/anomaly_detection_dataset` or multiunit).

**Step 3: Commit (if you added dataset default)**

```bash
git add configs/experiment/airbus_helicopter/
git commit -m "config: set dataset default for airbus_helicopter detection experiment"
```

---

## Phase 2: Rename Config Paths from `detection` to `anomaly_detection`

Use `anomaly_detection` as the directory name everywhere instead of `detection`.

### Task 3: Rename config directories

**Files:**
- Move: `configs/dataset/detection/` → `configs/dataset/anomaly_detection/`
- Move: `configs/task_definition/detection/` → `configs/task_definition/anomaly_detection/`
- Move: `configs/experiment/airbus_helicopter/detection/` → `configs/experiment/airbus_helicopter/anomaly_detection/`
- Move: `configs/model_configs/detection/` → `configs/model_configs/anomaly_detection/`

**Step 1: Create new dirs and move files**

```bash
cd /Users/leone/projects/TabPHM

mkdir -p configs/dataset/anomaly_detection
mv configs/dataset/detection/*.yaml configs/dataset/anomaly_detection/
rmdir configs/dataset/detection

mkdir -p configs/task_definition/anomaly_detection
mv configs/task_definition/detection/*.yaml configs/task_definition/anomaly_detection/
rmdir configs/task_definition/detection

mkdir -p configs/experiment/airbus_helicopter/anomaly_detection
mv configs/experiment/airbus_helicopter/detection/*.yaml configs/experiment/airbus_helicopter/anomaly_detection/
rmdir configs/experiment/airbus_helicopter/detection

mkdir -p configs/model_configs/anomaly_detection
mv configs/model_configs/detection/*.yaml configs/model_configs/anomaly_detection/
rmdir configs/model_configs/detection
```

**Step 2: Update all references to these paths**

- In `configs/experiment/airbus_helicopter/base.yaml`: change `override /task_definition: detection/airbus_helicopter` to `override /task_definition: anomaly_detection/airbus_helicopter`.
- In `configs/experiment/airbus_helicopter/anomaly_detection/isolation_forest.yaml`: change `- /model_configs/detection/isolation_forest` to `- /model_configs/anomaly_detection/isolation_forest`.
- Search codebase for any other references to `detection/airbus_helicopter`, `model_configs/detection`, `dataset/detection`, `task_definition/detection`, or `airbus_helicopter/detection` and update to use `anomaly_detection` in the path.

Run:

```bash
grep -r 'detection/airbus_helicopter\|model_configs/detection\|/dataset/detection\|task_definition/detection\|airbus_helicopter/detection' configs/ picid/ --include='*.yaml' --include='*.py'
```

Replace each with the corresponding `anomaly_detection` path.

**Step 3: Commit**

```bash
git add configs/
git commit -m "refactor(configs): rename detection to anomaly_detection in config paths"
```

---

## Phase 3: Standardise task_type and task_mode to `anomaly_detection`

### Task 4: Configs: task_type and task_mode

**Files:**
- Modify: `configs/datasource/airbus_helicopter.yaml`
- Modify: `configs/task_definition/anomaly_detection/concepts_n_cmapss.yaml`

**Step 1: Set datasource task_mode to anomaly_detection**

In `configs/datasource/airbus_helicopter.yaml` change:

```yaml
task_mode: detection # rul, fault
```

to:

```yaml
task_mode: anomaly_detection  # rul, fault
```

**Step 2: Set task_type to anomaly_detection in concepts_n_cmapss**

In `configs/task_definition/anomaly_detection/concepts_n_cmapss.yaml` change:

```yaml
task_type: detection
```

to:

```yaml
task_type: anomaly_detection
```

**Step 3: Commit**

```bash
git add configs/datasource/airbus_helicopter.yaml configs/task_definition/anomaly_detection/concepts_n_cmapss.yaml
git commit -m "config: use anomaly_detection for task_mode and task_type"
```

---

### Task 5: Code: use only `anomaly_detection`

**Files:**
- Modify: `picid/model/wrappers/fit_predict_isolation_forest_wrapper.py`
- Modify: `picid/data/datasources/toy_example.py` (optional: docstring/default)
- Modify: `picid/data/datasources/airbus_helicopter.py` (docstring)

**Step 1: Simplify isolation forest wrapper**

In `picid/model/wrappers/fit_predict_isolation_forest_wrapper.py`:

- Remove the local `DETECTION_TASKS = ["detection", "anomaly_detection"]`.
- Rely on `CLASSIFICATION_TASKS` from `picid.baselines.definitions` (which already includes `"anomaly_detection"`), or introduce a single constant e.g. `ANOMALY_DETECTION_TASK = "anomaly_detection"` and use that for validation if you want to restrict the wrapper to only anomaly_detection. For minimal change: keep supporting both by using `CLASSIFICATION_TASKS` only (since `anomaly_detection` is already there) and remove the duplicate `DETECTION_TASKS`; update the comment to say "anomaly_detection" is the canonical task type.

Concretely: delete lines 9–10 and 15; change the validation to use `CLASSIFICATION_TASKS` only (or `CLASSIFICATION_TASKS` + a list containing only `"anomaly_detection"` if you want to restrict this wrapper to anomaly_detection). Update the comment above the import to reference `anomaly_detection`.

**Step 2: Update docstrings (optional)**

In `picid/data/datasources/airbus_helicopter.py`, in the docstring that says `Task type (e.g. 'detection')`, change to `(e.g. 'anomaly_detection')`.

In `picid/data/datasources/toy_example.py`, change default `task_mode: str = "detection"` to `task_mode: str = "anomaly_detection"` and update the docstring that lists `'detection'` to `'anomaly_detection'` so docs and defaults are consistent.

**Step 3: Run tests**

Run:

```bash
cd /Users/leone/projects/TabPHM
uv run pytest test/ -v -k "isolation_forest or classification or airbus" --ignore=local_packages 2>/dev/null || uv run pytest test/ -v --ignore=local_packages -x
```

Fix any failures due to hardcoded `"detection"`.

**Step 4: Commit**

```bash
git add picid/model/wrappers/fit_predict_isolation_forest_wrapper.py picid/data/datasources/airbus_helicopter.py picid/data/datasources/toy_example.py
git commit -m "refactor: use anomaly_detection in code and docstrings"
```

---

## Phase 4: Re-run and document

### Task 6: Final run and plan update

**Files:**
- Modify: `docs/plans/2026-02-20-anomaly-detection-naming-and-config-check.md` (this file)

**Step 1: Run experiment with new paths**

```bash
cd /Users/leone/projects/TabPHM
uv run python picid/run.py experiment=airbus_helicopter/anomaly_detection/isolation_forest paths=rt_local debug=default
```

Expected: Config composes; if data is present, run proceeds; otherwise only note data/path errors.

**Step 2: Update this plan**

In the plan file, under a short "Completed" section, note:
- That config paths now use `anomaly_detection` (dataset, task_definition, experiment, model_configs).
- That task_type and task_mode are set to `anomaly_detection` in airbus_helicopter and concepts_n_cmapss.
- The exact experiment command: `experiment=airbus_helicopter/anomaly_detection/isolation_forest`.

**Step 3: Commit**

```bash
git add docs/plans/2026-02-20-anomaly-detection-naming-and-config-check.md
git commit -m "docs: complete anomaly_detection naming plan"
```

---

## Summary of renames

| Location | Before | After |
|----------|--------|--------|
| Dataset config dir | `configs/dataset/detection/` | `configs/dataset/anomaly_detection/` |
| Task definition dir | `configs/task_definition/detection/` | `configs/task_definition/anomaly_detection/` |
| Experiment subdir | `configs/experiment/airbus_helicopter/detection/` | `configs/experiment/airbus_helicopter/anomaly_detection/` |
| Model config dir | `configs/model_configs/detection/` | `configs/model_configs/anomaly_detection/` |
| base.yaml task_definition | `detection/airbus_helicopter` | `anomaly_detection/airbus_helicopter` |
| isolation_forest defaults | `/model_configs/detection/isolation_forest` | `/model_configs/anomaly_detection/isolation_forest` |
| datasource airbus_helicopter | `task_mode: detection` | `task_mode: anomaly_detection` |
| task_definition concepts_n_cmapss | `task_type: detection` | `task_type: anomaly_detection` |
| Code (wrapper, docs) | "detection" in comments/defaults | "anomaly_detection" |

No change to data keys in transforms: they already use `anomaly_detection` (e.g. `assign_to: anomaly_detection`, `apply_to: ["anomaly_detection"]`).

---

## Completed

- **Config paths** now use `anomaly_detection`: dataset (`configs/dataset/anomaly_detection/`), task_definition (`configs/task_definition/anomaly_detection/`), experiment (`configs/experiment/airbus_helicopter/anomaly_detection/`), model_configs (`configs/model_configs/anomaly_detection/`).
- **task_type** and **task_mode** are set to `anomaly_detection` in `configs/datasource/airbus_helicopter.yaml` and `configs/task_definition/anomaly_detection/concepts_n_cmapss.yaml`.
- **Experiment command:** `experiment=airbus_helicopter/anomaly_detection/isolation_forest` (with `paths=rt_local debug=default` for a minimal run). Config composes successfully; data/path errors (e.g. missing `ground_truth.csv`) are environment-specific, not config.
