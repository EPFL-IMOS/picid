# datasource.task_mode vs task_definition.task_type — Clarification and Simplification Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clarify when `datasource.task_mode` is required, how it relates to `task_definition.task_type`, and whether it can be removed or reduced in favour of a single source of truth.

**Architecture:** (1) Document the two concepts and all consumption points. (2) Define a clear rule: loader/dataset “which target” stays in datasource; experiment type and evaluation/model behaviour stay in task_definition (or evaluator/model config). (3) Optionally unify overlapping cases (e.g. evaluator/model reading from task_definition when possible) and rename for clarity. No change to loader semantics until we decide removal/rename.

**Tech Stack:** Hydra config, Python loaders/evaluators/models, YAML configs.

---

## 1. Current semantics (reference)

### 1.1 task_definition.task_type

- **Defined in:** `configs/task_definition/**/*.yaml` (e.g. `prognostics/ahrul.yaml` → `task_type: ahrul`, `prognostics/rul.yaml` → `task_type: rul`).
- **Meaning:** Experiment-level task kind. Drives model heads, dataset expected keys, and resolvers (e.g. `infer_data_dim`).
- **Consumers:** Model configs (`task_type: ${task_definition.task_type}`), dataset configs (`task_type: ${task_definition.task_type}`), `task_definition.model.data_requirements.input_tensors` (e.g. `features`, `ahrul`).
- **Values (examples):** `rul`, `ahrul`, `forecasting`, `anomaly_detection`, `fault_classification`, `concepts`, `state_forecasting`, `reconstruction`, `detection`.

### 1.2 datasource.task_mode

- **Defined in:** `configs/datasource/*.yaml` (e.g. `concepts_n_cmapss.yaml` → `task_mode: rul`, `n_cmapss.yaml` → `task_mode: rul`, `old/nb14.yaml` → `task_mode: ahRul`, `mzvav.yaml` → `task_mode: fault`).
- **Meaning:** Used in **two different ways** (source of confusion):
  1. **Dataset/loader “which target”:** For multi-task datasets (PHMD, N-CMAPSS, XJTU, HSF15, CBMv3, etc.), selects which task/target/fold to load (e.g. `rul` vs `fault`, `ahRul` vs `rul`, `cooler`/`valve`/`pump`/`accumulator`, `hdsc`/`ptdsc_stbd`, etc.).
  2. **Forecasting / evaluation “mode”:** For forecasting and some evaluators/models, indicates **univariate** vs **multivariate** (how to evaluate or which features mode). Used in ETTh, railway, MOVi-spheres, etc.
- **Consumers:**
  - **Loaders:** `picid/data/datasources/base/phmd_loader.py` (PHMDMultiSourceLoader), `phmd_battery.py`, `phmd_pronostia.py`, `phmd_n_cmapss.py`, `phmd_cbmv3.py`, `phmd_pubd16.py`, `xjtu_sy`, airbus_helicopter, etc. — all use `task_mode` to select which task/target from the dataset.
  - **Evaluators:** `configs/evaluator/default.yaml`, `rul.yaml`, `railway.yaml`, `per_unit.yaml`, `classification.yaml`, `reconstruction.yaml`, `forecasting.yaml` — `task_mode: ${datasource.task_mode}` (univariate/multivariate for forecasting; passthrough elsewhere).
  - **Models:** `configs/model/naive.yaml`, `mean.yaml`, `timepfn_batch_context.yaml` — `features_mode: ${datasource.task_mode}`.
  - **Transforms:** `configs/transforms/old_but_keep/bosello23_base_satistics.yaml` — `target_name: ${datasource.task_mode}` (column name: rul vs ahRul).
- **Values (examples):** `rul`, `fault`, `ahRul`, `cooler`, `valve`, `pump`, `accumulator`, `hdsc`, `univariate`, `multivariate`, `detection`/`anomaly_detection`, `SOC`.

---

## 2. When is datasource.task_mode needed?

| Use case | Needed? | Reason |
|----------|--------|--------|
| **PHMD-style loaders** (N-CMAPSS, NB14, Pronostia, XJTU, HSF15, CBMv3, etc.) | **Yes, cannot remove** | Loader must know which task name to request from the dataset API (e.g. `ds.get_task(task_mode)`). Same dataset can expose multiple tasks (rul, fault, ahRul, cooler, …). This is **dataset-level** configuration. |
| **Evaluator** (default, rul, railway, forecasting, etc.) | **Partially** | Evaluators take `task_mode` from datasource. For **forecasting**, this is univariate/multivariate. For RUL/prognostics it is often just passthrough. Univariate/multivariate could instead live in `task_definition` or evaluator config. |
| **Model** (naive, mean, timepfn_batch_context) `features_mode` | **Partially** | Same as evaluator: for forecasting, `features_mode` is univariate/multivariate. Could be moved to task_definition or model config. |
| **Transforms** (bosello23) `target_name` | **Could derive** | Uses `datasource.task_mode` as target column name (rul vs ahRul). When experiment is RUL/ahrul, this could be derived from `task_definition.task_type` (with a small mapping for casing, e.g. ahrul → ahRul) to avoid duplication. |

Summary: **datasource.task_mode is required for loaders** that support multiple tasks per dataset. It **cannot be removed** from datasource for those loaders. The open design choice is whether **evaluator and model** should keep reading univariate/multivariate from `datasource.task_mode` or from a single place like `task_definition` (e.g. `task_definition.features_mode` or `task_definition.evaluation_mode`).

---

## 3. Relationship to task_definition (e.g. ahrul.yaml)

- **task_definition** is chosen per **experiment** (e.g. `task_definition=prognostics/ahrul`). It defines `task_type: ahrul`, `seq_len`, `model.data_requirements.input_tensors: [features, ahrul]`, etc.
- **datasource** is chosen per **dataset/source** (e.g. `datasource=.../nb14` or concepts_n_cmapss). It defines how to load data and, for multi-task datasets, **which** task to load via `task_mode`.

For an **ahrul** experiment:

- You set `task_definition=prognostics/ahrul` (task_type: ahrul, input_tensors: ahrul).
- You set a datasource that provides that target, e.g. NB14 with `task_mode: ahRul`. The loader uses `task_mode` to select the "ahRul" task from the PHMD dataset; the pipeline then exposes it under the key expected by the model (e.g. `ahrul`).

So:

- **task_definition.task_type** = experiment type and tensor key semantics (rul, ahrul, forecasting, …).
- **datasource.task_mode** = which task/target to load from the **dataset** (must stay in datasource for multi-task loaders). For single-task datasets or when the only variant is univariate/multivariate, the overlap with “experiment type” or “evaluation mode” can be reduced by moving that concern to task_definition or evaluator.

---

## 4. Can datasource.task_mode be removed?

- **No** for **loader** use: multi-task datasets require a field in the datasource config that tells the loader which task to load. That field could be **renamed** (e.g. `dataset_task` or `target_task`) to avoid overloading the name “task_mode” with the unrelated “univariate/multivariate” meaning.
- **Reduction / simplification** is possible:
  1. **Evaluator and model:** For forecasting, stop using `datasource.task_mode` for univariate/multivariate. Introduce e.g. `task_definition.features_mode` and `task_definition.evaluation_mode` (or evaluator-level config) and have evaluator/model read from there when applicable. Then datasource.task_mode is only “which dataset task” for loaders.
  2. **Transforms:** Where the only use is “target column name” (e.g. rul/ahRul), consider deriving from `task_definition.task_type` via a small resolver or mapping (e.g. ahrul → ahRul) so transforms do not depend on datasource for that.
  3. **Naming:** Rename `datasource.task_mode` to something like `datasource.dataset_task` (or `target_task`) in configs and code to reflect “which task from the dataset,” and reserve “task_mode” for evaluation/features mode (univariate/multivariate) in task_definition or evaluator if desired.

---

## 5. Recommended next steps (bite-sized tasks)

### Task 1: Document current behaviour in codebase

**Files:**  
- Create: `docs/reference/config-task-mode-and-task-type.md`  
- Modify: None (optional: add a short comment in `configs/datasource/README.md` or equivalent if it exists)

**Step 1:** Add a short reference doc that: (1) defines `task_definition.task_type` and `datasource.task_mode`, (2) lists all consumer files (loaders, evaluators, models, transforms) with one line each, (3) states the two meanings of `task_mode` (dataset task selector vs univariate/multivariate).

**Step 2:** Commit with message: `docs: add reference for task_mode vs task_type`

---

### Task 2: Add validation / assertions (optional, low risk)

**Files:**  
- Modify: `picid/run.py` or the place where `cfg.datasource` and `cfg.task_definition` are first used together

**Step 1:** Add a small validation (or log warning) when both exist: if `task_definition.task_type` is one of `rul`, `ahrul`, and `datasource.task_mode` is set, check that they are consistent (e.g. ahrul ↔ ahRul, rul ↔ rul). This does not remove task_mode; it only catches mismatches.

**Step 2:** Add a test that loads config for one ahrul and one rul experiment and asserts that datasource has the expected task_mode.

**Step 3:** Commit with message: `config: validate task_type vs task_mode consistency for rul/ahrul`

---

### Task 3: Decide and document naming/unification (no code change)

**Files:**  
- Modify: `docs/plans/2026-02-20-datasource-task-mode-clarification.md` (this file) or `docs/reference/config-task-mode-and-task-type.md`

**Step 1:** After review, document the chosen policy: (A) keep `datasource.task_mode` as-is for all current uses, or (B) rename to `datasource.dataset_task` and move univariate/multivariate to task_definition/evaluator, or (C) another variant. Add a “Decision” section to the plan or reference doc.

**Step 2:** Commit with message: `docs: document decision on task_mode naming and unification`

---

### Task 4 (only if Task 3 decision is B or C): Unify evaluator/model onto task_definition

**Files:**  
- Modify: `configs/task_definition/forecasting/*.yaml` (add e.g. `features_mode`, `evaluation_mode`)  
- Modify: `configs/evaluator/*.yaml` to use `${task_definition.evaluation_mode}` or similar when present  
- Modify: `configs/model/naive.yaml`, `mean.yaml`, `timepfn_batch_context.yaml` to use task_definition when present  
- Test: `test/evaluator/`, `test/data/datasources/`

**Steps:** Implement the chosen unification (e.g. forecasting task_definitions set evaluation_mode/features_mode; evaluators and models read from task_definition when available and fall back to datasource.task_mode for backward compatibility). Add tests. Commit in small steps (configs, then code, then tests).

---

### Task 5 (only if Task 3 decision is B): Rename task_mode → dataset_task in datasource

**Files:**  
- Modify: All `configs/datasource/*.yaml` (replace `task_mode` with `dataset_task`)  
- Modify: All loader classes and interfaces that read `task_mode` to read `dataset_task` (or both for backward compat)  
- Modify: Transforms and evaluators that reference `datasource.task_mode` for “which target” to use `datasource.dataset_task`  
- Test: Full regression (run a few experiments per dataset type)

**Steps:** Rename in configs and code; keep backward compatibility or do a single breaking change and update all references; run tests and a few experiments; commit.

---

## 6. Summary table

| Concept | Where defined | When needed | Can remove? |
|--------|----------------|-------------|-------------|
| **task_definition.task_type** | task_definition config | Every experiment | No — core experiment type. |
| **datasource.task_mode** (loader: “which target”) | datasource config | All multi-task loaders (PHMD, N-CMAPSS, XJTU, HSF15, CBMv3, …) | No — required for dataset task selection. |
| **datasource.task_mode** (evaluator/model: univariate/multivariate) | datasource config | Forecasting (and similar) evaluators and some models | Optional — can move to task_definition or evaluator config. |
| **datasource.task_mode** (transform: target_name) | datasource config | Transforms that need target column name (e.g. rul/ahRul) | Optional — can derive from task_definition.task_type with mapping. |

---

## Execution handoff

Plan complete and saved to `docs/plans/2026-02-20-datasource-task-mode-clarification.md`. Two execution options:

1. **Subagent-Driven (this session)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.  
2. **Parallel Session (separate)** — Open a new session with executing-plans and run with checkpoints.

Which approach do you prefer?

If Subagent-Driven is chosen, use the **subagent-driven-development** skill in this session. If a parallel session is chosen, use **executing-plans** in the new session.
