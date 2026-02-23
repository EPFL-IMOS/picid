# Tasks

> **Workflow:** Plan first, verify before done. See `docs/plans/` for detailed plans.
> **Convention:** One task file per orchestration. When a task's goal is finished, move it to `tasks/archive/YYYY-MM-DD/<task_name>.md`.
> **Keep this file updated:** When new tasks are found, add them here. Mark checkboxes when done.

---

## Four-Task Orchestration (2026-02-23)

**File:** [four-task-orchestration-2026-02-23.md](four-task-orchestration-2026-02-23.md)

- [x] **Task 1** DatasetContainer `__repr__` — [Plan 1](../docs/plans/2026-02-23-plan-1-dataset-container-repr.md)
- [x] **Task 2** SplitDatasetContainer `is_multi_unit` — [Plan 2](../docs/plans/2026-02-23-plan-2-split-dataset-container-is-multi-unit.md)
- [ ] **Task 3** HydraConcatDataset unit_id from SliceInfo — [Plan 3](../docs/plans/2026-02-23-plan-3-hydra-concat-unit-id-from-slice-info.md) (investigation)
- [x] **Task 4** Archive autoencoder reconstruction — [Plan 4](../docs/plans/2026-02-23-plan-4-archive-autoencoder-reconstruction.md)

---

## Coverage to 95%

**File:** [coverage-95-percent.md](coverage-95-percent.md)

- [ ] **R1** Restore accidentally deleted `docs/plans/` files
- [ ] **R2** Add omit entries (Path C) to `pyproject.toml`
- [ ] **R3** Fix 2 failing picid_report tests (grid completion)
- [ ] **R4** Add tests for `loss/default.py` (target: 80%+)
- [ ] **R5** Add tests for `imputation_methods.py` (target: 85%+)
- [ ] **R6** Add tests for `timefeatures_railway.py` (target: 80%+)
- [ ] **R7** Add tests for `seasonal_dummy_railway.py` (target: 80%+)
- [ ] **R8** Add tests for `analytics/missing_values.py` (target: 80%+)
- [ ] **R9** Add tests for `transform_pipeline.py` (target: 85%+)
- [ ] **R10** Add tests for `concepts_n_cmapss.py` OR add to omit
- [ ] **R11** Run full coverage; verify ≥95%
- [ ] **R12** Enable `--cov-fail-under=95` in pytest config

---

## Archive

Completed tasks are in `tasks/archive/YYYY-MM-DD/`. See [archive/README.md](archive/README.md).
