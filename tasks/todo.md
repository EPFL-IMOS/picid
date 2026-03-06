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

- [x] **R1** Restore accidentally deleted `docs/plans/` files (not applicable — `docs/plans/` is gitignored by design; files exist locally)
- [x] **R2** Add omit entries (Path C) to `pyproject.toml`
- [ ] **R3** Fix 2 failing picid_report tests (grid completion)
- [x] **R4** Add tests for `loss/default.py` (target: 80%+)
- [x] **R5** Add tests for `imputation_methods.py` (target: 85%+)
- [x] **R6** Add tests for `timefeatures_railway.py` (target: 80%+)
- [x] **R7** Add tests for `seasonal_dummy_railway.py` (target: 80%+)
- [x] **R8** Add tests for `analytics/missing_values.py` (target: 80%+)
- [x] **R9** Add tests for `transform_pipeline.py` (target: 85%+)
- [x] **R10** Add tests for `concepts_n_cmapss.py` OR add to omit
- [x] **R11** Run full coverage; verify ≥95%
- [x] **R12** Enable `--cov-fail-under=95` in pytest config

---

## Publish / Submit to Leaderboard (2026-03-06)

**File:** [publish-submit-leaderboard-2026-03-06.md](publish-submit-leaderboard-2026-03-06.md)  
**Plan:** [docs/plans/2026-03-06-publish-submit-leaderboard.md](../docs/plans/2026-03-06-publish-submit-leaderboard.md)

- [x] **P1** Phase 1 Task 1: build_submission_message.py + README + test
- [ ] **P2** Phase 1 Task 2: push_run_to_gdrive.py + gdrive_config
- [ ] **P3** Phase 1 Task 3: Wire message + push; CLI (depends on P2)
- [ ] **P4** Phase 1 Task 4: Dataset-specific submission schema (stub)

---

## Config Diff and Metrics Comparison (2026-03-06)

**File:** [config-diff-2026-03-06.md](config-diff-2026-03-06.md)  
**Plan:** [docs/plans/2026-03-06-config-diff-and-metrics-comparison.md](../docs/plans/2026-03-06-config-diff-and-metrics-comparison.md)

- [x] **D1** picid/utils/config_diff.py + scripts/compare_configs.py + tests
- [ ] **D2** Test: config change produces different metrics and shows diff

---

## Docs Site & Hydra Experiment Generator (2026-03-07)

**File:** [docs-site-hydra-experiment-generator-2026-03-07.md](docs-site-hydra-experiment-generator-2026-03-07.md)  
**Plan:** [docs/plans/2026-03-07-docs-site-hydra-experiment-generator.md](../docs/plans/2026-03-07-docs-site-hydra-experiment-generator.md)

- [x] **T1** Phase 1: MkDocs placeholder, guide page, nav
- [x] **T2** Phase 1: Scripts directory README
- [x] **T3** Phase 2: generate_config_map.py + tests
- [x] **T4** Phase 2: Fix flatten logic, verify against real configs
- [x] **T5** Phase 3: Experiment Generator UI (HTML + JS)
- [x] **T6** Phase 4: Test instructions, README, docs links
- [x] **T7** Phase 4: Website README, GitHub Pages deployment
- [ ] **T8** Phase 5: Run & Upload workflow + submit_run_to_leaderboard guide
- [ ] **T9** Phase 5: Leaderboard page and data source (depends on Publish P1–P2)

---

## Archive

Completed tasks are in `tasks/archive/YYYY-MM-DD/`. See [archive/README.md](archive/README.md).

**Recently archived:** Test Coverage (picid/lr_scheduler & picid/utils) — C1–C9 done.
