# Test Configs Isolation (2026-03-06)

**Plan:** [docs/plans/archive/2026-03-06-test-configs-isolation.md](../docs/plans/archive/2026-03-06-test-configs-isolation.md)

## Status: Complete

All 6 tasks executed. Full verification passed.

## Completed Tasks

- [x] **T1** Create test/configs/ with experiment, datasource, task_definition, transforms (snapshot namespace)
- [x] **T2** Rename pipeline_regression_loader → synthetic_file_loader, fixtures pipeline_regression → snapshot
- [x] **T3** Add hydra.searchpath override to test and generate script
- [x] **T4** Rename fixtures/reference scripts, regenerate fixtures and reference
- [x] **T5** Delete pipeline_regression configs from configs/
- [x] **T6** Update docs, nox, remove old scripts

## Verification

- `uv run pytest test/pipeline/test_regression.py -v` — 3 passed
- `uv run pytest test/ -m "not slow" -q` — 2301 passed
- `configs/` contains no pipeline_regression references
