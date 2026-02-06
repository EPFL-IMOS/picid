# lmetk_report Refactoring Plan (Clarity & Best Practices)

**Goal**: Make the codebase clear, easy to read, and aligned with best practices. **No intentional removal of user-facing functionality**; optional cleanup of dead code only where it improves clarity.

**Status**: Phases 0–4 complete. 188 tests, ~81% coverage. Dead helpers removed; config.EXPECTED_SEARCH_SPACE kept as deprecated fallback.

---

## Principles

1. **Clarity first** — Code and comments should make data flow and intent obvious.
2. **Single responsibility** — One place for each concept (e.g. one helper for “get grid from search space”).
3. **Test coverage** — Critical paths and helpers covered by tests.
4. **Documentation** — Module and function docstrings plus targeted comments where logic is non-obvious.
5. **Preserve behavior** — Refactors must not change observable behavior; tests validate this.

---

## Phase 0: Test coverage ✅ COMPLETE

**Goal**: Bring coverage to a high level for all production code (excluding `not_used/`).

**Tasks**:
- [x] Add `test_utils.py`: `format_mean_std_count`, `flatten_aggregated_columns` (edge cases, NaN, MultiIndex).
- [x] Add `test_preprocess.py`: `clean_and_rename_models` (prefix strip, model type suffix, config defaults).
- [x] Add `test_run.py`: `_safe_basename`, `_save_outputs` (creates dirs and files), `main` (CLI parsing).
- [x] Add `test_config.py`: `PipelineConfig`, `PipelineConfig.from_default()`.
- [x] Fix 5 sort metric tests in `test_configs.py` to match current config (val_best_rerun/loss, global default).

**Validation**: 188 tests pass. Coverage (excluding `not_used/`): **81%** total; core modules 86–100%. `run.py` and `plots.py` have lower coverage (run_pipeline body and plotting paths); `plots.py` is optional and excluded from strict coverage target.

**Estimated**: 3–5 hours | **Risk**: Low

---

## Phase 1: Code clarity and shared helpers

**Goal**: Use shared helpers so logic lives in one place and call sites stay simple.

**Done**:
- [x] Introduce `get_model_grid_from_search_space(dataset, model, search_space)` in `configs/search_space.py` and use it in `analysis.py` (replacing inline legacy/new detection and lookup).

**Optional (no functionality change)**:
- [x] Removed dead code: `_convert_legacy_search_space`, `get_search_space_with_fallback`; tests now use `get_model_grid_from_search_space` only (Phase 4).

**Validation**: All tests pass; behavior unchanged.

**Estimated**: 0–1 hour | **Risk**: Low

---

## Phase 2: In-code documentation and comments

**Goal**: Add docstrings and comments so that “what is happening” is clear without reading implementation details.

**Tasks**:
- [x] **Module docstrings**: Each module has a short summary (purpose, main concepts, how it fits in the pipeline).
- [x] **Public functions**: Docstrings with purpose, args, returns, and notable behavior (e.g. “Returns None → auto-discovery”).
- [x] **Targeted comments**: Non-obvious steps (structure detection, fallbacks, aggregation choices) get one-line comments.
- [x] **Config modules**: Document structure of dicts (e.g. search space: new vs legacy shape; sort metric hierarchy).

**Files in scope** (excluding `not_used/`):
- `analysis.py`, `reporting.py`, `run.py`, `run_processor.py`
- `config.py`, `configs/__init__.py`, `configs/search_space.py`, `configs/sort_metrics.py`
- `preprocess.py`, `utils.py`, `validators.py`, `report_html.py`
- `__init__.py` (package public API)

**Validation**: Read-through; no behavior change; tests still pass.

**Estimated**: 3–4 hours | **Risk**: Low

---

## Phase 3: External documentation

**Goal**: Keep DOCS.md, README.md, and any migration/refactor notes accurate and helpful.

**Tasks**:
- [x] **DOCS.md**: Update for current config (search space, sort metrics), pipeline steps, and main entry points.
- [x] **README.md**: Quick start, examples, and pointers to DOCS.md.
- [x] **REFACTORING_PLAN.md / DEPRECATION_REMOVAL_PLAN.md**: Mark completed items; add short “current state” summary.

**Validation**: Docs match code and run instructions.

**Estimated**: 2–3 hours | **Risk**: Low

---

## Phase 4: Optional cleanup (no user-facing feature removal)

**Goal**: Simplify code without removing capabilities users might rely on.

**Optional items** (each can be done independently):
- [x] **Deprecated `config.EXPECTED_SEARCH_SPACE`**: Kept as deprecated fallback with strengthened comment; prefer `configs.search_space.EXPECTED_SEARCH_SPACE` and `get_model_grid_from_search_space`.
- [ ] **plots.py**: Keep unless explicitly decided otherwise; if removed, document and provide alternative (e.g. notebook or script).
- [x] **Dead helpers**: Removed `_convert_legacy_search_space` and `get_search_space_with_fallback`; tests updated to rely on `get_model_grid_from_search_space` (behavior preserved).

**Validation**: All tests pass; no undocumented breaking changes.

**Estimated**: 1–3 hours per item | **Risk**: Low–medium

---

## Summary

| Phase | Focus                    | Est.  | Risk  |
|-------|--------------------------|-------|-------|
| 0     | Test coverage            | 3–5 h | Low   |
| 1     | Clarity / shared helpers | 0–1 h | Low   |
| 2     | **In-code docs & comments** | 3–4 h | Low   |
| 3     | External docs            | 2–3 h | Low   |
| 4     | Optional cleanup         | 1–3 h | Low–med |

**Order**: 0 → 1 → 2 → 3 → 4 (with 4 items optional and independent).

---

## Current test and coverage summary

- **Test files**: `test_analysis`, `test_config`, `test_configs`, `test_integration`, `test_preprocess`, `test_report_html`, `test_reporting`, `test_run`, `test_run_processor`, `test_utils`, `test_validators`.
- **Total tests**: 188 (all passing).
- **Coverage**: Run `pytest test/lmetk_report/ --cov=lmetk_report --cov-report=term-missing`. Core modules (analysis, configs, preprocess, run_processor, utils, validators) are well covered; `run.py` and `plots.py` have lower coverage (pipeline body and optional plotting). Exclude `not_used/` from coverage target.

---

## Current state (post Phase 2 & 3)

- **Phase 0**: 188 tests, ~81% coverage (excluding `not_used/`); core modules 86–100%.
- **Phase 1**: `get_model_grid_from_search_space` in use.
- **Phase 2**: Module docstrings, public function docstrings, and config dict-shape comments in place.
- **Phase 3**: DOCS.md, README.md, and plan summaries updated.
- **Phase 4**: Dead helpers (`_convert_legacy_search_space`, `get_search_space_with_fallback`) removed; tests use `get_model_grid_from_search_space` only. `config.EXPECTED_SEARCH_SPACE` kept as deprecated fallback with clear comment. `plots.py` kept.
