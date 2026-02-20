# Transform migration: new structure evaluation and integration plan

This document (1) evaluates the draft `*_new.py` files (`data_transform_new`, `strategy_new`, `pipeline_new`) with pros and cons, and (2) outlines phased steps to integrate them into the codebase.

---

## 1. Evaluation of the new structure

### 1.1 High-level changes

| Area | Current | New |
|------|--------|-----|
| **Validation** | `DataTransform._validate_forward_output` runs after strategy returns; checks type, keys, splits, warnings. | Removed from DataTransform. Structural checks live in **IntegrityCheckAfterStep** (default after-checks). Soft warnings (type/unit-count) moved to **MergeStep._warn_on_changes**. |
| **Sanitization** | `_prepare_chunks_for_split` calls `NamedTransformInput(...).get_sanitized_data()` (NaN + length). | No sanitization in strategy. **IntegrityCheckBeforeStep** runs default before-checks (NaN + length) over apply_to chunks. |
| **Integrity API** | Single `integrity_checks` list; one **IntegrityCheckStep** before Fit (no-arg check fns). | **integrity_checks_before** / **integrity_checks_after**; **IntegrityCheckBeforeStep** and **IntegrityCheckAfterStep**; check fns receive scoped data (before: `Dict[str, List[NamedTransformInput]]`, after: `Dict[str, Any]` assign_to slice). |
| **Chunk flow** | FitStep and TransformStep call `strategy._prepare_chunks_for_split` directly; chunks built on demand but no explicit “builder” abstraction. | **RegisterChunkBuilderStep** stores `build_fit_chunks` and `build_transform_chunks` on context. Steps call the builder when needed; chunks are never stored on context → streamed, one split at a time. |
| **Manifest** | Payload: apply_to, assign_to, fit_on, log_splits, timings, slice_info. | Adds **integrity_before_run**, **integrity_after_run**, **integrity_before_checks**, **integrity_after_checks**. |
| **CopyStep** | Uses `get_instance_cls()` and `ak.Array` to decide deep vs shallow copy per key. | Uses `_first_leaf(data[key])` to unwrap nested dict/list and detect the first array; shallow copy for `ak.Array`, deep copy otherwise. Preserves awkward-aware semantics. |

---

### 1.2 Pros

- **Single place for structural validation:** Output contract (type, keys, splits) lives in pipeline integrity-after checks. Changing the container type (e.g. away from `SplitDatasetContainer`) only requires updating those checks, not every transform or DataTransform.
- **Clear before/after integrity:** Before-checks see only apply_to chunks; after-checks see only assign_to slice. Scope is enforced by the API (check signatures and what the steps pass), reducing misuse.
- **No sanitization in strategy:** `_prepare_chunks_for_split` only builds chunk wrappers; validation is explicit in IntegrityCheckBeforeStep. Aligns with the refactor plan (sanitization = integrity-before).
- **Memory model:** Chunks are never stored on context. RegisterChunkBuilderStep registers callables; FitStep and TransformStep call them and hold a local reference; when the reference goes out of scope, chunks can be GC’d before the next split. For large PHM datasets this keeps peak memory to one split’s chunk wrappers.
- **Manifest traceability:** Manifest records which integrity steps ran and which checks ran (by name), improving debugging and auditing.
- **Legacy support:** `integrity_checks` is mapped to `integrity_checks_before` with DeprecationWarning in both DataTransform and strategy.apply().
- **Testability:** Hand-built contexts (strategy=None or builders not registered) are still supported: FitStep and TransformStep fall back to building chunks inline via `strategy._prepare_chunks_for_split`, so existing tests that build context by hand can keep working with minimal changes.
- **Diagnostic warnings in MergeStep:** Type-change and unit-count warnings live where before/after data is already available, without passing extra args through DataTransform.
- **CopyStep awkward-aware copy:** CopyStep uses a `_first_leaf()` helper to unwrap nested structure (e.g. `{split: [arrays]}`) and detect the first concrete array; `ak.Array` gets shallow copy, other types get deep copy, matching current pipeline intent without relying on `get_instance_cls()`.

---

### 1.3 Cons and risks

- **CopyStep and container shape:** New CopyStep uses `_first_leaf(data[key])` to unwrap nested dict/list and reach the first array; if it is `ak.Array` it does shallow copy, otherwise deep copy. This preserves awkward-aware behaviour. Confirm that your container shape (e.g. `data[key]` is split-keyed and values are list-like or mapping) is compatible with that unwrapping so `_first_leaf` reaches a real array and does not raise or return something unexpected.
- **Two integrity steps and more context fields:** Context and docs must carry `integrity_checks_before` / `integrity_checks_after`, `integrity_before_ran` / `integrity_after_ran`, `integrity_before_check_names` / `integrity_after_check_names`. More surface area to maintain and document.
- **Check function signature change:** Current integrity uses no-arg callables (they read from context via closure). New pipeline passes a data slice as the first argument. The new code supports no-arg functions via introspection and emits DeprecationWarning; existing tests or configs that pass no-arg checks will need to migrate to the new signature when the deprecation is removed.
- **Pipeline import surface:** Strategy and other code import from `picid.transforms.base.pipeline`. Replacing pipeline.py with pipeline_new adds new exports (RegisterChunkBuilderStep, IntegrityCheckBeforeStep, IntegrityCheckAfterStep, build_default_before_checks, build_default_after_checks, _emit). Existing tests that import IntegrityCheckStep must switch to the new steps or a thin compatibility layer.
- **Unused import in pipeline_new:** `from picid.transforms.base.integrity import run_checks` is present but the new pipeline uses its own `_run_scoped_checks`; the import can be removed as a small cleanup.

---

### 1.4 Summary table

| Criterion | Verdict |
|-----------|--------|
| Aligns with SANITIZATION_AND_INTEGRITY_REFACTOR_PLAN | Yes: sanitization out of strategy, integrity before/after, no return from checks, manifest records steps and check names. |
| Backward compatibility | Partial: legacy `integrity_checks` supported with deprecation; no-arg check fns supported with deprecation; CopyStep preserves awkward shallow-copy via _first_leaf (confirm container shape compatible). |
| Testability | Good: fallback path when builders not registered preserves hand-built context usage. |
| Memory / streaming | Improved: explicit builder pattern, no chunks on context. |
| Maintainability | Improved: single place for structural validation; clearer scoping of integrity. |

---

## 2. Phased integration plan

Integration is done in phases so that each step is testable and reversible. The end state is: current `data_transform.py`, `strategy.py`, and `pipeline.py` are replaced by the logic in the `*_new.py` drafts (with any fixes noted below).

---

### Phase 0: Pre-migration checks and small fixes

**Goal:** Avoid surprises and fix known issues in the draft files.

1. **CopyStep and container shape**
   - The new pipeline’s CopyStep already implements awkward-aware copy: it uses `_first_leaf(data[key])` to unwrap nested structure and detect the first array; `ak.Array` gets `copy(deep=False)`, other types get `copy(deep=True)`. No need to port CopyStep logic from the current pipeline.
   - Confirm that your container shape is compatible: `data[key]` is split-keyed (e.g. `{split: [unit_arrays]}` or `{split: ...}` with values that unwrap to arrays) so `_first_leaf` reaches a concrete array and does not raise or return an unexpected type.

2. **Clean pipeline_new**
   - Remove unused `from picid.transforms.base.integrity import run_checks` from pipeline_new if still present (new code uses `_run_scoped_checks` only).

3. **Imports in strategy_new**
   - strategy_new imports from `picid.transforms.base.pipeline`. Ensure pipeline_new exports every symbol strategy_new needs: `TransformContext`, `CopyStep`, `RegisterChunkBuilderStep`, `IntegrityCheckBeforeStep`, `IntegrityCheckAfterStep`, `FitStep`, `TransformStep`, `PostprocessStep`, `MergeStep`, `RecordManifestStep`, `build_default_before_checks`, `build_default_after_checks`, `_emit`.

4. **Run existing transform tests against current code**
   - Establish a green baseline: `pytest test/transforms/ -v` (and any integration tests that use DataTransform/strategy/pipeline).

**Checkpoint:** All current tests pass; container shape is confirmed compatible with CopyStep’s _first_leaf unwrapping; draft files are cleaned.

---

### Phase 1: Replace pipeline.py with new pipeline

**Goal:** New pipeline steps and context are in place; strategy and DataTransform still use the current versions (so they will be temporarily broken if they depend on the new steps).

1. **Back up and replace**
   - Copy current `pipeline.py` to e.g. `pipeline_legacy.py` (or keep in git history).
   - Replace `pipeline.py` with the contents of `pipeline_new.py` (CopyStep already has awkward-aware copy via _first_leaf; no extra fix needed).

2. **Compatibility shim for old strategy**
   - Current `strategy.py` imports `IntegrityCheckStep` and runs it once. The new pipeline has no `IntegrityCheckStep`; it has `IntegrityCheckBeforeStep` and `IntegrityCheckAfterStep`.
   - **Option A:** Add a thin `IntegrityCheckStep` in pipeline.py that runs the same single list as before (e.g. run no-arg checks via current `run_checks` from integrity.py) so existing strategy continues to work without change. Then in Phase 2 when we switch to the new strategy, we remove or repurpose IntegrityCheckStep.
   - **Option B:** In Phase 1, change strategy.py to use the new step order and new context fields (integrity_checks_before / integrity_checks_after, build_fit_chunks / build_transform_chunks). That implies updating strategy in the same phase as pipeline (effectively doing Phase 1 + 2 together). Prefer Option A if you want a smaller first step.

3. **Context and tests**
   - Extend TransformContext in pipeline.py with any new fields the new steps expect (integrity_before_ran, integrity_after_ran, integrity_before_check_names, integrity_after_check_names, build_fit_chunks, build_transform_chunks, integrity_checks_before, integrity_checks_after). The new pipeline already defines these in its TransformContext.
   - Update tests that import from pipeline: those that use `IntegrityCheckStep` either (a) keep using a compatibility IntegrityCheckStep, or (b) are updated to use IntegrityCheckBeforeStep / IntegrityCheckAfterStep and the new context fields. Prefer (a) in Phase 1 if you added the shim.

4. **Run tests**
   - `pytest test/transforms/base/test_pipeline.py test/transforms/base/test_manifest.py test/transforms/base/test_integrity.py -v`
   - Fix any failures (imports, context shape, step order).

**Checkpoint:** Pipeline tests pass; strategy (and DataTransform) still run with the old flow plus any shim.

---

### Phase 2: Replace strategy.py with new strategy

**Goal:** Strategy uses the new pipeline order (RegisterChunkBuilderStep, IntegrityCheckBeforeStep, IntegrityCheckAfterStep), passes integrity_checks_before/after and validate_output, and no longer calls get_sanitized_data in _prepare_chunks_for_split.

1. **Back up and replace**
   - Copy current `strategy.py` to `strategy_legacy.py` (or rely on git).
   - Replace `strategy.py` with the contents of `strategy_new.py`.

2. **Import alignment**
   - New strategy imports from `picid.transforms.base.pipeline`. Pipeline was already replaced in Phase 1; ensure it exports RegisterChunkBuilderStep, IntegrityCheckBeforeStep, IntegrityCheckAfterStep, build_default_before_checks, build_default_after_checks. Remove the IntegrityCheckStep shim from pipeline if it was added only for Phase 1.

3. **_prepare_chunks_for_split**
   - New strategy’s `_prepare_chunks_for_split` does not call `get_sanitized_data()`. Confirm no other code path expects sanitization to happen inside this method; integrity-before now covers that.

4. **Run tests**
   - `pytest test/transforms/base/test_strategy.py test/transforms/base/test_pipeline.py test/transforms/base/test_data_transform.py test/transforms/base/test_manifest.py -v`
   - Fix failures (e.g. tests that expected IntegrityCheckStep or a single integrity_checks list).

**Checkpoint:** Strategy and pipeline tests pass; DataTransform still uses current forward (which may still call _validate_forward_output).

---

### Phase 3: Replace data_transform.py with new DataTransform

**Goal:** DataTransform no longer runs _validate_forward_output; it uses validate_output, integrity_checks_before, integrity_checks_after, and forwards everything to strategy.apply().

1. **Back up and replace**
   - Copy current `data_transform.py` to `data_transform_legacy.py`.
   - Replace `data_transform.py` with the contents of `data_transform_new.py`.

2. **Metadata and constructor**
   - New DataTransform reads integrity_checks_before, integrity_checks_after, validate_output; maps legacy integrity_checks → integrity_checks_before with DeprecationWarning. Confirm no external config or code relies on the old metadata shape in an incompatible way.

3. **forward()**
   - New forward() only calls strategy.apply(...) with the new parameters (integrity_checks_before, integrity_checks_after, validate_output). No _validate_forward_output. Structural validation is entirely in IntegrityCheckAfterStep.

4. **apply_to as mapping**
   - New DataTransform.forward raises NotImplementedError when apply_to_map is not None (kwargs-style transform_data). If that path is used anywhere, either implement it in the new code or skip that path in migration and document.

5. **Run tests**
   - `pytest test/transforms/ -v` and any integration tests (e.g. preprocessor, transform_manager, test_data_transform, test_coverage_expansion, test_phase1_stabilization).
   - Fix failures (e.g. tests that expected _validate_forward_output to run or that passed integrity_checks as a single list).

**Checkpoint:** All transform-related tests pass; no validation logic remains in DataTransform.

---

### Phase 4: Tests and documentation

**Goal:** Full test coverage for the new behaviour; docs updated; legacy files removed or archived.

1. **Tests**
   - Add or extend tests for: RegisterChunkBuilderStep (chunks not on context; builders called per split); IntegrityCheckBeforeStep / IntegrityCheckAfterStep (scoped data, default checks, manifest fields); MergeStep._warn_on_changes; manifest integrity_before_* / integrity_after_*.
   - Run the full test suite (transforms, data, integration) and fix any remaining failures.

2. **Documentation**
   - Update `docs/transforms/developer_guide.md` and related docs to describe: new pipeline order (RegisterChunkBuilderStep, IntegrityCheckBeforeStep, IntegrityCheckAfterStep); before/after integrity and default checks; check function signatures (before: `Dict[str, List[NamedTransformInput]]`, after: `Dict[str, Any]`); manifest integrity fields; removal of _validate_forward_output and where validation lives.
   - Update `SANITIZATION_AND_INTEGRITY_REFACTOR_PLAN.md` to mark implemented phases as done and point to this migration doc.

3. **Cleanup**
   - Remove `data_transform_new.py`, `strategy_new.py`, `pipeline_new.py` (or move to an archive folder).
   - If you kept `*_legacy.py` backups, remove them or move to docs/archive once confident.
   - In pipeline, remove the no-arg check compatibility path when you are ready to drop the deprecation (optional; can be a later minor version).

**Checkpoint:** Docs and tests are up to date; codebase uses a single pipeline/strategy/data_transform implementation.

---

### Phase 5 (optional): Integrity module alignment

**Goal:** Reuse or extend `picid.transforms.base.integrity` for the new scoped checks so that policy handling (fail/warn/allow) and check running live in one place.

- Current `integrity.run_checks` takes no-arg callables. New pipeline uses `_run_scoped_checks(slice)` in pipeline.py.
- Option: Add `run_scoped_checks(checks, data_slice)` in integrity.py (or similar), have pipeline call it, and deprecate no-arg run_checks. Then pipeline only imports and calls integrity; no duplicated policy logic.
- This phase can be skipped if you prefer to keep _run_scoped_checks in pipeline for now.

---

## 3. Dependency order

- **Phase 0** does not depend on others.
- **Phase 1** (pipeline) must be done before Phase 2 (strategy), because strategy imports pipeline and uses its steps and context.
- **Phase 2** (strategy) must be done before Phase 3 (DataTransform), because DataTransform calls strategy.apply() and the new signature is defined by the new strategy.
- **Phase 3** can be done right after Phase 2.
- **Phase 4** after Phase 3.
- **Phase 5** anytime after Phase 1 (optional).

---

## 4. Rollback

- Each phase replaces one or two modules. Rollback is to revert the corresponding file(s) from git (or restore from `*_legacy.py` if kept).
- If Phase 1 introduces a compatibility IntegrityCheckStep, rolling back Phase 2 but keeping Phase 1 requires that the old strategy still runs with the new pipeline (same step names and context fields the old strategy expects). Phase 1 should therefore add the minimal context fields and, if needed, the shim so that the old strategy continues to work until Phase 2 is applied.

---

## 5. References

- **Refactor plan:** `docs/transforms/SANITIZATION_AND_INTEGRITY_REFACTOR_PLAN.md`
- **Draft files:** `picid/transforms/base/data_transform_new.py`, `strategy_new.py`, `pipeline_new.py`
- **Current pipeline/strategy/data_transform:** `picid/transforms/base/pipeline.py`, `strategy.py`, `data_transform.py`
- **Integrity (current):** `picid/transforms/base/integrity.py`

---

## 6. Migration status (completed)

Phases 0–3 are done. Behaviour is preserved; interfaces updated as described.

- **Phase 0–1:** Pipeline replaced with new steps; `IntegrityCheckStep` and `context.integrity_checks` kept for backward compat.
- **Phase 2:** Strategy replaced; uses `integrity_checks_before` / `integrity_checks_after`, no sanitization in `_prepare_chunks_for_split`.
- **Phase 3:** DataTransform replaced; no `_validate_forward_output`; validation in pipeline (`IntegrityCheckAfterStep` and default after-checks). Tests updated: message regexes, mock-strategy validation tests now assert no raise (validation lives in pipeline), caplog checks use case-insensitive "implicit".

**Tests:** `pytest test/transforms/base/` — 318 passed, 1 skipped. One failure in `test/transforms/base_transforms/test_autoencoder.py::test_transform_data_integration` (StopIteration from mock `side_effect` exhaustion) is outside the refactor; that test calls `transform_data` directly and does not use pipeline/strategy/DataTransform.

**Phase 4 (optional):** Add tests for new steps (RegisterChunkBuilderStep, IntegrityCheckBefore/AfterStep, MergeStep warnings), update developer docs, remove `*_new.py` drafts and legacy `IntegrityCheckStep` when ready.
