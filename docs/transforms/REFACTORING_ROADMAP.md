# Transform Orchestration — Refactoring Roadmap (Phases 2–5)

**Based on:** Phase 0 Audit & Coverage Report, TRANSFORM_ARCHITECTURE_AUDIT.md  
**Prerequisite:** Phase 1 stabilization suite passing and coverage target met (or documented gaps accepted).

---

## Overview

| Phase | Goal | Risk | Depends on |
|-------|------|------|------------|
| **2** | Registry-based dispatching; decouple type-handling from strategy | Medium | Phase 1 |
| **3** | Composable pipeline steps and lifecycle hooks | Medium | Phase 2 |
| **4** | Unified metadata manifest and slice-aware orchestration | High | Phase 3 |
| **5** | Tiered caching and structural dry run | Medium | Phase 3 |

---

## Phase 2: Transition to Registry-Based Dispatching

**Objective:** Replace `isinstance`/`issubclass`/`get_instance_cls()` branches in strategy and multisource with a single dispatch key and handler registry. No change to external API of `DataTransform.forward()` or `TransformStrategy.apply()` signature.

### 2.1 Data kind and capability

- **Introduce `DataKind`:** `"dense"` | `"ragged"` | `"ragged_regular"`.  
  - Implement `infer_data_kind(data: SplitDatasetContainer, apply_to_keys) -> DataKind` in a single place (e.g. `lmetk/transforms/base/data_kind.py`), using existing `get_instance_cls()` and `find_singular_ragged_dim` logic. Data objects remain unchanged for now.
- **Introduce `TransformCapability`:** `"dense"` | `"ragged"` | `"both"`.  
  - Implement `get_capability(transform_instance: BaseTransform) -> TransformCapability` from marker classes (`DenseTransform`, `RaggedTransform`, `RaggedOrDenseTransform`). Markers can remain; this is a thin wrapper.

### 2.2 Handler interface and registry

- **Define `TransformHandler` protocol (or ABC):**
  - `fit_prepare(segments, metadata) -> prepared`
  - `transform_apply(segments, transform_instance, metadata) -> results`
  - Optionally: `merge(base_data, results, metadata)` if not kept in strategy.
- **Implement concrete handlers:**
  - `DenseDenseHandler`: current dense path (concatenate numpy, fit/transform).
  - `RaggedDenseHandler`: current ragged→dense path (ragged_index_tuples, to_numpy, transform, reassembly with ListOffsetArray/tolist).
  - `RaggedRaggedHandler`: current ragged-native path.
  - `RaggedRegularHandler` (or fold into DenseDense with a “wrap back to ak” step): ragged_but_regular path.
- **Registry:** `Dict[tuple[DataKind, TransformCapability], TransformHandler]`. Populate at module load or via a small `register_handler(data_kind, capability, handler)` API.

### 2.3 Strategy and multisource changes

- **TransformStrategy.apply():**
  - Keep: hybrid copy loop. Delegate copy decision to a **copy policy** per `DataKind` (or keep one function `copy_policy(container, keys) -> None` that mutates `transformed_data`), so that `get_instance_cls() is not ak.Array` lives in one place (copy policy or dense handler).
  - Replace: fit and transform branches. Call `data_kind = infer_data_kind(...)`, `capability = get_capability(transform_instance)`, `handler = registry[(data_kind, capability)]`, then `handler.fit_prepare` / `handler.transform_apply` (and merge if moved). Preserve `_prepare_chunks_for_split`, `postprocess_transformed_data`, `_merge_transformed_data` as-is or move merge into handler.
- **Multisource mixins:**
  - `fit_function_to_data_segments` and `transform_function_to_data_segments` become thin: compute `data_kind` and `capability`, look up handler, call handler. All concatenation/reassembly logic moves into handlers. This is the largest code move.

### 2.4 Testing and rollout

- Run full Phase 1 + existing transform base tests after each subtask.
- No new public API; run.py and preprocessor unchanged.
- **Rollback:** Keep old strategy/multisource logic behind a feature flag or branch until handler coverage is verified.

### 2.5 Difficulty (from Phase 0)

- **multisource.py** refactor is **high effort**: extract three main branches (dense, ragged→dense, ragged native) plus ragged_but_regular into handlers without regressions. Phase 1 tests must cover reassembly and fit concatenation.

---

## Phase 3: Composable Pipeline Steps and Lifecycle Hooks

**Objective:** Replace the monolithic `apply()` with a pipeline of steps (Copy, Fit, Transform, Postprocess, Merge) and add observable lifecycle hooks without changing outputs.

### 3.1 TransformContext

- **Introduce `TransformContext`** (dataclass or simple class): holds `data`, `transform_instance`, `metadata`, `log`, and optionally `step_id`, `cache_status`. Passed through the pipeline.

### 3.2 PipelineStep protocol

- **`PipelineStep.run(context: TransformContext) -> TransformContext`.**  
  - Implement: `CopyStep`, `FitStep`, `TransformStep`, `PostprocessStep`, `MergeStep`. Each step reads from context and updates context (and may call handlers from Phase 2). `TransformStrategy.apply()` becomes: build context, run steps in order, return `context.data`, `context.log`.

### 3.3 Lifecycle hooks (observer pattern)

- **Events:** `on_pipeline_start`, `before_transform`, `after_transform`, `on_transform_error`, `on_pipeline_end`.  
  - Emit from within steps (e.g. `after_transform` after `TransformStep`). Hooks receive read-only view of container + `StepContext` (step id, transform identity, scope, timing, cache status).  
  - **No output from hooks;** they must not modify the container.  
  - Add a **hook registry** on the strategy or on a new “PipelineScene” object that the facade holds.

### 3.4 Testing

- Unit tests per step (CopyStep, FitStep, etc.) with mock context.  
  - Integration: same Phase 1 + existing tests; ensure no behaviour change when steps are composed.

### 3.5 Difficulty

- **Medium.** Clear boundaries; most work is extracting existing logic into step classes and adding hook invocation. Risk is in ensuring hooks are called in all code paths (multi-unit, fit vs transform, error path).

---

## Phase 4: Unified Metadata Manifest and Slice-Aware Orchestration

**Objective:** Attach a versioned, queryable metadata manifest to the container; make orchestration slice-aware (coordinates, bounds, original→sliced index mapping).

### 4.1 Metadata manifest

- **Schema:** Versioned (schema_version, producer_version). Attachable at global / per split / per key / per unit / per cycle.  
  - Two categories: **datasource metadata** (upstream) and **transform metadata** (fit stats, feature definitions, inverse state, validation outcomes).  
  - Queryable by: step id/name, key, split, unit/cycle/time slice.  
  - Must survive slicing, cache restore, merge, postprocess.

### 4.2 Slice-awareness

- **Slice representation:** Coordinate selections (unit ids, cycle ids, timestamps), resulting bounds, mapping from original→sliced indices where applicable.  
  - Downstream transforms can query “what slice am I operating on?” and “what coordinate system is this input under?”  
  - Preserve coordinate integrity under split-level ops, unit/cycle selection, and ragged segmentation.

### 4.3 Integrity guards (invariants and validators)

- **Invariants (mandatory)** and **validators (optional)** as injectable check chain between transforms.  
  - NaN check remains mandatory by default.  
  - **IntegrityPolicy:** controls whether a violation fails, warns, or is allowed (scoped overrides with reason and audit log).  
  - Per-transform `metadata.validation_policy` in config (expectations, overrides).

### 4.3.1 Two layers of checks: data object vs pipeline integrity

We have two distinct places where checks run; both are needed and serve different roles.

| Layer | Where | When it runs | Purpose |
|-------|--------|----------------|---------|
| **Data-object checks** | `lmetk/data/data_objects/utils.py` (e.g. `check_for_nans`, `check_length_consistency`); used in container validation / `convert_and_validate` | At load or validation time, before or outside the transform pipeline | “Is this data valid for use?” — reject or warn on bad raw data so it never enters the pipeline. |
| **Pipeline integrity checks** | `lmetk/transforms/base/integrity.py` (`run_check` / `run_checks`); `IntegrityCheckStep` in `pipeline.py` (runs after CopyStep, before FitStep) | Between transform steps, as part of the pipeline | “After the last step, does the current container still satisfy invariants?” — policy-driven (fail/warn/allow). |

**Why both are needed**

- **Data-object checks** gate **input** data: they catch NaNs, length mismatches, or structural issues introduced by the datasource or load path. Without them, invalid data can enter the pipeline and cause hard-to-debug failures deep inside a transform.
- **Pipeline integrity checks** guard **post-step state**: they catch invariants violated by a transform (e.g. a scaler or buggy transform that introduces NaNs), or enforce orchestration invariants (e.g. manifest present after RecordManifestStep). They run in pipeline order and can be configured per environment (e.g. fail in prod, warn in dev).

**Strong use cases**

- **Data-object checks**
  - **Load/validation gate:** After loading a dataset (e.g. UNIBO from disk) into a `SplitDatasetContainer`, run `check_for_nans` and length consistency on the keys that will be used by the pipeline. Reject or warn before any transform runs so obviously bad data (sensor NaNs, CSV glitches, mismatched lengths) never reaches the first transform.
  - **Pre-pipeline contract:** Before calling the transform manager, ensure the container meets the pipeline’s assumed contract (e.g. `features` and `target` have no NaNs, keys exist). Run data-object checks once at load so we don’t waste time running transforms on invalid data.

- **Pipeline integrity checks**
  - **Post-step NaN invariant:** After a transform that might introduce NaNs (e.g. scaler, custom math), register a NaN check with policy `fail` so the pipeline fails immediately after that step instead of propagating bad data to later transforms or the evaluator.
  - **Environment-specific strictness:** Same check (e.g. “no NaNs in features/rul”) can use policy `fail` in production and `warn` in dev, so production stays strict while dev can continue for debugging.
  - **Orchestration invariants:** Checks that depend on pipeline state (e.g. “manifest is present and non-empty after RecordManifestStep”, “slice_info matches container splits”) belong in the pipeline integrity layer, not in data-object validation, because they refer to state produced by earlier steps.

Concrete check **implementations** (e.g. a reusable NaN check callable for the pipeline) are not yet centralized: the data layer has `check_for_nans()` in `utils.py`; the pipeline expects callables to be injected via `context.integrity_checks`. A follow-up can add a small set of built-in pipeline checks (e.g. “NaN in container keys X”) that wrap or reuse the data-layer logic where appropriate.

### 4.4 Data objects impact

- Containers may need optional fields or a wrapper for manifest and slice info. Phase 0 assessed this as **low–medium** if the container interface is unchanged; Phase 4 may add attributes or a separate manifest store that the orchestration updates.

### 4.5 Difficulty

- **High.** Cross-cutting concern; touches data objects, strategy, steps, and config. Recommend small increments: manifest only first, then slice info, then policy.

---

## Phase 5: Tiered Caching and Structural Dry Run

**Objective:** Cache boundaries (cache_point) and resume from latest valid boundary; structural dry run (no heavy math, shallow schema only).

### 5.1 Cache boundaries (done)

- **Implemented:** ConfigTransformManager exposes cache points and boundary helpers; PreProcessor restores from latest valid boundary and runs only the remaining transforms.
  - **Config:** Per-transform `metadata.cache_point: true` marks a step as a cache boundary.
  - **Manager API:** `get_cache_point_names()` (names in pipeline order), `get_transform_names_after(name)` (tail for resume), `get_config_up_to_and_including(name)` (for boundary cache key).
  - **Cache key:** Boundary key = `compute_cache_key({ datasource, transforms up to and including name }, library_dirs, [".py"])`. Full preprocessed key unchanged.
  - **PreProcessor.pipeline():** After Stage 1 (load/split), if cache_preprocessed and cache points exist: try loading from each boundary in reverse order; on first hit, restore container and manager, run only `get_transform_names_after(boundary_name)`, then save to preprocessed and return. When running all transforms, after each transform that is a cache point, save to stage `"boundary"` with the corresponding key.
  - **FileSystemCache:** Stage `"boundary"` uses nested layout by cache_key (same as `"preprocessed"`).
  - **Profiling:** Optional follow-up (hit/miss per step, restore time, time saved).

### 5.2 Structural dry run (done)

- **Implemented:** `dry_run_transforms(manager, container)` in `lmetk.transforms.base.dry_run`; validates apply_to keys and handler routing, no execution. See test_dry_run.py. Original goal: **Verify without executing heavy math:** input contract per step, apply_to/assign_to correctness, routing feasibility (handler availability for data_kind × capability), slice/coordinate preservation, validator coverage, cache boundary plan.  
  - **No exact tensor shapes;** only shallow traits (dense/ragged/unknown, coarse rank).  
  - Implement as a dedicated “dry run” mode that runs the pipeline with a no-op transform or mock that only validates contracts.

### 5.3 Inverse transform routing (from audit wish list)

- **Manager API:** `get_inverter_for_key(key, which="last"|"first") -> Optional[InverseTransformMixin]`. Matches transforms that **assign to** `key`; when several match, pipeline order + `which` pick first or last.  
  - **run.py:** Prefer `inverse_transform_key` (data key, e.g. `rul`) → `get_inverter_for_key(key)`; fallback to `inverse_transform_name` + `get_transform`.  
  - **Evaluator:** Standard case (target scaler) → set `inverse_transform_key: rul` (when assign_to is rul) or keep `inverse_transform_name: scaler_target`. Other scalers (e.g. feature) → use `inverse_transform_name` to pick a specific transform. See [INVERSE_TRANSFORM_ROUTING.md](INVERSE_TRANSFORM_ROUTING.md).
- **Follow-up (docs/clarity):** Document key = assign_to; add config example for inverse_transform_key; optionally expose `which` and add resolution logging (see INVERSE_TRANSFORM_ROUTING.md §4).

### 5.4 Difficulty

- **Medium.** Caching and dry run are additive. Inverse routing is a localised change to manager and run.py once registry exists.

---

## Execution Order and Gates

1. **Phase 1 (done):** Stabilization suite in place; coverage improved; transform base tests green.
2. **Phase 2 (done):** data_kind/capability, handlers, registry; multisource uses registry. **Gate:** Phase 1 + all existing tests green. ✅
3. **Phase 3 (done):** TransformContext, CopyStep/FitStep/TransformStep/PostprocessStep/MergeStep, lifecycle hooks; strategy.apply() runs pipeline. **Gate:** Phase 2 stable; no API change to DataTransform.forward. ✅
4. **Phase 4:** Metadata manifest and slice-awareness in small steps; add integrity policy. **Gate:** Phase 3 stable. **4.1 (done):** Manifest schema, MetadataManifest, optional manifest on DatasetContainer, RecordManifestStep, tests. ✅ **4.2 (done):** SliceInfo, optional slice_info on DatasetContainer, TransformContext.slice_info, manifest payload slice summary, tests. ✅ **4.3 (done):** IntegrityPolicy (fail/warn/allow), run_check/run_checks, IntegrityCheckStep, context.integrity_checks, tests. ✅
5. **Phase 5:** Cache boundaries, dry run, inverse registry. **Gate:** Phase 3 (and optionally Phase 4) stable. **5.1 (done):** Cache points via `metadata.cache_point`; `get_cache_point_names()`, `get_transform_names_after()`, `get_config_up_to_and_including()`; PreProcessor restores from latest boundary and runs tail; FileSystemCache stage `"boundary"`. ✅ **5.2 (done):** Structural dry run: `dry_run_transforms(manager, container)` validates apply_to keys, data_kind × capability handler availability; no fit/transform execution. ✅ **5.3 (done):** `get_inverter_for_key(key)`, run.py uses it when `inverse_transform_key` is set, evaluator can omit `inverse_transform_name`. ✅

---

## Status: What’s Done vs What’s Left

*Cross-reference: [TRANSFORM_ARCHITECTURE_AUDIT.md](TRANSFORM_ARCHITECTURE_AUDIT.md) for motivation, friction audit, and wish list.*

### Done ✅

| Area | Delivered |
|------|-----------|
| **Phase 1** | Stabilization suite; transform base tests green. |
| **Phase 2** | `DataKind` / `TransformCapability`; `infer_data_kind`, `get_capability`; handler registry; `DenseDenseHandler`, `RaggedDenseHandler`, `RaggedRaggedHandler`; strategy and multisource use registry. |
| **Phase 3** | `TransformContext`; `CopyStep`, `FitStep`, `TransformStep`, `PostprocessStep`, `MergeStep`, `RecordManifestStep`, `IntegrityCheckStep`, `MergeStep`; strategy runs pipeline of steps; lifecycle hooks (observer pattern) in pipeline. |
| **Phase 4.1** | Metadata manifest: schema, `MetadataManifest`, optional on `DatasetContainer`; `RecordManifestStep`; tests. |
| **Phase 4.2** | Slice-awareness: `SliceInfo`, optional on container; `TransformContext.slice_info`; manifest payload slice summary; tests. |
| **Phase 4.3** | Integrity: `IntegrityPolicy` (fail/warn/allow), `run_check` / `run_checks`, `IntegrityCheckStep`, `context.integrity_checks`; tests. Doc §4.3.1: data-object vs pipeline integrity checks. |
| **Phase 5.1** | Cache boundaries: `metadata.cache_point`; `get_cache_point_names()`, `get_transform_names_after()`, `get_config_up_to_and_including()`; PreProcessor restores from latest boundary, runs tail, optional `save_boundary_caches`; FileSystemCache stage `"boundary"`; pipeline doc. |
| **Phase 5.2** | Structural dry run: `dry_run_transforms(manager, container)`; apply_to keys (including pipeline-produced via assign_to); data_kind × capability handler check; tests. |
| **Phase 5.3** | Inverse routing: `get_inverter_for_key(key, which)`; run.py uses `inverse_transform_key`; [INVERSE_TRANSFORM_ROUTING.md](INVERSE_TRANSFORM_ROUTING.md). |
| **Caching tests** | `test_file_system_cache.py` (boundary stage); `test_preprocessor_caching.py` (disk cache, preprocessed/boundary, save_boundary_caches); `scripts/run_caching_coverage.sh`. |

### Not done / Optional follow-ups

| Area | What’s left |
|------|-------------|
| **Phase 4.4** | Data objects: optional manifest/slice fields are in place; any further container impact is low–medium (roadmap). |
| **Phase 4.3** | **Built-in pipeline checks:** No reusable check implementations in `integrity.py`; pipeline expects callables via `context.integrity_checks`. Follow-up: add built-in checks (e.g. NaN in container keys) wrapping data-layer `check_for_nans` where appropriate. |
| **Phase 4.3** | **Per-transform validation_policy in config:** Audit §2.3.4 — `metadata.validation_policy.expectations` / `overrides` per transform; scoped overrides with reason and audit. Not implemented as config-driven policy. |
| **Phase 5.1** | **Profiling:** Cache hit/miss per step, restore time, time saved (roadmap: optional follow-up). |
| **Phase 5.3** | **Docs/clarity:** Config example for `inverse_transform_key`; expose `which` and resolution logging (INVERSE_TRANSFORM_ROUTING.md §4). |
| **Audit §1.2 / §3.4** | **Protocols and typed metadata:** Capability protocols (SupportsDense/SupportsRagged) and `TransformMetadata` TypedDict/dataclass — not done; markers and `Dict[str, Any]` metadata remain. |
| **Audit §2.2** | **New data type (e.g. PointClouds):** Still closed; would require new handler + capability and registry entry. |
| **Audit §3) Wish list** | **StepRecord / Execution Report:** Per-step identity, scope, timing, cache info, validation outcomes; end-of-run summary and bottleneck hints — not implemented. |
| **Audit §3) Wish list** | **Validation-first execution:** Run invariants/validators before and after *each* transform (atomic transaction). Current design: single `IntegrityCheckStep` after CopyStep; not per-transform. |
| **Audit §1.1** | **Lifecycle hook registry:** Pipeline has steps and hook points; a formal hook registry and guaranteed emission of `on_pipeline_start`, `before_transform`, `after_transform`, `on_transform_error`, `on_pipeline_end` across all paths (e.g. cache hit) may need verification or completion. |

### Summary

- **Phases 2–5** of the roadmap are **implemented** in code: registry, pipeline steps, manifest, slice-info, integrity policy, cache boundaries, dry run, inverse routing. Remaining work is **follow-ups and audit wish-list items**: built-in integrity check implementations, config-driven validation policy, profiling, StepRecord/Execution Report, per-transform validation step, protocols/typed metadata, and documentation/examples.
- See [TRANSFORM_ARCHITECTURE_AUDIT.md](TRANSFORM_ARCHITECTURE_AUDIT.md) for the full friction audit, target state, and refined wish list (§3 and Appendix).

---

## Summary

- **Phase 2** removes type-branching from hot paths and centralises it in a registry and handlers; **multisource** is the main risk area.  
- **Phase 3** improves testability and observability without changing behaviour.  
- **Phase 4** and **Phase 5** deliver the audit’s advanced features (manifest, slicing, caching, dry run, inverse routing) on top of a decoupled, observable pipeline.
