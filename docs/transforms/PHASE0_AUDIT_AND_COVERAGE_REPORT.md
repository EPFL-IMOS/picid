# Phase 0: Discovery, Audit & Coverage Report

**Scope:** Transform orchestration refactor — lmetk library  
**Date:** 2026-02-16  
**Reference:** [TRANSFORM_ARCHITECTURE_AUDIT.md](TRANSFORM_ARCHITECTURE_AUDIT.md)

---

## 1. Codebase Deep-Scan

### 1.1 Navigation Logic (Data Flow Map)

End-to-end flow is traced as follows:

```
PreProcessor.pipeline()
  → fetch_data() → DatasetContainer (from datasource.get_data())
  → apply_transforms(data, transforms.get_transforms())
       for (transform_name, DataTransform) in transforms.items():
         data, log = transform.forward(data)
           → DataTransform.forward()
             → self.strategy.apply(transform_instance, data, apply_to_keys, assign_to_keys, ...)
                 → TransformStrategy.apply()  [decorated by inject_transform_context_to_strategy_apply]
                   → 1. Hybrid copy (per key: deep if not ak.Array, shallow if ak.Array)
                   → 2. Fit phase (if fit_on_split + fit_on_key):
                         fit_chunks = _prepare_chunks_for_split({fit_on_key: data[fit_on_key]}, fit_on_split, is_multi_unit)
                         if is_multi_unit: transform_instance.fit_multi_source(fit_chunks, metadata)
                         else:             transform_instance.fit_data(fit_chunks[0], metadata)
                   → 3. Transform phase (per split in available_splits):
                         chunks = _prepare_chunks_for_split(target_data_to_transform, split, is_multi_unit)
                         if is_multi_unit: transformed_chunks, log = transform_instance.transform_multi_source(chunks, metadata)
                         else:             transformed_chunks = [transform_instance.transform_data(chunks[0], metadata)]
                         transformed_chunks = postprocess_transformed_data(transformed_chunks, metadata)
                   → 4. _merge_transformed_data(base_data, transformed_results_for_new_key, available_splits)
             → Post-transformation validation (SplitDatasetContainer, keys, splits, type warnings)
  → return data
```

**Key files:**

| Step | File | Entry |
|------|------|--------|
| Entry | `lmetk/data/preprocessing/preprocessor.py` | `apply_transforms()`, `pipeline()` |
| Facade | `lmetk/transforms/base/data_transform.py` | `forward()` → `strategy.apply()` |
| Strategy | `lmetk/transforms/base/strategy.py` | `apply()`, `_prepare_chunks_for_split()`, `_merge_transformed_data()` |
| Multi-source | `lmetk/transforms/base/multisource.py` | `fit_multi_source()` → `fit_function_to_data_segments()`, `transform_multi_source()` → `transform_function_to_data_segments()` |
| Markers | `lmetk/transforms/base/base_transform.py` | `DenseTransform`, `RaggedTransform`, `RaggedOrDenseTransform` |
| Postprocess | `lmetk/transforms/base/strategy.py` | `postprocess_transformed_data()` |

### 1.2 Change Targets: Type-Branching and Marker Checks

All relevant occurrences in `lmetk/transforms/base/`:

| File | Line(s) | Pattern | Purpose |
|------|--------|---------|--------|
| **strategy.py** | 21 | `isinstance(obj, Sequence)`, `isinstance(..., (str, bytes))` | `is_sequence()` helper |
| **strategy.py** | 39, 46-47, 56 | `isinstance(data[0], Mapping)`, `isinstance(chunk, Mapping)` | Postprocess: single vs list vs named dicts |
| **strategy.py** | 81, 84, 87 | `isinstance(apply_to_keys, str)` etc. | Normalize keys to list |
| **strategy.py** | **96** | **`data[key].get_instance_cls() is not ak.Array`** | **Copy policy: deep vs shallow** |
| **strategy.py** | 116 | `isinstance(data[apply_key], Mapping)` | Validate split structure |
| **strategy.py** | 243 | `assert isinstance(data_to_prepare[keys[0]][split], list)` | Multi-unit structure |
| **multisource.py** | 103, 106 | `isinstance(d, (SortedDict))`, `isinstance(d, dict)` | `tolist()` recursion |
| **multisource.py** | 216 | `isinstance(apply_to_keys, list)` | Normalize apply_to_keys |
| **multisource.py** | **221-222** | **`issubclass(cls, RaggedTransform)`, `issubclass(cls, DenseTransform)`** | **Fit: transform capability** |
| **multisource.py** | **225, 234** | **`chunk.get_instance_cls().items()`, `data_types.pop() is ak.Array`** | **Fit: data kind** |
| **multisource.py** | 350, 354-357 | `isinstance(apply_to_keys, list)`, `issubclass(cls, RaggedTransform)` etc. | Transform: capability |
| **multisource.py** | 362, 371 | `chunk.get_instance_cls().values()`, `data_types.pop() is ak.Array` | Transform: data kind |
| **multisource.py** | 401 | `isinstance(out_segment, BaseDataObjectWithMetadata)` | Ragged-but-regular output wrap |
| **multisource.py** | 447, 472, 487, 495-500, 515, 518 | `ak.Array(layout)`, `ak.from_numpy`, `ListOffsetArray`, `ak_find_var_dims` | Ragged→dense reassembly |
| **multisource.py** | 456-461 | `index_depth == 1` vs else `ak.Array(tolist(v))` | Reassembly branch |
| **multisource.py** | 720, 747 | `data.get_instance_cls()`, `data_types[key] is ak.Array` / `np.ndarray` | ConcatenateUnitsMixing |
| **data_transform.py** | 40 | `issubclass(ann, expected)` | Signature validation |
| **data_transform.py** | 70, 75, 91, 103, 107 | `isinstance(apply_to, ListConfig)`, `Mapping`, `str`, etc. | Metadata processing |
| **data_transform.py** | 173, 182, 186, 191 | `isinstance(transform, BaseTransform)` | Transform instance validation |
| **data_transform.py** | 300, 309, 313, 325, 347, 353 | `isinstance(transformed_data, SplitDatasetContainer)` etc. | Post-forward validation |
| **transform_manager.py** | 44 | `isinstance(config, dict)` | Config normalization |

**Summary:** The main **change targets** for registry-based dispatching are:

- **strategy.py:** Replace `get_instance_cls() is not ak.Array` with a copy policy from a registry (or data-kind handler).
- **multisource.py:** Replace all `issubclass(cls, RaggedTransform/DenseTransform/RaggedOrDenseTransform)` and `data_types.pop() is ak.Array` with a single dispatch key `(data_kind, transform_capability)` and handler lookup. Replace explicit `np.concatenate`/`ak.concatenate`/reassembly logic with handler methods.

---

## 2. Secondary Impact Audit

### 2.1 Data Objects (`lmetk/data/data_objects/`)

| File | Role | Impact for refactor |
|------|------|---------------------|
| **base.py** | `BaseDataObject.get_instance_cls()`, allowed types `(pd.DataFrame, pd.Series, np.ndarray, ak.Array)`, `__setitem__` type recording | **Medium.** Dispatch will still need “data kind” (dense/ragged). Options: (1) Keep `get_instance_cls()` and derive kind in one place; (2) Add `get_data_kind()` that returns `"dense"` \| `"ragged"` \| `"ragged_regular"`. Adding a new type (e.g. PointCloud) would require extending allowed types and kind. |
| **data.py** | `NamedTransformInput`, `SplitDatasetContainer`, `DatasetContainer`, `to_split_dict()`, `group_by_split()`, `is_multi_unit` | **Low–Medium.** Orchestration assumes this structure. Slice-aware manifest (Phase 4) may add attributes or wrappers; no change required for Phases 2–3 if container interface is unchanged. |

**Metadata/slicing:** Today there is no unified “metadata manifest” or slice coordinates. Containers do not carry explicit coordinate bounds or slice maps. Phase 4 will add slice-aware orchestration and metadata manifest; data objects may need optional fields or a wrapper.

### 2.2 Run & Evaluation (`lmetk/run.py`, evaluators)

| Location | Usage | Impact for refactor |
|----------|--------|---------------------|
| **run.py** ~734–772 | Reads `apply_inverse_scaling`, `inverse_transform_name` from evaluator config; gets transform via `transforms_manager.get_transform(inverse_transform_name)`; passes `transform_instance` and checks `isinstance(..., InverseTransformMixin)` | **Medium.** Audit wish list: “Intelligent Inverse Transform Routing” — manager should expose `get_inverter_for_key(key)`. Refactor: replace manual `inverse_transform_name` + `get_transform` by a registry on the manager; run.py would call `get_inverter_for_key("target_variable")` and no longer need to know transform names. |
| **evaluator/default.py** | Uses `scaling_wrapper.apply_inverse`, `inverse_transform_if_needed` | **Low.** Interface is “inverse_transform + apply_inverse flag”; no change if the object passed in still implements the same interface. |
| **evaluator/base.py** | Holds `inverse_transform: Optional[InverseTransformMixin]` | **Low.** Same as above. |
| **evaluator/scaling_wrapper.py** | Calls `inverse_transform.inverse_transform(...)` | **Low.** Depends on interface, not on how run.py obtained the transform. |

### 2.3 Config / Transform Manager (`lmetk/transforms/base/transform_manager.py`)

| Concern | Current | Impact for refactor |
|--------|--------|---------------------|
| Cache boundaries | Not implemented | Phase 5: add `cache_point` and resume-from-boundary logic; manager (or roadmap) must expose which steps are cache points. |
| Registry init | None; strategy is created inside `DataTransform.__init__` | Phase 2: registry can be global or owned by a “scene”; manager may need to pass registry into DataTransform or strategy factory. |
| Lazy vs eager | `_instantiate_all()` builds `DataTransform` list | No change for Phase 2–3 if we only swap strategy internals. |

### 2.4 Decorators (`lmetk/utils/decorators.py`)

- `inject_transform_context_to_strategy_apply`: Wraps `TransformStrategy.apply`; adds context to exceptions. **No change** for registry refactor if `apply()` signature is unchanged.
- `check_transform_output_consistency`: Wraps `transform_data`; checks assign_to_map vs output keys. **No change** for Phase 2–3.

### 2.5 Conversion Utils (`lmetk/transforms/utils.py`)

- `_convert_to_numpy` / `_convert_from_numpy`: Explicit branches for `np.ndarray`, `pd.DataFrame`, `pd.Series`, `torch.Tensor`. **Abstraction leak:** orchestration (multisource) does not call these directly; they are used by transforms that need numpy. For a new backend (e.g. GPU), these would need extension; not blocking Phase 2.

---

## 3. Difficulty Assessment & Abstraction Leaks

| Module | Refactor difficulty (1–5) | Notes |
|--------|---------------------------|--------|
| **strategy.py** | 2 | Single place for copy policy; replace with data-kind lookup. Small, clear boundary. |
| **multisource.py** | 5 | Large, branch-heavy methods; fit and transform paths mirror each other; ragged→dense reassembly is complex and must stay correct. Highest risk. |
| **base_transform.py** | 1 | Markers can remain during transition; later add Protocols. Low risk. |
| **data_transform.py** | 2 | Keeps calling `strategy.apply()`; validation stays. May later accept optional strategy or registry. |
| **transform_manager.py** | 2 | Add registry/cache_point later; no change for Phase 2. |
| **data_objects (base + data)** | 2 | Optional `get_data_kind()` or keep using `get_instance_cls()` behind a single abstraction. |
| **run.py (inverse)** | 2 | New method `get_inverter_for_key` on manager; evaluator config simplified. |

**Abstraction leaks (where orchestration knows concrete types):**

- **strategy.py:** `get_instance_cls() is not ak.Array` → copy policy.
- **multisource.py:** `ak.Array`, `np.ndarray`, `np.concatenate`, `ak.concatenate`, `ListOffsetArray`, `to_numpy()`, `ak.from_regular`, `ak_find_var_dims` — all in the orchestration layer.
- **utils.py:** `torch.Tensor`, `.device` in conversion helpers (used by transforms, not strategy).

---

## 4. Test Coverage Audit

### 4.1 Coverage (from `pytest test/transforms/base/ --cov=lmetk/transforms/base`)

| File | Stmts | Miss | Cover | Missing lines (representative) |
|------|-------|------|------|--------------------------------|
| base_transform.py | 28 | 1 | **96%** | 33 |
| data_transform.py | 131 | 6 | **95%** | 52, 92, 183, 192, 348, 360 |
| strategy.py | 90 | 3 | **97%** | 59, 99, 117 |
| **multisource.py** | **287** | **131** | **54%** | 119, 127, 238-242, 257-309, 351, 367, 395-396, 401-405, 447, 472, 487, 495-500, 515, 518, 533-598, 661, 679-684, 709-793, 830-832, 845-852 |
| transform_manager.py | 119 | 30 | **75%** | 40, 46, 51, 64, 73-76, 124, 139, 148, 160, 201-207, 217-218, 230-231, 247-248, 300, 304, 308, 341-348 |

**Overall (transforms/base):** 74% (655 statements, 171 missed).

Secondary modules were not exercised by `test/transforms/base/` (data_objects, preprocessor, utils are used but not measured in that run). Achieving “100% functional coverage for orchestration and secondary dependencies” requires either extending the same coverage run to those modules or adding integration tests that import them.

### 4.2 Dark Zones (Critical Paths Lacking Robust Assertions)

1. **Ragged → Dense fallback (multisource.transform_function_to_data_segments)**  
   - Partially covered: `test_transform_multi_source_ragged_to_dense` runs the branch but does not assert on **output shapes**, **dtypes**, or **ragged segmentation/reassembly correctness**.  
   - Missing: multi-segment ragged input; assertion that reassembled awkward array has correct `var * ...` layout and values; no assertion on `index_depth > 1` reassembly (`tolist` path).

2. **Multi-source fit reassembly**  
   - Fit path for **ragged + dense-only** (lines 257–282): concatenate ragged, then `to_numpy()`, then `fit_func`. No test that fits on concatenated ragged and that subsequent transform sees consistent state.  
   - Fit path for **ragged + ragged-supporting** (284–309): concatenate along `var_dim`; no test that checks shape/type of concatenated awkward array passed to `fit_data`.

3. **Ragged-but-regular (multisource)**  
   - Branch where `data_is_ragged_but_regular` is True (regular awkward): convert to numpy for transform, then wrap back to `ak.Array` (lines 386–405). Not explicitly tested with a regular awkward array and a dense-only transform.

4. **postprocess_transformed_data unrecognized format (strategy line 59)**  
   - Tests expect ValueError/TypeError/AssertionError but do not assert exact message or contract for “unrecognized format.”

5. **Copy behavior (strategy)**  
   - Tests check that result has features and that ragged uses shallow copy; no explicit assertion that **non-ak keys are deep-copied** (e.g. that modifying result does not modify input for dense keys).

6. **TransformStrategy with transform_on_keys**  
   - `test_strategy_transform_on_keys_filtering` runs the path but does not assert that **test** split is omitted from transformation when only `["train","val"]` are in `transform_on_keys`.

7. **ConcatenateUnitsMixing**  
   - Full path (axis-based concatenation for ak/numpy, then single `transform_data`) and **InverseTransformMixin** (`inverse_transform_multi_source`, `inverse_transform`) are not tested in base tests.

8. **Exception logging in transform_function_to_data_segments**  
   - Lines 533–598 (exception handler with detailed logging) are not covered; no test that triggers an exception inside the ragged-to-dense loop and asserts that the re-raised exception or message contains the expected context.

9. **DataTransform: apply_to as mapping (line 281)**  
   - Covered by test; line 52 (implicit assign_to logging) and 183/192 (redundant isinstance) are minor gaps.

10. **PreProcessor.apply_transforms**  
    - Not under test in `test/transforms/base/`; integration coverage would be required for “100% functional coverage” of preprocessing entry point.

---

## 5. Phase 0 Summary: Affected Code and Risk

### 5.1 Affected Code (Concise)

- **Primary (orchestration):** `strategy.py`, `multisource.py`, `data_transform.py`, `base_transform.py`, `transform_manager.py`.
- **Secondary:** `lmetk/data/data_objects/base.py`, `lmetk/data/data_objects/data.py`, `lmetk/run.py` (inverse routing), `lmetk/evaluator/base.py`, `lmetk/evaluator/scaling_wrapper.py`, `lmetk/transforms/utils.py`, `lmetk/utils/decorators.py`, `lmetk/data/preprocessing/preprocessor.py`.

### 5.2 Refactor Difficulty Score (per module)

| Module | Score (1–5) | Rationale |
|--------|-------------|-----------|
| strategy.py | 2 | Small surface; copy policy and apply() structure. |
| multisource.py | 5 | Many branches, reassembly, fit/transform symmetry; most “math vs routing” coupling. |
| data_transform.py | 2 | Facade; stable API. |
| base_transform.py | 1 | Markers only. |
| transform_manager.py | 2 | Config + future cache/registry. |
| data_objects | 2 | Optional get_data_kind; structure unchanged for Phase 2–3. |
| run.py / evaluators | 2 | Inverse routing API change. |
| preprocessor | 1 | No change for Phase 2–3. |

### 5.3 Risks

- **High:** Regressions in ragged→dense and multi-source fit/transform if refactor is done without characterization tests that lock shapes, dtypes, and metadata.
- **Medium:** Run/evaluator config breakage if inverse routing is changed without backward compatibility or migration path.
- **Low:** Manager and data object changes are incremental and testable in isolation.

---

## 6. Next Step: Phase 1 (Stabilization)

Phase 1 must:

1. Add **characterization tests** that lock current behavior with **deep assertions** (shapes, dtypes, metadata, ragged layout, copy semantics, fit side-effects).
2. Target **100% functional coverage** of orchestration (strategy + multisource + data_transform + base_transform) and, where feasible, secondary impact code paths used by transforms.
3. Run the stabilization suite and **block Phase 2** until all tests pass and coverage target is met.

The concrete Phase 1 test suite is provided in `test/transforms/base/test_phase1_stabilization.py`.

### 4.3 Phase 1 verification (after adding stabilization suite)

After adding `test_phase1_stabilization.py`:

- **All 137 tests** in `test/transforms/base/` pass (116 existing + 21 Phase 1).
- **Coverage improvement:** multisource 54% → 74%, strategy 97% → 98%, **total 74% → 83%**.
- **Remaining gaps:** multisource still has ~75 missed lines (exception logging 533–598, ConcatenateUnitsMixing branches, InverseTransformMixin init, some edge branches). To approach 100% functional coverage, add tests for: exception path in `transform_function_to_data_segments`, full ConcatenateUnitsMixing with ak/numpy and axis, and NoFitMultiSourceTransformMixin/InverseTransformMixin init paths.

### 4.4 Coverage expansion (test_coverage_expansion.py)

After adding `test_coverage_expansion.py` (34 tests targeting missing lines):

- **171 tests** in `test/transforms/base/` pass.
- **Coverage:** strategy **100%**, transform_manager **94%**, data_transform **96%**, base_transform **96%**, multisource **90%**, **total 94%**.
- **Remaining missing:** base_transform 1 line (default `fit_data` body); data_transform 5 (implicit assign_to log, DictConfig path, redundant check, type/source warnings); multisource 29 (tolist docstrings, fit ValueError branch, ragged-but-regular BaseDataObjectWithMetadata branch, index_depth>1 reassembly, exception sub-branches, ConcatenateUnitsMixing edges); transform_manager 7 (empty config loop, add/remove/update after instantiation, get_data_transform KeyError, get_transforms_by_*).

---

## 5. Phase 2 & 3 completion (2026-02-16)

### 5.1 Phase 2: Registry-based dispatching (done)

- **2.1 Data kind and capability**
  - `lmetk/transforms/base/data_kind.py`: `DataKind` (`"dense"` | `"ragged"` | `"ragged_regular"`), `TransformCapability`, `infer_data_kind()`, `get_capability()`.
  - Tests: `test/transforms/base/test_data_kind.py`.
- **2.2 Handler interface and registry**
  - `lmetk/transforms/base/handlers.py`: `TransformHandler` protocol, `DenseDenseHandler`, `RaggedDenseHandler`, `RaggedRaggedHandler`, `get_handler(data_kind, capability)`.
  - Tests: `test/transforms/base/test_handlers.py`.
- **2.3 Multisource uses registry**
  - `fit_function_to_data_segments` and `transform_function_to_data_segments` in `multisource.py` use `infer_data_kind`, `get_capability`, `get_handler`, then handler `fit_prepare` / `transform_apply`.
- **2.4 Testing**
  - Transform base: 214 passed, 1 skipped (pre–Phase 3). No API change to `DataTransform.forward()` or `TransformStrategy.apply()` signature.

### 5.2 Phase 3: Composable pipeline steps and lifecycle hooks (done)

- **3.1 TransformContext**
  - `lmetk/transforms/base/pipeline.py`: `TransformContext` dataclass with `data`, `transform_instance`, keys, fit/transform options, `log`, and mutable state (`transformed_data`, `available_splits`, `transformed_results_for_new_key`, `_raw_transformed_by_split`); optional `strategy`, `postprocess_fn`.
- **3.2 PipelineStep protocol and concrete steps**
  - `PipelineStep` protocol: `run(context) -> TransformContext`.
  - Concrete steps: `CopyStep`, `FitStep`, `TransformStep`, `PostprocessStep`, `MergeStep`. Each uses `context.strategy` and `context.postprocess_fn` (no direct strategy import in pipeline for steps).
  - `TransformStrategy.apply()` refactored to: normalize keys, build `TransformContext` (strategy=self, postprocess_fn=postprocess_transformed_data), emit `on_pipeline_start`, run the five steps in order, emit `on_transform_error` on exception, emit `on_pipeline_end`, return `(context.transformed_data, context.log)`.
- **3.3 Lifecycle hooks**
  - Hook registry: `register_hook(event, callback)`, `clear_hooks(event?)`, `_emit(event, context)`. Events: `on_pipeline_start`, `before_transform`, `after_transform`, `on_transform_error`, `on_pipeline_end`. Hooks receive (event, context); context is read-only for hooks; hook exceptions are swallowed so the pipeline does not break.
- **3.4 Testing**
  - `test/transforms/base/test_pipeline.py`: TransformContext, CopyStep (valid and error paths), hook register/clear/emit, hook exception isolation, strategy.apply via pipeline smoke.
  - **Transform base:** 223 passed, 1 skipped. Behaviour unchanged.

### 5.3 Summary

- **Phase 2** removes type-branching from multisource hot paths and centralises dispatch in `data_kind` + handlers.
- **Phase 3** replaces the monolithic `apply()` with a five-step pipeline and observable hooks; `apply()` is now a thin orchestrator. No change to external API or return behaviour.

---

## 6. Phase 4.1: Metadata manifest (2026-02-16)

### 6.1 Implemented

- **Manifest schema and API** (`lmetk/data/data_objects/manifest.py`)
  - `ManifestEntry`: dataclass with `schema_version`, `producer_version`, `category` (`"datasource"` | `"transform"`), `payload`, optional `step_id`, `key`, `split`. Validates category.
  - `MetadataManifest`: container with `add(entry)`, `query(step_id=..., key=..., split=..., category=...)`, `copy(deep=...)`, `__len__`, `__iter__`.
  - `MANIFEST_SCHEMA_VERSION = "1.0"`.
- **Container integration** (`lmetk/data/data_objects/data.py`)
  - `DatasetContainer`: optional `manifest=` in `__init__` (popped from kwargs); `manifest` attribute; `copy()` copies manifest via `object.__setattr__(new, "manifest", ...)` so the copy retains the manifest.
- **Pipeline recording** (`lmetk/transforms/base/pipeline.py`)
  - `RecordManifestStep`: if `context.transformed_data` has a `manifest` (and it is a `MetadataManifest`), appends a transform entry (step_id from transform name, payload with apply_to_keys, assign_to_keys, fit_on_split, fit_on_key, log_splits). Producer version from `importlib.metadata.version("lmetk")` or `"0.0.0"`.
  - `TransformStrategy.apply()` runs `RecordManifestStep` after `MergeStep`.
- **Tests** (`test/transforms/base/test_manifest.py`)
  - ManifestEntry (valid/invalid category), MetadataManifest (add, query, copy deep/shallow), container (optional manifest, copy preserves manifest, copy without manifest), RecordManifestStep (no manifest no-op, with manifest appends entry after strategy.apply).

### 6.2 Summary

- **Phase 4.1** adds an optional, versioned metadata manifest on the container; when present, each transform application appends one entry. No change to callers that do not attach a manifest. Transform base: 235 passed, 1 skipped.

---

## 7. Phase 4.2: Slice-awareness (2026-02-16)

### 7.1 Implemented

- **SliceInfo** (`lmetk/data/data_objects/slice_info.py`)
  - Dataclass: optional `split`, `unit_ids`, `cycle_ids`, `bounds`, `index_map`. Enables “what slice am I operating on?” and future coordinate bounds / original→sliced index mapping.
  - `copy(deep=...)` for use when copying the container.
- **Container integration** (`lmetk/data/data_objects/data.py`)
  - `DatasetContainer`: optional `slice_info=` in `__init__` (popped from kwargs); `slice_info` attribute; `copy()` copies slice_info via `object.__setattr__` (same pattern as manifest).
- **Pipeline / context** (`lmetk/transforms/base/pipeline.py`, `strategy.py`)
  - `TransformContext.slice_info`: optional; set from `data.slice_info` when building context so transforms/hooks can read it.
  - `RecordManifestStep`: when recording a transform entry, if `context.slice_info` is set, adds a `slice_info` summary to the payload (`split`, `n_units`, `n_cycles`).
- **Tests** (`test/transforms/base/test_slice_info.py`)
  - SliceInfo (defaults, copy deep/shallow), container (accepts slice_info, copy preserves it, copy without slice_info), context (slice_info from data, manifest payload includes slice_info when present; slice_info None when data has none).

### 7.2 Summary

- **Phase 4.2** adds optional slice-awareness: `SliceInfo` on the container, copied through pipeline copy; `TransformContext.slice_info` for reads; manifest transform entries record a slice summary when present. No change to callers that do not set `slice_info`. Transform base: 244 passed, 1 skipped.

---

## 8. Phase 4.3: Integrity guards (2026-02-16)

### 8.1 Implemented

- **IntegrityPolicy and helpers** (`lmetk/transforms/base/integrity.py`)
  - Policy: `"fail"` (re-raise), `"warn"` (warn and continue), `"allow"` (suppress).
  - `run_check(name, check_fn, policy)`: runs a no-arg check; on exception applies policy.
  - `run_checks(checks)`: runs a list of `(name, check_fn, policy)` in order; stops on first `"fail"` raise.
- **Pipeline** (`lmetk/transforms/base/pipeline.py`, `strategy.py`)
  - `TransformContext.integrity_checks`: optional list of `(name, check_fn, policy)`; default None.
  - `IntegrityCheckStep`: runs `run_checks(context.integrity_checks)` if set; no-op otherwise. Placed after CopyStep, before FitStep.
- **Tests** (`test/transforms/base/test_integrity.py`)
  - run_check (fail raises, warn warns, allow suppresses, pass no-op), run_checks (order, stop on fail, warn continues), IntegrityCheckStep (no checks no-op, with fail raises, with allow no raise).

### 8.2 Summary

- **Phase 4.3** adds optional integrity checks: policy-driven (fail/warn/allow) and injectable via `context.integrity_checks`; NaN/length checks can be registered with policy `"fail"` for mandatory-by-default. No checks registered by default; behaviour unchanged. Transform base: 253 passed, 1 skipped.

---

## 9. Phase 5.3: Inverse transform routing (2026-02-16)

### 9.1 Implemented

- **Manager API** (`lmetk/transforms/base/transform_manager.py`)
  - `get_inverter_for_key(key: str) -> Optional[InverseTransformMixin]`: returns the last transform (in pipeline order) that assigns to `key` and whose `transform_instance` implements `InverseTransformMixin`; returns `None` if no match.
- **run.py** (`lmetk/run.py`)
  - When `apply_inverse_scaling` is True, prefer `inverse_transform_key` (e.g. `"target"`): call `transforms_manager.get_inverter_for_key(inverse_transform_key)`; if a transform is found, use it. If not set or no match, fall back to `inverse_transform_name` (existing behaviour). If neither key nor name is provided, raise a clear error asking for one of them.
- **Evaluator config**
  - Can set `inverse_transform_key` to the **data key** (assign_to) where the scaled values live, e.g. `rul` when the target scaler has `assign_to: rul`; inverter is then discovered and `inverse_transform_name` is not required. Use `inverse_transform_name` when you need a specific transform (e.g. scaler_features vs final_scaler_features). See [INVERSE_TRANSFORM_ROUTING.md](INVERSE_TRANSFORM_ROUTING.md).
- **Tests**
  - `DummyInverseTransform` in conftest (implements InverseTransformMixin). `TestGetInverterForKey` in test_transform_manager.py: empty manager returns None; non-inverse transform returns None; assign_to match returns inverter; other key returns None.

### 9.2 Summary

- **Phase 5.3** adds key-based inverse lookup: `get_inverter_for_key(key)` and optional `inverse_transform_key` in evaluator config so inverse can be discovered without naming the transform. Backward compatible: `inverse_transform_name` still works. Transform base: 257 passed, 1 skipped.

---

## 10. Phase 5.2: Structural dry run (2026-02-16)

### 10.1 Implemented

- **dry_run module** (`lmetk/transforms/base/dry_run.py`)
  - `DryRunResult(transform_name, success, issues)`: one result per transform.
  - `dry_run_transforms(manager, container) -> List[DryRunResult]`: for each transform in pipeline order, checks apply_to keys in container, builds one segment from first split/unit, infers data_kind, gets capability, verifies handler exists for (data_kind, capability). No fit/transform execution.
  - Helper `_get_first_segment(container, apply_to_keys)` for segment extraction.
- **Tests** (`test/transforms/base/test_dry_run.py`): DryRunResult str, _get_first_segment (missing key, present), dry_run_transforms (empty manager, single transform success, missing apply_to key failure).

### 10.2 Summary

- **Phase 5.2** adds structural dry run: validate pipeline contracts (keys, handler routing) without running transforms. Transform base: 268 passed, 1 skipped.
