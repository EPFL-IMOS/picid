# Transform System — Architectural Audit & System Health Report

**Role:** Senior Software Architect (HPC & Deep Learning)  
**Scope:** `docs/transforms`, `docs`, `lmetk/transforms/base`, usage in `lmetk/data/preprocessing`  
**Date:** 2026-02-16

---

## Executive Summary

The transform system implements a **sequential, configuration-driven pipeline** with a **Facade (DataTransform)** and a **single, monolithic Strategy** that branches on metadata and runtime type checks. Type handling is done via **marker inheritance** (Dense/Ragged/RaggedOrDense) and **explicit `isinstance` / `get_instance_cls()`** in hot paths. The design is **partially open for new transforms** but **closed for new data types** (e.g. PointClouds) without editing core modules. Below are the technical reverse-engineering, friction audit, and refactoring recommendations.

---

## Phase 1: Technical Reverse-Engineering

### 1.1 Orchestration Analysis

**Mechanism: Sequential pipeline with a single Strategy and internal branching.**

End-to-end flow:

```
Raw input (datasource)
    → PreProcessor.pipeline()
    → fetch_data() → DatasetContainer
    → apply_transforms(data, transforms.get_transforms())
        → for each (name, DataTransform): data, log = transform.forward(data)
            → DataTransform.forward()
                → self.strategy.apply(transform_instance, data, apply_to_keys, ...)
                    → TransformStrategy.apply()
                        → [Fit phase] fit_on_split/fit_on_key → fit_data or fit_multi_source
                        → [Transform phase] per split: transform_data or transform_multi_source
                        → postprocess_transformed_data() → _merge_transformed_data()
    → validated DatasetContainer
```

- **Pipeline shape:** Strictly sequential. No DAG, no parallel branches, no recursive dispatch. Order is defined by `ConfigTransformManager` (OrderedDict) and the loop in `PreProcessor.apply_transforms`.
- **Strategy usage:** There is a single concrete `TransformStrategy` instance per `DataTransform` (created in `DataTransform.__init__`: `self.strategy = TransformStrategy()`). The docstring alludes to "DirectTransformStrategy, FitTransformStrategy, MultiSourceFitTransformStrategy" but the code uses one class that branches on:
  - `is_multi_unit` (from `data.is_multi_unit`)
  - Presence of `fit_on_split` / `fit_on_key`
  - Whether the transform instance has `fit_multi_source` / `transform_multi_source` (used inside the strategy and in multisource mixins).
- **Conclusion:** Orchestration is **sequential pipeline + single strategy with internal conditionals**, not a recursive or functional map over a tree of transforms.

### 1.2 Type Management

**Mechanism: Marker base classes + runtime type inspection and branching.**

- **Declared capabilities (markers):**
  - `BaseTransform` → `DenseTransform`, `RaggedTransform`, `RaggedOrDenseTransform` (in `base_transform.py`). Used only as inheritance markers; no methods.
- **Runtime type discovery:**
  - Data: `NamedTransformInput` / `BaseDataObjectWithMetadata` store `_instance_cls: Dict[str, Type[T]]` and expose `get_instance_cls()`. Types are inferred on `__setitem__` from allowed types: `pd.DataFrame`, `pd.Series`, `np.ndarray`, `ak.Array`.
  - Strategy: `strategy.py` uses `data[key].get_instance_cls() is not ak.Array` to decide deep vs shallow copy.
  - Multisource: `multisource.py` uses `chunk.get_instance_cls().items()/values()` and `data_types.pop() is ak.Array` to choose code paths (dense vs ragged, and ragged-but-regular).
- **Type-specific behaviour:**
  - Dense: `np.concatenate`, direct `transform_func(segment, metadata)`.
  - Ragged (transform supports ragged): `ak.concatenate`, direct `transform_func(segment, metadata)`.
  - Ragged (dense-only transform): find `find_singular_ragged_dim`, iterate `ragged_index_tuples`, convert segments to NumPy, transform, reassemble with `ListOffsetArray` / `tolist` + `ak.Array`.
- **Conversion helpers:** `lmetk/transforms/utils.py` provides `_convert_to_numpy` / `_convert_from_numpy` with explicit branches for `np.ndarray`, `pd.DataFrame`, `pd.Series`, `torch.Tensor` (including `.cpu().numpy()` and device preservation on convert-back).
- **Conclusion:** Type handling is **marker-based + centralised branching on `get_instance_cls()` and `isinstance(..., ak.Array)`**. There is no registry, no `functools.singledispatch`, and no visitor-style double dispatch. Adding a new backend type (e.g. PointClouds) requires touching `multisource.py`, `strategy.py`, and possibly `base_transform.py` and `utils.py`.

### 1.3 State & Metadata

**State:**

- **Fit state:** Held on the transform instance (e.g. `MinMaxScalerSklearn.scaler`, or custom `self.mean_`, `self.std_`). No global transform state store.
- **Random seeds / interpolation:** Not managed by the framework. Individual transforms may use `np.random` or similar; there is no standardised RNG or interpolation API in the base.
- **Copy behaviour:** Strategy decides per key: deep copy for non–`ak.Array`, shallow for `ak.Array`, based on `get_instance_cls()`.

**Metadata (per call):**

- Passed through `metadata: Dict[str, Any]` into `transform_data` / `fit_data` / `transform_multi_source` / `fit_multi_source`. Typical keys: `mode`, `apply_to_keys`, `assign_to_keys`, `assign_to_map`. Originates from `DataTransform` (from config: `apply_to`, `assign_to`, `fit_on`, `fit_on_key`, `transform_on_keys`, `multi_source_fit`).
- No versioning or schema for metadata; validation is ad hoc (e.g. `assign_to_map` in decorator `check_transform_output_consistency`).

**Conclusion:** State is **instance-local**; metadata is **dictionary-based and unversioned**. Lifecycle of state is the lifecycle of the transform instance (and manager/cache if used).

---

## Phase 2: The "SWE Friction" Audit

### 2.1 Coupling & Cohesion

| Aspect | Assessment |
|--------|------------|
| **Math vs routing** | **Tightly coupled.** "How to transform" lives in `BaseTransform.transform_data` (and mixins), while "where to route" lives in `TransformStrategy.apply` and `ConcatinatedFitMultiSourceTransformMixin.fit_function_to_data_segments` / `transform_function_to_data_segments`. Routing depends on marker *classes* (`issubclass(cls, RaggedTransform)` etc.) and on data types inferred at runtime. Adding a new path (e.g. "ragged but stride-based") forces edits in the same large methods. |
| **Cohesion** | Strategy does: copy, fit dispatch, per-split iteration, single vs multi-source dispatch, postprocess, merge. Multisource mixin does: type detection, concatenation, per-segment transform, and ragged reassembly. Both mix "what to do" with "how to dispatch." Cohesion is moderate within each class but low when considering "all routing and data-shape logic" as one concern. |

**Code smells:**

- Long methods: `transform_function_to_data_segments` (and fit analogue) are large and branch-heavy.
- Repeated type checks: `transform_supports_ragged = issubclass(cls, RaggedTransform) or issubclass(cls, RaggedOrDenseTransform)` and similar in multiple places.
- Duplicated "data_is_ragged" and "ragged_but_regular" logic in fit and transform paths.

### 2.2 Extensibility (Open/Closed Principle)

| Extension | Open/Closed? | Notes |
|-----------|--------------|-------|
| **New transform (same types)** | **Open.** User subclasses `BaseTransform` (+ marker + optional mixins), implements `transform_data` (and optionally `fit_data`, `fit_multi_source`, `transform_multi_source`). No change to base engine. |
| **New data type (e.g. PointClouds)** | **Closed.** Would require: (1) extending `BaseDataObject` allowed types and `get_instance_cls()`; (2) new marker or capability in `base_transform.py`; (3) new branches in `TransformStrategy.apply` (copy logic); (4) new branches in `ConcatinatedFitMultiSourceTransformMixin.fit_function_to_data_segments` and `transform_function_to_data_segments` (concatenate, iterate, reassemble); (5) possibly new postprocess/merge behaviour. |
| **New "strategy" (e.g. different fit/apply order)** | **Closed.** Only one `TransformStrategy` is used; there is no pluggable strategy interface. To add a new strategy you would subclass or edit `TransformStrategy` and then change `DataTransform` to accept/instantiate it. |
| **New backend (e.g. GPU-only transform)** | **Partially open.** Transform can use GPU internally. But `_convert_to_numpy` / `_convert_from_numpy` and the rest of the pipeline assume CPU-friendly arrays for routing/copying; device placement is not part of the abstraction. |

### 2.3 Performance Hot-Paths

| Location | Issue |
|----------|-------|
| **Type checks in hot path** | `get_instance_cls()` and `issubclass(cls, RaggedTransform)` (and similar) are used inside per-split and per-segment loops. Not expensive per se but repeated and scattered. |
| **Ragged → Dense fallback** | When data is ragged and the transform is dense-only, the code iterates over `ragged_index_tuples` in Python, converts each subsegment to NumPy, calls `transform_func`, then reassembles. No vectorization; many small allocations and Python loop overhead. Docstring and logs warn that this "might be slow for large datasets." |
| **Copy strategy** | Per-key copy (deep for non-ak, shallow for ak) is done once per transform in `apply()`. Reasonable, but the condition `data[key].get_instance_cls() is not ak.Array` is a form of type dispatch that could be centralized. |
| **postprocess_transformed_data** | Called per split; does sequence/dict checks and wraps in `SimpleReturnObject` / `NamedDictReturnObject`. No obvious redundant copies but adds indirection. |
| **Signature validation** | Done once at `DataTransform.__init__` via `inspect.signature`; not a runtime hot path. |

### 2.4 Abstraction Leaks

| Leak | Where | Comment |
|------|-------|---------|
| **Awkward vs NumPy in strategy** | `strategy.py`: `if data[key].get_instance_cls() is not ak.Array: ... deep=True else ... deep=False`. High-level "apply transform" logic depends on a specific array backend. |
| **NumPy/Awkward in multisource** | `multisource.py`: Explicit `np.concatenate`, `ak.concatenate`, `ak.Array(layout)`, `ListOffsetArray`, `to_numpy()`. Backend types are visible in the orchestration layer. |
| **Torch in utils** | `utils.py`: `_convert_to_numpy` / `_convert_from_numpy` know about `torch.Tensor` and `.device`. Device handling is ad hoc; no abstract "array backend" or "device" in the base. |
| **Structure expectations** | Pipeline assumes `DatasetContainer` / `SplitDatasetContainer` with splits and optional list-of-units. Not a generic "batch of samples" abstraction; structure is part of the contract. |

---

## Phase 3: Refactoring Recommendations

### 3.1 Target State Overview

- **Dispatching:** Registry or single-dispatch for (transform capability, data type) → handler; avoid scattering `issubclass` and `get_instance_cls()` in long methods.
- **Composition:** Replace the single monolithic strategy with composable steps (e.g. fit step, transform step, merge step) or injectable handlers; keep "math" in transforms and "routing" in a small dispatcher.
- **Interface safety:** Use Protocols and typed metadata so that navigation logic relies on capabilities and input/output types rather than concrete base classes.

### 3.2 Dispatching: Registry / Single-Dispatch

**Current:** Branching on marker classes and `get_instance_cls()` inside Strategy and multisource mixins.

**Proposed:**

- Introduce a **dispatch key** (e.g. `(data_kind, transform_capability)`) where:
  - `data_kind`: `"dense"` | `"ragged"` | `"ragged_regular"` (and later e.g. `"pointcloud"`).
  - `transform_capability`: `"dense"` | `"ragged"` | `"both"` (derived from marker or explicit protocol).
- Implement a **registry** of handlers (functions or small objects) that know how to:
  - prepare chunks for fit (e.g. concatenate dense, concatenate ragged, or flatten ragged to dense),
  - prepare chunks for transform (per segment or concatenated),
  - merge results back into the container.
- Keep `TransformStrategy.apply` thin: compute `data_kind` and `transform_capability` once, look up handler, call it. No long if/elif chains.

**UML-style:**

```
«interface»
TransformHandler
  + fit_prepare(segments, metadata) -> prepared
  + transform_apply(segments, transform_instance, metadata) -> results
  + merge(base_data, results, metadata) -> void

DenseDenseHandler : TransformHandler
DenseRaggedHandler : TransformHandler   (ragged data + dense-only transform)
RaggedRaggedHandler : TransformHandler

TransformStrategy
  - registry: Dict[(DataKind, TransformCapability), TransformHandler]
  + apply(...)  -> (data, log)
      1. data_kind = infer_data_kind(data)
      2. capability = get_capability(transform_instance)
      3. handler = registry[(data_kind, capability)]
      4. return handler.run(...)
```

**Alternative:** Use `functools.singledispatch` (or `singledispatchmethod`) on a small set of types (e.g. a wrapper type that carries `DataKind`) so that adding a new data kind is "implement a new overload" rather than editing a central if/elif.

### 3.3 Composition: Strategy as Composable Steps

**Current:** One `TransformStrategy` with a long `apply()` that does copy → fit → loop over splits → transform → postprocess → merge.

**Proposed:**

- Split into **small, composable steps** (or a small pipeline of handlers):
  1. **CopyStep:** Clone container (current hybrid copy logic), optionally delegated to a copy policy per data kind.
  2. **FitStep:** Given `fit_on_split` / `fit_on_key`, prepare chunks and call either `fit_data` or `fit_multi_source` (dispatch via registry or capability).
  3. **TransformStep:** Per split, prepare chunks and call `transform_data` or `transform_multi_source`; dispatch as above.
  4. **PostprocessStep:** Current `postprocess_transformed_data` (or a small set of normalisers).
  5. **MergeStep:** Current `_merge_transformed_data`.
- `TransformStrategy` (or a new "PipelineStrategy") composes these steps and holds the dispatch registry. Each step can be tested and overridden independently.

**UML-style:**

```
«interface»
PipelineStep
  + run(context: TransformContext) -> TransformContext

CopyStep : PipelineStep
FitStep : PipelineStep      // uses TransformHandler for fit_prepare + fit call
TransformStep : PipelineStep // uses TransformHandler for transform_apply
PostprocessStep : PipelineStep
MergeStep : PipelineStep

TransformContext
  - data: SplitDatasetContainer
  - transform_instance: BaseTransform
  - metadata: dict
  - log: dict

CompositeTransformStrategy
  - steps: List[PipelineStep]
  + apply(...) -> (data, log)
      context = TransformContext(...)
      for step in steps: context = step.run(context)
      return context.data, context.log
```

This keeps "where to route" in the steps and registry, and "how to transform" in `BaseTransform` and handlers.

### 3.4 Interface Safety: Protocols and Typed Metadata

**Current:** `BaseTransform` is an ABC; capabilities are expressed by inheriting marker classes. Metadata is `Dict[str, Any]`; signature validation is ad hoc.

**Proposed:**

- **Capability protocol** (instead of or in addition to marker inheritance):

```python
class SupportsDense(Protocol):
    def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any: ...

class SupportsRagged(Protocol):
    def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any: ...
# Optional: SupportsMultiSourceFit, SupportsInverse, etc.
```

- **Typed metadata:** Introduce a `TransformMetadata` TypedDict or a small dataclass (e.g. `mode`, `apply_to_keys`, `assign_to_keys`, `assign_to_map`) and pass it through the pipeline so that handlers and transforms do not rely on string keys and ad hoc checks.
- **Input/Output protocols:** Define `TransformInput` and `TransformOutput` protocols (or reuse `NamedTransformInput` and return types) so that the dispatcher and postprocess step depend on interfaces, not concrete `NamedTransformInput` / `SimpleReturnObject`.
- **Registration:** When a transform is registered (or instantiated), it can declare capabilities (e.g. `[SupportsDense, SupportsMultiSourceFit]`) so that the dispatch registry can be built or validated without relying on `issubclass(cls, RaggedTransform)`.

**UML-style:**

```
«protocol»
TransformInput
  + get_instance_cls() -> Dict[str, Type]

«protocol»
TransformOutput
  + keys() -> Iterable[str]
  + __getitem__(key) -> Any

«protocol»
SupportsDense
  + transform_data(data: TransformInput, metadata: TransformMetadata) -> TransformOutput

«protocol»
SupportsRagged
  + transform_data(data: TransformInput, metadata: TransformMetadata) -> TransformOutput

TransformMetadata (TypedDict or dataclass)
  mode: str
  apply_to_keys: List[str]
  assign_to_keys: List[str]
  assign_to_map: List[str]
```

This makes the "navigation" logic depend on capabilities and typed metadata rather than concrete base classes and string keys.

---

## Summary Tables

### Current Implementation

| Dimension | Current |
|-----------|---------|
| **Orchestration** | Sequential pipeline; single Strategy with internal branching. |
| **Type dispatch** | Marker classes (Dense/Ragged/RaggedOrDense) + `get_instance_cls()` + `isinstance(ak.Array)` in Strategy and multisource. |
| **State** | Instance-local fit state; unversioned dict metadata. |
| **Extensibility** | New transforms: open. New data types / new strategy: closed. |
| **Performance** | Type checks in loops; ragged→dense path is Python-loop heavy. |
| **Abstraction** | NumPy/Awkward/Torch and container structure leak into orchestration. |

### Architectural Flaws

| Flaw | Severity | Location |
|------|----------|----------|
| Routing and math coupled in long methods | High | `strategy.py`, `multisource.py` |
| New data types require editing core | High | `multisource.py`, `strategy.py`, `base_transform.py` |
| Single hardcoded strategy | Medium | `data_transform.py`, `strategy.py` |
| Repeated type/capability checks | Medium | `multisource.py`, `strategy.py` |
| No typed metadata | Medium | All call sites |
| Backend-specific logic in orchestration | Medium | `strategy.py`, `multisource.py`, `utils.py` |
| Ragged→dense performance | Medium | `multisource.transform_function_to_data_segments` |

### Proposed Refactoring Strategy

| Area | Change |
|------|--------|
| **Dispatching** | Registry or single-dispatch keyed by (data_kind, transform_capability); handlers for fit/transform/merge. |
| **Composition** | Replace monolithic `apply()` with composable pipeline steps (Copy, Fit, Transform, Postprocess, Merge). |
| **Interfaces** | Protocols for SupportsDense/SupportsRagged, TypedDict/dataclass for metadata, TransformInput/TransformOutput. |
| **Extensibility** | New data type = new handler + optional new capability; new strategy = new step composition. |

---

## Refactoring Impact on Test Suite

**Scope:** `test/transforms/base/` (116 tests, all passing as of this audit).

### Test Layout vs. Current Architecture

| Test file | What it exercises | Coupling to current design |
|-----------|-------------------|----------------------------|
| `test_base_transform.py` | `BaseTransform`, markers (Dense/Ragged/RaggedOrDense), `__call__`, `fit_data`, `__repr__` | **High** — direct inheritance and marker classes. |
| `test_data_transform.py` | `DataTransform` init, metadata processing, `forward()`, validation, signature checks, **mock strategy** for error paths | **High** — `DataTransform` API, single strategy, validation logic. |
| `test_strategy.py` | `TransformStrategy.apply`, `_prepare_chunks_for_split`, `_merge_transformed_data`, `postprocess_transformed_data`, copy behaviour (ak vs numpy) | **Very high** — strategy is the main orchestration unit. |
| `test_multisource.py` | `find_singular_ragged_dim`, `tolist`, mixins (`ConcatinatedFitMultiSourceTransformMixin`, `NoFitMultiSourceTransformMixin`), fit/transform multi-source (dense, ragged→dense, ragged native) | **Very high** — type branches and mixin behaviour. |
| `test_transform_manager.py` | `ConfigTransformManager` (config, lazy/eager, add/remove/clear, skip `__` keys) | **Low** — manager is mostly independent of dispatch. |
| `conftest.py` | Dummy transforms (Dense/Ragged/RaggedOrDense + mixins), container factories | **High** — mirrors current marker + mixin design. |

### Impact of Proposed Refactoring

#### 1. Registry-based / handler-based dispatching

| Change | Test impact |
|--------|-------------|
| Replace internal type branches with a **registry** keyed by `(data_kind, transform_capability)` and handler objects. | **Strategy tests:** Tests that assert current behaviour (e.g. copy deep vs shallow by `ak.Array`, fit-on-train, per-split apply) remain valid **if** the public contract of `TransformStrategy.apply` (inputs/outputs, split structure) is unchanged. Tests that rely on **internal** branching (e.g. that a specific code path runs for "ragged + dense-only transform") would need to target the **handler** for that (data_kind, capability) instead of the strategy. |
| Add new handlers (e.g. `DenseDenseHandler`, `RaggedRaggedHandler`). | **New tests:** Add tests per handler (fit_prepare, transform_apply, merge) and optionally integration tests that the registry returns the correct handler for a given data + transform. **Existing tests:** Integration tests (e.g. `test_strategy_apply_*`, `test_data_transform_forward_*`) should still pass if the composed behaviour is equivalent. |

**Recommendation:** Keep current strategy tests as **integration tests** for the full pipeline. Introduce **handler-level unit tests** as you add the registry; avoid asserting on the strategy's internal if/elif structure.

#### 2. Composable pipeline steps (Copy, Fit, Transform, Postprocess, Merge)

| Change | Test impact |
|--------|-------------|
| Replace monolithic `apply()` with a pipeline of steps (e.g. `CopyStep`, `FitStep`, `TransformStep`, …). | **Strategy tests:** If the pipeline is hidden behind the same `apply(transform_instance, data, ...)` API, existing tests can stay as-is. If you expose steps (e.g. for reuse or testing), add **unit tests per step** (e.g. `CopyStep` preserves structure, `MergeStep` writes to the right keys). |
| Introduce `TransformContext` (or similar) passed between steps. | Tests that **mock the strategy** (e.g. in `test_data_transform.py`: `test_forward_missing_assign_to_key_error`, `test_forward_not_split_container_error`) will still work as long as the facade still calls a single entry point that returns `(data, log)`. If the facade starts calling steps directly, mocks may need to target the step interface or a new "pipeline runner." |

**Recommendation:** Preserve the current `apply(...)` signature and return shape so that all existing `DataTransform` and strategy integration tests remain valid. Add new tests for each step and for the pipeline runner.

#### 3. Protocols and typed metadata

| Change | Test impact |
|--------|-------------|
| Replace marker **inheritance** with **Protocols** (e.g. `SupportsDense`, `SupportsRagged`) or explicit capability flags. | **Base + conftest:** Tests that use `isinstance(transform, DenseTransform)` (e.g. `test_dense_transform_marker`, `test_ragged_transform_marker`) will need to be updated to check the new capability model (e.g. protocol or attribute). **Conftest** dummy transforms should implement the new protocols or set the new flags. |
| Introduce **TypedDict or dataclass** for metadata instead of `Dict[str, Any]`. | **DataTransform and strategy tests:** Most tests pass `metadata` as dicts (e.g. `{"apply_to": "features", "assign_to": "features"}`). If the public API continues to accept dicts and only internally converts to a typed object, tests need no change. If the API switches to requiring a typed object, all call sites in tests must be updated. |

**Recommendation:** Prefer a **gradual** move: keep accepting dicts at the boundary and add a small adapter to a typed structure inside the pipeline. Then tests stay valid and you can add tests for the adapter. Update marker-based tests only when you actually remove the marker classes.

#### 4. Summary: what stays vs. what to touch

| Area | Likely unchanged | Likely to change |
|------|------------------|------------------|
| **ConfigTransformManager** | All manager tests (config, lazy, add/remove, skip `__`) | None if manager only builds `DataTransform` from config. |
| **DataTransform (facade)** | Init, metadata processing, `forward()` signature and return type, validation (splits, keys, types) | If strategy becomes pluggable or steps are exposed, tests for strategy selection or step calls. |
| **TransformStrategy** | Integration tests that call `apply(...)` and assert on final container shape and content | Tests that assert on internal branches or private helpers; these move to handler/step tests. |
| **Multisource mixins** | Tests that call `fit_multi_source` / `transform_multi_source` and assert on results | Tests that assert on "dense vs ragged" or "ragged_but_regular" branches; these move to handler tests. |
| **BaseTransform / markers** | Tests for `transform_data`, `fit_data`, `__call__`, `__repr__` | Tests that rely on `isinstance(..., DenseTransform)`; switch to protocol/capability. |
| **conftest** | Container factories, basic dummy transform behaviour | Dummy transforms: add protocol impl or capability flags when markers are replaced. |

### Test Health and Gaps

- **All 116 tests in `test/transforms/base/` pass** with the current implementation.
- **Fixed:** `test_multisource.py::TestMultisourceUtils::test_find_singular_ragged_dim_multiple_error` now asserts `ValueError` for an input with two variable dimensions (single array of type `var * var * float64`).
- **Mock usage:** Several tests in `test_data_transform.py` replace `dt.strategy` with a mock to force error paths (e.g. missing key, wrong type, split mismatch). This is appropriate for testing the facade's validation. After refactoring, keep one "strategy" entry point that returns `(data, log)` so these mocks remain simple.
- **Coverage:** The suite already stresses metadata handling, single vs multi-unit, fit-on-train, assign_to/apply_to, and postprocess/merge. Refactoring should preserve these scenarios and add tests for new units (handlers, steps, registry).

### Suggested test strategy during refactor

1. **Keep** all existing tests running as integration tests; fix or remove only those that assert on removed internals.
2. **Add** handler tests when introducing the dispatch registry; add step tests when introducing the pipeline.
3. **Migrate** marker-based assertions to protocol/capability checks when removing marker classes.
4. **Retain** strategy mocks in `DataTransform` tests by keeping a single delegate (e.g. "pipeline runner") that returns `(data, log)`.

---

## Appendix: File Reference

| File | Role |
|------|------|
| `lmetk/data/preprocessing/preprocessor.py` | Entry point; calls `apply_transforms(data, transforms.get_transforms())`; iterates over DataTransforms. |
| `lmetk/transforms/base/transform_manager.py` | Builds OrderedDict of DataTransforms from config (Hydra). |
| `lmetk/transforms/base/data_transform.py` | Facade; metadata processing; single `TransformStrategy()`; forwards to strategy and validates output. |
| `lmetk/transforms/base/strategy.py` | Single strategy; copy, fit, per-split transform, postprocess, merge. |
| `lmetk/transforms/base/multisource.py` | Mixins for multi-source fit/transform; type-based branches for dense/ragged and reassembly. |
| `lmetk/transforms/base/base_transform.py` | ABC BaseTransform; markers DenseTransform, RaggedTransform, RaggedOrDenseTransform. |
| `lmetk/data/data_objects/data.py` | NamedTransformInput, DatasetContainer, SplitDatasetContainer, return types. |
| `lmetk/data/data_objects/base.py` | BaseDataObject; `get_instance_cls()`; allowed types. |
| `lmetk/transforms/utils.py` | _convert_to_numpy / _convert_from_numpy (NumPy, Pandas, Torch). |
| `lmetk/utils/decorators.py` | check_transform_output_consistency; inject_transform_context_to_strategy_apply. |
| `docs/transforms.md`, `docs/transforms/developer_guide.md` | High-level and developer documentation. |

---

## Refined Wish List: Strategic Refactoring — Transforms & Orchestration Logic

### 1) Observability & Lifecycle Hooks

**Motivation**

Transforms currently operate on units and cycles independently, but the pipeline lacks a formal lifecycle that enables "observation without interference." We want predictable lifecycle events that support debugging, telemetry, profiling, and inspection without affecting pipeline outputs.

**Requirements**

**1.1 Standard Lifecycle Events** The orchestration layer (Dispatcher/Scene) must define and emit consistent lifecycle events:

- `on_pipeline_start`
- `before_transform`
- `after_transform`
- `on_transform_error`
- `on_pipeline_end`

These must fire consistently across:

- dense/ragged data
- single- vs multi-source paths
- fit vs transform phases
- cache hit/miss paths

**1.2 end_of_transform_hook (Observer Pattern)** Goal: Enable users to execute side-effect-free observer logic (plotting, telemetry, debugging) at defined points in the pipeline.

**Distinction:** A Hook is read-only. Unlike a Transform, it:

- has no output
- cannot modify the DatasetContainer
- cannot affect downstream execution

**Standard:** Use the Observer Pattern. The pipeline maintains a registry/list of hook functions that are triggered after a transform's `forward()` pass. Hooks receive:

- a read-only view of the container
- a StepContext (step id, transform identity/version, scope, routing path, timing, cache status)

Hooks must also support:

- `on_transform_error` to emit diagnostics without mutating state

### 2) Advanced Metadata & Data Handling

**2.1 Metadata Manifest (Upstream + Transform-Generated)**

We must support two categories of metadata:

**Datasource Metadata (Upstream)**

- asset/sensor identifiers (when applicable)
- units, sampling assumptions, acquisition context
- coordinate scope (unit/cycle/time bounds)
- provenance (source system, ingest time)

**Transform Metadata (Generated During Execution)**

- fit statistics / normalization params
- feature definitions (window specs, STFT parameters, etc.)
- transform state required for inverse operations
- validator outcomes, invariant violations, warnings

**Requirement:** Maintain a Metadata Manifest attached to the container and queryable by:

- transform step id/name
- key (features / targets / context)
- split + unit/cycle/time slice scope

**Design constraints**

- metadata must be versioned (schema_version, producer_version)
- metadata must be attachable at multiple granularities: global / per split / per key / per unit / per cycle
- metadata must survive slicing, caching restore, merging, and postprocessing

**2.2 Indexing & Slicing (Slice-Aware Orchestration)**

We prefer Xarray-style coordinates to define units and cycles. The orchestration must be slice-aware, ensuring that slicing across time indices, samples, or features does not lose metadata or break coordinate alignment.

**Requirements**

- slice operations must carry: coordinate selections (unit ids, cycle ids, timestamps), resulting bounds, and mapping from original → sliced indices (when applicable).
- downstream transforms must be able to query: "what slice am I operating on?" and "what coordinate system is this input under?"
- coordinate integrity must be preserved under: split-level operations, unit/cycle selection, and ragged segmentation / awkward-array scenarios.

**2.3 Integrity Guards (Mandatory vs Optional, Policy-Driven)**

**Motivation** We need correctness checks to be safe by default, but also support controlled experiments (e.g., MissingData injection) without disabling safety globally.

**Requirements**

**2.3.1 Checks are Separated into Invariants and Validators** Introduce an injectable check chain between transforms:

- **Invariants (mandatory checks):** always executed; by default, pipeline fails if violated.
- **Validators (optional checks):** always executed when enabled; may warn or annotate, typically not failing execution.

Both types operate on read-only views and record outcomes into the Metadata Manifest and Execution Report (profiling + audit).

**2.3.2 NaN Checking Remains Mandatory by Default** NaN detection is a mandatory invariant today and must remain so by default (safe baseline).

**2.3.3 Policy-Driven Enforcement (Scoped Overrides / Expectations)** Mandatory checks must not be "hard-coded unchangeable." Instead, enforcement is controlled by an IntegrityPolicy that supports scoped overrides.

Key principle: Mandatory invariants always run and always produce results, but policy controls whether a violation fails, warns, or is allowed — in a scoped and auditable way.

**2.3.4 Per-Transform Validation Policy via Config** Metadata Each transform may specify a `validation_policy` under its `metadata:` block.

Example of config shape (conceptual):

- `metadata.validation_policy.expectations`: declare what violations are expected/allowed after this step.
- `metadata.validation_policy.overrides`: downgrade enforcement for a specific check within a scope.

Scopes must support at least keys and splits. Optionally: units/cycles/time slices.

**2.3.5 Guardrails (Safety Properties)** * default is ENFORCE

- overrides must be scoped (not global)
- overrides must include a human-readable reason
- overrides must be logged in StepRecord + Execution Report
- fail closed on invalid/unknown check IDs

### 3) Tiered Caching Integration

**Motivation**

The preprocessing loop should support cache boundaries to avoid recomputing heavy operations unnecessarily. Caching must be tiered because caching after every transform is often prohibitively expensive.

**Existing Entry Point**

A ConfigTransformManager already maps config → ordered transform roadmap and is a natural entry point for cache integration because it is where step ordering, metadata, and flags can be interpreted consistently.

**Default Behavior (important)**

If the user does not specify cache boundaries, caching occurs after the entire transform chain completes (cache after the final transform only). This default minimizes memory/storage overhead.

**Requirement: Cache Boundaries (cache_point)**

Users can mark a transform as a `cache_point`. A cache point indicates that:

- the container state after that transform is eligible for caching
- future runs may resume from the latest valid boundary

**Primary Use Case** Pipeline: T1 → T2 → T3 → T4

T3 is heavy and stable; T4 is fast but frequently varied in parameters.

Desired behavior: mark T3 as cache_point. On future runs, reuse cached state after T3 and execute only T4.

**Dispatcher / Orchestration Behavior**

The dispatcher must:

- read cache boundaries from the roadmap (via ConfigTransformManager).
- before execution: find the latest boundary with a valid cache entry, restore cached container state, and skip upstream transforms.
- resume execution from the next transform after the restored boundary.
- record cache behavior in profiling: cache hit/miss per step, restore time, and estimated time saved.

**Cache Keys / Correctness**

Cache keys must be computed per boundary and include:

- pipeline version/hash
- transform versions + parameters hashes up to boundary
- input fingerprint + coordinate selection (splits/units/cycles/time window)
- (optional) upstream datasource provenance hash

### 4) Architectural Enhancements (SWE Best Practices)

**A) Dry Run Mode (Structural Verification)**

**Motivation** A strict "shape proof" dry run is expensive and impractical in PHM because schema can be data-dependent (2D vs 3D, variable STFT frames, ragged segment lengths, awkward-array variability).

**Requirement: Structural Verification Dry Run (Shallow Schema)** Dry run must validate orchestration correctness without executing heavy math and without deep shape inference.

What it verifies:

- input contract satisfaction at each step (lane-aware)
- assignment correctness (apply_to / assign_to / maps)
- routing feasibility (dense/ragged/multisource handler availability)
- slice-awareness and coordinate preservation
- validator/invariant injection coverage
- cache boundary plan coherence (resume feasibility)

Non-goals: exact tensor shapes are not proven; deep awkward typing is not required. Only shallow traits (dense/ragged/unknown, coarse rank hints) are tracked.

**B) Intelligent Inverse Transform Routing**

**Current Issue** Inverse routing logic is overly manual and tightly coupled to preprocessing configuration.

**Requirement** Shift responsibility for discovering inverse transforms into the TransformManager.

**Strategy: Service-Oriented Registry** Evaluators request an inverter by key:

```
inverter = transforms_manager.get_inverter_for_key("target_variable")
if inverter:
    original_scale = inverter.inverse(scaled_data)
```

**Mechanism** Any transform implementing InverseTransformMixin registers itself with the manager as the provider of inverse operations for its target keys, removing inverse_transform_name requirements from evaluator config.

**C) Unified Performance Profiling (Dispatcher-Owned)**

**Observation** Execution timing exists today but is fragmented. Since every transform is executed through orchestration, profiling must be centralized in the Dispatcher/Scene.

**Requirement: Step Records (mandatory)** For every transform execution (including cached/skipped), emit a StepRecord capturing identity, scope, timing breakdown, shallow data fingerprints, cache info, validation outcomes, and status/error codes.

**Requirement: Execution Report (mandatory)** At end of preprocessing, generate a standard report that summarizes total time and per-step costs, highlights bottlenecks, and recommends cache points based on observed bottlenecks.

**D) Validation-First Execution (Atomic + Predictable)**

**Motivation** Given the complexity of lazy execution/graph optimization, we prefer atomic execution with explicit validation.

**Requirement** Treat each transform as an atomic transaction:

- run invariants/validators before transform
- execute transform
- run invariants/validators after transform

Failures must be deterministic, localized, and recorded in the Metadata Manifest and Execution Report.

### Non-Functional Requirements (Global)

- No actor changes required for profiling, routing logs, lifecycle hooks, caching decisions, or validation policy enforcement.
- Structured outputs only (JSON records / structured events).
- Stable identifiers: run_id, step_id, and artifact_id.
- Version everything: pipeline, transforms, schema/manifest, and validation policies.
- Low overhead by default: dry run avoids deep shape inference.
- Adapter-ready lineage: logs must reconstruct: what ran, when, with what scope/config, producing what artifacts.
