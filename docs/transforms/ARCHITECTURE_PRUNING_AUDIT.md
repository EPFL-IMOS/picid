# Transform Base — Architectural Pruning Audit

**Role:** Pragmatic Systems Architect / Code Minimalist  
**Focus:** Complexity debt and "gold-plating" in the refactored `lmetk/transforms/base`.  
**Constraint:** Do not revert to monolithic if/else; aim for a decoupled but boring, predictable middle path.

---

## 1. The "YAGNI" (You Ain't Gonna Need It) Test

### 1.1 Unused or Single-Use Abstractions

| Item | Status | Evidence |
|------|--------|----------|
| **`before_transform` / `after_transform` hooks** | **Unused** | Registered in `_HOOK_REGISTRY` in `pipeline.py` but **never emitted**. Only `on_pipeline_start`, `on_transform_error`, `on_pipeline_end` are called from `strategy.apply()`. |
| **`context.integrity_checks`** | **Unused on default path** | `TransformContext` is built only in `TransformStrategy.apply()` and never sets `integrity_checks` (defaults to `None`). So `IntegrityCheckStep` is always a no-op unless some external code builds a context with checks. No caller in the codebase does this. |
| **`context.step_id` / `context.cache_status`** | **Optional / future** | Set nowhere in the current flow; reserved for observability. Low cost but add cognitive load when reading the context. |
| **Handler registry** | **Used** | Only used in **multi-source** path: `ConcatinatedFitMultiSourceTransformMixin.fit_function_to_data_segments` and `transform_function_to_data_segments` call `get_handler(data_kind, capability)`. Single-unit path never touches the registry. |

### 1.2 Is Registry-Based Dispatching Too "Magical"?

**Traceability from DataTransform to execution:**

- **Single-unit:**  
  `DataTransform.forward` → `strategy.apply()` → **FitStep** calls `transform_instance.fit_data(chunks[0], metadata)` or **TransformStep** calls `transform_instance.transform_data(chunks[0], metadata)`.  
  **No registry.** Path is: Facade → Strategy → Pipeline → Step → Transform instance. Easy to follow.

- **Multi-unit:**  
  `DataTransform.forward` → `strategy.apply()` → **FitStep** calls `transform_instance.fit_multi_source(fit_chunks, metadata)` → mixin's `fit_function_to_data_segments` → `infer_data_kind()` + `get_capability(self)` → `get_handler(data_kind, capability)` → `handler.fit_prepare(...)`.  
  So the chain is: **Facade → Strategy → Step → Transform (mixin) → data_kind/capability → registry → handler**. That’s 4–5 hops; the “magic” is that the **pipeline doesn’t know about the registry**—the transform’s mixin does. A developer debugging “why did this transform use dense path?” must know: (1) strategy runs steps, (2) FitStep/TransformStep call the transform’s multi_source methods, (3) the mixin uses data_kind × capability and the handler registry.  

**Verdict:** Traceability is acceptable but not trivial. One improvement: document in code or docstring that “multi-source behavior is dispatched via (data_kind, capability) in the handler registry, used inside the multi-source mixin.”

---

## 2. Cognitive Load & Traceability

### 2.1 Indirection-to-Value Ratio

| Task | Files / concepts involved | Comment |
|------|----------------------------|--------|
| **Add a new transform (user)** | 1–2: implement `BaseTransform` (+ marker if needed), config. | Good. |
| **Add a new data kind (e.g. PointClouds)** | 3+: `data_kind.py` (new kind + infer), `handlers.py` (new handler + registry), possibly `base_transform.py` (new capability marker). | Roadmap already notes this is closed; acceptable for a rare change. |
| **Wire a new pipeline step** | 2: add step class, add `Step().run(context)` in `strategy.apply()`. | Clear. |
| **Use integrity checks** | 3: define checks, pass `integrity_checks` into context, but **context is only built inside strategy**—so either strategy must accept optional checks from outside or a different entry point is needed. | Currently no supported way to inject checks without changing strategy. So the abstraction exists but is not wired to value. |
| **Use lifecycle hooks** | 2: `register_hook(event, cb)` + understand which events are emitted. | Two events are never emitted; documentation would be wrong if it listed them. |

**Verdict:** For “add a custom transform,” the ratio is good. For “add integrity checks” and “use before/after_transform,” the ratio is poor because the wiring is missing or the events don’t fire.

### 2.2 Entry Barrier for a Simple Custom Transform

A new contributor who only needs to **use** the system:

- Must understand: `DataTransform` (config + metadata), `BaseTransform` (+ marker: Dense/Ragged/RaggedOrDense), `transform_data` (and optionally `fit_data`). No need to touch registry, steps, or context.
- For **multi-source**: must know their transform will go through a mixin that uses data_kind × capability and handlers. They don’t need to implement handlers.

So the **entry barrier for writing a simple custom transform is low**. The **entry barrier for debugging “which path ran?”** is higher: they need the mental model above (strategy → steps → transform → mixin → registry → handler).

---

## 3. Abstraction Leaks

Orchestration was **not** fully decoupled from concrete data types:

| Location | Leak | Notes |
|----------|------|--------|
| **`pipeline.CopyStep`** | `data[key].get_instance_cls() is not ak.Array` | Direct type check for copy depth. |
| **`data_kind.infer_data_kind`** | `next(iter(data_types)) is ak.Array` | Ragged = awkward. |
| **`handlers.DenseDenseHandler`** | `ak.Array in cls_map.values()`, `ak.Array(val)`, `segment_dict[key].to_numpy()` | Knows Awkward and numpy. |
| **`handlers.RaggedDenseHandler`** | `ak.concatenate`, `to_numpy()`, `ak.Array(layout)`, `ListOffsetArray` | Heavy Awkward usage. |
| **`handlers.RaggedRaggedHandler`** | `ak.concatenate` | Awkward. |
| **`multisource`** | `get_instance_cls()`, `ak.Array` in error logging | Type checks and logging. |

So: **data kind and copy policy are still tied to `ak.Array` and `get_instance_cls()`**. The refactor moved **branching** into handlers and registry, but **type knowledge** stayed in pipeline, data_kind, and handlers. Adding a new backend (e.g. `torch.Tensor` or point clouds) would still require editing these modules. This is an intentional leak for now; the audit already calls out “new data type requires core edits.”

---

## 4. Performance Overhead

- **TransformContext:** One dataclass per `apply()`; fields are references or small collections. No large allocations.
- **Steps:** CopyStep does container copy (existing); others mostly read/write context. No extra heavy work.
- **RecordManifestStep:** Single `manifest.add(entry)` when manifest exists; negligible.
- **Hooks:** `_emit` iterates over a small list of callbacks; no-op if no hooks registered.
- **Registry:** Built lazily on first `get_handler()`; thereafter dict lookup. Handler instances are reused (same handler instance per (data_kind, capability)).

No significant latency or memory pressure was identified from the new observability or metadata layers. The main cost remains in the actual fit/transform and data movement (e.g. RaggedDenseHandler’s per-segment loop).

---

## 5. Simplicity Score Card

**Scale: 1 = Lean, 5 = Overengineered**

| Module / concept | Score | Rationale |
|------------------|--------|-----------|
| **Registry + data_kind** | **2** | Single lookup (data_kind × capability) → handler. Used only on multi-source path. Clear constants and one place to extend. |
| **Handlers** | **2** | Three concrete handlers; protocol is small (fit_prepare, transform_apply). Logic is where it belongs (dense vs ragged). |
| **Lifecycle hooks** | **4** | Five events, three emitted. Two slots are dead. No in-repo consumers of hooks. Overbuilt for current use. |
| **Pipeline steps** | **3** | Six steps in a fixed order; each step has a clear job. TransformContext is large; IntegrityCheckStep is no-op by default. |
| **TransformContext** | **4** | Many optional fields (step_id, cache_status, integrity_checks, slice_info, strategy, postprocess_fn). “Kitchen sink” for future features. |
| **Integrity (run_check / IntegrityCheckStep)** | **3** | run_check/run_checks are simple. IntegrityCheckStep is never fed checks on the default path, so it’s dead surface area. |
| **Manifest + RecordManifestStep** | **2** | Optional; no-op if no manifest. Adds one step and one add() when manifest exists. Lean. |
| **Strategy** | **3** | Builds context and runs pipeline; also holds `_prepare_chunks_for_split` and `_merge_transformed_data` used by steps. Coupling between strategy and steps is acceptable but visible. |

**Overall:** The base is **moderately lean** where it matters (registry, handlers, manifest). The **hooks** and **context/IntegrityCheck** surface are **more complex than current use** and are the best candidates for pruning or wiring.

---

## 6. Keep, Collapse, or Cut

### Keep (essential or clearly on the roadmap)

- **Registry + data_kind + get_capability:** Centralizes multi-source dispatch; avoids branching in the mixin.
- **Handlers (DenseDense, RaggedDense, RaggedRagged):** Encapsulate fit/transform by (data_kind, capability); keep as-is.
- **Pipeline steps (Copy, Fit, Transform, Postprocess, Merge, RecordManifest):** Composable and testable; keep.
- **TransformContext:** Keep, but consider trimming or documenting “reserved” fields.
- **IntegrityPolicy + run_check/run_checks:** Keep; they’re small and support future injection.
- **Dry run:** Keep; validates apply_to/assign_to and handler availability without running transforms.
- **Manifest + RecordManifestStep:** Keep; optional and low cost.

### Collapse (merge layers that don’t need to be separate)

- **IntegrityCheckStep:** Consider merging into **CopyStep**: after setting `transformed_data` and `available_splits`, “if context.integrity_checks: run_checks(context.integrity_checks)”. This removes a dedicated no-op step and keeps one place for “copy + optional checks.” Alternatively, keep as a separate step but document that it’s no-op unless the caller injects checks (and then add a supported injection point, e.g. strategy accepting optional `integrity_checks`).
- **Hook events:** Either **emit** `before_transform` and `after_transform` (e.g. around TransformStep) so the advertised API is real, or **collapse** the hook surface to the three events that are actually used and remove the two unused names from the registry and docs.

### Cut (remove or hide to reduce complexity without immediate functional loss)

- **`before_transform` and `after_transform`:** If the team does not plan to use them soon, remove from `_HOOK_REGISTRY` and from docstrings (or any external docs). Restore when a concrete use case appears.
- **Unused context fields:** Do not cut yet; they’re cheap. If a “reserved for observability” doc or comment is added, cognitive load is reduced without removal.

---

## 7. One-Page Developer Path: How Data Flows

### 7.1 Single-Page Picture

```
DataTransform.forward(data)
  → TransformStrategy.apply(transform_instance, data, apply_to_keys, assign_to_keys, ...)
       → Build TransformContext(data, transform_instance, apply_to_keys, ...)
       → _emit("on_pipeline_start", context)
       → CopyStep.run(context)           → copy container; set transformed_data, available_splits
       → IntegrityCheckStep.run(context) → if context.integrity_checks: run_checks(...); else no-op
       → FitStep.run(context)            → if fit_on_split/fit_on_key: prepare chunks
                                          → single: transform_instance.fit_data(chunk, metadata)
                                          → multi:  transform_instance.fit_multi_source(chunks, metadata)
                                               → mixin: infer_data_kind, get_capability, get_handler
                                               → handler.fit_prepare(segments, ..., fit_func)
       → TransformStep.run(context)      → for each split: prepare chunks
                                          → single: transform_instance.transform_data(chunk, metadata)
                                          → multi:  transform_instance.transform_multi_source(chunks, metadata)
                                               → mixin: get_handler → handler.transform_apply(...)
       → PostprocessStep.run(context)    → postprocess_fn(raw results) → transformed_results_for_new_key
       → MergeStep.run(context)          → strategy._merge_transformed_data(...)
       → RecordManifestStep.run(context) → if container.manifest: manifest.add(entry)
       → _emit("on_pipeline_end", context)
  ← return context.transformed_data, context.log
```

### 7.2 Where the Registry Lives

The **handler registry is not used by the pipeline**. It is used **inside the transform instance** when it’s a multi-source transform (mixin):

- **FitStep** / **TransformStep** call `transform_instance.fit_multi_source` / `transform_multi_source`.
- The mixin’s `fit_function_to_data_segments` / `transform_function_to_data_segments` call `infer_data_kind`, `get_capability`, `get_handler(data_kind, capability)`, then `handler.fit_prepare` / `handler.transform_apply`.

So: **Pipeline → Transform (mixin) → (data_kind, capability) → registry → handler.**

### 7.3 Can This Be Explained Simply?

Yes: one page (above) is enough for “how data flows” and “where dispatch happens.” The only non-obvious part is that **dispatch is inside the transform’s multi_source methods**, not in the strategy or steps. A short note in the module docstring or a dedicated “Transform pipeline” section in the docs would make this the single source of truth.

### 7.4 Optional Refactor for Even Simpler Explanation

If the goal is “a developer sees the pipeline and immediately sees dispatch,” one could **move handler lookup into the strategy** (or a small helper used by FitStep/TransformStep): e.g. strategy or step computes data_kind and capability and calls a “run_fit” / “run_transform” helper that uses the registry. Then the mixin would only “prepare segments” and call that helper. That would make the pipeline the single place that “knows” about data_kind and handlers, at the cost of one more coupling (strategy or steps to data_kind + registry). Current design keeps the pipeline data-agnostic and puts type-aware dispatch inside the transform; both are defensible. Recommendation: **keep current design** and document the flow (e.g. this one-pager) rather than refactor for this alone.

---

## 8. Summary

- **YAGNI:** Two lifecycle events and the default integrity-check path are unused; registry and pipeline steps are used and justified.
- **Traceability:** Acceptable; multi-source path is 4–5 hops with dispatch inside the mixin. Document that in one place.
- **Indirection-to-value:** Good for “add a transform”; poor for “add integrity checks” and “use before/after_transform” until wired or pruned.
- **Abstraction:** Orchestration still knows about `ak.Array` and `get_instance_cls()` in CopyStep, data_kind, and handlers; no regression, but no full decoupling.
- **Performance:** No concerning overhead from context, steps, manifest, or hooks.
- **Simplicity:** Registry and handlers are lean; hooks and unused context/IntegrityCheck surface are the main complexity debt.
- **Actions:** Keep registry, handlers, steps, manifest, dry run. Collapse or wire IntegrityCheckStep and before/after_transform (emit or remove). Cut unused hook names from the public surface if not planning to use them soon. Add the one-page developer path to the repo (e.g. this section or a link from `lmetk/transforms/base`).
