# Proposal: Reorganize `lmetk/transforms/base` into subfolders

**Status: Implemented.** All three phases are done; `test/transforms/base/` — 318 passed, 1 skipped. Full `test/transforms/` — 729 passed, 1 skipped, 1 pre-existing failure (autoencoder).

## 1. Current state

All modules live in a single flat directory:

| File | Lines | Responsibility |
|------|--------|----------------|
| **pipeline.py** | **919** | Context, protocol, hook registry, default integrity checks, **9 step classes** (Copy, RegisterChunkBuilder, IntegrityCheck legacy, IntegrityCheckBefore, Fit, Transform, Postprocess, Merge, IntegrityCheckAfter, RecordManifest), version helper. |
| **multisource.py** | **549** | Multi-unit/multi-source: helpers (`find_singular_ragged_dim`, `tolist`, `_concatenate_segments`), mixins (FitByConcatenation, NoFit, PerSegment, ConcatenateBefore, StandardFitTransform, StatelessPerSegment, StatelessConcatenate), InverseTransformMixin. |
| **transform_manager.py** | 485 | ConfigTransformManager, cache boundaries, get_inverter_for_key. |
| **handlers.py** | 327 | TransformHandler protocol, DenseDenseHandler, RaggedDenseHandler, RaggedRaggedHandler, registry, get_handler. |
| **data_transform.py** | 341 | DataTransform facade: metadata, apply_to/assign_to, signature validation, strategy delegation. |
| **strategy.py** | 301 | TransformStrategy, postprocess_transformed_data, _prepare_chunks_for_split, _merge_transformed_data. |
| **dry_run.py** | 188 | DryRunResult, dry_run_validation (structural validation without execution). |
| **data_kind.py** | 100 | DataKind, TransformCapability, infer_data_kind, get_capability. |
| **integrity.py** | 49 | run_check, run_checks, policy (legacy no-arg checks). |
| **hooks.py** | 54 | Default hook implementations (timing, error logging), register_defaults. |
| **base_transform.py** | 76 | BaseTransform, RaggedTransform, DenseTransform, RaggedOrDenseTransform. |

**Total:** ~3.4k lines in 11 files. The main pain points are:

- **pipeline.py** is very large and mixes context, hook registry, default checks, and many step classes.
- No clear grouping: “pipeline”, “strategy”, “multisource”, “handlers”, “manager” sit next to small shared pieces (integrity, data_kind, base_transform).
- External code imports from `lmetk.transforms.base.<module>` (e.g. `base.pipeline`, `base.strategy`); any move must preserve those import paths (e.g. via packages and `__init__.py` re-exports).

---

## 2. Proposed structure

Use **subpackages under `base/`** so that existing imports keep working:

- `from lmetk.transforms.base.pipeline import TransformContext`  
  → `base/pipeline/` becomes a package; `base/pipeline/__init__.py` re-exports the same names.
- Same idea for `strategy`, `data_transform`, `transform_manager`, `multisource`, `handlers`, etc.

Two levels of change:

1. **Subfolders only** – move each current module into a package of the same name; no split of files.
2. **Subfolders + split large files** – additionally break `pipeline.py` and optionally `multisource.py` into smaller modules by functionality.

---

### 2.1 Option A – Subfolders only (minimal churn)

Turn each logical “module” into a package and keep a single “content” file inside it. No splitting of pipeline or multisource.

```
lmetk/transforms/base/
├── __init__.py                      # Optional: re-export main public API (see 2.3)
├── base_transform/
│   ├── __init__.py                  # Re-export BaseTransform, marker classes
│   └── base_transform.py
├── data_kind/
│   ├── __init__.py
│   └── data_kind.py
├── integrity/
│   ├── __init__.py
│   └── integrity.py
├── pipeline/
│   ├── __init__.py                  # Re-export context, steps, checks, hooks
│   └── pipeline.py                  # Current pipeline.py (unchanged content)
├── strategy/
│   ├── __init__.py
│   └── strategy.py
├── data_transform/
│   ├── __init__.py
│   └── data_transform.py
├── handlers/
│   ├── __init__.py
│   └── handlers.py
├── multisource/
│   ├── __init__.py
│   └── multisource.py
├── transform_manager/
│   ├── __init__.py
│   └── transform_manager.py
├── dry_run/
│   ├── __init__.py
│   └── dry_run.py
└── hooks.py                          # Single module at base level (used by pipeline)
```

**Import path preservation:**  
- `from lmetk.transforms.base.pipeline import X` → `base/pipeline/__init__.py` imports from `base.pipeline.pipeline` and re-exports `X`.  
- So the file that is currently `base/pipeline.py` becomes `base/pipeline/pipeline.py` (or a single module name kept for clarity), and `base/pipeline/__init__.py` does `from lmetk.transforms.base.pipeline.pipeline import ...` (or from `.pipeline` if we name the file `pipeline.py` inside the package).  
- To avoid `base.pipeline.pipeline`, a common pattern is: put content in `pipeline/_steps.py` + `pipeline/_context.py` etc., and `pipeline/__init__.py` re-exports. Then no module is named “pipeline” twice. See Option B.

**Simpler variant:** keep `base/pipeline.py` as a single file and add only `base/pipeline/` as a package that re-exports:

- `base/pipeline/__init__.py` does `from lmetk.transforms.base.pipeline_impl import *` and we rename current `pipeline.py` → `pipeline_impl.py` at base level. Then imports are `from lmetk.transforms.base.pipeline import X` (package) and the real code stays in `base/pipeline_impl.py`. That avoids a nested `pipeline/pipeline.py` but adds a new name `pipeline_impl`.  

**Cleaner:** have `base/pipeline/` with a single module inside, e.g. `base/pipeline/flow.py`, and `__init__.py` re-exports from `.flow`. Then the “content” file has a different name (e.g. `flow.py`) so we never write `from base.pipeline.pipeline import`. So:

- `base/pipeline/flow.py` = current pipeline.py content  
- `base/pipeline/__init__.py` = `from .flow import *` (or explicit list)

Then existing `from lmetk.transforms.base.pipeline import TransformContext` still works. Same for other packages: e.g. `base/strategy/strategy.py` → redundant name; better `base/strategy/apply.py` or keep `strategy.py` inside `base/strategy/` and accept `from lmetk.transforms.base.strategy import TransformStrategy` (import from package, which loads `base/strategy/strategy.py`). So `base/strategy/strategy.py` is fine: the import is `from lmetk.transforms.base.strategy import ...`, not `from lmetk.transforms.base.strategy.strategy import ...`. So we can have:

- `base/pipeline/` with `pipeline.py` inside → `from lmetk.transforms.base.pipeline import X` loads `base/pipeline/__init__.py` which imports from `.pipeline` (the module `base.pipeline.pipeline`). So the full module name is `lmetk.transforms.base.pipeline.pipeline`. That’s a bit redundant but standard in Python (e.g. `xml.etree.ElementTree`). So Option A is:

- `base/pipeline/__init__.py`  
- `base/pipeline/pipeline.py`  (current content)

and same for strategy, etc. Clean.

---

### 2.2 Option B – Subfolders + split large files

Same as Option A, but split **pipeline** and optionally **multisource** by responsibility.

#### Pipeline (919 lines → 4–5 files)

| New file under `base/pipeline/` | Content | ~Lines |
|---------------------------------|--------|--------|
| `context.py` | TransformContext, PipelineStep (Protocol), type aliases (BeforeCheckFn, AfterCheckFn, …) | ~100 |
| `hooks_registry.py` | register_hook, clear_hooks, _emit, _register_default_hooks | ~80 |
| `checks.py` | _run_scoped_checks, _default_before_check_*, _default_after_check_*, build_default_before_checks, build_default_after_checks | ~120 |
| `steps.py` | CopyStep, RegisterChunkBuilderStep, IntegrityCheckStep, IntegrityCheckBeforeStep, FitStep, TransformStep, PostprocessStep, MergeStep, IntegrityCheckAfterStep, RecordManifestStep, _get_producer_version | ~550 |
| `__init__.py` | Re-export all public types and functions used by strategy and tests | ~40 |

Internal steps can import from `context`, `checks`, and `hooks_registry` as needed. Strategy and other code keep importing from `lmetk.transforms.base.pipeline`; only the package’s internal structure changes.

#### Multisource (549 lines → 2 files, optional)

| New file under `base/multisource/` | Content | ~Lines |
|------------------------------------|--------|--------|
| `utils.py` | find_singular_ragged_dim, tolist, _normalize_keys, _assert_*, _build_flags, _log_transform_error, _concatenate_segments | ~280 |
| `mixins.py` | MultiSourceTransformInterface, _BaseDispatcher, FitByConcatenationMixin, NoFitMixin, PerSegmentTransformMixin, ConcatenateBeforeTransformMixin, StandardFitTransformMixin, StatelessPerSegmentMixin, StatelessConcatenateMixin, InverseTransformMixin | ~270 |
| `__init__.py` | Re-export all public mixins and functions | ~30 |

Splitting multisource is optional; it’s already a single concern (multi-source behaviour). Priority is splitting pipeline.

---

### 2.3 Public API and backward compatibility

- **Preserve current import paths** by making each “module” a package and re-exporting from its `__init__.py`:
  - `from lmetk.transforms.base.pipeline import TransformContext, CopyStep, ...`
  - `from lmetk.transforms.base.strategy import TransformStrategy, postprocess_transformed_data`
  - `from lmetk.transforms.base.data_transform import DataTransform`
  - `from lmetk.transforms.base.multisource import InverseTransformMixin, find_singular_ragged_dim, ...`
  - `from lmetk.transforms.base.transform_manager import ConfigTransformManager`
  - `from lmetk.transforms.base.handlers import get_handler`
  - `from lmetk.transforms.base.base_transform import BaseTransform, DenseTransform, ...`
  - `from lmetk.transforms.base.data_kind import infer_data_kind, get_capability, ...`
  - `from lmetk.transforms.base.integrity import run_check, run_checks`
  - `from lmetk.transforms.base.dry_run import dry_run_validation, DryRunResult`
- Optional: add `base/__init__.py` that re-exports the main public symbols so that `from lmetk.transforms.base import DataTransform, TransformStrategy, ConfigTransformManager, BaseTransform, ...` works as a convenience. Do not remove or change the existing `base.<module>` imports so that existing code and tests keep working.

---

## 3. Recommended layout (concrete)

**Phase 1 – Subfolders only (low risk)**  
- Add packages: `core/`, `pipeline/`, `strategy/`, `facade/`, `dispatch/`, `multisource/`, `manager/`, `validation/`.  
- Move current files into the matching package (e.g. `pipeline.py` → `pipeline/pipeline.py`), add `__init__.py` that re-exports the same names.  
- Keep `hooks.py` and, if you prefer, `integrity.py` and `data_kind.py` at `base/` level or under `core/` (see below).  
- Update all internal imports (e.g. strategy importing from pipeline) to use the new package paths (e.g. `from lmetk.transforms.base.pipeline import ...` stays the same; only package-internal imports like `from .context import TransformContext` change).  
- Run tests; fix any broken imports.

**Phase 2 – Split pipeline (optional)**  
- Inside `base/pipeline/`, introduce `context.py`, `hooks_registry.py`, `checks.py`, `steps.py` and move the corresponding bits from `pipeline.py`.  
- `pipeline/__init__.py` re-exports everything that was previously in `pipeline.py`.  
- Run tests again.

**Phase 3 – Split multisource (optional)**  
- Inside `base/multisource/`, add `utils.py` and `mixins.py`, move code, re-export from `__init__.py`.

Suggested placement of small shared modules:

- **integrity** – Used by pipeline only. Either keep at `base/integrity.py` or move to `base/core/integrity.py` and have `base/core/__init__.py` re-export it; pipeline then imports from `lmetk.transforms.base.core` or `lmetk.transforms.base.integrity` if you keep a top-level re-export in `base/__init__.py`.
- **data_kind** – Used by handlers and dry_run. Keep under `core/` and re-export from `core/__init__.py`; dispatch and validation import from `lmetk.transforms.base.core` (or keep `base/data_kind.py` as a single file for minimal churn).
- **hooks** – Stays at `base/hooks.py`; pipeline (or pipeline’s steps) imports it. No need to put under core.

To minimise churn and preserve every current import path, use **package names = current module names**:

- `base/base_transform/` – base_transform.py + `__init__.py`
- `base/data_kind/` – data_kind.py + `__init__.py`
- `base/integrity/` – integrity.py + `__init__.py`
- `base/pipeline/` – pipeline.py + `__init__.py` (no split yet)
- `base/strategy/` – strategy.py + `__init__.py`
- `base/data_transform/` – data_transform.py + `__init__.py`
- `base/handlers/` – handlers.py + `__init__.py`
- `base/multisource/` – multisource.py + `__init__.py`
- `base/transform_manager/` – transform_manager.py + `__init__.py`
- `base/dry_run/` – dry_run.py + `__init__.py`
- `base/hooks.py` – unchanged at top level (single module, not a package).

Then existing imports must change **only** in the sense that the *file* that defines a symbol moves; the *import path* can stay the same if we use package names that match the old module names:

- Old: `lmetk.transforms.base.pipeline` (module = base/pipeline.py).  
- New: `lmetk.transforms.base.pipeline` (package = base/pipeline/ with __init__.py importing from base/pipeline/pipeline.py).  
So `from lmetk.transforms.base.pipeline import TransformContext` is unchanged. We only need to ensure that nothing in the codebase does `from lmetk.transforms.base import pipeline` and then uses `pipeline.TransformContext`; if they do, that still works because `base.pipeline` is now a package and its `__init__.py` exposes the same names.

The only breaking change would be if something did `from lmetk.transforms.base import pipeline` and then `pipeline.SomePrivateThing` that we don’t re-export. So we should re-export everything that is currently part of the public API of each module (and avoid removing re-exports of previously public names).

---

## 4. Summary

| Approach | Pros | Cons |
|----------|------|------|
| **Subfolders only** | Clear grouping, minimal risk, same file contents | pipeline.py and multisource.py stay large |
| **+ Split pipeline** | Easier to navigate and test pipeline logic | More files and internal imports to maintain |
| **+ Split multisource** | Clear separation of utils vs mixins | Optional; benefit smaller than for pipeline |

Recommendation: do **Phase 1 (subfolders)** first and keep existing imports via package `__init__.py` re-exports. Then do **Phase 2 (split pipeline)** so the 919-line file becomes a small set of modules under `base/pipeline/`. Optionally add **Phase 3 (split multisource)** later. Keep `base/__init__.py` as an optional convenience re-export so that `from lmetk.transforms.base import DataTransform, ConfigTransformManager, ...` works without changing the rest of the codebase.
