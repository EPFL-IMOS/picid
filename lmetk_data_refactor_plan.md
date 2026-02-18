# lmetk/data Refactoring Plan — Data Layer Architectural Pruning

**Role:** Senior Software Architect / Systems Engineer (Architectural Pruning & Clean Code)  
**Scope:** `lmetk/data` — datasources, data_objects, cache  
**Context:** Transforms base has been refactored to a registry-based, decoupled architecture; the data layer remains tightly coupled and branch-heavy.  
**Goal:** Move to a unified, agnostic, PEP8-compliant data orchestration system that is scalable, maintainable, and accessible for new contributors.  
**Constraint:** Do not revert to monolithic if/else blocks; aim for a decoupled but "boring" and predictable middle path.

**Cross-layer audit:** This plan also considers how datasource and data objects are **used downstream** (PreProcessor, transform pipeline, docs/transforms) and aligns **validation and integrity** between the data layer (NaN/length checks, container.validate()) and the transform pipeline (IntegrityCheckStep, run_checks)—so both are used wisely and simplification does not come at the cost of functionality.

**Document status:** Living document. Re-validated for consistency, clarity, and optimality; recommendations are actionable and phased. Terminology: *load/split/get* refer to the target protocol; current code uses *load_data/split_data/get_data* until Phase 1.

---

## 1. Findings & Audit Results

### 1.1 Current Coupling: data → transforms

| Location | Dependency | Nature |
|----------|------------|--------|
| `lmetk/data/preprocessing/preprocessor.py` | `DataTransform` | Direct import; applies transforms to container; type-check `isinstance(t, DataTransform)`. |
| `lmetk/data/preprocessing/base.py` | `ConfigTransformManager` | Type hint and constructor; interface requires a transform manager. |
| `lmetk/data/optimization/sequencer.py` | `find_singular_ragged_dim` from `lmetk.transforms.base.multisource` | Uses transform-layer ragged-dim utility. |
| `lmetk/data/optimization/ak_jagged_sequencer.py` | Same `find_singular_ragged_dim` | Same. |

**Transforms → data (reverse direction):**

| Location | Dependency | Nature |
|----------|------------|--------|
| `lmetk/transforms/base/data_transform.py` | `NamedTransformInput`, `SplitDatasetContainer` from `lmetk.data.data_objects.data` | Core types for forward() contract. |
| `lmetk/transforms/base/strategy.py` | `NamedTransformInput`, `ReturnObject`, `SimpleReturnObject`, `NamedDictReturnObject`, `SplitDatasetContainer` | Pipeline and postprocess. |
| `lmetk/transforms/base/handlers.py` | `NamedTransformInput`, `BaseDataObjectWithMetadata` | Handler protocol and implementations. |

**Verdict:** The data layer does **not** import transform strategy/handlers/registry directly; the main coupling is (1) PreProcessor owning and applying `DataTransform`/ConfigTransformManager, and (2) optimization sequencers depending on `find_singular_ragged_dim`. Data objects are the **contract** between data and transforms; that dependency is intentional. The refactor should **remove** data → transform imports where possible (e.g. move `find_singular_ragged_dim` to a neutral place or duplicate a minimal version in data) and **define a clean interface** so datasources expose raw data + metadata only, with no knowledge of which transformation strategy will run.

### 1.2 Loader Complexity and Branching

- **SingleSourceLoader:**  
  - State: `_is_loaded`, `_is_splitted`, `is_part_of_multisource`, `data_splitter`.  
  - `get_data()` branches: if not `is_part_of_multisource` → require split, normalize metadata to `{train, val, test}`, return `SplitDatasetContainer`; else return `DatasetContainer`.  
  - Splitting is key-based (`create_splits_for`), updates `data_dict` in place with `split_masks` and per-variable `{train, val, test}`.

- **MultiSourceLoader:**  
  - Holds `data_source_dict: dict[SingleSourceLoader]`, `source_names`, `multisource_data_splitter` (optional).  
  - `load_data()` delegates to each child `load_data()`; `split_data()` either (a) uses `BySourceSplitter` to merge all sources into one `SplitDatasetContainer`, or (b) calls each source’s `split_data()` and concatenates per-split lists into one container.  
  - `get_data()` returns `SplitDatasetContainer` only; container built in `split_data()`.

- **PHMDMultiSourceLoader:**  
  - Bypasses Single/Multi pattern: implements `AbstractDataSourceLoader` directly, loads pre-split PHMD data, returns `SplitDatasetContainer`; no per-unit splitter, `split_data()` is a no-op with a warning.

**Observations:**  
- Single and Multi share the same abstract interface but different lifecycles: Single can return unsplit `DatasetContainer` when used as a child of Multi; Multi always returns split.  
- Logic is fragmented (who splits, when, and what container type is returned) and relies on `is_part_of_multisource` and multiple flags.  
- PHMD is a special case that doesn’t use SingleSourceLoader at all.

### 1.3 Data Objects and Metadata

- **BaseDataObject / BaseDataObjectWithMetadata:**  
  - Dict-like container with `_instance_cls` (type per key), `_data`; allowed types: `pd.DataFrame`, `pd.Series`, `np.ndarray`, `ak.Array`.  
  - Metadata is a dict (`_metadata`) with optional `column_map`; no dependency on MetadataManifest here.

- **DatasetContainer / SplitDatasetContainer:**  
  - DatasetContainer: optional `manifest: MetadataManifest`, `slice_info: SliceInfo`; normalizes unit data to lists.  
  - SplitDatasetContainer: split-level structure `key → {train, val, test} → list of per-unit data`; `validate()`, `to_split_dict()`, `group_by_split()`.  
  - `get_instance_cls()` is used by MultiSourceLoader (instance_cls consistency across sources) and by transforms (data_kind, copy policy).

- **MetadataManifest / SliceInfo:**  
  - Manifest: versioned entries, category `datasource` | `transform`, queryable by step_id/key/split.  
  - SliceInfo: split, unit_ids, cycle_ids, bounds, index_map.  
  - Both are optional on the container; not yet populated consistently by datasources.

- **NamedTransformInput, ReturnObject variants:**  
  - Used by transforms; names are transform-centric. Data layer only needs to provide structures that satisfy the transform contract (e.g. dict-like with keys and optional metadata).

### 1.4 Cache and Transform Coupling

- **FileSystemCache (offline):**  
  - Stages: `loaded_and_splitted_data`, `boundary`, `preprocessed`.  
  - Cache key for preprocessed/boundary = `compute_cache_key(config, library_dir, extensions)` where `config` includes **both** `datasource` and `transforms` (full or up-to-boundary).  
  - So cache **key** is tied to transform config; cache **value** stores `data` + `metadata` (including `transforms` manager and `meta_data`).  
  - Data-only stage (`loaded_and_splitted_data`) uses only `datasource` config + data library path; no transform config.

**Conclusion:** Cache is already staged. A "Data Signature" that identifies loaded-and-split data (datasource config + code hash) is already in place for stage 1. For preprocessed/boundary, the signature intentionally includes transforms; decoupling means we keep a **stable data signature** for stage 1 and document that stage 2+ is "data + transform pipeline outcome" so that the data layer doesn’t need to know transform internals—only the PreProcessor (or a single pipeline facade) combines them.

### 1.5 PEP8 and Clean Code Notes

- Inconsistent naming: `_is_splitted` (typo: "split" not "splitted"), `get_chached_transform_manager` (typo: "cached").  
- `AbstractDataSourceLoader.split_data()` docstring describes MultiSourceLoader-style args (data_lst, source_names) but the abstract method has no parameters.  
- Some long methods (e.g. PreProcessor.pipeline) and deep nesting; could be broken into smaller functions.  
- Type hints: partial (e.g. `data_dict: Dict[str, Union[...]] | Box`); some returns use union of container types.

### 1.6 Downstream Usage: How Data Flows Through the Pipeline

Understanding how datasource and data objects are used **after** load is essential for refactoring without breaking behaviour. The following traces the pipeline from config to consumption (sources: `docs/transforms`, `lmetk/transforms/base`, `lmetk/data/preprocessing`, `run.py`).

**End-to-end flow:**

1. **Config** → Run specifies a **datasource** (Single/Multi/PHMD) and a **transform pipeline** (OrderedDict of DataTransform from ConfigTransformManager).
2. **PreProcessor** is constructed with that datasource and ConfigTransformManager.
3. **Pipeline execution (PreProcessor.pipeline() or equivalent):**
   - **Stage 1 (load):** `datasource.load_data()` → `datasource.split_data()` → `datasource.get_data()` returns a **DatasetContainer** or **SplitDatasetContainer**. PreProcessor stores it in `self.data` via `fetch_data()` (no validation step today).
   - **Stage 2 (transform):** For each DataTransform in order, `transform.forward(data)` is called. Data is the **same container** (reference), mutated in place by each transform.
4. **Inside each DataTransform.forward():**
   - **TransformStrategy.apply()** builds a **TransformContext** (data, transform_instance, apply_to_keys, assign_to_keys, …). It does **not** set `integrity_checks`; so **IntegrityCheckStep is always a no-op** on the default path.
   - **CopyStep** copies the container (per-key deep vs shallow based on `get_instance_cls()` and `ak.Array`).
   - **IntegrityCheckStep** runs only if `context.integrity_checks` is set → **never in production**.
   - **FitStep / TransformStep** call `_prepare_chunks_for_split()` which, for each unit, builds `NamedTransformInput(**unit_data).get_sanitized_data()`. So **NaN and length checks run per unit, per split, per transform** inside chunk preparation—not at a single “data boundary” or “pipeline integrity” moment.
   - PostprocessStep → MergeStep → RecordManifestStep complete the step.
5. **After all transforms:** PreProcessor holds the final container; `get_processed_data_dict()` / `.data` feeds into **datamodules** and **datasets** (e.g. RUL context dataset, sliding window), which build PyTorch DataLoaders.

**Implications for refactoring:**

- Any change to container shape (e.g. manifest, slice_info, or key layout) must remain compatible with TransformContext, CopyStep (which uses `get_instance_cls()` and apply_to_keys), FitStep/TransformStep (which use `_prepare_chunks_for_split` and expect list of NamedTransformInput), and MergeStep (which writes by assign_to_keys).
- **Validation is currently scattered:** SplitDatasetContainer.validate() (structure: same number of units per split, same keys per unit) is **never called** in the codebase. NaN/length checks exist only inside `get_sanitized_data()`, invoked repeatedly during chunk preparation. There is no single “validate once after load” gate in PreProcessor.

### 1.7 Validation & Integrity: Data Layer vs Transforms (Unified Strategy)

This subsection audits **who checks what** and proposes a **single, wise split** of responsibilities so both layers are used without duplication or loss of functionality. It directly addresses the overlap between data-object checks (NaNs, validate) and the transform pipeline’s unused integrity machinery (IntegrityCheckStep, run_checks).

**Current state:**

| Layer | Where | When it runs | Used? |
|-------|--------|----------------|--------|
| **Data-object checks (implementation)** | `lmetk/data/data_objects/utils.py`: `check_for_nans`, `check_length_consistency` | — | **Yes**: called from BaseDataObjectWithMetadata.validate() and from NamedTransformInput.get_sanitized_data(). |
| **Data-object validation (container)** | `DatasetContainer.validate()` (empty); `SplitDatasetContainer.validate()` (structure: units/splits/keys consistent) | — | **No**: validate() is never called in the codebase; only documented (“run .validate() first”). |
| **Chunk-level sanitization** | `NamedTransformInput.get_sanitized_data(check_nans, check_lengths)` | **Every** FitStep and TransformStep, for **every** unit, when building chunks in `_prepare_chunks_for_split` | **Yes**: same NaN/length logic runs N×M times (N units × M transforms × splits). |
| **Pipeline integrity** | `IntegrityCheckStep`; `context.integrity_checks`; `run_check`/`run_checks` in `integrity.py` | After CopyStep, before FitStep—**only if** integrity_checks is set | **No**: strategy never sets `integrity_checks`; no caller injects checks. So the step is always a no-op. |

**Gap (from docs):** REFACTORING_ROADMAP §4.3.1 and ARCHITECTURE_PRUNING_AUDIT state that (1) data-object checks should gate **input** (after load, before transforms) and (2) pipeline integrity checks should guard **post-step state** (e.g. after a transform that might introduce NaNs). Today we have neither a single “after load” validation nor any pipeline checks; we only have repeated chunk-level checks inside the pipeline.

**Proposed unified strategy (simplification without losing functionality):**

1. **Single implementation (canonical checks):**  
   Keep `check_for_nans` and `check_length_consistency` in **`lmetk/data/data_objects/utils.py`** as the only implementation. Both “data boundary” and “pipeline integrity” will call these; no duplicate logic elsewhere.

2. **Data boundary (after load, before transforms):**  
   - In **PreProcessor.fetch_data()** (or immediately after `get_data()` in `pipeline()`), add an optional **one-time validation** step:  
     - For SplitDatasetContainer: call `container.validate()` (structure: same units per split, same keys per unit).  
     - Optionally: run NaN/length checks on a configurable set of keys (e.g. `features`, `target`) using the data-layer utils on the **container** (e.g. iterate splits and units and call the same checks).  
   - This gates “bad raw data” before any transform runs and satisfies the “data-object checks gate input” requirement. One place, one time.

3. **Pipeline integrity (per-transform, optional):**  
   - **Wire** IntegrityCheckStep so it can receive checks from the outside: e.g. **TransformStrategy.apply()** (or a factory used by DataTransform/PreProcessor) accepts an optional `integrity_checks: List[(name, check_fn, policy)]`.  
   - **Built-in pipeline checks** that wrap data-layer utils: e.g. in `lmetk/transforms/base/integrity.py` add helpers such as `make_container_nan_check(container, keys, split)` returning a no-arg `check_fn` that calls `check_for_nans` on the relevant slices of `context.transformed_data`. Policy remains fail/warn/allow (PHM: use `"fail"`).  
   - Strategy (or config) injects one such check per transform, e.g. “no NaN in apply_to keys after copy,” so we catch transform-introduced NaNs **once per transform** instead of only at chunk preparation.

4. **Chunk preparation (simplify):**  
   - **Option A (recommended):** Make **get_sanitized_data()** **optional** in chunk preparation: when the pipeline is configured with integrity checks (e.g. strategy or DataTransform receives an optional `integrity_checks` list and it is non-empty), call `NamedTransformInput(**unit_data).get_sanitized_data(check_nans=False, check_lengths=False)` so redundant N×M checks are skipped; otherwise keep the current default (check_nans=True, check_lengths=True) for safety. Single decision point: “integrity_checks configured ⇒ skip chunk-level sanitization.”  
   - **Option B:** Remove get_sanitized_data() from _prepare_chunks_for_split entirely and rely on (1) one-time validation after load and (2) IntegrityCheckStep with container-level checks. Chunk preparation then only builds NamedTransformInput(**unit_data) without sanitization. Simpler but requires that pipeline checks cover all keys/splits that transforms read.

**Responsibility matrix (target):**

| Check type | Who implements | When / where | Policy |
|------------|----------------|--------------|--------|
| Structure (units/splits/keys) | Data layer (SplitDatasetContainer.validate) | Once after load (PreProcessor) | Fail (recommended) |
| NaN / length on raw data | Data layer (utils) | Once after load, on chosen keys (PreProcessor) | Configurable (fail for PHM) |
| NaN / length after transform | Data layer (utils) wrapped by pipeline | IntegrityCheckStep, once per transform (container-level) | Policy-driven (fail for PHM) |
| Chunk-level (per unit) | Data layer (get_sanitized_data) | Optional in _prepare_chunks_for_split; default off when boundary + pipeline checks are on | Optional defense-in-depth |

**Benefits:**  
- **Single source of truth:** All concrete checks (NaN, length) live in data/utils; pipeline only orchestrates “when” and “policy.”  
- **Clear mental model:** “Validate once at the boundary; optionally re-check after each transform via IntegrityCheckStep.”  
- **Simplification:** No repeated per-unit checks in hot path when boundary + pipeline checks are enabled; IntegrityCheckStep becomes useful instead of dead code.  
- **PHM-aligned:** Use policy `"fail"` for both data-boundary and pipeline checks in production so execution stops on first violation.

**Refactor plan impact:**  
- Phase 1 (Interface Unification): No change to validation.  
- Phase 2 (Object & Cache): Add optional “data boundary” validation in PreProcessor (or unified SourceLoader) after get_data(); document container.validate() as part of the contract.  
- **Phase 2.5 (Validation & Integrity — optional dedicated phase):** (1) Add injection point for `integrity_checks` (strategy or DataTransform/PreProcessor); (2) Add built-in pipeline check helpers in `integrity.py` that wrap data-layer utils; (3) Make get_sanitized_data() default to check_nans=False, check_lengths=False when integrity_checks are configured; (4) Document the responsibility matrix in REFACTORING_ROADMAP and in this plan.

### 1.8 Datasource Metadata Propagation to Transforms

Understanding how datasource metadata flows—and whether it is reachable from transform instances—is needed for refactoring and for any transform that might depend on dataset-level or unit-level metadata.

**Definitions:**

- **Datasource-level metadata:** The dict returned by `datasource.get_meta_data()`. Content is loader-specific: e.g. PHMD loaders set `task.meta`, `unit_names`, `unit_ids` (per split); UNIBO/NASA loaders set `unit_ids`, `unit_names` and similar; SingleSourceLoader returns `{}`; MultiSourceLoader returns `{}`.
- **Per-unit metadata in the container:** The container (SplitDatasetContainer) can have a key `"metadata"` with structure `metadata[split][i]` = dict for unit *i* in that split (e.g. `unit_id`, `unit_name`, `features_columns`, `column_map`). This is populated by the datasource when it builds the container (e.g. in PHMD `_process_unit` each unit has a `"metadata"` dict).
- **Transform metadata:** The dict passed as the second argument to `fit_data(data, metadata=...)` and `transform_data(data, metadata=...)`. It is **built inside the pipeline** (FitStep, TransformStep) from `TransformContext` and contains only: `mode` (split name), `apply_to_keys`, `assign_to_keys`, `assign_to_map`. It does **not** contain datasource-level metadata.

**Flow of datasource-level metadata:**

1. **Production:** Loaders implement `get_meta_data()` and return a dict (e.g. after `_load_data()` or when building the container).
2. **Capture:** During `PreProcessor.pipeline()`, stage 1 either calls `build_data_stage()` (which returns `(datasource, datasource.get_meta_data())`) or loads from cache; in both cases PreProcessor ends up with `self.meta_data` = that dict (from cache it is `saved_objects["meta_data"]`).
3. **Consumption:** PreProcessor exposes it via `get_meta_data_dict()`. **Only** `run.py` and `simplified_run_objects.py` call that; they pass it as `meta_data_dict` into **dataset** construction (e.g. `HydraConcatDataset`, `RulContextDataset`). Datasets use it for unit IDs, unit names, current split, concat index, etc.
4. **Transform pipeline:** `DataTransform.forward(data)` receives only the **container** (`data`). It does **not** receive PreProcessor.meta_data or any other datasource-level metadata. `TransformStrategy.apply()` builds `TransformContext` from `data`, `transform_instance`, `apply_to_keys`, etc.; there is **no** field on `TransformContext` for datasource metadata. FitStep and TransformStep build the `metadata` dict only from context (mode, apply_to_keys, assign_to_keys, assign_to_map). So **datasource-level metadata is never passed into the transform pipeline or into transform instances.**

**Reachability of per-unit metadata (container key `"metadata"`):**

- Chunks passed to `fit_data` / `transform_data` are built by `_prepare_chunks_for_split`: for each unit index *i*, `unit_data = { key: data_to_prepare[key][split][i] for key in keys }` where `keys` come from `data_to_prepare`, which is built from **apply_to_keys** only (e.g. `context.data[key]` for `key in apply_to_keys`). So the chunk contains only the **apply_to** keys (typically `"features"`, `"target"`). The container key `"metadata"` is included in the chunk **only if** `"metadata"` is one of the transform’s **apply_to** keys in config.
- **If** a transform is configured with `apply_to` that includes `"metadata"` (e.g. `["features", "metadata"]`), then each chunk will have `chunk["metadata"]` = that unit’s metadata from the container. Transforms can then read it (e.g. `lmetk/transforms/battery/nb1_transforms.py` uses `data["metadata"]["features_columns"]`). So **per-unit metadata in the container is reachable by transform instances only when "metadata" is in apply_to**—i.e. **conditionally reachable** and config-dependent.

**Summary: extent of reachability**

| Metadata kind | Produced / stored | Passed to pipeline? | Reachable from transform instance? |
|---------------|-------------------|---------------------|------------------------------------|
| **Datasource-level** (get_meta_data()) | Loader; PreProcessor.meta_data; cache | **No** | **No.** Only flows to run.py → datasets as meta_data_dict. |
| **Per-unit in container** (data["metadata"][split][i]) | Datasource when building container | Yes (it is part of the container) | **Only if** apply_to includes `"metadata"` → then chunk["metadata"] is that unit’s metadata. |
| **Transform metadata** (mode, apply_to_keys, assign_to_map) | Built in FitStep/TransformStep from context | Yes (second argument to fit_data/transform_data) | **Yes.** Every transform receives it as the `metadata` parameter. |

**Implications for refactoring:**

- If a transform needs **datasource-level** metadata (e.g. list of feature names, task identifier, unit_ids for the whole dataset), it **cannot** get it today from the pipeline or from the transform instance. Options for a future refactor: (1) add an optional `datasource_metadata: Optional[Dict[str, Any]]` to `TransformContext` and pass it from PreProcessor when calling the pipeline (e.g. via DataTransform or a wrapper); (2) keep the current design and document that such metadata is only available to datasets, not to transforms.
- If a transform needs **per-unit** metadata (e.g. unit_id, features_columns for that unit), it can get it today by having `"metadata"` in apply_to and reading `chunk["metadata"]`; this should be documented so config authors know to include `"metadata"` when transforms need it.
- Cache already stores `meta_data` next to the preprocessed data (saved_objects["meta_data"]); that is datasource metadata for use after restore (e.g. run.py), not for transforms.

### 1.9 Main Usage Example: run.py

The primary consumer of the datasource and transform pipeline is **`lmetk/run.py`**. Documenting this flow grounds the refactor plan in real usage and constrains breaking changes.

**Flow (concise):**

1. **Config:** Hydra supplies `cfg.datasource` (target + params) and `cfg.transforms` (list of transform configs).
2. **Instantiate datasource:**  
   `dataset_source = hydra.utils.instantiate(cfg.datasource)`  
   Type is one of: `SingleSourceLoader`, `MultiSourceLoader`, `PHMDMultiSourceLoader`, `MOViLoader`, `NB14Loader`, `UNIBO21Loader`, `MZVAVLoader`.
3. **Transform manager:**  
   `transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)`  
   Built once before preprocessing.
4. **Preprocessor mode:**  
   - If `dataset_source` is in `(SingleSourceLoader, PHMDMultiSourceLoader, MOViLoader, NB14Loader, UNIBO21Loader, MZVAVLoader)` → `preprocessor_mode = "per_unit"`.  
   - Else if `isinstance(dataset_source, MultiSourceLoader)` → `preprocessor_mode = "cross_unit"`.  
   - Else → `ValueError("Invalid dataset source type")`.  
   So run.py **branches on concrete loader types** to set mode; it does not use an abstract interface or a property on the loader.
5. **PreProcessor:**  
   `preprocessor = PreProcessor(preprocessor_mode, dataset_source, transforms_manager)`  
   PreProcessor holds the datasource and the transform manager for the rest of the run.
6. **Pipeline:**  
   `preprocessor.pipeline(data_cache_path=..., data_library_part_path=..., transform_library_part_path=..., cache_preprocessed=...)`  
   This runs load/split (or restores from cache), then applies all transforms (or restores preprocessed + manager from cache). On preprocessed cache hit, `preprocessor`’s internal state (data, transforms manager) is restored from cache.
7. **Outputs:**  
   - `data_dict = preprocessor.get_processed_data_dict(return_splits_on_first_level=True)` → `{ "train": { key: ... }, "val": { ... }, "test": { ... } }`.  
   - `meta_data_dict = preprocessor.get_meta_data_dict()` → datasource metadata (unit_ids, unit_names, etc.).  
   - If `cache_preprocessed`: `transforms_manager = preprocessor.get_chached_transform_manager()` so the **fitted** manager (e.g. fitted scalers) is used for the rest of the run.
8. **Datasets:** For each split, run.py builds a dataset via Hydra with `data_dict={ key: data_dict[split][key] for key in input_data_task_keys }` and `meta_data_dict=meta_data_dict` (with `current_data_split` set). So datasets receive **processed** data and **datasource** metadata.
9. **Datamodule:** Instantiated with `dataset_train`, `dataset_val`, `dataset_test`.
10. **Evaluator / inverse transform:** For each split, if the evaluator needs inverse scaling, run.py gets the inverter from `transforms_manager.get_inverter_for_key_with_name(inverse_transform_key, ...)` or `transforms_manager.get_transform(inverse_transform_name)`. So **transforms_manager must remain the single source of truth for fitted transforms** after pipeline (or after cache restore).

**Refactor constraints implied by run.py:**

- **Loader type check:** run.py uses `isinstance(dataset_source, single_source_types)` and `isinstance(dataset_source, MultiSourceLoader)` to set `preprocessor_mode`. If loaders are unified into a single `SourceLoader` (or interface), run.py must either: (a) keep supporting the same concrete types via thin wrappers that are `isinstance`-compatible, or (b) switch to a **property** on the loader (e.g. `get_split_mode()` already on AbstractDataSourceLoader) and drop the explicit type list. Option (b) is preferable for a unified loader: use `dataset_source.get_split_mode()` (or equivalent) instead of branching on loader class.
- **PreProcessor API:** `PreProcessor(mode, datasource, transforms_manager)`, `.pipeline(...)`, `.get_processed_data_dict(return_splits_on_first_level=True)`, `.get_meta_data_dict()`, `.get_chached_transform_manager()` are all used. Names and signatures should be preserved or migrated with a deprecation path.
- **Cache restore:** After loading from preprocessed cache, run.py relies on `get_chached_transform_manager()` to obtain the fitted manager. The cache must keep storing and restoring the manager next to the preprocessed data so that inverse transforms and any later use of transforms refer to fitted instances.
- **data_dict shape:** run.py assumes `data_dict` has keys `train`/`val`/`test` and, per split, keys from `input_data_task_keys` (from `cfg.task_definition.model.data_requirements.input_tensors`). The refactor must not change the contract of `get_processed_data_dict(return_splits_on_first_level=True)`.

**Summary:** run.py is the main usage example; the refactor plan (unified loader, PreProcessor, cache, metadata) must keep this flow working and should prefer deriving `preprocessor_mode` from the loader instead of listing concrete loader types. **AbstractDataSourceLoader** already defines **`get_split_mode()`** (returns `"per_unit"` or `"cross_unit"` from `_init_kwargs.multisource_data_splitter`). **Caveat:** Today run.py treats *any* MultiSourceLoader as `"cross_unit"`, whereas the base implementation of `get_split_mode()` returns `"cross_unit"` only when `multisource_data_splitter` is set (otherwise `"per_unit"`). So before run.py can switch to `preprocessor_mode = dataset_source.get_split_mode()`, **MultiSourceLoader** should override `get_split_mode()` to return `"cross_unit"` unconditionally (or the unified SourceLoader must expose an equivalent that matches run.py’s current semantics). Then run.py can drop the `isinstance` checks safely.

---

## 2. Unified Loader Assessment

### 2.1 Is SingleSourceLoader a Special Case of MultiSourceLoader (N=1)?

**Functionally:**

- **SingleSourceLoader (standalone):** load → split (per-unit splitter) → get_data → `SplitDatasetContainer`.  
- **SingleSourceLoader (as child of Multi):** load only; no split; get_data → `DatasetContainer` (raw dict).  
- **MultiSourceLoader:** load all children; split_data() either (a) joint BySourceSplitter, or (b) split each child then concatenate lists; get_data → `SplitDatasetContainer`.

If we model "source" as an abstraction that can **either** own a splitter (single standalone) **or** delegate splitting to a parent (multi), then:

- A **unified SourceLoader** could be: one object that represents one or many "sources."  
  - N=1: one internal source, one optional splitter; load → split (if splitter and not delegated) → return container.  
  - N>1: N internal sources, optional multisource splitter; load all → split (joint or per-source then merge) → return container.

So **yes**, Single can be seen as the N=1 case of Multi **if** we unify the lifecycle:

- **Unified contract:** `load()` → `split()` (may no-op if splitting is delegated or pre-split) → `get_data()` returns one of: unsplit container (single chunk) or `SplitDatasetContainer`.  
- Caller (PreProcessor) always calls load → split → get_data; for N=1 with splitter we get SplitDatasetContainer; for N=1 as child we’d never use it standalone; for N>1 we get SplitDatasetContainer.

**Design options:**

1. **Collapse into one class `SourceLoader`:**  
   - Internal: `sources: List[SingleSourceLike]` with length 1 or more.  
   - Splitter: either per-source (N=1 and not delegated) or multisource (N>1).  
   - Single class simplifies the API (one loader type) but the internal branching (N=1 vs N>1, splitter type) remains; we just move it under one roof.

2. **Keep two classes but derive Single from Multi:**  
   - `MultiSourceLoader` is the base; `SingleSourceLoader` = Multi with a single source and a default per-unit splitter.  
   - Reduces duplication but keeps two public types for backward compatibility and clarity of intent.

3. **Single interface, two implementations (current shape, refined):**  
   - Keep AbstractDataSourceLoader; Single and Multi both implement it; standardize `.load()` / `.split()` / `.get_data()` semantics and return types so that PreProcessor always gets a `DatasetContainer` or `SplitDatasetContainer` and doesn’t branch on loader type.

**Recommendation:**  
**Unify the protocol** (load → split → get_data with clear guarantees) and **prefer Option 1 (single SourceLoader)** for new code, with a compatibility shim so that existing code that expects SingleSourceLoader or MultiSourceLoader can get a thin wrapper. This reduces cognitive load ("there is one way to load") and makes it easier to add PHMD as a different "backend" (same interface, different internal implementation) rather than a parallel hierarchy.

### 2.2 Performance

- **N=1 as Multi:** One-element list of sources; one splitter; no measurable overhead.  
- **Unified load/split/get:** Same number of actual load/split steps as today; no extra layer of indirection if we implement via a single class.  
- **Risk:** Low, provided we don’t add unnecessary abstraction (e.g. generic "source" objects that wrap every single-unit loader in another list).

---

## 3. Phase-by-Phase Roadmap

### 3.0 Potential Usage Interfaces (Reference for Phases 0–1)

These interfaces are targets for the refactor and are used to specify Phase 0 snapshot format and test contracts. No production code is required to implement them in Phase 0; they serve as a specification.

**Loader protocol (target, Phase 1):**

```python
# Conceptual interface for SourceLoader (single or multi).
# Current code uses load_data/split_data/get_data until Phase 1.

class SourceLoaderProtocol(Protocol):
    def load_data(self) -> None: ...
    def split_data(self) -> None: ...
    def get_data(self) -> DatasetContainer | SplitDatasetContainer: ...
    def get_meta_data(self) -> Dict[str, Any]: ...
    def get_split_mode(self) -> str:  # "per_unit" | "cross_unit"
```

**PreProcessor pipeline contract (unchanged):**

- Input: `datasource` (loader), `transforms_manager` (ConfigTransformManager), `mode` ("per_unit" | "cross_unit").
- Pipeline steps: load/split (or cache restore) → fetch_data() → apply_transforms().
- Output: `get_processed_data_dict(return_splits_on_first_level=True)` → `dict[str, dict[str, Any]]` with keys `train`/`val`/`test`; `get_meta_data_dict()` → datasource metadata.

**Snapshot (Phase 0 real-data fixtures):**

- **Purpose:** Save deterministic “slices” of data after load and after transforms so that tests can compare current pipeline output against these slices and detect regressions.
- **Slice format:** For each stage (`loaded` | `transformed`) and each split (`train` | `val` | `test`), save a small, comparable representation (e.g. shapes, dtypes, and a checksum or a small subset of values—e.g. first unit’s first N rows for selected keys). Exact format is defined in the snapshot script (see Phase 0 Task B.1 below).
- **Location:** Versioned under `test/data/fixtures/pipeline_snapshots/` (or similar), e.g. `{datasource_name}/{stage}/slices_manifest.json` + optional `.npy`/`.npz` for numeric slices so tests stay deterministic and fast.

**Helper signatures (Phase 0 snapshot/comparison):**

```python
# Extract a comparable slice representation from a container or split dict.
# Used by both the snapshot script and the comparison tests.

def container_to_slice_manifest(
    data: SplitDatasetContainer | dict[str, dict[str, Any]],
    keys: list[str],
    max_rows_per_unit: int = 100,
    splits: list[str] | None = None,  # default ["train", "val", "test"]
) -> dict[str, Any]:
    """Produce a manifest (shapes, dtypes, and paths or inline arrays for a small subset)."""
    ...

def compare_to_saved_slices(
    current_manifest: dict[str, Any],
    saved_manifest_path: Path,
    saved_slices_dir: Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert current_manifest matches saved manifest and numeric slices; raise on mismatch."""
    ...

# Timing (Phase 0 real-data): saved manifest includes "preprocessing_time_seconds".
# In tests, measure transform time and assert current_time <= baseline * max_time_ratio (e.g. 2.0).

def assert_preprocessing_time_within_bounds(
    current_time_seconds: float,
    saved_manifest_path: Path,
    max_time_ratio: float = 2.0,
    max_absolute_slack_seconds: float | None = None,
) -> None:
    """Load baseline from manifest; fail if current_time exceeds baseline * max_time_ratio (or baseline + slack)."""
    ...
```

---

### Phase 0: Stabilization (The Safety Net)

**Goal:** 100% functional coverage of current data loading and caching behavior before changing production code. Include both **mock-based** characterization tests and **real-data** tests that run the full pipeline (datasource + transforms + caching) and compare outputs to saved slices.

**Tasks:**

---

#### A. Mock-based characterization tests

1. **Characterization tests for loaders (mocks and in-memory)**  
   - SingleSourceLoader: load_data → split_data → get_data (standalone); load only, get_data as part of multi (mock MultiSourceLoader that holds one source).  
   - MultiSourceLoader: 2 sources, no multisource splitter (per-source split then concat); 2 sources with BySourceSplitter.  
   - Edge cases: missing metadata; metadata not in `{train, val, test}` (SingleSourceLoader normalizes); ragged/per-unit list lengths; empty splits.

2. **Characterization tests for data objects**  
   - DatasetContainer: single-unit vs multi-unit normalization; manifest/slice_info optional.  
   - SplitDatasetContainer: validate(), to_split_dict(), group_by_split(); get_instance_cls() consistency across keys.

3. **Characterization tests for cache**  
   - FileSystemCache: save/load for stages `loaded_and_splitted_data`, preprocessed, boundary; hash match → load; hash mismatch → rebuild; load_metadata when missing.  
   - Pipeline (with mocks): load-only cache hit; preprocessed cache hit; boundary restore then run remaining transforms; cache key = f(datasource_config, transform_config, library paths).

4. **Pipeline + transforms (mocked)**  
   - PreProcessor with mock datasource and real ConfigTransformManager (minimal transform config, e.g. identity or one scaler): run pipeline(), then assert on get_processed_data_dict() structure and that transforms were applied (e.g. data shape preserved or scaling applied).  
   - With cache enabled (tmp_path): run pipeline twice; second run restores from cache; assert output matches first run.

5. **Run.py contract (mocked):** Characterization test that mirrors the main run.py path with mocked or in-memory datasource: instantiate datasource, set preprocessor_mode (or get_split_mode()), build PreProcessor, run pipeline(), assert get_processed_data_dict(return_splits_on_first_level=True) shape and get_meta_data_dict() presence. Locks the contract for Phase 1 and run.py simplification.

---

#### B. Real-data pipeline tests: snapshot generation and comparison

Two real datasource configs are used so that both a **single-source** and a **multi-unit** (or multisource) path are covered by real pipeline runs. **Meaningful transform configs** are used so that the pipeline exercises real scaling and feature logic:

- **Single-source (per_unit):**  
  - Datasource: `configs/datasource/phme20.yaml` → `PHME20Loader` (PHMD, pre-split).  
  - Transforms: **`configs/transforms/phme20/normalize_feature_target.yaml`** (MinMaxScaler on features and on rul, fit_on train).
- **Multi-unit:**  
  - Datasource: `configs/datasource/unibo21_bosello.yaml` → `UNIBO21Loader`.  
  - Transforms: **`configs/transforms/battery/unibo/combined.yaml`** (scalers, time/spectral stats, ragged→regular, concatenate, target subsample/reshape, etc.).

**Task B.1 — Snapshot generation script (separate file)**

- **Objective:** Run the **current** pipeline (PreProcessor + datasource + transforms) for each of the two configs and save **slices** of the data at two stages: (1) **after load** (before transforms), and (2) **after transforms** (final preprocessed data). Also **measure and save** the wall-clock time taken to apply the transforms so tests can assert on performance.
- **Location:** A dedicated script or module, e.g. `scripts/generate_pipeline_snapshots.py` or `test/data/pipeline_snapshots/generate_snapshots.py`, that can be run manually or in CI to refresh fixtures when the pipeline or data is intentionally changed.
- **Approach:**
  1. Resolve config for datasource and transforms: **phme20** → datasource `configs/datasource/phme20.yaml`, transforms **`configs/transforms/phme20/normalize_feature_target.yaml`**; **unibo21_bosello** → datasource `configs/datasource/unibo21_bosello.yaml`, transforms **`configs/transforms/battery/unibo/combined.yaml`**.
  2. Instantiate datasource and ConfigTransformManager; set mode (per_unit for both as per run.py).
  3. Build PreProcessor and run **stage 1 only** (load_data, split_data, get_data; no transforms). Export **slices** of the container to e.g. `test/data/fixtures/pipeline_snapshots/phme20/loaded/` and `.../unibo21_bosello/loaded/`.
  4. Run the **full pipeline** (with transforms). **Measure** the time taken for the transform step only (e.g. wall-clock around PreProcessor.apply_transforms() or the transform loop inside pipeline()). Record this as **`preprocessing_time_seconds`** (or `transform_apply_seconds`). Export slices of the **transformed** container to `.../phme20/transformed/` and `.../unibo21_bosello/transformed/` and **include the recorded time** in the manifest (see slice format below).
  5. Optionally include meta_data (get_meta_data_dict()) in the snapshot (e.g. keys and list lengths).
- **Slice format (concrete):**  
  - **JSON manifest:** `slices_manifest.json` with structure such as `{ "split": { "train"|"val"|"test": { "key": { "shape": [...], "dtype": "...", "slice_path": "train_features_0.npy" } } }, "preprocessing_time_seconds": <float> }` and optional `meta_data_keys`, `meta_data_unit_counts`. The **`preprocessing_time_seconds`** field is the baseline transform time used later in tests.  
  - **Numeric slices:** For each (split, key, unit_index) a small `.npy` or a single `.npz` for the chosen subset.  
  - **Version:** Include a schema version or snapshot format version in the manifest.

**Task B.2 — Real-data comparison tests**

- **Objective:** Load data through the **current** pipeline for phme20 and unibo21_bosello and **compare** the resulting data to the saved slices for both (1) **just after load** and (2) **after transforms**. In addition, **measure** the current preprocessing (transform) time and **fail the test if it deviates considerably** from the saved baseline so that performance regressions are caught.
- **Location:** e.g. `test/data/preprocessing/test_pipeline_real_data.py` (or under `test/data/integration/`).
- **Approach:**
  1. **After-load comparison:** For each config, instantiate datasource, run load_data(), split_data(), get_data(). Build the same slice representation as in the snapshot; assert shapes, dtypes, and numeric slices match the saved manifest (e.g. `np.allclose` with a small tolerance).
  2. **After-transforms comparison:** For each config, run the full PreProcessor.pipeline() (no cache, or with a dedicated tmp cache). **Time** the transform step (same scope as in the snapshot script: e.g. wall-clock around the apply_transforms call or the transform loop). Build the same slice representation and assert vs. saved transformed slices (shapes, dtypes, numeric comparison). **Then assert preprocessing time:** load the saved **`preprocessing_time_seconds`** from the transformed-stage manifest; require that the current run’s transform time does **not** exceed `baseline_time * max_time_ratio` (and optionally `baseline_time + max_absolute_slack_seconds`). If it does, **fail the test** (e.g. `pytest.fail(...)` or `assert current_time <= baseline_time * max_time_ratio`). Recommended default: **`max_time_ratio = 2.0`** (current run must not be more than 2× slower than baseline); make this configurable (e.g. pytest param or env var `PIPELINE_SNAPSHOT_MAX_TIME_RATIO`) so CI can be relaxed if needed. Document in the test file that a failing time assertion indicates a performance regression and that snapshots can be regenerated if the slowdown was intentional.
  3. **Caching (optional but recommended):** Run pipeline twice with cache enabled (tmp_path); compare output to first run and to saved transformed slices; optionally assert that cache restore is faster than full run (or at least not slower than baseline * ratio).
  4. **Skip when fixtures missing:** If the snapshot directories or manifest are missing, skip with a clear reason and document that the snapshot generation script must be run once. Use a pytest marker (e.g. `@pytest.mark.requires_snapshots`) for selective runs.
- **Determinism:** Use fixed seeds where the pipeline or datasource allows so that regenerated snapshots are stable. Document any non-determinism and compare only deterministic parts.
- **Timing summary:**  
  - **Snapshot script:** Record and store `preprocessing_time_seconds` when generating transformed-stage fixtures.  
  - **Test:** Measure the same transform step; load baseline from manifest; **fail if `current_time > baseline_time * max_time_ratio`** (e.g. 2.0). This ensures the test does not pass when preprocessing time deviates considerably (performance regression).

**Deliverables (Phase 0 overall):**

- Mock-based characterization tests for loaders, data objects, cache, pipeline with transforms and caching, and run.py contract.
- Snapshot generation script (separate file) that produces slices for phme20 and unibo21_bosello at "loaded" and "transformed" stages, using `configs/transforms/phme20/normalize_feature_target.yaml` and `configs/transforms/battery/unibo/combined.yaml`, and records `preprocessing_time_seconds` in the transformed-stage manifest.
- Real-data comparison tests that assert loaded and transformed outputs against those slices, and assert that current preprocessing time does not exceed baseline × max_time_ratio (test fails if it deviates considerably); optional cache round-trip test.
- All tests marked or documented as characterization/snapshot tests; no production code refactor in this phase.

**Phase 0 approach summary:**

| Layer | Mock tests | Real-data tests |
|-------|------------|-----------------|
| **Loaders** | In-memory / mock loaders; edge cases (metadata, ragged, empty). | — |
| **Data objects** | Containers, validate(), to_split_dict(), get_instance_cls(). | — |
| **Cache** | FileSystemCache stages; pipeline cache hit/miss with mocks. | Optional: cache round-trip in real-data test. |
| **Pipeline + transforms** | PreProcessor + mock datasource + real transform manager; cache on tmp_path. | Full pipeline for phme20 (normalize_feature_target) and unibo21_bosello (unibo/combined); compare to saved slices; **compare preprocessing time to baseline, fail if > baseline × max_time_ratio**. |
| **Run.py contract** | Mock path: datasource → mode → PreProcessor → pipeline → data_dict shape + meta_data. | Covered by real-data pipeline test output shape. |

| Real-data flow | Step 1 (script) | Step 2 (tests) |
|----------------|-----------------|-----------------|
| **Snapshot generation** | Run pipeline with phme20 and unibo transform configs; save slices at loaded/transformed; **record and save `preprocessing_time_seconds`** in transformed manifest. | — |
| **Comparison** | — | Build same slice representation; assert vs. saved manifest + numeric slices; **measure transform time, assert ≤ baseline × max_time_ratio (e.g. 2.0)**; optional cache round-trip. |

**Complexity:** Medium–High (fixture format, two real configs, pipeline invocation).  
**Risk:** Low for mocks; medium for real-data (dependencies on data dir, config resolution). Mitigation: skip real-data tests when paths or snapshots are missing; document data and config requirements clearly.

---

### Phase 1: Interface Unification

**Goal:** Single, predictable SourceLoader abstraction and standardized load/split/get protocol.

**Tasks:**

1. **Define a single SourceLoader interface (or class)**  
   - Methods: `load_data()` (or `load()`), `split_data()` (or `split()`), `get_data()` → `DatasetContainer | SplitDatasetContainer`.  
   - Semantics: after load, data is in memory; after split, data is split (or no-op for pre-split/delegated); get_data() returns the canonical container.  
   - **Naming:** Either (a) standardize on `load()`/`split()` and add thin aliases `load_data`/`split_data` for backward compatibility (PreProcessor and run.py call `load_data()`/`split_data()`), or (b) keep `load_data()`/`split_data()` as the canonical names to avoid changing PreProcessor and configs. Option (b) is lower risk; Option (a) yields a shorter protocol. Choose one and use it consistently in docs and Section 7.

2. **Implement unified SourceLoader**  
   - Internal: list of "source" objects (each can load and optionally split); optional multisource splitter.  
   - N=1: one source, optional per-unit splitter; if splitter present, split after load; else no split (for use as child).  
   - N>1: load all; if multisource splitter, joint split; else split each and concatenate.  
   - Return: always SplitDatasetContainer when split was applied; otherwise DatasetContainer (single chunk).

3. **Migrate SingleSourceLoader and MultiSourceLoader**  
   - Option A: SingleSourceLoader and MultiSourceLoader become thin wrappers that construct the unified SourceLoader with 1 or N sources.  
   - Option B: Remove them and replace all usages with SourceLoader(sources=[...], ...).  
   - PreProcessor and any code that types `Union[SingleSourceLoader, MultiSourceLoader]` should use `SourceLoader` (or the new interface) only.

4. **Standardize .load() and .get()**  
   - All loaders: `load()` and `get_data()` (or `get()`); ensure docstrings and type hints are aligned.  
   - Abstract base: same method names and return type.

**Complexity:** Medium–High.  
**Risk:** Medium (many call sites; datamodules, run.py, simplified_run_objects). Mitigation: Phase 0 tests; backward-compat wrappers if needed.

---

### Phase 2: Object & Cache Modernization

**Goal:** Data objects as "smart carriers" aligned with MetadataManifest; cache keyed by stable data signature where appropriate; no transform-internal variables in the data layer.

**Tasks:**

1. **Data objects as smart carriers**  
   - DatasetContainer/SplitDatasetContainer: ensure `manifest` and `slice_info` are first-class; document that datasources may add manifest entries (category=datasource) and transforms add (category=transform).  
   - Optionally add a small helper to append a ManifestEntry from the datasource (e.g. `container.add_datasource_manifest_entry(entry)`).  
   - Keep `get_instance_cls()` for compatibility with transforms and MultiSource consistency check; document it as part of the data contract.

2. **Data signature for stage 1 cache**  
   - Already in place: `config = datasource config`, `library_dir = data_library_part_path`.  
   - Document that "data signature" = hash(datasource config + data library path); no transform config in stage 1.  
   - Ensure FileSystemCache and PreProcessor use this consistently; no references to transform internals in the data-layer cache module.

3. **Decouple cache from transform-internal variables**  
   - Cache module should not import ConfigTransformManager or DataTransform.  
   - PreProcessor (or a pipeline facade in preprocessing) is the only place that builds "preprocessed" config = datasource + transforms and calls cache.save/load.  
   - So: cache remains generic (save/load by key); the **key** is computed outside (in PreProcessor) from datasource + transform config. Data layer only needs to know "stage 1 = loaded/split data" and the key for that is data-only.

4. **PHMD and other loaders**  
   - PHMDMultiSourceLoader: either (a) implement the same SourceLoader interface (load/split no-op/get_data) and register as one "backend," or (b) keep as a special loader that returns SplitDatasetContainer and document it as pre-split.  
   - PreProcessor should depend on the interface (get_data() returns container), not on Single/Multi.

5. **Validation at data boundary (see §1.7):**  
   - After get_data() (in PreProcessor or SourceLoader), optionally call container.validate() and optionally run NaN/length checks on chosen keys using data-layer utils. Document as the single “gate” for raw data.

**Complexity:** Medium.  
**Risk:** Low–Medium (manifest/slice_info adoption is incremental).

---

### Phase 2.5: Validation & Integrity (Optional)

**Goal:** Use both data-layer checks and pipeline IntegrityCheckStep wisely; single implementation, clear “when”; simplification without losing functionality.

**Tasks:** See §1.7: (1) Injection point for integrity_checks (strategy or PreProcessor/DataTransform accepts optional list); (2) Built-in pipeline check helpers in integrity.py wrapping data-layer utils; (3) When integrity_checks is configured, use get_sanitized_data(check_nans=False, check_lengths=False) in chunk prep to avoid N×M redundancy; (4) Document responsibility matrix in REFACTORING_ROADMAP and this plan.

**Complexity:** Low–Medium.  
**Risk:** Low.

---

### Phase 3: Final Decoupling & Standards

**Goal:** No cross-layer leaks from data to transforms (except the agreed contract types); PEP8; strict type hints; updated docs.

**Tasks:**

1. **Remove data → transform imports**  
   - Move `find_singular_ragged_dim` to `lmetk.data.utils` or `lmetk.utils` (or a small `lmetk.data.optimization.ragged_utils`) and use it from both data and transforms, so that data no longer imports from transforms.  
   - PreProcessor remains in data and will still use ConfigTransformManager and DataTransform; that is the single "orchestration" boundary. So the only remaining data→transform imports should be in PreProcessor (and optionally a single pipeline facade).  
   - All other data subpackages (datasources, data_objects, cache, optimization) should not import from transforms.

2. **PEP8 and style**  
   - Fix: `_is_splitted` → `_is_split` (or keep name but fix docstring); `get_chached_transform_manager` → `get_cached_transform_manager`.  
   - Line length, naming (e.g. `data_dict` vs `data`), consistent use of type hints.

3. **Type hints**  
   - AbstractDataSourceLoader.get_data() → `DatasetContainer | SplitDatasetContainer`.  
   - All public loader and container methods: strict annotations; use `from __future__ import annotations` where helpful.

4. **Documentation**  
   - One-pager "Data flow" (see Section 7).  
   - Module-level docstrings for datasources, data_objects, cache; update any existing architecture docs to reference the unified loader and the clean interface (datasources provide raw data + metadata; PreProcessor applies transforms).

**Complexity:** Low–Medium.  
**Risk:** Low.

### Overall risks and mitigations

| Risk | Mitigation |
|------|------------|
| **Phase 1 breaks run.py / PreProcessor** | Phase 0 characterization tests; keep load_data/split_data names or add aliases; run.py can switch to get_split_mode() only after MultiSourceLoader (or unified loader) returns the correct mode. |
| **Cache restore loses fitted transforms** | Cache already stores saved_objects["transforms"]; PreProcessor and run.py rely on get_chached_transform_manager() after restore. No change to cache payload in this plan. |
| **Validation change weakens guarantees** | Keep data-layer checks as single implementation; add boundary validation once; make chunk-level optional only when pipeline integrity_checks are configured. PHM: use policy "fail". |
| **Unified loader hides N=1 vs N>1 bugs** | Phase 0 tests cover both N=1 and N>1 paths; internal branching stays explicit in one class; get_split_mode() (or equivalent) remains testable. |
| **find_singular_ragged_dim move breaks transforms** | Move to shared utils (lmetk.utils or lmetk.data.utils); both data and transforms import from the same place; no behavior change. |

---

## 4. Potential Approaches & Discussion

### 4.1 Data-Object Hierarchy

- **Current:** BaseDataObject → BaseDataObjectWithMetadata; DatasetContainer, SplitDatasetContainer; NamedTransformInput, ReturnObject variants.  
- **Options:**  
  - **Keep:** Retain hierarchy; add optional manifest/slice_info and document.  
  - **Collapse:** Merge DatasetContainer and SplitDatasetContainer into one class with an optional "split view" (e.g. `splits: Optional[Dict[str, ...]]`); more flexible but bigger change.  
  - **Rename:** Rename `NamedTransformInput` to something data-neutral (e.g. `DataChunk` or `UnitData`) in the data package and let transforms alias it; reduces transform-centric naming in data.

**Recommendation:** Keep the hierarchy for now; add clear documentation and optional manifest/slice_info. Rename `NamedTransformInput` in data to a neutral name (e.g. `UnitData`) and re-export for transforms to avoid data layer sounding transform-specific (see §6 Keep/Collapse/Cut).

### 4.2 Registry-Based Dispatching (Transforms) — Traceability

From the transforms audit: single-unit path has no registry (Facade → Strategy → Step → Transform instance). Multi-unit path goes through data_kind × capability → handler registry. So **data layer** doesn’t touch the registry; only transform layer does. For the data refactor we don’t change that. We only ensure that the **data contract** (container shape, get_instance_cls, optional manifest) is stable so that a developer tracing from PreProcessor → get_data() → container → transform.forward(container) can follow the flow. The "magic" stays inside transforms; data stays boring.

### 4.3 Cognitive Load & Indirection-to-Value

- **Unified loader:** One concept (SourceLoader) instead of two (Single vs Multi) plus special (PHMD) reduces "which loader do I use?"  
- **Clean interface:** "Datasources provide raw data + metadata; PreProcessor applies transforms" keeps the mental model simple.  
- **Cache:** Two stages (loaded vs preprocessed/boundary) and "data signature vs full pipeline signature" documented in one place keeps cache predictable.

If a simple operation required changing registry, strategy, and manifest in three places, that would be high overhead—but those live in **transforms**; the data refactor doesn’t add more indirection there. We only reduce branches in the **data** layer.

### 4.4 Abstraction Leaks

- **Data layer:** Data objects already accept `np.ndarray`, `ak.Array`, `pd.DataFrame`, etc.; `get_instance_cls()` and type checks are in data_objects and in transforms. The refactor doesn’t remove those; we document that the contract is "container with these types and optional manifest."  
- **Transforms:** As in the transforms audit, handlers and data_kind still know `ak.Array`; that’s an intentional leak. The data refactor doesn’t move those dependencies into "more obscure" places; we just avoid adding new ones from data to transforms.

### 4.5 Performance Overhead

- Metadata manifest: list of entries, queried by filter; negligible unless thousands of entries.  
- SliceInfo: small dataclass; negligible.  
- Unified loader: no extra copy of large arrays if we implement without redundant list wrapping.  
- Cache: same hash and I/O as today.  
No significant latency or memory pressure is expected from the refactor if we avoid deep copies and unnecessary wrappers.

---

## 5. Simplicity Score Card (Module-by-Module)

| Module | Score (1=Lean, 5=Overengineered) | Notes |
|--------|-----------------------------------|--------|
| **datasources/base/interfaces.py** | 2 | Thin ABC; docstring for split_data() describes MultiSourceLoader-style args but the abstract method has no parameters—fix in Phase 3. |
| **datasources/base/single_source_loader.py** | 3 | Branchy get_data and split logic; two modes (standalone vs part of multi). |
| **datasources/base/multi_source_loader.py** | 3 | Clear but duplicated state and lifecycle vs Single. |
| **datasources/base/phmd_loader.py** | 3 | Different paradigm (pre-split); many abstract methods. |
| **data_objects/base.py** | 2 | Solid dict-like base; type enforcement is useful. |
| **data_objects/data.py** | 3 | Many classes (NamedTransformInput, ReturnObject variants, DatasetContainer, Split); some dead/commented code. |
| **data_objects/manifest.py** | 1 | Lean dataclass + container. |
| **data_objects/slice_info.py** | 1 | Lean dataclass. |
| **cache/base.py** | 1 | Minimal ABC. |
| **cache/offline.py** | 2 | Clear stages and hash; slightly long. |
| **cache/online.py** | 1 | Simple stream cache. |
| **preprocessing/base.py** | 2 | Interface only. |
| **preprocessing/preprocessor.py** | 4 | Long pipeline(); tightly coupled to transforms and cache; many branches. |

**Overall:** Data layer is mostly 2–3; PreProcessor is the heaviest (4). Unifying loaders and clarifying the pipeline (without adding features) can bring datasources and PreProcessor down.

---

## 6. Keep, Collapse, or Cut

| Item | Recommendation | Rationale |
|------|----------------|-----------|
| **AbstractDataSourceLoader** | **Keep** | Essential; unify protocol (load/split/get_data) here. |
| **SingleSourceLoader / MultiSourceLoader** | **Collapse** | Merge into one SourceLoader (N=1 or N>1) to reduce duplication and branching. |
| **PHMDMultiSourceLoader** | **Keep** | Different backend (pre-split); implement same interface (load/split/get_data) so PreProcessor doesn’t special-case. |
| **DatasetContainer / SplitDatasetContainer** | **Keep** | Both are used; splitting is a real distinction. |
| **NamedTransformInput** | **Collapse** | Rename in data to neutral (e.g. UnitData) and re-export for transforms; one concept. |
| **ReturnObject variants** | **Keep** | Used by strategy/postprocess; minimal cost. |
| **MetadataManifest / SliceInfo** | **Keep** | Lean; enable future observability and slice-aware transforms. |
| **FileSystemCache stages** | **Keep** | loaded / boundary / preprocessed are useful. |
| **PreProcessor.pipeline()** | **Collapse** | Extract helpers (e.g. _try_load_preprocessed(), _run_stage1(), _run_stage2_with_boundary()) to shorten and clarify. |
| **find_singular_ragged_dim in transforms** | **Cut (from data)** | Move to shared utils so data doesn’t depend on transforms. |
| **Data-layer check utils (check_for_nans, check_length_consistency)** | **Keep** | Single implementation; used at data boundary and (via pipeline helpers) in IntegrityCheckStep. |
| **container.validate()** | **Keep + Wire** | Keep implementation; wire a single call after load in PreProcessor (or SourceLoader). |
| **IntegrityCheckStep / integrity_checks** | **Wire (transforms)** | Add injection point and built-in checks wrapping data utils; see §1.7. |
| **get_sanitized_data() in chunk prep** | **Collapse** | Make optional (default off when pipeline integrity checks are configured) to avoid N×M redundant checks. |

---

## 7. One-Page Developer Path: Data Flow in the New System

**Goal:** A new contributor can understand how data moves in under 30 minutes.

### 7.1 Current State (Simplified)

1. **Configuration:** A run specifies a **datasource** (single unit, multiple units, or PHMD) and a **transform pipeline** (list of transforms with apply_to, fit_on, etc.).
2. **PreProcessor** is created with a **loader** (Single/Multi/PHMD) and a **ConfigTransformManager**.
3. **Pipeline:**  
   - **Load:** `loader.load_data()` loads raw data into memory.  
   - **Split:** `loader.split_data()` splits by time/unit or uses predefined splits (or no-op for PHMD).  
   - **Fetch:** `loader.get_data()` returns a **DatasetContainer** or **SplitDatasetContainer** (splits → train/val/test, each a list of units).  
   - **Transform:** PreProcessor applies each DataTransform in order; each transform’s strategy runs (fit/transform) on the container and returns an updated container.  
   - **Cache (optional):** Stage 1 = loaded+split data (stage `"loaded_and_splitted_data"`; key = datasource config + data library path). Stage 2 = preprocessed (key = datasource + full transforms + library paths). Boundary stages allow resuming after a heavy transform.
4. **Output:** PreProcessor.get_processed_data_dict() (or .data) gives the final container; datamodules build PyTorch datasets from it.

### 7.2 After Refactor (Target)

- **One loader type (conceptually):** SourceLoader with 1 or N sources; same protocol: load_data() → split_data() → get_data() (or load()/split() if Phase 1 adopts the shorter names).  
- **Clean interface:** Datasources and loaders only deal with raw data and metadata (and optional manifest entries). They do not know about DataTransform, strategy, or registry.  
- **Single boundary:** PreProcessor (or one pipeline facade) is the only place that composes "data + transforms" and computes cache keys that include transform config.  
- **Data contract:** Containers have keys (e.g. features, target), per-split lists of units, optional manifest and slice_info, and get_instance_cls() for compatibility. Transforms consume and produce these containers.

If this one-pager is not enough for a new contributor, the next step is to add a short "Data layer" section to the project docs with a diagram: Config → SourceLoader → load/split → Container → PreProcessor → Transforms → Container → Cache (optional) → Datamodule/Datasets.

---

## 8. Summary

- **Findings:** Data layer couples to transforms only in PreProcessor and in optimization (find_singular_ragged_dim). Loaders are branchy and fragmented; cache is staged and already uses a data-only key for stage 1.  
- **Downstream usage (§1.6):** Data flows from datasource → load/split → get_data() → PreProcessor.fetch_data() (no validation today) → apply_transforms (each transform: CopyStep → IntegrityCheckStep no-op → Fit/TransformStep, which call get_sanitized_data() per unit). Container shape and get_instance_cls() are part of the contract with transforms; any refactor must preserve compatibility.  
- **Validation & integrity (§1.7):** Data layer has the only implementation (check_for_nans, check_length_consistency, container.validate()); transform pipeline has IntegrityCheckStep but no checks are ever injected. Proposal: **one implementation in data**, **data boundary** (validate once after load in PreProcessor), **pipeline integrity** (wire IntegrityCheckStep with built-in checks wrapping data utils), and **optional** get_sanitized_data() in chunk prep to avoid N×M redundant checks—simplification without losing functionality.  
- **Datasource metadata propagation (§1.8):** Datasource-level metadata (get_meta_data()) is stored in PreProcessor and passed only to **datasets** (run.py); it is **not** passed to the transform pipeline or to transform instances. Per-unit metadata in the container (data["metadata"][split][i]) is **reachable by transforms only if** apply_to includes `"metadata"` (then chunk["metadata"] is available). Transform instances always receive only **transform metadata** (mode, apply_to_keys, assign_to_map).  
- **Main usage (run.py, §1.9):** The main flow is: instantiate datasource and ConfigTransformManager → set preprocessor_mode from loader type → PreProcessor(mode, datasource, transforms_manager) → pipeline() → get_processed_data_dict() + get_meta_data_dict() → datasets + datamodule → trainer; after cache hit, transforms_manager is taken from preprocessor for inverse transforms. Refactor must preserve this API; run.py can be simplified to use `dataset_source.get_split_mode()` instead of isinstance checks once loader types are unified or get_split_mode() is guaranteed.  
- **Unified loader:** Yes—SingleSourceLoader can be the N=1 case of a unified SourceLoader; merging into one interface (and one implementation) simplifies the codebase without expected performance regression.  
- **Phases:** 0 (characterization tests: mock + real-data; snapshot script for phme20 and unibo21_bosello at loaded/transformed; comparison tests vs. slices; pipeline + caching) → 1 (unified SourceLoader and protocol) → 2 (data objects + cache cleanup + optional data-boundary validation) → 2.5 optional (validation & integrity wiring) → 3 (decoupling, PEP8, types, docs).  
- **Keep/Collapse/Cut:** Collapse Single/Multi into SourceLoader; collapse PreProcessor.pipeline into smaller helpers; move find_singular_ragged_dim out of data→transforms; keep containers, manifest, slice_info, cache stages, and data-layer check utils; wire container.validate() and IntegrityCheckStep; make get_sanitized_data() optional in chunk prep when pipeline checks are on.  
- **One-pager:** Load → Split → Get container → [optional validate at boundary] → PreProcessor applies transforms → optional cache → output container; datasources stay agnostic of transforms.

This plan keeps the data layer decoupled but "boring" and predictable, aligns with the existing transforms refactor, and uses both data-layer and pipeline integrity mechanisms wisely for a single, clear validation story.

**Re-validation (audit pass):** The plan was iterated for optimality and clarity: (1) Document status and terminology note added; (2) get_split_mode() caveat clarified (§1.9)—MultiSourceLoader must return "cross_unit" for run.py to switch safely; (3) Phase 0 extended with run.py contract test; (4) Phase 1 naming options made explicit (keep load_data/split_data vs. standardize to load/split); (5) Phase 2.5 and §1.7 chunk-preparation decision point clarified ("integrity_checks configured"); (6) Section 7 aligned with current vs. target method names; (7) Overall risks-and-mitigations table added; (8) Cross-references (e.g. §6 from §4.1, interfaces.py docstring) tightened.
