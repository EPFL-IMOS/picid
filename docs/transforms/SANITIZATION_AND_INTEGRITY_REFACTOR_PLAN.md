# Sanitization and Integrity Checks Refactor — Plan

This document outlines the phased changes required to:

1. Move sanitization out of `_prepare_chunks_for_split` into pipeline steps.
2. Run integrity checks **before** and **after** the transform, scoped to input and output data respectively.
3. Record in the manifest that integrity steps ran (before/after).
4. Migrate parts of `DataTransform._validate_forward_output` into pipeline integrity checks and/or `on_pipeline_end`.

**Constraints:** Integrity checks before transform must run only over data that participates in the current transformation (apply_to). Integrity checks after transform must run only over data produced by the transform (assign_to / transformed output). Between phases, run the transform and data-object test suites to ensure nothing is broken.

---

## Sanitization vs integrity — clarification

In this codebase, **“sanitization” is effectively input integrity checking.**  

`NamedTransformInput.get_sanitized_data()` does **not** modify or clean data (no imputation, no dropping rows). It only:

- Runs `check_for_nans(values, keys, ...)` and optionally `check_length_consistency(values, keys, ...)`.
- Returns the same object (or a copy); it does not change values.

So the current “sanitization” is **a set of integrity checks on input chunks** (no NaNs, consistent lengths across keys). There is no separate concept of “sanitize then integrity-check”: we can treat NaN + length as **default integrity-before checks** and run them in the pipeline over the apply_to data. That simplifies the design: we do not need a distinct “SanitizeStep” that produces “sanitized” chunks; we have **integrity checks before transform** that include, by default, these same checks. After they pass, we pass the same chunks to fit/transform (no need to call `get_sanitized_data()` in `_prepare_chunks_for_split`).

---

## Classic and potential integrity checks

A concise list of what “integrity” can mean, for both **before** and **after** transform.

**Input-side (before transform, over apply_to data):**

| Check | Description | Current location |
|-------|-------------|-------------------|
| No NaNs | Numeric arrays must not contain NaN (or policy: warn/allow). | `get_sanitized_data()` → `check_for_nans` |
| Length consistency | All arrays in a chunk (e.g. features, target) have compatible lengths. | `get_sanitized_data()` → `check_length_consistency` |
| No infinities | No ±inf in numeric arrays. | Not implemented; easy to add. |
| Value range | Values within expected bounds (e.g. [0, 1] for normalized). | Not implemented; domain-specific. |
| Shape / ndim | Expected number of dimensions or shape pattern. | Not implemented. |
| Dtype | Expected dtypes (e.g. float32, int). | Not implemented. |
| Required keys present | Chunk has all keys in apply_to. | Implicit (strategy builds chunks with those keys). |

**Output-side (after transform, over assign_to data):**

| Check | Description | Current location |
|-------|-------------|-------------------|
| Structure | Result is SplitDatasetContainer; assign_to keys present; values are split-keyed. | `DataTransform._validate_forward_output` |
| Split preservation | Set of splits unchanged (original vs transformed). | `_validate_forward_output` |
| No new NaNs | Transform did not introduce NaNs (if input had none). | Not implemented. |
| Length / shape | Per-split lengths or shapes as expected (e.g. same number of units). | Partially (warnings in _validate_forward_output). |
| Dtype / type | Output type consistent (e.g. still array-like). | Warning only in _validate_forward_output. |

**Structural (container-level, often after transform):**

- Type of `transformed_data` (e.g. SplitDatasetContainer).
- Each `transformed_data[key]` is split-keyed (train/val/test).
- Splits match between original and transformed for the keys that were transformed.

So: **default “sanitization” = default input integrity checks (NaN + length).** Other checks above can be added as optional or default integrity checks (before/after) as needed.

---

## Integrity checks: no return value, pipeline throws on failure

Integrity checks **do not return anything**. They only validate the data. If something is wrong, the check raises an exception; the pipeline strategy (e.g. policy `"fail"`) then propagates that error so that the pipeline throws and the error message outlines what is wrong (check name + exception message). So:

- Check functions are **void**: they run, and either complete (data is OK) or raise (data is invalid).
- The pipeline does not collect a “result” from checks; it just runs them. If a check fails with policy `"fail"`, the exception bubbles up and the pipeline fails with a clear error.
- No need for integrity checks to return a structure (passed/failed, message) for the manifest. Any “result” in the manifest can be inferred: if we reached RecordManifestStep, the integrity steps completed without raising (so we can record that they ran, and implicitly passed).

---

## Current State (Summary)

- **`strategy._prepare_chunks_for_split`** builds one `NamedTransformInput` per unit and calls `.get_sanitized_data()` on each. Sanitization today: NaN check and length-consistency check on the chunk’s values (with optional warn-only and size threshold).
- **IntegrityCheckStep** runs once after CopyStep, before FitStep. It runs `context.integrity_checks` (list of `(name, check_fn, policy)`). It has no notion of “which data” (apply_to vs full container).
- **DataTransform.forward** calls `strategy.apply()`, then optionally runs `_validate_forward_output` (structural checks: type, keys, splits, type/source-count warnings).
- **RecordManifestStep** appends a single manifest entry per transform with payload (apply_to_keys, assign_to_keys, fit_on_split, timings, etc.). No integrity result fields.

---

## Phase 1: Move “sanitization” into integrity-before (no integrity split yet)

**Goal:** Stop calling `.get_sanitized_data()` in `_prepare_chunks_for_split`. Run the same checks (NaN, length consistency) as **default integrity checks before transform** in the pipeline. FitStep and TransformStep then receive the same chunks (from `_prepare_chunks_for_split` without sanitization); the pipeline has already validated them via integrity-before.

**Clarification:** The strategy **keeps** `_prepare_chunks_for_split`; we only remove the call to `.get_sanitized_data()` from it. Chunk building (per split, per unit, from `context.data` and `apply_to_keys`) stays the same; validation is moved to pipeline integrity checks instead of being done inside that function.

Because sanitization is just validation (see above), we do **not** need to produce a separate “sanitized” copy of the data: we only need to run the checks over the apply_to chunks before Fit/Transform. Optionally we can still cache chunks in context after running checks to avoid building them twice (FitStep and TransformStep both need them); that is an optimization, not a requirement for correctness.

### 1.1 Where to sanitize

**Option A — New “SanitizeStep” after CopyStep, before IntegrityCheckStep**

- Add a step that, for the **current transform’s apply_to data only**, iterates over the same structure that `_prepare_chunks_for_split` would (per split, per unit), builds `NamedTransformInput` per chunk, calls `get_sanitized_data()`, and **writes back** sanitized arrays into `context.transformed_data` (or a dedicated buffer) so that when `_prepare_chunks_for_split` runs later it reads already-sanitized data.
- **Problem:** `_prepare_chunks_for_split` reads from `context.data`, not `context.transformed_data`. So we must either (i) write sanitized data back into `context.data` (mutates input), or (ii) have FitStep/TransformStep read from a different source (e.g. `context.sanitized_data`) and have the strategy use that in `_prepare_chunks_for_split` when present.

**Option B — Sanitize inside IntegrityCheckStep (before transform)**

- Rename or extend IntegrityCheckStep so that its first duty is “prepare and sanitize the slice of data that will be used for this transform.” It would:
  - Build chunks from `context.data` for `context.apply_to_keys` (same logic as `_prepare_chunks_for_split` but inside the step).
  - Call `get_sanitized_data()` on each chunk.
  - Store the list of sanitized chunks per split somewhere on context (e.g. `context._sanitized_chunks_by_split`).
- Then FitStep and TransformStep call a strategy method that returns `context._sanitized_chunks_by_split[split]` instead of calling `_prepare_chunks_for_split` when sanitization has been run (or always use sanitized cache when present).
- **Advantage:** Single place for “input preparation + validation” before transform. **Disadvantage:** IntegrityCheckStep becomes heavier and mixes “sanitize” with “run optional integrity checks.”

**Option C — Sanitize in CopyStep**

- CopyStep, after building `transformed_data`, also builds sanitized chunks for apply_to_keys and stores them on context. FitStep and TransformStep use those when calling the transform.
- **Advantage:** CopyStep already has the container and keys. **Disadvantage:** CopyStep grows; it currently does not know about “per-unit chunks” (that’s strategy’s job).

**Recommended for Phase 1:** Option B, implemented as a clear sub-step: first “sanitize apply_to data and cache,” then “run optional integrity_checks.” So the step could be split into two conceptual parts or two sub-steps; for simplicity we can keep one step that (1) populates sanitized chunks, (2) runs integrity_checks. Chunks are then read from context by FitStep/TransformStep.

**Concrete steps (Phase 1):**

1. Add on `TransformContext` something like `_sanitized_chunks_by_split: Optional[Dict[str, List[NamedTransformInput]]] = None`.
2. In the step that runs before FitStep (either a new SanitizeStep or the current IntegrityCheckStep):
   - For each split in `context.available_splits`, call the same logic as `_prepare_chunks_for_split` (using `context.data`, `context.apply_to_keys`) to build chunks, then call `chunk.get_sanitized_data()` for each chunk and store the result in `context._sanitized_chunks_by_split[split]`.
   - This requires access to `context.strategy._prepare_chunks_for_split` or inlining that logic (without `.get_sanitized_data()`) and then sanitizing the list.
3. Change `_prepare_chunks_for_split` to **not** call `.get_sanitized_data()`. When `context._sanitized_chunks_by_split` is set (e.g. when called from pipeline), FitStep/TransformStep should use `context._sanitized_chunks_by_split[split]` instead of calling `_prepare_chunks_for_split`. So either:
   - FitStep and TransformStep take a path that uses `context._sanitized_chunks_by_split` when non-None; or
   - Strategy has a method like `get_chunks_for_split(context, split)` that returns `context._sanitized_chunks_by_split[split]` if present, else `_prepare_chunks_for_split(...)` (and then we still need to sanitize in the step for the case when we’re in the pipeline).
4. Remove `.get_sanitized_data()` from the single place in `strategy._prepare_chunks_for_split` (line 227). Leave the rest of `_prepare_chunks_for_split` unchanged (it still builds and returns chunks; it just no longer sanitizes them).
5. Run tests: `test/transforms/base/test_strategy.py`, `test/data/data_objects/test_data_integrity.py`, and any integration tests that run the full pipeline. Ensure no regressions.

**Deliverable:** Sanitization happens in a pipeline step; chunks passed to fit/transform are sanitized; strategy no longer calls `get_sanitized_data()`.

---

## Phase 2: Scope “before” integrity checks to apply_to data and add “after” integrity

**Goal:** (1) Integrity checks that run before the transform see only the data that participates in the transform (apply_to). (2) Add a second integrity step that runs **after** the transform and sees only the transformed output (assign_to keys in `context.transformed_data`).

### 2.1 Before-transform integrity (current IntegrityCheckStep)

- **Current:** `IntegrityCheckStep.run(context)` runs `run_checks(context.integrity_checks)`. Check functions are no-arg; they typically close over something or read from context.
- **Desired:** Checks run only over the “current transform input” slice: i.e. `context.data` restricted to `context.apply_to_keys`, and optionally per-split or as a single concatenation depending on check semantics.
- **Options:**
  - **A)** Keep check_fns as no-arg; document that they must read `context.data` and `context.apply_to_keys` and restrict themselves to that slice. No API change; just documentation and possibly a helper that, given context, returns a view/slice of data for apply_to_keys.
  - **B)** Change the signature of “check” to receive context (or a small DTO with `data_slice`, `apply_to_keys`, `stage="before"`). Then IntegrityCheckStep passes that into each check. Existing checks would need to be adapted.
- **Recommendation:** A for minimal change. Optionally add a helper in `integrity.py` such as `get_apply_to_slice(context)` returning a structure (e.g. `{split: list of NamedTransformInput}` or a single concatenated structure) so custom checks can use it. Default integrity checks (if we add any) would use this slice.

### 2.2 After-transform integrity (new step or second run)

- **Placement:** After MergeStep so that `context.transformed_data` contains the final merged result. So order becomes: … → MergeStep → **IntegrityCheckAfterStep** (or a second invocation of integrity over “output”) → RecordManifestStep.
- **Scope:** Only keys that were written by this transform: `context.assign_to_keys`. Data to check: for each assign_to_key, `context.transformed_data[assign_to_key]` (split → list of arrays or single array).
- **Implementation options:**
  - **A)** New step `IntegrityCheckAfterStep` that runs a separate list `context.integrity_checks_after` (list of (name, check_fn, policy)). Each check_fn receives or closes over the “output slice” (assign_to data only).
  - **B)** Same `context.integrity_checks` list but run twice: once before FitStep (with a “stage” or data_slice = apply_to) and once after MergeStep (with data_slice = assign_to from transformed_data). Check_fns would need to accept a stage or data_slice argument.
  - **C)** Single step class that runs two lists: `integrity_checks_before` and `integrity_checks_after`, or a single list with a flag per entry (before/after/both).
- **Recommendation:** A or C. Prefer **C** with two lists on context: `integrity_checks_before` and `integrity_checks_after`. So we have one step class that runs “before” checks after sanitization (and before FitStep) and a second step class that runs “after” checks after MergeStep. Alternatively one step class that runs two phases when present. For clarity, two steps are easier to document: **IntegrityCheckBeforeStep** (after sanitization, before Fit) and **IntegrityCheckAfterStep** (after Merge, before RecordManifest).

**Concrete steps (Phase 2):**

1. Add `integrity_checks_after` on context (and optionally rename current `integrity_checks` to `integrity_checks_before` for symmetry, or keep name and add `_after`).
2. Add **IntegrityCheckAfterStep** that runs after MergeStep: runs `context.integrity_checks_after` if present, with check_fns that are expected to inspect only `context.transformed_data` restricted to `context.assign_to_keys`. Optionally provide a helper `get_assign_to_slice(context)`.
3. Update strategy’s pipeline order to: … → MergeStep → IntegrityCheckAfterStep → RecordManifestStep.
4. DataTransform / metadata: support `integrity_checks_after` in addition to `integrity_checks` (before). Strategy.apply() passes both into context.
5. Run tests again (transforms, pipeline, integrity tests). Ensure existing tests that build context with `integrity_checks` still pass; add a test that runs with `integrity_checks_after` only.

**Deliverable:** Before-transform checks are documented (and optionally helpers) to scope to apply_to; after-transform checks run in a dedicated step over assign_to output only.

---

## Phase 3: Record in manifest that integrity steps ran

**Goal:** Manifest entry for each transform records that the integrity-before and integrity-after steps were executed. Since integrity checks do not return anything and the pipeline throws on failure, “reaching RecordManifestStep” means all integrity checks that run with policy `"fail"` have passed. We only need to record that the steps ran (and optionally a brief state, e.g. “passed” or “no checks configured”).

### 3.1 What to store

- In `RecordManifestStep`, when appending the transform entry, add to the payload:
  - `integrity_before_run`: `true` if the integrity-before step ran (e.g. checks were configured and the step completed without raising). If no checks were configured, `false` or omit.
  - `integrity_after_run`: same for integrity-after.
  - **Check identifiers:** For traceability, also store the names/identifiers of the checks that ran: `integrity_before_checks` and `integrity_after_checks` (e.g. list of strings: check name, or `f"{check.__module__}.{check.__qualname__}"` for function-based checks). That way the manifest records both that the steps ran and *which* checks ran (e.g. `["check_nan", "check_length"]` or `["lmetk.transforms.base.integrity.check_structure_assign_to"]`).
- No need to store a “result” object from the checks: checks do not return. If a check failed, the pipeline already raised and we never reach RecordManifestStep.

### 3.2 How to record without return values

- IntegrityCheckBeforeStep and IntegrityCheckAfterStep keep using the existing `run_checks(checks)` (no return value). When the step finishes without raising, we know checks passed (for any with policy `"fail"`). To record “we ran” in the manifest, either:
  - **Option A:** Steps set a simple flag on context when they run, e.g. `context.integrity_before_ran = True` (set at the start of the step) and `context.integrity_after_ran = True`. RecordManifestStep reads these and adds `integrity_before_run`, `integrity_after_run` to the payload. If the step raised, we never set the flag for the next step, but we also never reach RecordManifestStep.
  - **Option B:** RecordManifestStep infers from context: if `integrity_checks_before` was non-empty, set `integrity_before_run: true` (we must have reached the step and it didn’t raise); same for after. This avoids extra context fields but is slightly less explicit.
- **Recommendation:** Option A — steps set `context.integrity_before_ran` / `context.integrity_after_ran` at the end of their `run()`, and set `context.integrity_before_check_names` / `context.integrity_after_check_names` to the list of check names (or class/function identifiers) that were run. RecordManifestStep adds these to the manifest payload. No need for `run_checks_with_result` or any return value from checks.

**Concrete steps (Phase 3):**

1. IntegrityCheckBeforeStep: at the end of `run()`, if it ran (and did not return early), set e.g. `context.integrity_before_ran = True` and `context.integrity_before_check_names = [name for (name, _, _) in checks]` (or equivalent from the check list). Same for IntegrityCheckAfterStep: `context.integrity_after_ran`, `context.integrity_after_check_names`.
2. RecordManifestStep: add to the transform entry payload `integrity_before_run`, `integrity_after_run`, `integrity_before_checks`, `integrity_after_checks` from these context fields (default `False` / `[]` if unset).
3. Run tests; ensure manifest entries in existing tests still work and new payload keys are present when the steps run.

**Deliverable:** Manifest records that integrity-before and integrity-after steps ran and which checks ran (by name/identifier), and implicitly passed if we reached RecordManifestStep.

---

## Phase 4: Migrate _validate_forward_output into pipeline and on_pipeline_end

**Goal:** Move structural and consistency checks from `DataTransform._validate_forward_output` into (a) integrity-after checks where they fit, and (b) optional lightweight checks or logging in `on_pipeline_end` or in a final step.

### Take on keeping _validate_forward_output in DataTransform

**Recommendation: do not keep structural validation in DataTransform.** Reasons:

- **Single source of truth:** Checks like `isinstance(transformed_data[key], SplitDatasetContainer)` are about pipeline output shape. If you later change the container type (e.g. a different dict wrapper or container class), you must update that check. Having it in `DataTransform._validate_forward_output` means every transform’s validation can break when the pipeline evolves, and you have to remember to change DataTransform instead of (or as well as) the pipeline. Putting the same checks in **integrity-after** means there is one place to update when the output contract changes.
- **Strategy changes:** If you swap in a different strategy, either (i) that strategy runs the same pipeline steps (including integrity-after), so validation still runs, or (ii) the strategy is custom and does not use the full pipeline — then the caller is responsible for running equivalent checks or accepting no structural validation. Duplicating validation in DataTransform “to help when strategy is changed” also duplicates the risk: the DataTransform check can become wrong (e.g. still checking for `SplitDatasetContainer`) while the pipeline has moved on, and then it can **break** the pipeline or give false positives/negatives.
- **Conclusion:** Migrate the full structural validation (type, keys, splits) into default integrity-after checks. Then **remove** `_validate_forward_output` or make it a no-op when the pipeline is used (e.g. when `validate_output` is True, DataTransform does nothing extra; the pipeline runs integrity-after). Optionally keep a minimal escape hatch (e.g. a single “transformed_data is not None” or “assign_to keys present” check) only if you explicitly want a safety net for non-pipeline callers; document it as such. The plan below assumes we **remove or no-op** the structural part of `_validate_forward_output` and rely entirely on integrity-after for structural checks.

### 4.1 What lives where

**Current `_validate_forward_output` (data_transform.py 310–380):**

- Type check: `transformed_data` is `SplitDatasetContainer` → **Integrity after** (structural check on output).
- Keys present: each `assign_to` key in `transformed_data` → **Integrity after**.
- Type of values: each `transformed_data[key]` is SplitDatasetContainer-like (dict of splits) → **Integrity after**.
- Split keys match original: `transformed_data[apply_to_key].keys() == data[apply_to_key].keys()` → **Integrity after** (or after-step that has access to both original and transformed).
- Warnings: type change per split, source-count change → can stay as **DataTransform** post-return checks with `validate_output: true`, or move to **on_pipeline_end** hook / a small “consistency report” step that only warns (no raise).

**Proposal:**

- **IntegrityCheckAfterStep (default checks):** Add a small set of “default” structural checks that run when no custom `integrity_checks_after` is provided (or in addition): (1) transformed_data is SplitDatasetContainer, (2) each assign_to key present, (3) each value is split-keyed, (4) split set matches original for the keys that were transformed. These can be implemented as check_fns that read context and raise on failure, and be registered as default “after” checks (e.g. from DataTransform metadata or from strategy when building context).
- **DataTransform.forward:** When `validate_output` is True, **do not** run structural checks in DataTransform; rely on pipeline integrity-after. Optionally keep only the “warning” part (type change per split, source-count change) and call it after strategy returns, or move those warnings to an `on_pipeline_end` hook. Do not keep `isinstance(..., SplitDatasetContainer)` (or similar) in DataTransform — that belongs in integrity-after so the contract can evolve in one place.
- **on_pipeline_end:** Optionally, a default hook that logs or attaches to context a one-line summary of validation (e.g. “structural checks passed”) so that consumers can rely on it without parsing the manifest. This can be a brief state message that is also written into the manifest.

### 4.2 Default integrity checks

- Introduce “default” before-checks: e.g. none, or reuse sanitization outcome (sanitization already ensures NaN/length; we can record that the integrity-before step ran when it completes).
- Default after-checks: the structural checks above (type, keys, splits). They can live in `integrity.py` as named functions that take context (or a small DTO) and are added to `integrity_checks_after` by default when building the pipeline (e.g. in strategy or DataTransform when metadata does not disable them).

**Concrete steps (Phase 4):**

1. Add in `integrity.py` (or pipeline) helper functions that implement the structural checks currently in `_validate_forward_output`, taking context (or transformed_data + assign_to_keys + original data for split comparison).
2. When building context, if metadata does not set `validate_output: false`, add these as default `integrity_checks_after` entries (policy "fail"). So the pipeline always runs them unless the user opts out.
3. **Remove or no-op** `DataTransform._validate_forward_output` for structural checks: type (SplitDatasetContainer), keys present, split-keyed values, split match. All of that is now in default integrity-after. Keep only the warning-only parts (type change per split, source count) in DataTransform if desired, or move those to a small “warn only” step or on_pipeline_end; otherwise remove `_validate_forward_output` entirely when `validate_output` is True (pipeline is the single source of structural validation).
4. Optionally add a default hook for `on_pipeline_end` that sets `context.validation_summary` or similar for the manifest.
5. RecordManifestStep: already extended in Phase 3 (integrity_after_run, integrity_after_checks); ensure default structural checks are included in the list of checks so their names appear in `integrity_after_checks`.
6. Run full transform and pipeline tests; ensure `validate_output: true` (default) still enforces the same guarantees and that `validate_output: false` skips both DataTransform validation and (if we tie it) default integrity-after.

**Deliverable:** Structural validation is performed in the pipeline (integrity-after); DataTransform no longer duplicates it; optional warnings or summary in on_pipeline_end/manifest.

---

## Phase 5: Cleanup and documentation

- Remove any dead code (e.g. duplicate checks in DataTransform).
- Update docs: transform guide and developer guide to describe (1) sanitization step and when it runs, (2) integrity before (apply_to scope) and after (assign_to scope), (3) manifest fields `integrity_before_run`, `integrity_after_run`, `integrity_before_checks`, `integrity_after_checks`, (4) `validate_output` and default checks.
- Add a short “Validation and integrity” subsection or update the existing one to reflect the two-phase integrity and manifest reporting.
- Run full test suite (transforms, data objects, pipeline, integration) and fix any remaining failures.

---

## Summary: Pipeline order after refactor

1. CopyStep  
2. SanitizeStep (or first part of IntegrityCheckBeforeStep): build and sanitize apply_to chunks; store in context.  
3. IntegrityCheckBeforeStep: run `integrity_checks_before` over apply_to slice; set `context.integrity_before_ran = True` when step completes.  
4. FitStep (uses sanitized chunks from context).  
5. TransformStep (uses sanitized chunks from context).  
6. PostprocessStep  
7. MergeStep  
8. IntegrityCheckAfterStep: run `integrity_checks_after` over assign_to slice in transformed_data; set `context.integrity_after_ran = True` when step completes.  
9. RecordManifestStep (payload includes integrity_before_run, integrity_after_run, integrity_before_checks, integrity_after_checks).  

---

## Advantages and disadvantages (draft solution)

**Advantages**

- Sanitization is explicit in the pipeline and testable in isolation; chunks are sanitized once per transform application.  
- Integrity is clearly “before” (input) and “after” (output), each scoped to the relevant keys.  
- Manifest records that integrity steps ran, improving traceability and debugging.  
- Structural validation lives in one place (integrity-after) instead of being duplicated in DataTransform.  
- Default behaviour (sanitize + default structural checks) remains “safe”; opt-out via metadata is explicit.

**Disadvantages**

- More steps and context fields; pipeline and context become slightly more complex.  
- FitStep and TransformStep must agree on where to get chunks from (context cache vs _prepare_chunks_for_split); need to avoid bugs when context is built by hand in tests (no sanitization step run).  
- Two integrity lists (before/after) and two run flags on context to document and maintain.  
- Backward compatibility: existing code that passes `integrity_checks` only must be mapped to `integrity_checks_before` (or we keep `integrity_checks` as alias for before).

---

## Test checkpoints (between phases)

- **After Phase 1:** `pytest test/transforms/base/test_strategy.py test/data/data_objects/test_data_integrity.py test/transforms/base/test_integrity.py test/transforms/base/test_data_transform.py -v`  
- **After Phase 2:** Same plus any pipeline tests that run the full step sequence.  
- **After Phase 3:** Same plus tests that assert on manifest payload (add or extend tests for integrity_before/after).  
- **After Phase 4:** Full transform and pipeline test suite; sanity-run one end-to-end experiment if available.  
- **After Phase 5:** Full project test suite and doc build if applicable.
