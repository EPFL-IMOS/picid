# Multi-Source Mixin Renaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace confusing mixin names (`StatelessPerSegmentMixin`, `StandardFitTransformMixin`, `StatelessConcatenateMixin`) with clearer alternatives and optionally update backward-compat aliases.

**Architecture:** Rename the three composition mixins in `picid/transforms/base/multisource/mixins.py`, update all imports and references across the codebase and docs, and keep (or adjust) existing aliases so configs and tests that use old names keep working.

**Tech Stack:** Python, pytest, existing transform base classes.

---

## 1. Naming alternatives (choose one per mixin)

### 1.1 Current: `StatelessPerSegmentMixin`

**Behavior:** No fit; transform each segment independently (NoFit + PerSegment).

| Alternative | Rationale |
|-------------|-----------|
| **`NoFitPerSegmentMixin`** | Describes composition literally: no fit + per-segment transform. Clear and consistent with `FitByConcatenationMixin` / `PerSegmentTransformMixin`. |
| **`TransformOnlyPerSegmentMixin`** | Emphasizes “transform only” (no fit) and “per segment.” Slightly longer. |
| **`PerSegmentOnlyMixin`** | Short; “only” hints no fit. May be read as “only per segment” (exclusive) rather than “no fit, per segment.” |
| **`ElementWiseMultiSourceMixin`** | “Element-wise” over segments. Familiar to numpy users; doesn’t say “no fit” explicitly. |
| **`IndependentSegmentTransformMixin`** | Stresses that each segment is transformed independently. Longer; “stateless” is implicit. |

**Recommendation:** **`NoFitPerSegmentMixin`** — clearest and aligns with existing `NoFitMixin` / `PerSegmentTransformMixin` naming.

---

### 1.2 Current: `StandardFitTransformMixin`

**Behavior:** Fit on all segments concatenated; transform each segment (FitByConcatenation + PerSegment). Common for scalers and fittable per-unit transforms.

| Alternative | Rationale |
|-------------|-----------|
| **`FitConcatTransformPerSegmentMixin`** | Explicit: fit on concatenated data, transform per segment. Long but unambiguous. |
| **`FitOnConcatTransformPerSegmentMixin`** | Same idea, “fit on concat” reads naturally. |
| **`FittablePerSegmentMixin`** | Short; “fittable” implies fit then transform; “per segment” for transform. Doesn’t spell out “fit on concatenated.” |
| **`GlobalFitPerSegmentTransformMixin`** | “Global fit” = one fit over all data (concatenated); “per segment” = transform. Clear for “fit once, apply per unit.” |
| **`ConcatFitAndPerSegmentTransformMixin`** | Explicit: fit on concatenated data, then transform per segment. Reads naturally; “Concat” + “Fit” + “And” + “PerSegment” + “Transform”. |
| **`ConcatenatedFitPerSegmentMixin`** | Matches current alias `ConcatinatedFitMultiSourceTransformMixin` (minus typo) and describes behavior. |
| **`FitAllThenPerSegmentMixin`** | Very explicit: fit on all, then transform per segment. Readable. |

**Recommendation:** **`ConcatFitAndPerSegmentTransformMixin`** — explicit about the mechanism (fit on concatenated data, then per-segment transform) and reads clearly. Alternative: **`GlobalFitPerSegmentTransformMixin`** if you prefer a shorter, domain-friendly name.

---

### 1.3 Current: `StatelessConcatenateMixin`

**Behavior:** No fit; concatenate all segments along an axis, then one `transform_data` call. Subclass must set `axis` in `__init__`.

| Alternative | Rationale |
|-------------|-----------|
| **`NoFitConcatenateThenTransformMixin`** | Literal: no fit, concatenate, then single transform. Long. |
| **`ConcatenateAlongAxisMixin`** | Short; “along axis” matches `axis` in API. Doesn’t say “no fit” or “then one transform.” |
| **`NoFitConcatAlongAxisMixin`** | No fit + concat along axis. Clear; “transform once” implied by base. |
| **`MergeThenTransformMixin`** | “Merge” = concatenate; “then transform” = one call. Doesn’t mention axis or no-fit. |
| **`ConcatenateThenTransformMixin`** | Clear sequence. Ambiguous whether there is a fit step; “stateless” not in name. |
| **`AxisConcatenateTransformMixin`** | Highlights axis and concatenation. “No fit” implicit from behavior. |

**Recommendation:** **`NoFitConcatAlongAxisMixin`** or **`ConcatenateAlongAxisMixin`**. Use **`NoFitConcatAlongAxisMixin`** if you want “no fit” explicit; **`ConcatenateAlongAxisMixin`** if you prefer brevity and are okay with no-fit being documented rather than in the name.

---

## 2. Summary table (recommended choices)

| Current | Recommended new name | Optional shorter |
|--------|-----------------------|------------------|
| `StatelessPerSegmentMixin` | `NoFitPerSegmentMixin` | — |
| `StandardFitTransformMixin` | `ConcatFitAndPerSegmentTransformMixin` | `GlobalFitPerSegmentTransformMixin` |
| `StatelessConcatenateMixin` | `NoFitConcatAlongAxisMixin` | `ConcatenateAlongAxisMixin` |

---

## 3. Implementation tasks (after names are chosen)

Assume renames:

- `StatelessPerSegmentMixin` → `NoFitPerSegmentMixin`
- `StandardFitTransformMixin` → `ConcatFitAndPerSegmentTransformMixin`
- `StatelessConcatenateMixin` → `NoFitConcatAlongAxisMixin`

(Replace with your chosen alternatives if different.)

### Task 1: Add new names and keep old names as aliases

**Files:**
- Modify: `picid/transforms/base/multisource/mixins.py`

**Steps:**

1. Rename the three class definitions to the new names.
2. Add backward-compat aliases at the bottom so existing code keeps working, e.g.:
   - `StatelessPerSegmentMixin = NoFitPerSegmentMixin`
   - `StandardFitTransformMixin = ConcatFitAndPerSegmentTransformMixin`
   - `StatelessConcatenateMixin = NoFitConcatAlongAxisMixin`
3. Keep existing aliases (`NoFitMultiSourceTransformMixin`, `ConcatinatedFitMultiSourceTransformMixin`, `ConcatenateUnitsMixing`, etc.) pointing to the **new** class names (so one canonical name).
4. Update the module docstring at the top of `mixins.py` to use the new names in the composition list.

**Commit:** `refactor(mixins): add clearer mixin names and keep old names as aliases`

---

### Task 2: Export new names from package

**Files:**
- Modify: `picid/transforms/base/multisource/__init__.py`

**Steps:**

1. Import the new class names from `mixins`.
2. Add them to `__all__` (and keep old names in exports if you still want them public).
3. Optionally re-export old names as aliases only (no need to add old names to `__all__` if deprecating).

**Commit:** `refactor(multisource): export new mixin names`

---

### Task 3: Switch codebase to new names (optional, can be gradual)

**Files (imports / inheritance):**

- `picid/transforms/base_transforms/spectral.py`
- `picid/transforms/base_transforms/moving_average.py`
- `picid/transforms/base_transforms/scaler.py`
- `picid/transforms/base_transforms/reshaping.py`
- `picid/transforms/base_transforms/identity.py`
- `picid/transforms/base_transforms/stfft.py`
- `picid/transforms/base_transforms/time_statistics.py`
- `picid/transforms/base_transforms/subsample.py`
- `picid/transforms/base_transforms/shape_manipulations.py`
- `picid/transforms/base_transforms/concatenate.py`
- `picid/transforms/base_transforms/data_cleaning.py`
- `picid/transforms/base_transforms/padding2length.py`
- `picid/transforms/base_transforms/autoencoder_transform.py`
- `picid/transforms/base_transforms/plot_transforms.py`
- `picid/transforms/base_transforms/tabularizers.py`
- `picid/transforms/signal_processing/cumsum.py`
- `picid/transforms/signal_processing/rms.py`
- `picid/transforms/n_cmapss/n_cmapss_scalers.py`
- `picid/transforms/n_cmapss/concept_classes_builder.py`
- `picid/transforms/battery/sequence2statistics.py`
- `picid/transforms/battery/nb1_transforms.py`
- `picid/transforms/bearings/health_index.py`
- `picid/transforms/bearings/health_models/li2019.py`
- `picid/transforms/bearings/health_models/ahmad2019.py`
- `picid/transforms/bearings/old/cut_till_first_zero.py`
- `picid/transforms/bearings/old/simple_stat.py`
- `picid/transforms/bearings/old/statistics.py`
- `picid/transforms/bearings/old/filter_target.py`
- `picid/transforms/building/mzvav_scaler.py`
- `picid/transforms/building/rugged_dimension_swap.py`
- `picid/transforms/time_transforms/sequence2table.py`
- `picid/transforms/visualization/features_target_plotter.py`
- `picid/transforms/rigid_body_dynamics/transforms.py`
- `picid/transforms/base/base_transform/base_transform.py` (comment only)

**Steps:**

1. In each file, replace:
- `StatelessPerSegmentMixin` → `NoFitPerSegmentMixin`
- `StandardFitTransformMixin` → `ConcatFitAndPerSegmentTransformMixin`
- `StatelessConcatenateMixin` → `NoFitConcatAlongAxisMixin`
2. In `concatenate.py`, update `StatelessConcatenateMixin.__init__(self, axis=axis, **kwargs)` to `NoFitConcatAlongAxisMixin.__init__(self, axis=axis, **kwargs)` (or chosen name).
3. Run tests: `pytest test/transforms/ -v --tb=short` (or project’s test command). Expect: PASS (aliases preserve behavior).

**Commit:** `refactor(transforms): use new multisource mixin names in picid`

---

### Task 4: Update tests to new names (optional)

**Files:**
- Modify: `test/transforms/base/test_multisource.py`
- Modify: `test/transforms/base/test_coverage_expansion.py`
- Modify: `test/transforms/base/test_phase1_stabilization.py`
- Modify: `test/transforms/base/conftest.py`
- Modify: `test/transforms/base/test_strategy.py`
- Modify: `test/transforms/base/test_pipeline.py`

**Steps:**

1. Replace old mixin names with new names in imports and class bodies (and in test class names / docstrings if they reference the mixin).
2. Tests that still use aliases (e.g. `NoFitMultiSourceTransformMixin`) can be left as-is or updated to new names for consistency.
3. Run: `pytest test/transforms/base/ -v --tb=short`. Expect: PASS.

**Commit:** `test(transforms): use new mixin names in tests`

---

### Task 5: Update documentation

**Files:**
- Modify: `docs/transforms/developer_guide.md`
- Modify: `docs/transforms/intro_developer.md`
- Modify: `docs/reference/transforms.md`
- Modify: `picid/transforms/base/base_transform/base_transform.py` (docstring/comment if it references mixins)

**Steps:**

1. Replace all mentions of `StandardFitTransformMixin`, `StatelessPerSegmentMixin`, and `StatelessConcatenateMixin` with the new names.
2. In tables and “behavior” descriptions, use the new names as primary; mention old names as “formerly X” or “alias: X” where helpful.
3. Fix typo in docs: `ConcatinatedFitMultiSourceTransformMixin` → note it as legacy alias of the new “fit-concat-per-segment” mixin.

**Commit:** `docs: update transform docs to new mixin names`

---

### Task 6: Optional — deprecation warnings for old names

**Files:**
- Modify: `picid/transforms/base/multisource/mixins.py`

**Steps:**

1. Create wrapper classes or use `warnings.warn(DeprecationWarning, ...)` when old names are used (e.g. when `StatelessPerSegmentMixin` is referenced). Alternatively, only document in release notes that old names are deprecated and will be removed in a future version.
2. Run tests again; if using warnings, consider filtering in tests or updating remaining usages.

**Commit:** `chore(mixins): deprecate old mixin names (optional)`

---

## 4. Verification

- Grep for `StatelessPerSegmentMixin`, `StandardFitTransformMixin`, `StatelessConcatenateMixin` (except in `mixins.py` aliases and deprecation) — should be zero in code, or only in docs as “formerly”.
- Full transform test suite passes.
- Docs build (if applicable) and read correctly.

---

## 5. Execution options

**Plan complete and saved to `docs/plans/2026-02-20-multisource-mixin-renaming.md`.**

Two execution options:

1. **Subagent-Driven (this session)** — One subagent per task, review between tasks, fast iteration.
2. **Parallel Session (separate)** — New session with executing-plans, batch execution with checkpoints.

**Which approach?**

If Subagent-Driven is chosen, use superpowers:subagent-driven-development. If Parallel Session is chosen, open a new session in the worktree and use superpowers:executing-plans.
