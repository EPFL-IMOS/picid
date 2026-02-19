# Audit: Unused and Old Code (Framework, Configs, Docs)

This document lists files and folders in the TabPHM framework (`lmetk`), configs, and docs that contain **old, unused, or dead code**, or configs that reference **non-existent or misplaced** code. Use it to prioritize cleanup, refactors, or migration.

**Scope:** `lmetk/`, `configs/`, `docs/`.  
**Method:** Cross-check of imports, `_target_` references in configs, and explicit “old” / “not_used” / “_archive” layout.

---

## 1. Transform base: `*_new.py` (draft / not wired)

| Location | Status | Notes |
|----------|--------|--------|
| `lmetk/transforms/base/data_transform_new.py` | **Unused** | No imports anywhere. `transform_manager` uses `data_transform.DataTransform` only. |
| `lmetk/transforms/base/strategy_new.py` | **Unused** | No imports. Imports from `pipeline` (current), not `pipeline_new`. |
| `lmetk/transforms/base/pipeline_new.py` | **Unused** | No imports. Current pipeline used by `strategy.py` and `transform_manager`. |

**Recommendation:** Either complete the migration described in `docs/transforms/transform_migration.md` and switch the runtime to the `*_new` chain, or remove/archive the `*_new` files to avoid confusion. The migration plan is already documented.

---

## 2. Explicit “old” and “not_used” code

### 2.1 Transforms

| Path | Referenced from config? | Notes |
|------|-------------------------|--------|
| `lmetk/transforms/bearings/old/statistics.py` | No | Never referenced in configs or other code. |
| `lmetk/transforms/bearings/old/simple_stat.py` | No | Same. |
| `lmetk/transforms/bearings/old/filter_target.py` | No | Same. |
| `lmetk/transforms/bearings/old/cut_till_first_zero.py` | **Yes, but wrong path** | Config uses `lmetk.transforms.bearings.cut_till_first_zero.RULTaskUpToFirstZero`; implementation lives under `bearings/old/`, so correct path is `lmetk.transforms.bearings.old.cut_till_first_zero`. The current config will **fail at runtime** when loading `bearings/pronostia/residual_autoencoder` (used by `experiment/pronostia/prognostics/autoencoder/stf.yaml`). |
| `lmetk/transforms/not_used/autocorr_drop.py` | No | Never referenced. |

**Recommendation:**  
- Fix config: in `configs/transforms/bearings/pronostia/residual_autoencoder.yaml`, set `_target_` to `lmetk.transforms.bearings.old.cut_till_first_zero.RULTaskUpToFirstZero`, **or** move `cut_till_first_zero.py` out of `old/` and keep the current `_target_`.  
- Consider removing or archiving the other `bearings/old/*` and `not_used/autocorr_drop.py` if no longer needed.

---

### 2.2 Configs: “old” and “old_but_keep”

| Path | Notes |
|------|--------|
| `configs/datasource/old/nb14.yaml` | Uses `PHMDBatteryLoader`. Active nb14 experiments use `configs/datasource/nb14_bosello.yaml` (NB14Loader). Likely legacy. |
| `configs/datasource/old/nb1.yaml` | Uses `PHMDBatteryLoader`. nb1 experiments under `_archive` use `datasource: nb1` (resolves to this or a non-old nb1). |
| `configs/datasource/old/unibo21_battery.yaml` | Uses `PHMDBatteryLoader`. Unibo experiments use `unibo21_bosello` (UNIBO21Loader), not this. |
| `configs/transforms/old_but_keep/bosello23_base_satistics.yaml` | References `lmetk.transforms.battery.nb1_transforms.*`. Kept by name; confirm if any experiment still uses it. |
| `configs/transforms/old_but_keep/bosello_fit_predict.yaml` | Same idea; confirm usage. |
| `configs/transforms/old_but_keep/default_residual_autoencoder.yaml` | Same; confirm usage. |

**Recommendation:** If no experiment references these, move to a single `configs/_archive/...` or delete. If kept, add a one-line comment in each file that they are legacy.

---

### 2.3 Configs: `_archive`

| Path | Notes |
|------|--------|
| `configs/experiment/_archive/` | 40 experiment configs (nb1, etth, picasso, cbmv3, concepts_n_cmapss_ds03_da, movi_spheres, etc.). By name, these are archived and not part of the main experiment surface. |

**Recommendation:** Leave as-is for history, or document in `docs/` that `_archive` is excluded from the default experiment list/CI.

---

## 3. Model configs pointing to non-existent code

| Config | `_target_` | Issue |
|--------|------------|--------|
| `configs/model/persistence.yaml` | `lmetk.models.persistence.Persistence` | Package `lmetk.models` (plural) does **not** exist. No class `Persistence` in the repo. |
| `configs/model/similar_period.yaml` | `lmetk.models.similar_period.SimilarPeriod` | Same: no `lmetk.models` and no `SimilarPeriod` class. |

**Recommendation:** Remove these two model configs or implement the missing modules under `lmetk.model` (singular) and fix the `_target_` to the real path (e.g. `lmetk.model.methods.persistence.Persistence` if you add it there). Until then, any run that selects `model=persistence` or `model=similar_period` will fail at instantiation.

---

## 4. Synthetic data and optional entry points

| Path | Referenced from configs? | Notes |
|------|--------------------------|--------|
| `lmetk/synthetic_data/` (tabicl_style, timepfn_style) | No | No experiment or transform config references these. Used programmatically (e.g. `genload.py`). |
| `lmetk/simplified_run.py` + `lmetk/simplified_run_objects.py` | Yes (code only) | Alternative entry point; not referenced from main `run.yaml` or docs quick start. |
| `lmetk/verify_environment.py` | No | Not referenced in configs or main run flow. |

**Recommendation:** Treat `synthetic_data` as optional tooling; document in `docs/` if it’s part of the supported surface. For `simplified_run*`, either document as an alternative entry point or deprecate. `verify_environment` can stay as a dev/CI script; no need to remove.

---

## 5. Summary table (actions)

| Category | Action |
|----------|--------|
| **Transform `*_new`** | Migrate and switch, or remove/archive `data_transform_new`, `strategy_new`, `pipeline_new`. |
| **bearings/old** | Fix `cut_till_first_zero` config path or move module; consider removing other `old/` transform files if unused. |
| **not_used/autocorr_drop** | Remove or move to `_archive` if not needed. |
| **configs/datasource/old/** | Archive or delete if no experiment uses them. |
| **configs/transforms/old_but_keep/** | Confirm no usage; then archive or delete. |
| **configs/model persistence & similar_period** | Fix or remove; implement missing `lmetk.models.*` or point to correct `lmetk.model.*` and class. |
| **configs/experiment/_archive/** | Document as archived; exclude from default experiment discovery if applicable. |

---

## 6. References

- **Transform migration:** `docs/transforms/transform_migration.md`  
- **Sanitization / integrity refactor:** `docs/transforms/SANITIZATION_AND_INTEGRITY_REFACTOR_PLAN.md`  
- **Transform manager** (only uses non-`_new`): `lmetk/transforms/base/transform_manager.py` → `DataTransform` from `data_transform.py`, strategy from `strategy.py`, pipeline from `pipeline.py`.
