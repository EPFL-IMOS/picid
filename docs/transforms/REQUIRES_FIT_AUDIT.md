# Audit: Transforms with `fit_data` / `fit_multi_source` and Mixin Inheritance

**Purpose:** Ensure every class that overrides `fit_data` or `fit_multi_source` inherits from `ConcatinatedFitMultiSourceTransformMixin` (fittable) or `NoFitMultiSourceTransformMixin` (stateless). The mixins set `requires_fit = True` or `False` so the pipeline can raise a clear error when a fittable transform is used without `fit_on` / `fit_on_key`.

**Scope:**  
`lmetk/transforms/base_transforms`, `battery`, `bearings`, `building`, `n_cmapss`, `not_used`, `railway`, `rigid_body_dynamics`, `signal_processing`, `time_transforms`, `visualization`.

---

## Summary

| Status | Count | Meaning |
|--------|--------|--------|
| **OK** | All listed below with ✓ | Overrides fit and has correct mixin (ConcatinatedFit for fittable, NoFit for stateless no-op fit). |
| **MISSING_MIXIN** | 1 | Overrides `fit_multi_source` but does not inherit from `ConcatinatedFitMultiSourceTransformMixin`. |
| **NEEDS_REVIEW** | 1 | Overrides `fit_data` with non-trivial logic but uses `NoFitMultiSourceTransformMixin` (possible inconsistency). |

---

## base_transforms

| File | Class | Overrides | Inherits from mixin? | Status |
|------|--------|-----------|----------------------|--------|
| scaler.py | ConstantScaler | fit_data | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| scaler.py | MinMaxScalerSklearn | fit_data | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| scaler.py | StandardScalerSklearn | fit_data | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| plot_transforms.py | PlotFeatures | fit_data | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| identity.py | IdentityPassThrough | fit_data | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| padding2length.py | PadToLength | fit_data (pass) | NoFitMultiSourceTransformMixin | ✓ OK |
| autoencoder_transform.py | BaseAutoencoderTransform | fit_data (pass) | NoFitMultiSourceTransformMixin | ✓ OK |
| data_cleaning.py | FillTransform | — | ConcatinatedFitMultiSourceTransformMixin | ✓ OK (no override, uses base fit_data) |
| concatenate.py | ConcatenateTransform | — | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| concatenate.py | RuggedToDenseTransform | — | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| concatenate.py | MultiDatasetRuggedToDenseTransform | — | ConcatenateUnitsMixing (fit raises NotImplemented) | ✓ OK |
| stfft.py | STFTTransform | — | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| tabularizers.py | TimeseriesTabularizer | — | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| shape_manipulations.py | RegularizeRaggedDataTransform | — | NoFitMultiSourceTransformMixin | ✓ OK |
| shape_manipulations.py | ExpandScalarToReferenceFeatureSize | — | NoFitMultiSourceTransformMixin | ✓ OK |
| shape_manipulations.py | RaggedToDenseTransform | — | NoFitMultiSourceTransformMixin | ✓ OK |
| All other base_transforms (time_statistics, subsample, spectral, reshaping, moving_average, timefeatures) | — | no fit override | NoFitMultiSourceTransformMixin where used | ✓ OK |

---

## battery

| File | Class | Overrides | Inherits from mixin? | Status |
|------|--------|-----------|----------------------|--------|
| nb1_transforms.py | FilterDataFrameColumns | fit_data (pass) | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| nb1_transforms.py | UpdateTarget | fit_data (pass) | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| nb1_transforms.py | AssignNaiveTimefeatures | fit_data commented out | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| nb1_transforms.py | Cycles2Statistics | fit_data (with logic) | NoFitMultiSourceTransformMixin | **NEEDS_REVIEW** (has fitting logic but declares NoFit) |
| sequence2statistics.py | Sequence2Statistics | fit_data (pass) | NoFitMultiSourceTransformMixin | ✓ OK |
| battery_transform.py | BatteryTransform | — | DenseTransform only (no multi-source) | ✓ N/A |
| autoencoder_transform.py | AutoencoderTransform | — | extends BaseAutoencoderTransform (NoFit) | ✓ OK |

---

## bearings

| File | Class | Overrides | Inherits from mixin? | Status |
|------|--------|-----------|----------------------|--------|
| old/statistics.py | StatisticsBearingsTransform | — | NoFitMultiSourceTransformMixin | ✓ OK |
| old/simple_stat.py | BearingStat | — | NoFitMultiSourceTransformMixin | ✓ OK |
| old/filter_target.py | FilterTarget | — | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| old/cut_till_first_zero.py | RULTaskUpToFirstZero | — | NoFitMultiSourceTransformMixin | ✓ OK |
| health_models/li2019.py | Li2019FPTCutTransform | — | NoFitMultiSourceTransformMixin | ✓ OK |
| health_models/ahmad2019.py | Ahmad2019FPTCutTransform | — | NoFitMultiSourceTransformMixin | ✓ OK |
| health_index.py | HealthIndexTransform | — | NoFitMultiSourceTransformMixin | ✓ OK |
| autoencoder_transform.py | BearingAutoencoderTransform | — | extends AutoencoderTransform | ✓ OK |

---

## building

| File | Class | Overrides | Inherits from mixin? | Status |
|------|--------|-----------|----------------------|--------|
| mzvav_scaler.py | MinMaxScalerMZVAV | fit_data | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| rugged_dimension_swap.py | RuggedDimensionSwap | — | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |

---

## n_cmapss

| File | Class | Overrides | Inherits from mixin? | Status |
|------|--------|-----------|----------------------|--------|
| n_cmapss_scalers.py | N_CMAPSSDescriptorsScaler | fit_data | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| n_cmapss_scalers.py | N_CMAPSSFeaturesScaler | fit_data | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |
| concept_classes_builder.py | ConceptClassesBuilder | fit_data | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |

---

## not_used

| File | Class | Overrides | Inherits from mixin? | Status |
|------|--------|-----------|----------------------|--------|
| autocorr_drop.py | AutocorrFeatureDropTransform | fit_multi_source | **DenseTransform only** | **MISSING_MIXIN** — should add `ConcatinatedFitMultiSourceTransformMixin` and `requires_fit` (or use mixin’s `requires_fit`). |

---

## railway

| File | Class | Overrides | Inherits from mixin? | Status |
|------|--------|-----------|----------------------|--------|
| timefeatures_railway.py | RailwayTimeFeaturesTransform | — | DenseTransform only | ✓ N/A (no fit override) |
| seasonal_dummy_railway.py | SeasonalDummyTransform | — | DenseTransform only | ✓ N/A |
| dummy_railway.py | RailwayDummyTransform | — | DenseTransform only | ✓ N/A |

---

## rigid_body_dynamics

| File | Class | Overrides | Inherits from mixin? | Status |
|------|--------|-----------|----------------------|--------|
| transforms.py | First | — | ConcatinatedFitMultiSourceTransformMixin | ✓ OK |

---

## signal_processing

| File | Class | Overrides | Inherits from mixin? | Status |
|------|--------|-----------|----------------------|--------|
| rms.py | MovingRMSTransform | — | NoFitMultiSourceTransformMixin | ✓ OK |
| cumsum.py | CumSumTransform | — | NoFitMultiSourceTransformMixin | ✓ OK |

---

## time_transforms

| File | Class | Overrides | Inherits from mixin? | Status |
|------|--------|-----------|----------------------|--------|
| sequence2table.py | ExtractSequenceStartTime | fit_data (pass) | NoFitMultiSourceTransformMixin | ✓ OK |

---

## visualization

| File | Class | Overrides | Inherits from mixin? | Status |
|------|--------|-----------|----------------------|--------|
| features_target_plotter.py | FeatureTargetPlotter | — | NoFitMultiSourceTransformMixin | ✓ OK |

---

## base/multisource.py (definitions, not user transforms)

- `ConcatinatedFitMultiSourceTransformMixin`: defines `requires_fit = True` and `fit_multi_source` / `fit_function_to_data_segments`.
- `NoFitMultiSourceTransformMixin`: subclasses it, sets `requires_fit = False`, `fit_multi_source` raises `NotImplementedError`.
- `ConcatenateUnitsMixing`: sets `requires_fit = False`; has `fit_multi_source` that raises `NotImplementedError` (used by e.g. `MultiDatasetRuggedToDenseTransform`).

---

## Recommended follow-ups

1. **not_used/autocorr_drop.py — AutocorrFeatureDropTransform**  
   Add `ConcatinatedFitMultiSourceTransformMixin` to the class and ensure the pipeline can call the existing `fit_multi_source` via the mixin (or align the method signature with the mixin). Then the transform will have `requires_fit = True` and the pipeline will enforce `fit_on` / `fit_on_key` when used.

2. **battery/nb1_transforms.py — Cycles2Statistics**  
   It overrides `fit_data` with real logic but uses `NoFitMultiSourceTransformMixin` (`requires_fit = False`). Decide:  
   - If it is intended to be fittable (e.g. on train), switch to `ConcatinatedFitMultiSourceTransformMixin` and set `fit_on` in metadata.  
   - If the fit is only for a single-source path and multi-source should not fit, document that and keep NoFit (current behavior).
