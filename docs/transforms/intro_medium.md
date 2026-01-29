# Medium-Level Introduction: How Transforms Are Orchestrated

This guide explains how configuration turns into executed transforms. It is meant for users who can read code but mainly work in YAML.

---

## Configuration Contract

Each entry in `configs/transforms/**` is a small contract:

```yaml
my_transform:
  transform:
    _target_: lmetk.transforms.path.to.Class
    arg1: 123
  metadata:
    apply_to: features
    assign_to: features_scaled
    fit_on: train
    transform_on_keys: ["train", "val"]
```

Key behaviors:

- `apply_to` can be a string or list
- `assign_to` defaults to `apply_to` if omitted (in-place)
- `fit_on` triggers a fit pass on the given split
- `transform_on_keys` filters which splits are transformed
- entries whose name starts with `__` are treated as config helpers and are skipped

If `fit_on` is set and `apply_to` is a list, you must also set `fit_on_key` (to avoid ambiguity).

---

## What Happens at Runtime

The execution path is:

1. **`ConfigTransformManager`** instantiates the raw Python transform with Hydra.
2. **`DataTransform`** parses metadata and validates the transform signature.
3. **`TransformStrategy.apply()`** performs:
   - an efficient copy of the dataset
   - optional fitting on one split
   - transform application on selected splits
   - merge results into `assign_to`

The data passed around is a `SplitDatasetContainer` with split keys such as
`train`, `val`, `test`. Each split may contain **one unit** (single-source) or a
**list of units** (multi-source).

---

## Single-Source vs Multi-Source

If the data is multi-unit, `TransformStrategy` builds a list of per-unit chunks
(`NamedTransformInput`) and calls:

- `fit_multi_source(...)` for fitting
- `transform_multi_source(...)` for transformation

Single-source data calls `fit_data(...)` and `transform_data(...)` directly.

---

## Split Filtering and Branching

`transform_on_keys` filters which splits are transformed. This is how you implement
train-only augmentation or different settings for train vs test:

```yaml
__base_meta: &__base_meta
  apply_to: features
  assign_to: features

tabularize_train:
  transform:
    _target_: lmetk.transforms.base_transforms.tabularizers.TimeseriesTabularizer
    seq_len: 50
    stride: 1
  metadata:
    <<: *__base_meta
    transform_on_keys: ["train"]

tabularize_eval:
  transform:
    _target_: lmetk.transforms.base_transforms.tabularizers.TimeseriesTabularizer
    seq_len: 50
    stride: 50
  metadata:
    <<: *__base_meta
    transform_on_keys: ["val", "test"]
```

Use `assign_to` to branch outputs into a new key, or to map dict outputs:

```yaml
classify_faults:
  transform:
    _target_: lmetk.transforms.base_transforms.identity.IdentityTransform
  metadata:
    apply_to: target
    assign_to:
      target: fault_classification
```

---

## Output Normalization (Why Your Return Type Matters)

`TransformStrategy` accepts the following return shapes and normalizes them:

- a single array/object
- a list of arrays (one per unit)
- a list of dict-like objects

The standardization happens in `postprocess_transformed_data(...)`, which
maps outputs to `assign_to_map` before merging back into the dataset.

---

## Ragged vs Dense Data (Awkward vs NumPy)

Transforms support:

- **Dense (NumPy)** data
- **Ragged (Awkward)** data

`multisource.py` determines which pathway is used based on marker base classes
(`DenseTransform`, `RaggedTransform`, `RaggedOrDenseTransform`). For ragged data
without native support, the framework will flatten segments to NumPy, apply the
transform, then reassemble the ragged structure (slower but safe).

---

## Practical Debugging Hooks

Each transform produces a per-split `transform_log` including flags such as:

- `data_is_ragged`
- `transform_supports_ragged`
- `transform_supports_dense`

These flags help diagnose ragged-array conversion behavior and performance issues.

---

## Where to Read the Code

- `lmetk/transforms/base/data_transform.py` for metadata parsing and signature validation
- `lmetk/transforms/base/strategy.py` for orchestration and split handling
- `lmetk/transforms/base/multisource.py` for multi-unit and ragged logic
