# 5-Minute Introduction: Transforms

This is the quick-start guide. It tells you how to **use** transforms, not how they are built.

---

## What a Transform Is

Each entry in `configs/transforms/**` has two parts:

- **`transform`**: the Python class and its constructor args (`_target_`)
- **`metadata`**: how the transform is applied to the dataset

Minimal example:

```yaml
scale_features:
  transform:
    _target_: picid.transforms.base_transforms.scaler.MinMaxScalerSklearn
  metadata:
    apply_to: features
    fit_on: train
```

This:
- fits the scaler on `features.train`
- applies it to all splits
- overwrites `features` in place

---

## Mental Model (Tiny)

- The dataset is a dict-like container with keys such as `features`, `target`, `timestamps`.
- Each key holds a **split map** (`train`, `val`, `test`).
- Each split is either a single array (single-unit) or a list of arrays (multi-unit).

---

## Common Metadata Knobs

Use these in almost every config:

- `apply_to`: which key(s) to read from
- `assign_to`: where to write results (omit to overwrite input)
- `fit_on`: which split to fit on (`train`, `val`, `test`)
- `transform_on_keys`: restrict which splits to transform

Examples:

```yaml
# Branch results into a new key
normalize_features:
  transform:
    _target_: picid.transforms.base_transforms.scaler.StandardScalerSklearn
  metadata:
    apply_to: features
    assign_to: features_scaled
    fit_on: train

# Apply only on train (e.g., data augmentation)
augment_train_only:
  transform:
    _target_: picid.transforms.base_transforms.subsample.RandomSubsampleTransform
    fraction: 0.5
  metadata:
    apply_to: features
    transform_on_keys: ["train"]
```

If `apply_to` is a list and you set `fit_on`, you also need `fit_on_key`
(to specify which input should be used for fitting).

---

## Multi-Input and Multi-Output

If a transform needs multiple inputs, pass a list in `apply_to`:

```yaml
concat_features:
  transform:
    _target_: picid.transforms.base_transforms.concatenate.ConcatenateTransform
    axis: 1
  metadata:
    apply_to: ["spectral_features", "time_domain_features"]
    assign_to: features
```

If the transform returns a dict, you can remap keys with `assign_to`:

```yaml
classify_faults:
  transform:
    _target_: picid.transforms.base_transforms.identity.IdentityTransform
  metadata:
    apply_to: target
    assign_to:
      target: fault_classification
```

---

## Where to Find Real Examples

Look in `configs/transforms/**` for dataset-specific pipelines:

- `configs/transforms/bearings/pronostia/`
- `configs/transforms/battery/`
- `configs/transforms/railway/`

Start from a config that matches your dataset, then modify the transform list.
