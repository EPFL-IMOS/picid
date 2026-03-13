# Transforms User Guide

**Reading time: ~15-20 minutes**

This guide provides a comprehensive overview of the transforms system, covering configuration, usage patterns, and common scenarios.

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Metadata](#metadata)
4. [Transform Manager](#transform-manager)
5. [Fitting Strategies](#fitting-strategies)
6. [Data Types](#data-types)
7. [Common Patterns](#common-patterns)
8. [Examples](#examples)

## Overview

The transforms system provides a flexible, configuration-driven approach to data preprocessing. It handles:

- **Multiple data formats**: Dense NumPy arrays and ragged Awkward arrays
- **Multiple data sources**: Single-unit and multi-unit datasets
- **Fitting strategies**: Fit-on-train, stateless, multi-source fitting
- **Flexible application**: Apply to specific keys, assign to different keys

## Configuration

Transforms are configured using YAML files. Each transform entry has two main parts:

1. **`transform`**: The transform class and its parameters
2. **`metadata`**: How the transform should be applied

### Basic Structure

```yaml
transform_name:
  transform:
    _target_: picid.transforms.base_transforms.scaler.StandardScalerSklearn
    # Additional parameters for the transform class

  metadata:
    apply_to: features
    fit_on: train
```

### Configuration with Hydra

The system uses Hydra for configuration management. You can:

- Use `defaults` to compose configurations
- Use YAML anchors (`&`) and references (`<<: *anchor`) for reusability
- Reference other config values with `${variable}`

**Example:**

```yaml
defaults:
  - bearings/pronostia/combined@_here_

# Define reusable transform config
__scaler_config: &scaler_config
  _target_: picid.transforms.base_transforms.scaler.StandardScalerSklearn

# Use it multiple times
scale_features:
  transform:
    <<: *scaler_config
  metadata:
    apply_to: features
    fit_on: train

scale_target:
  transform:
    <<: *scaler_config
  metadata:
    apply_to: target
    fit_on: train
```

## Metadata

Metadata controls how transforms are applied to your data. Key metadata fields:

### `apply_to` (Required)

Specifies which data keys the transform operates on. Can be:

- **String**: Single key
  ```yaml
  apply_to: features
  ```

- **List**: Multiple keys
  ```yaml
  apply_to: [features, target]
  ```

- **Mapping**: Map input keys to transform parameters (advanced)
  ```yaml
  apply_to:
    features: history
    target: present
  ```

### `assign_to` (Optional)

Specifies where to store transformed data. Defaults to `apply_to` if not specified.

- **String**: Single output key
  ```yaml
  apply_to: features
  assign_to: normalized_features
  ```

- **List**: Multiple output keys (must match length of `apply_to`)
  ```yaml
  apply_to: [x0, x1]
  assign_to: [features]
  ```

- **Mapping**: Map input keys to output keys
  ```yaml
  apply_to:
    x0: history
    x1: history
  assign_to:
    x0: features
    x1: features
  ```

### `fit_on` (Optional)

Specifies which data split to use for fitting. Options: `train`, `val`, `test`, or `None` (no fitting).

```yaml
metadata:
  apply_to: features
  fit_on: train  # Fit on training data only
```

### `fit_on_key` (Optional)

When `apply_to` is a list, specifies which key to use for fitting.

```yaml
metadata:
  apply_to: [features, target]
  fit_on: train
  fit_on_key: features  # Fit only on features, not target
```

### `transform_on_keys` (Optional)

Specifies which splits to transform. Defaults to all splits.

```yaml
metadata:
  apply_to: features
  transform_on_keys: [train, val]  # Skip test split
```

### `multi_source_fit` (Optional)

If `True`, the transform will be fitted across multiple sources (units). Defaults to `False`.

```yaml
metadata:
  apply_to: features
  fit_on: train
  multi_source_fit: true  # Fit across all units in training split
```

### Metadata reference (summary)

| Field | Required | Description |
|-------|----------|-------------|
| `apply_to` | Yes | Data key(s) to read from: string, list, or mapping. |
| `assign_to` | No | Where to write results; defaults to `apply_to`. String, list, or mapping. |
| `fit_on` | No* | Split to fit on: `train`, `val`, or `test`. Required if the transform has `requires_fit=True`. |
| `fit_on_key` | No** | When `apply_to` is a list and `fit_on` is set, which key to use for fitting. |
| `transform_on_keys` | No | Restrict which splits are transformed; default is all splits. |
| `multi_source_fit` | No | If `true`, fit across all units in the fit split (concatenate then fit). Default `false`. |
| `validate_output` | No | If `false`, skip built-in structural validation after the transform. Validation is performed by the pipeline's IntegrityCheckAfterStep when enabled. Default `true`. |
| `include_slice_info_in_metadata` | No | If `true`, the pipeline adds the container's slice (split, unit_ids, cycle_ids, bounds, index_map) to the `metadata` dict passed to `transform_data` and `fit_data` under the key `"slice_info"`. Use when the transform needs to know which slice of the dataset it is processing. The container must have `slice_info` set. See [Data objects — SliceInfo](../reference/dataobject.md). Default `false`. |
| `integrity_checks_before` | No | Optional list of `(name, check_fn, policy)` run before fit/transform on apply_to chunks. Check fn receives `Dict[str, List[NamedTransformInput]]`. Set programmatically (not from YAML). See [Developer Guide: Validation and integrity](developer_guide.md#validation-and-integrity). |
| `integrity_checks_after` | No | Optional list of `(name, check_fn, policy)` run after merge on assign_to slice. Check fn receives `Dict[str, Any]`. Set programmatically (not from YAML). See [Developer Guide: Validation and integrity](developer_guide.md#validation-and-integrity). |
| `integrity_checks` | No | *(Deprecated)* Legacy single list; maps to `integrity_checks_before`. Use `integrity_checks_before` / `integrity_checks_after` instead. |

\* Transforms that learn from data (e.g. scalers) set `requires_fit=True`; the pipeline will raise if `fit_on` is missing.
\** Required when `apply_to` is a list and `fit_on` is set.

### Slice info in transforms

When the container has optional [SliceInfo](../reference/dataobject.md) set (by the datasource or preprocessing when building the container), transforms can receive it in the `metadata` dict passed to `transform_data` and `fit_data`. Set `include_slice_info_in_metadata: true` in the transform's metadata to get `metadata["slice_info"]` (a dict with `split`, `unit_ids`, `cycle_ids`, `bounds`, `index_map`). This is useful when the transform needs to know which slice of the dataset it is processing.

### Turning off post-transform validation

To disable the built-in structural validation (e.g. for debugging or trusted pipelines), set `validate_output: false` in metadata:

```yaml
my_transform:
  transform:
    _target_: picid.transforms.path.ToSomeTransform
  metadata:
    apply_to: features
    validate_output: false
```

## Transform Manager

The `ConfigTransformManager` is the main interface for working with transforms.

### Initialization

```python
from picid.transforms.base.transform_manager import ConfigTransformManager
from omegaconf import OmegaConf

# From dictionary
config = {
    "scale_features": {
        "transform": {"_target_": "picid.transforms.base_transforms.scaler.StandardScalerSklearn"},
        "metadata": {"apply_to": "features", "fit_on": "train"}
    }
}
manager = ConfigTransformManager(transforms_config=config)

# From YAML file
config = OmegaConf.load("path/to/transforms.yaml")
manager = ConfigTransformManager(transforms_config=config)

# Lazy instantiation (transforms created on first access)
manager = ConfigTransformManager(transforms_config=config, lazy_instantiation=True)
```

### Accessing Transforms

```python
# Get all transforms as OrderedDict
transforms = manager.get_data_transforms()

# Get specific transform
scale_transform = manager.get_data_transform("scale_features")

# Get transforms as list
transform_list = manager.prepare_transforms()

# Filter by apply_to
feature_transforms = manager.get_transforms_by_apply_to("features")

# Filter by fit_on
train_fitted = manager.get_transforms_by_fit_on("train")
```

### Modifying Configuration

```python
# Add new transforms
new_config = {
    "new_transform": {
        "transform": {...},
        "metadata": {...}
    }
}
manager.add_transforms_config(new_config)

# Update configuration (must clear cache first)
manager.clear_cache()
manager.update_transforms_config(new_config)

# Remove transform
manager.clear_cache()
manager.remove_transform("transform_name")
```

## Fitting Strategies

Transforms can be fitted in different ways depending on your needs:

### 1. Fit-on-Train (Most Common)

Fit the transform on training data, then apply to all splits:

```yaml
scale_features:
  transform:
    _target_: picid.transforms.base_transforms.scaler.StandardScalerSklearn
  metadata:
    apply_to: features
    fit_on: train  # Fit on train, transform all splits
```

### 2. Stateless Transforms

Transforms that don't require fitting (e.g., reshaping, identity):

```yaml
reshape_features:
  transform:
    _target_: picid.transforms.base_transforms.reshaping.ReshapeTransform
    new_shape: [100, 10]
  metadata:
    apply_to: features
    # No fit_on needed
```

### 3. Multi-Source Fitting

Fit across multiple units/sources:

```yaml
scale_features:
  transform:
    _target_: picid.transforms.base_transforms.scaler.StandardScalerSklearn
  metadata:
    apply_to: features
    fit_on: train
    multi_source_fit: true  # Concatenate all units before fitting
```

## Data Types

The system supports two main data types:

### Dense Arrays (NumPy)

Regular NumPy arrays with fixed dimensions:

```python
import numpy as np
data = {
    "features": {
        "train": np.array([[1, 2, 3], [4, 5, 6]]),
        "val": np.array([[7, 8, 9]])
    }
}
```

### Ragged Arrays (Awkward)

Variable-length arrays using Awkward Array:

```python
import awkward as ak
data = {
    "features": {
        "train": ak.Array([[1, 2], [3, 4, 5]]),  # Variable length
        "val": ak.Array([[6, 7, 8, 9]])
    }
}
```

### Transform Support

Transforms declare their data type support:

- **`DenseTransform`**: Only supports dense arrays
- **`RaggedTransform`**: Only supports ragged arrays
- **`RaggedOrDenseTransform`**: Supports both

The system automatically handles conversion when needed (e.g., ragged → dense for dense-only transforms).

## Common Patterns

### Pattern 1: Feature Scaling

Scale features to [0, 1] using training statistics:

```yaml
scale_features:
  transform:
    _target_: picid.transforms.base_transforms.scaler.MinMaxScalerSklearn
  metadata:
    apply_to: features
    fit_on: train
```

### Pattern 2: Multiple Transforms in Sequence

Apply multiple transforms in order:

```yaml
clean_features:
  transform:
    _target_: picid.transforms.base_transforms.data_cleaning.RemoveOutliers
  metadata:
    apply_to: features

scale_features:
  transform:
    _target_: picid.transforms.base_transforms.scaler.StandardScalerSklearn
  metadata:
    apply_to: features
    fit_on: train
```

### Pattern 3: Transform to Different Key

Transform data and store in a new key:

```yaml
extract_statistics:
  transform:
    _target_: picid.transforms.base_transforms.time_statistics.TimeStatistics
  metadata:
    apply_to: features
    assign_to: statistics
```

### Pattern 4: Conditional Transformation

Transform only specific splits:

```yaml
tabularize_train:
  transform:
    _target_: picid.transforms.base_transforms.tabularizers.TimeseriesTabularizer
    seq_len: 100
  metadata:
    apply_to: features
    transform_on_keys: [train]  # Only transform train split

tabularize_val_test:
  transform:
    _target_: picid.transforms.base_transforms.tabularizers.TimeseriesTabularizer
    seq_len: 100
  metadata:
    apply_to: features
    transform_on_keys: [val, test]  # Transform val and test
```

### Pattern 5: Multi-Key Transformation

Transform multiple keys simultaneously:

```yaml
scale_features_and_target:
  transform:
    _target_: picid.transforms.base_transforms.scaler.StandardScalerSklearn
  metadata:
    apply_to: [features, target]
    fit_on: train
    fit_on_key: features  # Fit only on features
```

## Examples

### Example 1: Simple Scaling Pipeline

```yaml
# transforms.yaml
normalize_features:
  transform:
    _target_: picid.transforms.base_transforms.scaler.StandardScalerSklearn
  metadata:
    apply_to: features
    fit_on: train

normalize_target:
  transform:
    _target_: picid.transforms.base_transforms.scaler.MinMaxScalerSklearn
  metadata:
    apply_to: target
    fit_on: train
```

### Example 2: Feature Extraction Pipeline

```yaml
# Extract time statistics
extract_stats:
  transform:
    _target_: picid.transforms.base_transforms.time_statistics.TimeStatistics
  metadata:
    apply_to: features
    assign_to: time_stats

# Extract spectral features
extract_spectral:
  transform:
    _target_: picid.transforms.base_transforms.spectral.SpectralTransform
  metadata:
    apply_to: features
    assign_to: spectral_features

# Concatenate all features
concatenate_features:
  transform:
    _target_: picid.transforms.base_transforms.concatenate.ConcatenateTransform
  metadata:
    apply_to: [features, time_stats, spectral_features]
    assign_to: features
```

### Example 3: Time Series Tabularization

```yaml
# Tabularize training data with stride
tabularize_train:
  transform:
    _target_: picid.transforms.base_transforms.tabularizers.TimeseriesTabularizer
    seq_len: 100
    stride: 10
  metadata:
    apply_to: features
    transform_on_keys: [train]

# Tabularize validation/test with different stride
tabularize_val_test:
  transform:
    _target_: picid.transforms.base_transforms.tabularizers.TimeseriesTabularizer
    seq_len: 100
    stride: 100  # No overlap for val/test
  metadata:
    apply_to: features
    transform_on_keys: [val, test]
```

## Best Practices

1. **Order matters**: Transforms are applied in configuration order
2. **Fit before transform**: Ensure fitting transforms come before transforms that depend on them
3. **Use meaningful names**: Choose clear transform names for debugging
4. **Validate outputs**: Check that transforms produce expected shapes/types
5. **Test incrementally**: Add transforms one at a time to debug issues

## Troubleshooting

### Transform not applying

- Check that `apply_to` keys exist in your data
- Verify transform is in the manager: `manager.has_transform("name")`
- Check transform logs for errors

### Shape mismatches

- Verify input/output shapes match expectations
- Check `assign_to` vs `apply_to` configuration
- Ensure transforms are applied in correct order

### Fitting issues

- Verify `fit_on` split exists in your data
- Check `multi_source_fit` setting matches your data structure
- Ensure fitting happens before transformation

## Next Steps

- **Want to understand the implementation?** → [Developer Guide](developer_guide.md)
- **Need to create custom transforms?** → [Developer Guide: Creating Custom Transforms](developer_guide.md#creating-custom-transforms)
- **Want to see more examples?** → Check `configs/transforms/` directory
- **Pipeline hooks (debugging, error handling)?** → [Hooks examples](HOOKS_EXAMPLES.md)

---

For deep technical details, see the [Developer Guide](developer_guide.md).
