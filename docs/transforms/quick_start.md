# Transforms Quick Start Guide

**Reading time: ~5 minutes**

This guide will get you up and running with transforms quickly. For more detailed information, see the [User Guide](guide.md) or [Developer Guide](developer_guide.md).

## What are Transforms?

Transforms are data preprocessing operations that modify your data before it's used for training or inference. Think of them as a pipeline of operations that:

- **Normalize** your data (e.g., scaling features to [0, 1])
- **Reshape** your data (e.g., converting sequences to tabular format)
- **Extract** features (e.g., computing statistics, spectral features)
- **Clean** your data (e.g., removing outliers, handling missing values)

## Why Use Transforms?

Transforms help you:

1. **Prepare data** for machine learning models
2. **Standardize** data across different datasets
3. **Extract** meaningful features from raw data
4. **Handle** different data formats (dense arrays, ragged arrays, time series)

## Basic Example

Here's a simple transform configuration that scales features:

```yaml
scale_features:
  transform:
    _target_: lmetk.transforms.base_transforms.scaler.StandardScalerSklearn
  
  metadata:
    apply_to: features
    fit_on: train
```

This transform:
- **Applies to** the `features` key in your data
- **Fits** on the training split (computes mean/std)
- **Transforms** all splits (train, val, test) using the fitted parameters

## How Transforms Work

1. **Configuration**: You define transforms in YAML files
2. **Instantiation**: The system creates transform objects from your config
3. **Fitting**: Some transforms learn parameters from training data
4. **Transformation**: Transforms modify your data according to their logic

## Common Transform Types

### Scaling Transforms
Normalize your data to a specific range or distribution:
- `StandardScalerSklearn`: Zero mean, unit variance
- `MinMaxScalerSklearn`: Scale to [0, 1]

### Shape Manipulation
Change the shape or structure of your data:
- `TimeseriesTabularizer`: Convert sequences to tabular format
- `ReshapeTransform`: Change array dimensions

### Feature Extraction
Extract new features from your data:
- `TimeStatistics`: Compute statistical features
- `SpectralTransform`: Extract frequency domain features

## Simple Usage Pattern

```python
from lmetk.transforms.base.transform_manager import ConfigTransformManager

# Load transform configuration
manager = ConfigTransformManager(transforms_config=your_config)

# Get transforms
transforms = manager.get_data_transforms()

# Apply transforms to your data
for transform in transforms.values():
    data = transform.forward(data)
```

## Key Concepts

- **`apply_to`**: Which data keys the transform operates on
- **`assign_to`**: Where to store the transformed data (defaults to `apply_to`)
- **`fit_on`**: Which split to use for fitting (usually `train`)
- **`transform_on_keys`**: Which splits to transform (defaults to all)

## Next Steps

- **Want to understand configuration better?** → [User Guide: Configuration](guide.md#configuration)
- **Need to create custom transforms?** → [Developer Guide: Creating Custom Transforms](developer_guide.md#creating-custom-transforms)
- **Want to see more examples?** → [User Guide: Examples](guide.md#examples)

## Common Questions

**Q: Do I need to fit transforms?**  
A: Only if the transform requires it (e.g., scalers). Stateless transforms (like reshaping) don't need fitting.

**Q: Can I apply multiple transforms?**  
A: Yes! Transforms are applied in the order they appear in your configuration.

**Q: What data formats are supported?**  
A: Dense NumPy arrays and ragged Awkward arrays are both supported.

---

Ready to learn more? Continue to the [User Guide](guide.md) for detailed information about configuration and usage.
