# Transforms Documentation

Welcome to the Transforms documentation! This documentation is organized into three levels to help you understand the transforms system at the depth you need.

## Documentation Structure

### 🚀 [Quick Start Guide](../transforms/quick_start.md) (5 minutes)
**Perfect for:** Getting started quickly, understanding the basics, seeing simple examples.

This guide provides a high-level overview of what transforms are, why they're useful, and how to use them in the simplest way possible. Read this first if you're new to the system.

### 📚 [User Guide](../transforms/guide.md) (Medium Level)
**Perfect for:** Understanding how transforms work, configuring them, and using them effectively.

This guide covers the transform system in detail, including configuration, metadata, fitting strategies, and common patterns. Read this when you need to understand how to configure and use transforms effectively.

### 📚 [Developer Guide](../transforms/developer_guide.md) (Deep Dive)
**Perfect for:** Implementing custom transforms, understanding the architecture, debugging issues.

This guide dives deep into the inner workings of the transform system, including the architecture, base classes, strategies, and how to extend the system. Read this when you need to create custom transforms or understand the implementation details.

## Quick Navigation

- **What are transforms?** → [Quick Start](../transforms/quick_start.md#what-are-transforms)
- **How do I configure transforms?** → [User Guide](../transforms/guide.md#configuration)
- **How do transforms work internally?** → [Developer Guide](../transforms/developer_guide.md#architecture-overview)
- **How do I create a custom transform?** → [Developer Guide](../transforms/developer_guide.md#creating-custom-transforms)
- **Pipeline hooks (debugging, error handling)?** → [Hooks examples](../transforms/HOOKS_EXAMPLES.md)

## Overview

The transforms system in this codebase provides a flexible, configuration-driven approach to data preprocessing and transformation. It supports:

- **Multiple data types**: Dense NumPy arrays and ragged Awkward arrays
- **Multiple data sources**: Single-unit and multi-unit transformations
- **Fitting strategies**: Fit-on-train, stateless transforms, multi-source fitting
- **Flexible configuration**: YAML-based configuration with Hydra integration
- **Validation**: Automatic validation of transform outputs

Choose the guide that matches your needs and dive in!
# LMEtk Transformation Framework Documentation

The LMEtk framework is designed to decouple **transformation logic** (Python code) from **orchestration logic** (Configuration). This allows you to chain, branch, and manipulate data flows purely through YAML configuration without altering the codebase.

This guide covers how to configure transforms, manage data flow via metadata, understand the underlying strategies (Mixins), and leverage advanced patterns for complex pipelines.

---

## 1. The Anatomy of a Transform

Every transformation in the pipeline is defined by two distinct blocks:

1.  **`transform`**: Defines *what* the logic is (Class instantiation and hyperparameters).
2.  **`metadata`**: Defines *how* it applies to the data (Input keys, output keys, fitting strategies, split selection).

### Basic Configuration Structure

```yaml
my_transform_name:
  # The Logic Block
  transform:
    _target_: lmetk.transforms.path.to.YourClass
    arg1: value
    arg2: value
  
  # The Control Block
  metadata:
    apply_to: input_feature_name
    assign_to: output_feature_name
    fit_on: train
```

---

## 2. The Metadata Directive: Controlling Data Flow

The `metadata` block is the control center. It dictates how data moves in and out of the transform instance.

### 2.1 Inputs and Outputs (`apply_to` & `assign_to`)

* **In-Place Transformation (Overwrite):**
    If `assign_to` is omitted, the framework assumes you want to overwrite the input.
    ```yaml
    metadata:
      apply_to: features  # Reads 'features', transforms it, writes back to 'features'
    ```

* **Branching (Create New Features):**
    To keep the original data and save the result as a new key.
    ```yaml
    metadata:
      apply_to: features
      assign_to: features_scaled # Original 'features' remains; result goes here
    ```

* **Multi-Input Transformation:**
    For transforms that require multiple inputs (e.g., Concatenation, Plotting, Autoencoders), pass a list.
    ```yaml
    metadata:
      apply_to: ["spectral_features", "time_domain_features"]
      assign_to: features
    ```

* **Renaming Outputs (Mapping):**
    If the transform returns a dictionary or specific keys, you can map them to new names in your dataset using a dictionary format.
    ```yaml
    metadata:
      apply_to: target
      assign_to: 
        target: fault_classification  # Renames internal 'target' output to 'fault_classification'
    ```

### 2.2 Fitting Strategy (`fit_on`)

Stateful transforms (like Scalers, PCA, Autoencoders) need to learn parameters from a specific dataset split to avoid data leakage.

* **Standard Fitting:**
    Fits the transform on the training set, then applies the learned parameters to all splits (train/val/test).
    ```yaml
    metadata:
      apply_to: features
      fit_on: train
    ```
    *Note: If `fit_on` is omitted, the transform is treated as stateless (applied independently to every batch/item).*

### 2.3 Execution Scope (`transform_on_keys`)

Sometimes you want to apply a transform *only* to specific splits, or apply the same transform with different parameters to different splits (e.g., different sliding window strides).

* **Filter Splits:**
    ```yaml
    metadata:
      apply_to: features
      transform_on_keys: ["train"] # This transform will be skipped for val/test
    ```

---

## 3. Advanced Patterns & Use Cases ("Tricks")

### 3.1 The "Training vs. Inference" Split Pattern
**Use Case:** You want data augmentation (overlapping windows) during training, but non-overlapping windows during testing/validation.

**Trick:** Use YAML Anchors (`&`) and Inheritance (`<<:`) to define the base logic once, then override specific parameters for different splits.

```yaml
# 1. Define the Base Logic (Anchor)
__tabularize_base: &__tabularize_base
  _target_: lmetk.transforms.base_transforms.tabularizers.TimeseriesTabularizer
  seq_len: 50

# 2. Define Base Metadata (Anchor)
__meta_base: &__meta_base
  apply_to: features
  assign_to: features

# 3. Instantiate for TRAIN (Small stride = more data)
tabularize_train:
  transform:
    <<: *__tabularize_base
    stride: 1  # Overriding stride
  metadata:
    <<: *__meta_base
    transform_on_keys: ["train"] # Only applies to train split

# 4. Instantiate for TEST (Large stride = no overlap)
tabularize_test:
  transform:
    <<: *__tabularize_base
    stride: 50 # Overriding stride
  metadata:
    <<: *__meta_base
    transform_on_keys: ["val", "test"] # Only applies to val/test
```

### 3.2 The "Corruption-Imputation" Loop (Denoising)
**Use Case:** Training robust models or Denoising Autoencoders by artificially corrupting data and attempting to restore it.

**Trick:** Branch the data flow using `assign_to`.

1.  **Corrupt:** Apply noise to `features` -> assign to `features_corrupted`.
2.  **Impute:** Apply imputer to `features_corrupted` -> assign to `features_imputed`.
3.  **Result:** You now have the original clean data (`features`), the noisy data (`features_corrupted`), and the restored data (`features_imputed`) available simultaneously for loss calculation or visualization.

```yaml
mcar_corruption:
  transform:
    _target_: lmetk.transforms.base_transforms.mcar_corruption.MCARCorruptorTransform
    missing_fraction: 0.1
  metadata:
    apply_to: features
    assign_to: features_corrupted  # Branch created here

imputation:
  transform:
    _target_: lmetk.transforms.base_transforms.imputation_methods.ImputationTransform
    strategy: "mean"
  metadata:
    apply_to: features_corrupted   # Work on the corrupted branch
    assign_to: features_imputed    # Save result to new key
```

### 3.3 Handling Ragged Data (Variable Lengths)
**Use Case:** Your raw data consists of run-to-failure cycles of different lengths (ragged arrays), but your model requires fixed-size tensors.

**Trick:** Use a Statistics Transform followed by a Regularizer.

1.  **Calculate Stats:** `TimeStatsTransform` or `SpectralStatsTransform` computes fixed-size statistics (mean, max, etc.) over the ragged dimension.
2.  **Regularize:** `RegularizeRaggedDataTransform` converts the list of statistics into a dense Numpy/Tensor format.

```yaml
time_domain_features:
  transform:
    _target_: lmetk.transforms.base_transforms.time_statistics.TimeStatsTransform
    stats_to_compute: ["mean", "variance"]
  metadata:
    apply_to: "features"
    assign_to: "time_domain_features"

ragged2regular:
  transform:
    _target_: lmetk.transforms.base_transforms.shape_manipulations.RegularizeRaggedDataTransform
  metadata:
    apply_to: "time_domain_features" # Converts the ragged stats to dense array
```

### 3.4 Multi-Source Transforms (Concatenation & Autoencoders)
**Use Case:** Merging different feature sets or training models that require both features and targets (like reconstruction tasks).

**Trick:** Pass a list to `apply_to`. The order in the list determines the order of arguments passed to the Python class.

```yaml
concatenate_features:
  transform:
    _target_: lmetk.transforms.base_transforms.concatenate.ConcatenateTransform
    axis: 1
  metadata:
    # Concatenates spectral features with time features
    apply_to: ["spectral_features", "time_domain_features"] 
    assign_to: features
```

---

## 4. Advanced: Mixins & Multi-Source Strategies

While every transform inherits from `BaseTransform`, specific behaviors—particularly regarding how data from multiple sources (e.g., multiple battery units, different experiment runs) is handled—are dictated by **Mixins**.

These classes determine how the transform "sees" the data: does it learn from all units combined? Does it merge them? Or does it treat them independently?

### Class Overview & Selection Guide

Below is a guide to the mixins available in `lmetk.transforms.base.base_transform`. Your custom transform should inherit from `BaseTransform` **and** one of these mixins depending on the desired behavior.

#### A. `ConcatFitAndPerSegmentTransformMixin`
**"Fit Global, Transform Local"** (legacy alias: `ConcatinatedFitMultiSourceTransformMixin`)

* **Behavior:**
    * **Fitting:** Concatenates data from *all* sources (units) into one large dataset to learn global parameters (e.g., a global min/max for scaling).
    * **Transforming:** Applies the logic to each source *individually*. The output structure matches the input structure (N input sources -> N output sources).
    * **Ragged Support:** Contains smart logic. If your transform only supports Dense (Numpy) data but is fed Ragged (Awkward) data, this mixin automatically flattens the ragged dimension, applies the transform, and reconstructs the ragged structure.
* **Use Case:** Scalers (`MinMaxScaler`, `StandardScaler`), PCA, Autoencoders, or any model that needs to learn the statistical properties of the *entire* dataset but preserve the original split structure.

#### B. `NoFitPerSegmentMixin`
**"Stateless Multi-Source"** (legacy alias: `NoFitMultiSourceTransformMixin`)

* **Behavior:**
    * No fitting; transform applied per segment.
    * Iterates over all sources and applies the transform logic.
* **Use Case:** Stateless operations that don't need to "learn" anything but still need to handle list-of-dicts input. Examples: Casting data types, log-transforms, simple mathematical operations.

#### C. `NoFitConcatAlongAxisMixin`
**"Many-to-One Aggregator"** (legacy alias: `ConcatenateUnitsMixing`)

* **Behavior:**
    * Takes a list of data segments (N sources).
    * Concatenates them along a specific `axis` *before* the transform logic sees them.
    * **Result:** The output is a list containing a **single** dataset (1 source).
* **Use Case:** preparing a final dataset for training where unit boundaries no longer matter, or merging specific features before a dimensionality reduction step.

#### D. `InverseTransformMixin`
**"Reversible Logic"**

* **Behavior:** Adds a contract for `inverse_transform`.
* **Use Case:** Required for transforms like Scalers or Normalizers where you need to project model predictions back to the original physical units for evaluation.

### Configuration Differences by Mixin

While the Python implementation changes based on the parent class, the YAML configuration changes slightly to accommodate specific requirements (like `axis` for concatenation).

#### Scenario 1: Global Learning (Scaler)
**Inherits:** `ConcatFitAndPerSegmentTransformMixin` + `BaseTransform`

This transform learns from all units (Train set) but keeps them separate.

```yaml
scaler_global:
  transform:
    # Python class inherits ConcatFitAndPerSegmentTransformMixin
    _target_: lmetk.transforms.base_transforms.scaler.MinMaxScalerSklearn
    
  metadata:
    apply_to: features
    fit_on: train
    # The framework detects the Mixin and knows to stitch all train units 
    # together before calling fit().
```

#### Scenario 2: Aggregation (Merging Units)
**Inherits:** `NoFitConcatAlongAxisMixin` + `BaseTransform`

This transform merges all units into one. It **requires** an `axis` parameter.

```yaml
merge_units:
  transform:
    # Python class inherits NoFitConcatAlongAxisMixin
    _target_: lmetk.transforms.base_transforms.concatenate.MultiDatasetRuggedToDenseTransform
    axis: 0  # <--- REQUIRED: How to stitch them together?
    
  metadata:
    apply_to: features
    assign_to: features
    # Note: After this transform, your data is no longer a list of units, 
    # but a single dataset.
```

#### Scenario 3: Stateless Operation
**Inherits:** `NoFitPerSegmentMixin` + `BaseTransform`

```yaml
log_transform:
  transform:
    # Python class inherits NoFitPerSegmentMixin
    _target_: lmetk.transforms.math.LogTransform
    
  metadata:
    apply_to: features
    # fit_on is NOT needed (and will raise error/warning if provided)
```

---

## 5. Developer Guide: Creating New Transforms

To add new functionality, create a Python class that adheres to the framework's contract.

### Step 1: Inherit from BaseTransform
Your class must inherit from `lmetk.transforms.base.base_transform.BaseTransform`.

### Step 2: Implement Required Methods
* `__init__`: Handle configuration arguments.
* `fit_data(self, data, metadata)`: (Optional) Learn parameters from data.
* `transform_data(self, data, metadata)`: Apply the logic.

### Step 3: Argument Matching
The signature of `transform_data` must match your `metadata.apply_to` configuration.

| YAML `apply_to` | Python Signature | Notes |
| :--- | :--- | :--- |
| `apply_to: features` | `transform_data(self, data, metadata)` | `data` contains the features object. |
| `apply_to: [a, b]` | `transform_data(self, data, metadata)` | `data` is a list `[a, b]`. |
| `apply_to: {img: x, lbl: y}` | `transform_data(self, img, lbl, metadata)` | Arguments are unpacked by name. |

### Example Class

```python
from lmetk.transforms.base.base_transform import BaseTransform

class MyCustomScaler(BaseTransform):
    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def transform_data(self, data, metadata):
        # 'data' comes from the key specified in apply_to
        return data * self.factor
```

### Step 4: Developer Notes: Handling Ragged Data

The `ConcatFitAndPerSegmentTransformMixin` contains sophisticated logic for handling Ragged (Variable-length) data. If you are writing a transform:

1.  **If your math works on Ragged Arrays (Awkward Array):**
    Inherit from `RaggedTransform`. The mixin will pass the raw Awkward Array to you. This is the fastest method.

2.  **If your math ONLY works on Dense Arrays (Numpy):**
    Inherit from `DenseTransform`.
    * The mixin will detect ragged input.
    * It will **flatten** the ragged dimension.
    * It will pass you a standard Numpy array to `fit`/`transform`.
    * It will **reconstruct** the ragged structure from your numpy output.
    * *Warning:* This reconstruction process has overhead. If performance is critical, try to write a native Ragged implementation.

[← Reference index](index.md) | [Back to documentation index](../index.md)
