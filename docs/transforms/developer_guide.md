# Transforms Developer Guide

**Reading time: ~30-40 minutes**

This guide provides a deep dive into the transforms system architecture, implementation details, and how to extend it. It's intended for developers who need to understand the inner workings or create custom transforms.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [Base Classes](#base-classes)
5. [Strategy Pattern](#strategy-pattern)
6. [Multi-Source Transforms](#multi-source-transforms)
7. [Data Type Handling](#data-type-handling)
8. [Creating Custom Transforms](#creating-custom-transforms)
9. [Pipeline steps and hooks](#pipeline-steps-and-hooks)
10. [Validation and integrity](#validation-and-integrity)
11. [Advanced Topics](#advanced-topics)
12. [Debugging](#debugging)

## Architecture Overview

The transforms system follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│              ConfigTransformManager                      │
│  (Configuration loading, transform instantiation)        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  DataTransform                           │
│  (Metadata processing, validation, facade pattern)      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              TransformStrategy                           │
│  (Fitting logic, transformation orchestration)           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              BaseTransform (User Implementation)         │
│  (transform_data, fit_data methods)                     │
└─────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Separation of Concerns**: Configuration, orchestration, and transformation logic are separated
2. **Strategy Pattern**: Different fitting/transformation strategies are encapsulated
3. **Facade Pattern**: `DataTransform` provides a simple interface to complex internals
4. **Type Safety**: Marker classes (`DenseTransform`, `RaggedTransform`) declare capabilities
5. **Validation**: Multiple validation layers ensure correctness

## Core Components

### 1. ConfigTransformManager

**Location**: `picid/transforms/base/transform_manager.py`

The `ConfigTransformManager` is responsible for:

- Loading transform configurations (YAML or dict)
- Instantiating transform objects using Hydra
- Managing transform lifecycle (lazy vs eager instantiation)
- Providing access to transforms

**Key Methods**:

```python
def _instantiate_all(self) -> None:
    """Instantiate all DataTransform objects from configuration."""
    # Iterates through config, instantiates BaseTransform objects
    # Wraps them in DataTransform objects with metadata
```

**Instantiation Flow**:

1. Parse configuration (YAML → DictConfig)
2. For each transform entry:
   - Extract `transform` config (class + parameters)
   - Extract `metadata` config (how to apply)
   - Instantiate `BaseTransform` using Hydra
   - Wrap in `DataTransform` with metadata
3. Store in `OrderedDict` for efficient access

### 2. DataTransform

**Location**: `picid/transforms/base/data_transform.py`

`DataTransform` acts as a facade that:

- Processes metadata (`apply_to`, `assign_to`, `fit_on`, etc.)
- Validates transform signatures
- Delegates to `TransformStrategy` for actual work
- Validates outputs

**Initialization Process**:

```python
def __init__(self, transform_name: str, transform: dict, metadata: dict):
    # 1. Process apply_to (string/list/mapping → list)
    self.apply_to, self.apply_to_map = self._process_apply_to(apply_to)
    
    # 2. Process assign_to (defaults to apply_to if None)
    self.assign_to, self.assign_to_map = self._process_assign_to(assign_to, apply_to)
    
    # 3. Process fit_on configuration
    self.fit_on_split = metadata.get("fit_on", None)
    self.fit_on_key = metadata.get("fit_on_key", None)
    
    # 4. Instantiate transform instance
    self.transform_instance = instantiate(transform)
    
    # 5. Validate transform signature matches metadata
    self._validate_signature()
```

**Signature Validation**:

The system validates that transform signatures match the metadata:

- **Standard signature**: `transform_data(data: NamedTransformInput, metadata: dict)`
- **Named parameters**: When `apply_to` is a mapping, transform can accept named parameters matching the keys

**Forward Method**:

```python
def forward(self, data: SplitDatasetContainer) -> SplitDatasetContainer:
    # 1. Delegate to strategy
    transformed_data, transform_log = self.strategy.apply(...)
    
    # 2. Validate outputs
    # - Check structure matches input
    # - Check splits are preserved
    # - Check types are consistent
    
    return transformed_data, transform_log
```

### 3. TransformStrategy

**Location**: `picid/transforms/base/strategy.py`

`TransformStrategy` orchestrates the transformation process:

1. **Efficient Copying**: Creates shallow copies, deep copies only when needed
2. **Fitting**: Handles fit-on-train logic
3. **Transformation**: Applies transform to all splits
4. **Post-processing**: Standardizes outputs
5. **Merging**: Merges transformed data back into structure

**Apply Method Flow**:

```python
def apply(self, transform_instance, data, apply_to_keys, ...):
    # 1. Efficient hybrid copy
    transformed_data = data.copy(deep=False)
    # Deep copy only keys that will be modified
    
    # 2. Fit transformer (if needed)
    if fit_on_split and fit_on_key:
        fit_chunks = self._prepare_chunks_for_split(...)
        if is_multi_unit:
            transform_instance.fit_multi_source(fit_chunks, metadata)
        else:
            transform_instance.fit_data(fit_chunks[0], metadata)
    
    # 3. Transform all splits
    for split in available_splits:
        chunks_to_transform = self._prepare_chunks_for_split(...)
        if is_multi_unit:
            transformed_chunks = transform_instance.transform_multi_source(...)
        else:
            transformed_chunks = [transform_instance.transform_data(...)]
        
        # 4. Post-process outputs
        transformed_chunks = postprocess_transformed_data(...)
        
        # 5. Merge results
        self._merge_transformed_data(...)
    
    return transformed_data, transform_log
```

**Chunk Preparation**:

For multi-unit data, chunks are prepared by extracting each unit:

```python
def _prepare_chunks_for_split(self, data_to_prepare, split, is_multi_unit):
    chunks = []
    num_units = len(data_to_prepare[keys[0]][split])
    
    for i in range(num_units):
        unit_data = {
            key: data_to_prepare[key][split][i]
            for key in keys
        }
        chunks.append(NamedTransformInput(**unit_data))
    
    return chunks
```

## Data Flow

### Input Data Structure

Data flows through the system in a structured format:

```python
data = {
    "features": {
        "train": [unit1_data, unit2_data, ...],  # Multi-unit
        "val": [unit1_data, unit2_data, ...],
        "test": [unit1_data, unit2_data, ...]
    },
    "target": {
        "train": [unit1_target, unit2_target, ...],
        "val": [unit1_target, unit2_target, ...],
        "test": [unit1_target, unit2_target, ...]
    }
}
```

### Transformation Flow

1. **Input**: `SplitDatasetContainer` (dict of splits)
2. **Chunk Preparation**: Extract individual units → `List[NamedTransformInput]`
3. **Fitting** (if needed): Concatenate chunks → Fit transform
4. **Transformation**: Apply transform to each chunk → `List[ReturnObject]`
5. **Post-processing**: Standardize outputs → `List[SimpleReturnObject]`
6. **Merging**: Reassemble into `SplitDatasetContainer`
7. **Output**: `SplitDatasetContainer` with transformed data

### Output Standardization

The `postprocess_transformed_data` function standardizes outputs:

- **Single array**: `np.ndarray` → `[SimpleReturnObject(key=array)]`
- **Multiple arrays**: `[np.ndarray, ...]` → `[SimpleReturnObject(...), ...]`
- **Named outputs**: `[dict, ...]` → `[NamedDictReturnObject(...), ...]`

## Base Classes

### BaseTransform

**Location**: `picid/transforms/base/base_transform.py`

The abstract base class all transforms must inherit from:

```python
class BaseTransform(ABC):
    @abstractmethod
    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> Any:
        """Transform a single data segment."""
        pass
    
    def fit_data(self, data: NamedTransformInput, metadata: Dict[str, Any]) -> Any:
        """Fit the transform on data. Default: no-op."""
        pass
```

**Key Points**:

- `transform_data` is abstract and must be implemented
- `fit_data` has a default no-op implementation (for stateless transforms)
- Decorated with `@check_transform_output_consistency` for validation

### Marker Classes

Transforms declare their data type support:

```python
class DenseTransform(BaseTransform):
    """Marker: Only supports dense (NumPy) arrays."""

class RaggedTransform(BaseTransform):
    """Marker: Only supports ragged (Awkward) arrays."""

class RaggedOrDenseTransform(BaseTransform):
    """Marker: Supports both ragged and dense arrays."""
```

**Why Marker Classes?**

The system uses these markers to:
- Route data through appropriate code paths
- Handle conversions automatically
- Provide clear documentation of capabilities

## Strategy Pattern

The system uses the Strategy pattern to handle different transformation scenarios:

### Single-Unit vs Multi-Unit

```python
if is_multi_unit:
    transformed_chunks = transform_instance.transform_multi_source(chunks, metadata)
else:
    transformed_chunks = [transform_instance.transform_data(chunks[0], metadata)]
```

### Dense vs Ragged Data

The multi-source mixins (in `picid/transforms/base/multisource.py`) handle data type routing and fit/transform policy. Use **exactly one** fit-policy mixin and **one** transform-policy mixin, or use a ready-made composition.

## Multi-Source Transforms

**Location**: `picid/transforms/base/multisource.py`

### Ready-made mixins (recommended)

| Mixin | Use case |
|-------|----------|
| `ConcatFitAndPerSegmentTransformMixin` | Fit on concatenated segments; transform per segment (e.g. scalers). Sets `requires_fit=True`. |
| `NoFitPerSegmentMixin` | No fit; transform applied per segment (e.g. reshape, concatenate inputs). |
| `NoFitConcatAlongAxisMixin` | No fit; concatenate all segments along an axis, then one `transform_data` call. Requires `axis` in `__init__`. |

### Composable mixins (advanced)

Combine one fit-policy and one transform-policy mixin:

- **Fit:** `FitByConcatenationMixin` (fit on concatenated data), `NoFitMixin` (no fitting).
- **Transform:** `PerSegmentTransformMixin` (transform each segment), `ConcatenateBeforeTransformMixin` (concatenate then transform once; requires `axis`).

Example: `ConcatFitAndPerSegmentTransformMixin` = `FitByConcatenationMixin` + `PerSegmentTransformMixin`.

### Fit and transform behavior (ConcatFitAndPerSegmentTransformMixin / FitByConcatenationMixin)

**Fitting Logic**:

```python
def fit_function_to_data_segments(self, data_segments, fit_func, metadata):
    # Determine data type (dense vs ragged)
    data_is_ragged = data_types.pop() is ak.Array
    
    if not data_is_ragged and transform_supports_dense:
        # Concatenate all sources along axis=0
        train = {key: np.concatenate([chunk[key] for chunk in data_segments], axis=0)}
        fit_func(train, metadata)
    
    elif data_is_ragged and not transform_supports_ragged:
        # Find variable dimension, concatenate, convert to numpy
        var_dim = find_singular_ragged_dim(data_segments[0])
        train = {key: ak.concatenate([...], axis=var_dim-1).to_numpy()}
        fit_func(train, metadata)
    
    elif data_is_ragged and transform_supports_ragged:
        # Concatenate ragged arrays directly
        train = {key: ak.concatenate([chunk[key] for chunk in data_segments], axis=var_dim-1)}
        fit_func(train, metadata)
```

**Transformation Logic**:

```python
def transform_function_to_data_segments(self, data_segments, transform_func, metadata):
    if not data_is_ragged and transform_supports_dense:
        # Direct transformation
        out = [transform_func(segment, metadata) for segment in data_segments]
    
    elif data_is_ragged and not transform_supports_ragged:
        # Process each ragged segment individually
        # Convert to numpy, transform, reassemble
        for segment in data_segments:
            var_dim = find_singular_ragged_dim(segment)
            multi_index = ragged_index_tuples(segment, var_dim)
            for coord in multi_index:
                subsegment = {k: segment[k][coord].to_numpy() for k in segment.keys()}
                transformed = transform_func(subsegment, metadata)
                # Reassemble into ragged structure
            out.append(reassembled_segment)
    
    elif data_is_ragged and transform_supports_ragged:
        # Direct transformation on ragged arrays
        out = [transform_func(segment, metadata) for segment in data_segments]
```

### Ragged Array Handling

The system handles ragged arrays through several utilities:

**Finding Variable Dimensions**:

```python
def find_singular_ragged_dim(nti: NamedTransformInput) -> Optional[int]:
    """Find the single variable-length dimension."""
    var_dims = [ak_find_var_dims(arr) for arr in nti.values()]
    unique_var_dims = set([x for row in var_dims for x in row])
    
    if len(unique_var_dims) == 0:
        return None  # Regular array
    elif len(unique_var_dims) == 1:
        return unique_var_dims.pop()
    else:
        raise ValueError("Multiple variable dimensions found")
```

**Reassembling Ragged Arrays**:

When a dense-only transform processes ragged data, the system:
1. Flattens along the variable dimension
2. Transforms each sub-segment
3. Reassembles into ragged structure using offsets

## Data Type Handling

### Type Detection

The system detects data types by checking instance classes:

```python
data_types = {v for chunk in data_segments for k, v in chunk.get_instance_cls().items()}
data_is_ragged = data_types.pop() is ak.Array
```

### Automatic Conversion

When a dense-only transform receives ragged data:

1. **Detection**: System detects ragged data + dense-only transform
2. **Warning**: Logs a warning about potential slowness
3. **Processing**: Iterates through ragged segments, converts to numpy
4. **Transformation**: Applies transform to each segment
5. **Reassembly**: Reconstructs ragged structure from transformed segments

### Regularization

Ragged arrays that are structurally regular (all dimensions fixed) are treated as dense:

```python
ragged_dims = {find_singular_ragged_dim(ds) for ds in data_segments}
data_is_ragged_but_regular = ragged_dims == {None}
if data_is_ragged_but_regular:
    data_is_ragged = False  # Treat as dense
```

## Creating Custom Transforms

### Step 1: Choose Base Class

Select the appropriate marker class:

```python
from picid.transforms.base.base_transform import DenseTransform

class MyTransform(DenseTransform):
    """My custom transform that only works with dense arrays."""
    pass
```

### Step 2: Implement transform_data

```python
def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any:
    """
    Transform a single data segment.
    
    Parameters
    ----------
    data : NamedTransformInput
        Dictionary-like object with data keys
    metadata : dict
        Contains 'mode', 'apply_to_keys', 'assign_to_keys', etc.
    
    Returns
    -------
    Any
        Transformed data (array, dict, or ReturnObject)
    """
    # Extract data
    key = list(data.keys())[0]  # Or use metadata['apply_to_keys']
    values = data[key]
    
    # Apply transformation
    transformed = your_transformation_logic(values)
    
    # Return (will be wrapped in SimpleReturnObject automatically)
    return transformed
```

### Step 3: Implement fit_data (if needed)

```python
def fit_data(self, data: NamedTransformInput, metadata: dict) -> None:
    """Fit the transform on data."""
    key = list(data.keys())[0]
    values = data[key]
    
    # Learn parameters
    self.mean_ = np.mean(values, axis=0)
    self.std_ = np.std(values, axis=0)
```

### Step 4: Add Multi-Source Support (if needed)

```python
from picid.transforms.base.multisource import ConcatFitAndPerSegmentTransformMixin

class MyTransform(
    DenseTransform,
    ConcatFitAndPerSegmentTransformMixin
):
    """Transform with multi-source fitting support."""
    
    def fit_data(self, data: NamedTransformInput, metadata: dict):
        # fit_data will be called with concatenated data
        pass
    
    def transform_data(self, data: NamedTransformInput, metadata: dict):
        # transform_data will be called per-segment
        pass
```

### Step 5: Handle Multiple Keys (if needed)

```python
def transform_data(self, data: NamedTransformInput, metadata: dict) -> dict:
    """Transform multiple keys."""
    apply_to_keys = metadata['apply_to_keys']
    
    result = {}
    for key in apply_to_keys:
        result[key] = your_transformation(data[key])
    
    # Return dict to match assign_to mapping
    return result
```

### Complete Example

```python
import numpy as np
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import (
    ConcatFitAndPerSegmentTransformMixin,
    InverseTransformMixin
)
from picid.data.data_objects.data import NamedTransformInput

class CustomScaler(
    DenseTransform,
    ConcatFitAndPerSegmentTransformMixin,
    InverseTransformMixin
):
    """Custom scaler that scales to [-1, 1] range."""
    
    def __init__(self):
        super().__init__()
        self.min_ = None
        self.max_ = None
    
    def fit_data(self, data: NamedTransformInput, metadata: dict) -> None:
        """Fit on concatenated training data."""
        key = list(data.keys())[0]
        values = data[key]
        
        self.min_ = np.min(values, axis=0)
        self.max_ = np.max(values, axis=0)
    
    def transform_data(
        self, data: NamedTransformInput, metadata: dict
    ) -> np.ndarray:
        """Transform a single data segment."""
        key = list(data.keys())[0]
        values = data[key]
        
        # Scale to [-1, 1]
        scaled = 2 * (values - self.min_) / (self.max_ - self.min_) - 1
        return scaled
    
    def inverse_transform(
        self, data: NamedTransformInput, metadata: dict
    ) -> np.ndarray:
        """Inverse transform (required by InverseTransformMixin)."""
        key = list(data.keys())[0]
        values = data[key]
        
        # Reverse scaling
        original = (values + 1) / 2 * (self.max_ - self.min_) + self.min_
        return original
```

`InverseTransformMixin` provides `inverse_transform_data` and `inverse_transform_multi_source`; you implement the abstract `inverse_transform`.

### Pipeline steps and hooks

The pipeline runs in order: **CopyStep** → **IntegrityCheckStep** → **FitStep** → **TransformStep** → **PostprocessStep** → **MergeStep** → **RecordManifestStep**. Lifecycle events (`on_pipeline_start`, `before_transform`, `after_transform`, `on_transform_error`, `on_pipeline_end`) can be subscribed to via `register_hook(event, callback)` in `picid.transforms.base.pipeline`. For examples (e.g. error logging, debugging), see [Hooks examples](HOOKS_EXAMPLES.md).

### Validation and integrity

Transform metadata supports two validation-related options:

1. **Built-in post-transform validation**  
   After each transform, `DataTransform.forward()` runs structural checks (return type, keys present, splits preserved). You can turn this off per transform with **`validate_output: false`** in metadata (e.g. for debugging or trusted pipelines). Default is `true`.

2. **Optional pipeline integrity checks**  
   The pipeline runs an **IntegrityCheckStep** after CopyStep. It executes a list of `(name, check_fn, policy)` where `policy` is `"fail"`, `"warn"`, or `"allow"`. You can pass **`integrity_checks`** in metadata: a list of `(name, check_fn, policy)` tuples. Each `check_fn` must be a no-arg callable that may raise; it is run during the step. Because callables cannot be expressed in YAML, **`integrity_checks`** is set **programmatically** when building the transform config (e.g. when constructing the metadata dict before passing it to `DataTransform` or `ConfigTransformManager`). Helpers: `picid.transforms.base.integrity.run_check`, `run_checks`. Integrity checks run **before** FitStep and TransformStep.

## Advanced Topics

### Signature Validation

The system validates transform signatures match metadata:

**Standard Signature**:
```python
def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any:
    pass
```

**Named Parameters** (when `apply_to` is a mapping):
```python
def transform_data(self, features: np.ndarray, target: np.ndarray, metadata: dict) -> Any:
    pass
```

The validation checks:
- Parameter names match `apply_to_map` keys
- `metadata` parameter is present
- Correct number of parameters

### Decorators

**@check_transform_output_consistency**:

Validates that transform outputs match `assign_to_map` expectations:

```python
@check_transform_output_consistency
def transform_data(self, data, metadata):
    # If assign_to_map = ['features'], output dict must have 'features' key
    return {'features': transformed_data}
```

**@inject_transform_context_to_strategy_apply**:

Adds context to exceptions during transformation:

```python
@inject_transform_context_to_strategy_apply
def apply(self, transform_instance, apply_to_keys, ...):
    # Exceptions include transform name and apply_to keys
```

### Return Objects

The system uses return objects to standardize outputs:

- **SimpleReturnObject**: Single array output
- **NamedDictReturnObject**: Dictionary output with named keys
- **ReturnObject**: Base class for all return types

### Copying Strategy

The system uses efficient copying:

```python
# Shallow copy for most data
transformed_data = data.copy(deep=False)

# Deep copy only for keys that will be modified
for key in keys_to_modify:
    if data[key].get_instance_cls() is not ak.Array:
        transformed_data[key] = data[key].copy(deep=True)  # NumPy: deep copy
    else:
        transformed_data[key] = data[key].copy(deep=False)  # Awkward: shallow
```

## Debugging

### Enable Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('picid.transforms')
```

### Common Issues

**1. Signature Mismatch**:

```
TypeError: Transform 'MyTransform': transform_data signature must be ...
```

**Solution**: Check that your signature matches the expected format.

**2. Output Key Missing**:

```
KeyError: OUTPUT KEY MISSING IN TRANSFORM MyTransform
```

**Solution**: Ensure output dict contains keys specified in `assign_to_map`.

**3. Ragged Dimension Ambiguity**:

```
ValueError: Expected at most one variable-length dimension, found 2
```

**Solution**: Ensure all arrays in `NamedTransformInput` have the same ragged structure.

**4. Data Type Mismatch**:

```
ValueError: Transform MyTransform does not support the data type {ak.Array}
```

**Solution**: Use appropriate marker class (`RaggedTransform` for ragged data).

### Debugging Tools

**Describe Data Type**:

```python
from picid.utils.rich_output import describe_data_type

describe_data_type(data, calculate_stat=True)
```

**Check Transform State**:

```python
# Check if transform is instantiated
manager.is_instantiated

# Get transform configuration
config = manager.get_transform_config("transform_name")

# Inspect transform instance
transform = manager.get_data_transform("transform_name")
print(transform.transform_instance)
```

### Error Messages

The system provides detailed error messages:

- **Transform context**: Transform name and apply_to keys
- **Data type details**: Ragged/dense flags, transform capabilities
- **Metadata**: Full metadata dictionary
- **Data segment**: Sample of problematic data

## Best Practices

1. **Inherit from appropriate marker class**: Declare data type support
2. **Use mixins for multi-source**: Don't reimplement multi-source logic
3. **Validate inputs**: Check data shapes/types in `transform_data`
4. **Return consistent types**: Use return objects for complex outputs
5. **Handle edge cases**: Empty arrays, single-element arrays, etc.
6. **Document assumptions**: Comment on expected input formats
7. **Test incrementally**: Test transforms in isolation before integration

## Performance Considerations

1. **Ragged → Dense conversion**: Can be slow for large datasets
2. **Multi-source concatenation**: Memory-intensive for many units
3. **Deep copying**: Minimize deep copies when possible
4. **Lazy instantiation**: Use for large transform pipelines

## Extension Points

The system is designed for extension:

1. **New base classes**: Create domain-specific base classes
2. **New mixins**: Add new multi-source strategies
3. **New return types**: Extend return object hierarchy
4. **New strategies**: Implement custom transformation strategies

---

For usage examples, see the [User Guide](guide.md). For quick reference, see the [Quick Start](quick_start.md).
