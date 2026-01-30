# Test Organization Structure

The test folder structure now mirrors the source code structure exactly.

## Structure Mapping

### Source Code → Test Code

```
lmetk/transforms/
├── base/                          test/transforms/
│   ├── base_transform.py      →   base/
│   ├── data_transform.py      →       test_base_transform.py
│   ├── strategy.py             →       test_data_transform.py
│   ├── multisource.py          →       test_strategy.py
│   └── transform_manager.py    →       test_multisource.py
│                                       test_transform_manager.py
│                                       conftest.py (shared fixtures)
│
└── base_transforms/                 base_transforms/
    ├── tabularizers.py          →       test_tabularizers.py
    ├── timefeatures.py           →       test_timefeatures.py
    ├── spectral.py               →       test_spectral.py
    ├── stfft.py                  →       test_stfft.py
    ├── time_statistics.py        →       test_time_statistics.py
    ├── autoencoder_transform.py  →       test_autoencoder.py
    ├── identity.py               →       test_identity.py
    ├── concatenate.py            →       test_concatenate.py
    ├── data_cleaning.py          →       test_data_cleaning.py
    ├── moving_average.py         →       test_moving_average.py
    ├── padding2length.py         →       test_padding2length.py
    ├── plot_transforms.py        →       test_plot_transforms.py
    ├── reshaping.py              →       test_reshaping.py
    ├── scaler.py                 →       test_scaler.py
    ├── shape_manipulations.py    →       test_shape_manipulations.py
    └── subsample.py              →       test_subsample.py
```

## Organization Principles

1. **One-to-One Mapping**: Each source module has a corresponding test file in the same relative location
2. **Shared Fixtures**: Common test utilities and dummy transforms are in `test/transforms/base/conftest.py`
3. **Module-Specific Tests**: Each test file tests exactly one source module
4. **Consolidated Tests**: All test classes for a module are in a single file

## Test Files by Module

### `test/transforms/base/` - Tests for `lmetk.transforms.base`

- `test_base_transform.py` - Tests for `base_transform.py` (BaseTransform, DenseTransform, RaggedTransform)
- `test_data_transform.py` - Tests for `data_transform.py` (DataTransform class)
- `test_strategy.py` - Tests for `strategy.py` (TransformStrategy, postprocess_transformed_data)
- `test_multisource.py` - Tests for `multisource.py` (multisource mixins and utilities)
- `test_transform_manager.py` - Tests for `transform_manager.py` (ConfigTransformManager)
- `conftest.py` - Shared fixtures and dummy transforms used across base module tests

### `test/transforms/base_transforms/` - Tests for `lmetk.transforms.base_transforms`

- One test file per transform module (e.g., `test_tabularizers.py` for `tabularizers.py`)
- All tests for base_transforms are consolidated into single files per module

## Benefits

1. **Clear Organization**: Easy to find tests for any given module
2. **Maintainability**: Changes to a module's tests are isolated to one file
3. **Consistency**: Test structure mirrors source structure exactly
4. **Discoverability**: Developers can easily locate relevant tests
