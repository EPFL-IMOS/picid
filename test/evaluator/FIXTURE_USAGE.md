# Data Fixture Usage Guide

This document explains how to use the shared data fixtures in `conftest.py` for testing evaluators.

## Overview

The `conftest.py` file provides comprehensive, realistic data fixtures for all evaluator types. These fixtures are designed to be "interesting" and challenging, testing:
- Edge cases (empty batches, single samples, large batches)
- Realistic PHM scenarios (degradation curves, fault signatures)
- Various data distributions (balanced, imbalanced, multi-unit)
- Different shapes and dimensions

## Available Fixtures

### Regression/Forecasting Fixtures

#### `regression_data_basic`
- **Shape**: `(batch=5, time=1, features=1)`
- **Use case**: Basic RUL predictions with linear degradation
- **Interesting aspects**: Small batch, single timestep, full normalized range

```python
def test_basic_regression(evaluator, regression_data_basic):
    evaluator.update(regression_data_basic)
    results = evaluator.compute()
    assert "mse" in results
```

#### `regression_data_large_batch`
- **Shape**: `(batch=1000, time=1, features=1)`
- **Use case**: Stress testing with many samples
- **Interesting aspects**: Large batch, realistic noise, outliers

#### `regression_data_single_sample`
- **Shape**: `(batch=1, time=1, features=1)`
- **Use case**: Edge case testing
- **Interesting aspects**: Minimum batch size

#### `forecasting_data_multi_step`
- **Shape**: `(batch=10, time=24, features=1)`
- **Use case**: Multi-step forecasting with degradation curves
- **Interesting aspects**: Multiple timesteps, exponential decay, varying degradation rates

#### `forecasting_data_fault_onset`
- **Shape**: `(batch=5, time=50, features=1)`
- **Use case**: Testing fault detection capability
- **Interesting aspects**: Sudden degradation change, model lag

#### `forecasting_data_multivariate`
- **Shape**: `(batch=8, time=20, features=5)`
- **Use case**: Multivariate forecasting with multiple sensors
- **Interesting aspects**: Correlated channels, different scales

### Classification Fixtures

#### `classification_data_balanced`
- **Shape**: `(batch=12, time=3, classes=4)` predictions, `(batch=12, time=3, 1)` targets
- **Use case**: Balanced class distribution
- **Interesting aspects**: Equal classes, clear predictions, multiple timesteps

```python
def test_balanced_classification(evaluator, classification_data_balanced):
    evaluator.update(classification_data_balanced)
    results = evaluator.compute()
    assert "accuracy" in results
```

#### `classification_data_imbalanced`
- **Shape**: `(batch=20, time=2, classes=3)` predictions, `(batch=20, time=2, 1)` targets
- **Use case**: Testing imbalanced data handling
- **Interesting aspects**: Highly imbalanced (60/30/10%), ambiguous predictions

#### `classification_data_misclassified`
- **Shape**: `(batch=8, time=4, classes=3)` predictions, `(batch=8, time=4, 1)` targets
- **Use case**: Testing error handling
- **Interesting aspects**: 25% misclassification rate, realistic mistakes

### Multi-Unit Fixtures

#### `multiunit_data_1d_units`
- **Shape**: `(batch=15, time=1, features=1)` + `unit_id: (15,)`
- **Use case**: Simple unit identification
- **Interesting aspects**: Multiple units (1-5), uneven distribution, different degradation rates

```python
def test_multiunit_1d(evaluator, multiunit_data_1d_units):
    evaluator.update(multiunit_data_1d_units)
    results = evaluator.compute()
    # Verify per-unit metrics
    assert 1 in evaluator.unit_metrics
```

#### `multiunit_data_2d_units`
- **Shape**: `(batch=12, time=1, features=1)` + `unit_id: (12, 2)`
- **Use case**: Hierarchical unit identification (dataset, unit)
- **Interesting aspects**: Multiple datasets, same unit IDs across datasets

#### `multiunit_data_many_units`
- **Shape**: `(batch=50, time=1, features=1)` + `unit_id: (50,)`
- **Use case**: Stress testing with many units
- **Interesting aspects**: 20 different units, sparse samples per unit

#### `multiunit_data_time_series`
- **Shape**: `(batch=20, time=10, features=1)` + `unit_id: (20,)`
- **Use case**: Multi-unit with time series
- **Interesting aspects**: Multiple timesteps, different degradation patterns per unit

### Edge Case Fixtures

#### `data_empty_batch`
- **Shape**: `(batch=0, time=1, features=1)`
- **Use case**: Testing graceful handling of empty input

#### `data_perfect_predictions`
- **Shape**: `(batch=10, time=1, features=1)`
- **Use case**: Testing lower bound of metrics (should be 0)

#### `data_extreme_values`
- **Shape**: `(batch=5, time=1, features=1)`
- **Use case**: Testing boundary conditions (0.0, 1.0)

## Helper Function

### `create_model_out(predictions, targets, unit_id=None)`

Helper function to create `model_out` dictionary:

```python
from test.evaluator.conftest import create_model_out

preds = np.array([[[0.5]]])
targets = np.array([[[0.52]]])
unit_ids = np.array([1])

model_out = create_model_out(preds, targets, unit_ids)
evaluator.update(model_out)
```

## Migration Guide

### Before (Manual Data Creation)
```python
def test_update(evaluator):
    preds = np.array([[[10.0]], [[20.0]]])
    targets = np.array([[[10.0]], [[20.0]]])
    evaluator.update({"predictions": preds, "targets": targets})
```

### After (Using Fixtures)
```python
def test_update(evaluator, regression_data_basic):
    evaluator.update(regression_data_basic)
    # More realistic, challenging data automatically
```

## Best Practices

1. **Use appropriate fixtures**: Match fixture to evaluator type
   - Regression → `regression_data_*`
   - Classification → `classification_data_*`
   - Multi-unit → `multiunit_data_*`
   - Forecasting → `forecasting_data_*`

2. **Test edge cases**: Use `data_empty_batch`, `data_extreme_values` for edge cases

3. **Combine fixtures**: Use multiple fixtures in one test for comprehensive coverage

4. **Verify fixture properties**: Assert on fixture characteristics (shapes, distributions)

5. **Document fixture usage**: Add comments explaining why a specific fixture is used

## Example: Comprehensive Test

```python
def test_evaluator_comprehensive(
    evaluator,
    regression_data_basic,
    regression_data_large_batch,
    data_perfect_predictions,
):
    """Test evaluator with various data scenarios."""
    
    # Test 1: Basic data
    evaluator.reset()
    evaluator.update(regression_data_basic)
    results1 = evaluator.compute()
    assert "mse" in results1
    
    # Test 2: Large batch
    evaluator.reset()
    evaluator.update(regression_data_large_batch)
    results2 = evaluator.compute()
    assert "mse" in results2
    
    # Test 3: Perfect predictions (should have zero error)
    evaluator.reset()
    evaluator.update(data_perfect_predictions)
    results3 = evaluator.compute()
    assert results3["mse"] < 1e-6  # Near zero
```

## Benefits

1. **Realistic testing**: Fixtures simulate real PHM scenarios
2. **Consistency**: Same data across tests for reproducibility
3. **Challenging**: "Interesting" data tests edge cases
4. **Maintainability**: Centralized data creation
5. **Documentation**: Fixtures document expected data formats
