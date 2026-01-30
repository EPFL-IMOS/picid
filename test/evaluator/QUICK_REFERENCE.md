# Quick Reference: Using Test Fixtures

## 🚀 Quick Start

### Import Fixtures (Automatic)
Fixtures are automatically available via `conftest.py`. Just add them as test parameters:

```python
def test_my_evaluator(evaluator_fixture, regression_data_basic):
    evaluator_fixture.update(regression_data_basic)
    results = evaluator_fixture.compute()
    assert "mse" in results
```

---

## 📦 Available Fixtures

### Regression/Forecasting
```python
regression_data_basic              # (5, 1, 1) - Basic RUL predictions
regression_data_large_batch        # (1000, 1, 1) - Stress test
regression_data_single_sample      # (1, 1, 1) - Edge case
forecasting_data_multi_step        # (10, 24, 1) - Degradation curves
forecasting_data_fault_onset       # (5, 50, 1) - Fault detection
forecasting_data_multivariate      # (8, 20, 5) - Multi-sensor
```

### Classification
```python
classification_data_balanced        # Balanced classes, clear predictions
classification_data_imbalanced      # 60/30/10% distribution
classification_data_misclassified   # 25% error rate
```

### Multi-Unit
```python
multiunit_data_1d_units           # 5 units, uneven distribution
multiunit_data_2d_units           # Hierarchical (dataset, unit)
multiunit_data_many_units         # 20 units, stress test
multiunit_data_time_series        # Multi-unit with time series
```

### Edge Cases
```python
data_empty_batch                  # Empty input
data_perfect_predictions          # Zero error
data_extreme_values               # Boundaries (0.0, 1.0)
```

---

## 💡 Usage Examples

### Basic Usage
```python
def test_regression(evaluator, regression_data_basic):
    evaluator.update(regression_data_basic)
    results = evaluator.compute()
    assert "mse" in results
```

### Multi-Unit Usage
```python
def test_multiunit(evaluator, multiunit_data_1d_units):
    evaluator.update(multiunit_data_1d_units)
    results = evaluator._compute_metrics()
    assert "mse_denormalized_mean" in results
```

### Classification Usage
```python
def test_classification(evaluator, classification_data_balanced):
    evaluator.update(classification_data_balanced)
    results = evaluator.compute()
    assert "accuracy" in results
```

### Edge Case Usage
```python
def test_empty_batch(evaluator, data_empty_batch):
    evaluator.update(data_empty_batch)  # Should not raise
    results = evaluator.compute()
    assert isinstance(results, dict)
```

---

## 🔧 Helper Function

```python
from test.evaluator.conftest import create_model_out

# Create model_out with unit IDs
model_out = create_model_out(
    predictions=preds,
    targets=targets,
    unit_id=unit_ids  # Optional
)
```

---

## 📋 Fixture Properties

### regression_data_basic
- **Shape**: (5, 1, 1)
- **Values**: Linear degradation 0.9 → 0.1
- **Use for**: Basic regression tests

### multiunit_data_1d_units
- **Shape**: (15, 1, 1) + unit_id: (15,)
- **Units**: 5 units (1-5)
- **Distribution**: Uneven (5, 4, 3, 2, 1 samples)
- **Use for**: Multi-unit routing tests

### classification_data_balanced
- **Shape**: (12, 3, 4) preds, (12, 3, 1) targets
- **Classes**: 4 classes, balanced
- **Use for**: Classification accuracy tests

---

## ⚠️ Important Notes

1. **Fixtures are automatically available** - No import needed
2. **Use appropriate fixtures** - Match fixture to evaluator type
3. **Fixtures are realistic** - They test real PHM scenarios
4. **Fixtures are challenging** - They catch bugs!

---

## 🎯 Best Practices

1. ✅ Use fixtures instead of manual data creation
2. ✅ Match fixture to evaluator type
3. ✅ Use edge case fixtures for edge case tests
4. ✅ Document why you chose a specific fixture
5. ✅ Verify fixture properties in assertions

---

## 📖 Full Documentation

See `FIXTURE_USAGE.md` for complete documentation.
