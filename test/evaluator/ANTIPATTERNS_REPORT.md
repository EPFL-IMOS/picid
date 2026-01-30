# Test Antipatterns Analysis Report

## Executive Summary

This report identifies test antipatterns in `test/evaluator/` that could mask bugs by:
1. Bypassing normal code paths
2. Over-mocking critical functionality
3. Missing assertions on side effects
4. Testing implementation details instead of behavior

---

## Critical Antipatterns

### 1. **Patching Parent `compute()` Method** ⚠️ HIGH RISK

**Location**: Multiple files

**Problem**: Tests patch `DefaultEvaluator.compute()` to return `{}`, bypassing the actual orchestration logic.

**Examples**:
- `test_multiunit_evaluator.py:398` - `test_plotting_triggers`
- `test_multiunit_evaluator.py:565` - `test_plotting_skips_when_no_logger`
- `test_reconstruction_evaluator.py:200` - `test_compute_plots_denormalized_if_scaling_on`
- `test_reconstruction_evaluator.py:229` - `test_compute_plots_normalized_if_scaling_off`

**Why This Masks Bugs**:
- `compute()` orchestrates: `_prepare_predictions_for_save()` → `_save_predictions_to_file()` → `_compute_metrics()`
- Patching it means we're not testing:
  - Whether `_save_predictions_to_file()` is called correctly
  - Whether the return value from `_compute_metrics()` is properly returned
  - Integration between parent and child `compute()` methods
  - Error handling in the orchestration

**Example Bug That Would Be Missed**:
```python
# If compute() had a bug like this:
def compute(self, mode, epoch, step):
    self._prepare_predictions_for_save()
    self._save_predictions_to_file(...)
    # BUG: Forgot to return _compute_metrics()
    return {}  # Always returns empty dict
```

**Recommendation**: 
- Don't patch `compute()` - test the full flow
- If you need to isolate, mock `_save_predictions_to_file()` and `_prepare_predictions_for_save()` instead
- Verify the actual return value from `compute()`

---

### 2. **Direct State Manipulation** ⚠️ MEDIUM RISK

**Location**: `test_multiunit_evaluator.py`

**Problem**: Tests directly set `eval.preds = [...]` instead of using `update()`, bypassing normal data flow.

**Examples**:
- `test_multiunit_evaluator.py:1145` - `test_log_pred_vs_target_curves_concatenation_error`
- `test_multiunit_evaluator.py:1174` - `test_log_pred_vs_target_curves_3d_unit_ids_warning`
- `test_multiunit_evaluator.py:1205` - `test_log_pred_vs_target_curves_2d_unit_ids`
- `test_multiunit_evaluator.py:1245` - `test_log_pred_vs_target_curves_subsampling`

**Why This Masks Bugs**:
- `update()` validates shapes, applies scaling, routes to metrics
- Direct manipulation skips:
  - Shape validation in `update()`
  - Scaling logic in `update()`
  - Unit ID processing in `update()`
  - Metric updates

**Example Bug That Would Be Missed**:
```python
# If update() had a bug:
def update(self, model_out):
    # BUG: Forgot to validate unit_ids shape
    unit_ids = model_out["unit_id"]
    # Would crash with 3D unit_ids, but test bypasses this
```

**Recommendation**:
- Use `update()` for normal test cases
- Only use direct manipulation for testing **defensive error handling** (corrupted state that shouldn't occur)
- Add comments explaining why direct manipulation is necessary

---

### 3. **Mocking Parent's `_compute_metrics()`** ⚠️ MEDIUM RISK

**Location**: `test_classification_evaluator.py`

**Problem**: Tests mock `DefaultEvaluator._compute_metrics()` instead of testing actual integration.

**Examples**:
- `test_classification_evaluator.py:545-553` - `test_compute_metrics_handles_denormalized_suffix`
- `test_classification_evaluator.py:581-586` - `test_compute_metrics_raises_on_unexpected_suffix`
- `test_classification_evaluator.py:614-622` - `test_compute_metrics_conflict_raises_assertion`

**Why This Masks Bugs**:
- `ClassificationEvaluator._compute_metrics()` calls `super()._compute_metrics()` and then processes the result
- Mocking the parent means we're not testing:
  - Whether `super()` is called correctly
  - Whether the parent's return value format matches expectations
  - Integration bugs between parent and child

**Example Bug That Would Be Missed**:
```python
# If ClassificationEvaluator had:
def _compute_metrics(self):
    # BUG: Called wrong parent method
    results = super().compute()  # Wrong! Should be _compute_metrics()
    # Process results...
```

**Recommendation**:
- Don't mock `_compute_metrics()` - let it run normally
- Mock the **metrics themselves** (the `self.metrics` dict) if you need control
- Or use real metrics with known inputs/outputs

---

### 4. **Missing Assertions on Side Effects** ⚠️ MEDIUM RISK

**Location**: Multiple files

**Problem**: Tests call methods but don't verify important side effects.

**Examples**:

#### `test_forecaster_evaluator.py:25-55` - `test_update_slices_effective_pred_len`
- ✅ Verifies what was passed to parent `update()`
- ❌ Doesn't verify that `self.preds`/`self.targets` were updated correctly
- ❌ Doesn't verify that metrics were updated

#### `test_multiunit_evaluator.py:239` - `test_update_splits_data_by_unit_1d`
- ✅ Verifies metric values
- ❌ Doesn't verify that `self.unit_ids` was appended
- ❌ Doesn't verify that `self.preds`/`self.targets` were collected

**Why This Masks Bugs**:
- Methods often have multiple side effects
- Only checking one aspect can miss bugs in other aspects

**Example Bug That Would Be Missed**:
```python
# If update() had:
def update(self, model_out):
    # Process predictions correctly
    self.preds.append(preds)
    # BUG: Forgot to append unit_ids
    # self.unit_ids.append(unit_ids)  # Missing!
```

**Recommendation**:
- Assert on all important side effects
- Verify state changes (collections, metrics, flags)
- Use helper functions to check multiple aspects

---

### 5. **Over-Mocking Critical Functionality** ⚠️ LOW-MEDIUM RISK

**Location**: `test_multiunit_evaluator.py`, `test_default_evaluator.py`

**Problem**: Tests mock `MetricFactory.create_metric()` to return `MagicMock()`, losing real metric behavior.

**Examples**:
- Most fixtures use `mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())`
- This means metrics don't actually accumulate, reset, or compute correctly

**Why This Masks Bugs**:
- Real metrics have state (counters, accumulators)
- Mock metrics don't test:
  - Whether `metric.update()` is called correctly
  - Whether `metric.compute()` returns correct values
  - Whether `metric.reset()` works
  - Accumulation logic

**Example Bug That Would Be Missed**:
```python
# If update() had:
def update(self, model_out):
    # BUG: Passed wrong arguments to metric.update()
    metric.update(preds, targets, extra_arg=5)  # Wrong signature
    # Mock would accept this, real metric would crash
```

**Recommendation**:
- Use real metric instances for integration tests
- Only mock metrics for unit tests that focus on orchestration
- Create a `MockMetric` class that mimics real behavior (like `test_multiunit_evaluator.py` does)

---

### 6. **Testing Implementation Details Instead of Behavior** ⚠️ LOW RISK

**Location**: `test_base_evaluator.py`

**Problem**: Some tests verify internal structure rather than external behavior.

**Examples**:
- `test_base_evaluator.py:306` - `test_abstract_methods_exist`
  - Tests that abstract methods are defined, but doesn't test that they work
- `test_base_evaluator.py:232` - `test_repr_excludes_complex_objects`
  - Tests internal representation format, not behavior

**Why This Masks Bugs**:
- Tests break on refactoring even if behavior is correct
- Don't catch bugs where methods exist but don't work correctly

**Recommendation**:
- Test behavior, not structure
- If testing structure is necessary, also test behavior

---

### 7. **Incomplete Error Path Testing** ⚠️ LOW RISK

**Location**: `test_forecaster_evaluator.py`

**Problem**: Tests verify exceptions are raised but don't verify error messages or context.

**Examples**:
- `test_forecaster_evaluator.py:146` - `test_update_raises_shape_mismatch`
  - ✅ Verifies `ValueError` is raised
  - ❌ Doesn't verify error message contains useful info
  - ❌ Doesn't verify that state wasn't corrupted

**Why This Masks Bugs**:
- Wrong exception type might be raised
- Error message might be unhelpful
- State might be corrupted before exception

**Recommendation**:
- Verify exception message with `match=` parameter
- Verify state is clean after exception
- Test that partial updates don't occur

---

## Summary of Recommendations

### High Priority
1. **Stop patching `compute()`** - Test full orchestration flow
2. **Use `update()` instead of direct state manipulation** - Only use direct manipulation for defensive error handling tests

### Medium Priority
3. **Don't mock parent `_compute_metrics()`** - Test actual integration
4. **Assert on all side effects** - Verify state changes comprehensively
5. **Use real metrics for integration tests** - Only mock for unit tests

### Low Priority
6. **Test behavior, not structure** - Focus on external contracts
7. **Verify error messages and state** - Complete error path testing

---

## Files Requiring Immediate Attention

1. **`test_multiunit_evaluator.py`** - Multiple antipatterns (patching compute, direct manipulation)
2. **`test_classification_evaluator.py`** - Mocking parent `_compute_metrics()`
3. **`test_reconstruction_evaluator.py`** - Patching `compute()`
4. **`test_forecaster_evaluator.py`** - Missing side effect assertions

---

## Example: How to Fix a Test

### Before (Antipattern):
```python
def test_plotting_triggers(mock_plt, multiunit_evaluator_with_plotting_fixture):
    eval = multiunit_evaluator_with_plotting_fixture
    eval.update({"predictions": preds, "targets": targets, "unit_id": unit_ids})
    eval.log_plot = MagicMock()
    
    # ANTIPATTERN: Patching compute() bypasses orchestration
    with patch("lmetk.evaluator.default.DefaultEvaluator.compute", return_value={}):
        eval.compute(mode="val", epoch=0, step=0)
    
    mock_plt.subplots.assert_called()
    eval.log_plot.assert_called()
```

### After (Fixed):
```python
def test_plotting_triggers(mock_plt, multiunit_evaluator_with_plotting_fixture):
    eval = multiunit_evaluator_with_plotting_fixture
    eval.update({"predictions": preds, "targets": targets, "unit_id": unit_ids})
    eval.log_plot = MagicMock()
    
    # FIXED: Mock only file operations, not orchestration
    with patch.object(eval, '_save_predictions_to_file'):
        results = eval.compute(mode="val", epoch=0, step=0)
    
    # FIXED: Verify full behavior
    mock_plt.subplots.assert_called()
    eval.log_plot.assert_called()
    assert isinstance(results, dict)  # Verify compute() returns metrics
    assert len(results) > 0  # Verify metrics were computed
```

---

## Conclusion

These antipatterns create a false sense of security. Tests pass even when critical bugs exist in:
- Orchestration logic (`compute()`)
- Data flow (`update()`)
- Integration between parent/child classes
- Side effects and state management

Addressing these issues will significantly improve test reliability and bug detection.
