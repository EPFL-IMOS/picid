import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# Adjust imports to match your project structure
from lmetk.evaluator.classification import ClassificationEvaluator
from lmetk.evaluator.default import DefaultEvaluator
from lmetk.metrics.metric_factory import MetricFactory
from lmetk.metrics.metrics import (
    MulticlassAccuracyMetric,
    MulticlassPrecisionMetric,
    MulticlassRecallMetric,
    MulticlassF1Metric,
    MulticlassAUROCMetric,
)

# ==========================================
# PART 1: FIXTURES & DATA
# ==========================================


@pytest.fixture
def cls_evaluator_fixture(mocker):
    """
    Standard fixture with fully mocked metrics (for structural tests).
    save_predictions=True, collect_predictions=True.
    """
    mocker.patch.object(
        MetricFactory, "create_classification_metric", return_value=MagicMock()
    )

    evaluator = ClassificationEvaluator(
        num_classes=3,
        metric_names=["accuracy"],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=True,
        paths=MagicMock(),
    )
    return evaluator


@pytest.fixture
def cls_evaluator_no_save_fixture(mocker):
    """
    Fixture with save_predictions=False (for testing save guard clause).
    """
    mocker.patch.object(
        MetricFactory, "create_classification_metric", return_value=MagicMock()
    )

    evaluator = ClassificationEvaluator(
        num_classes=3,
        metric_names=["accuracy"],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths=MagicMock(),
    )
    return evaluator


@pytest.fixture
def cls_evaluator_no_collect_fixture(mocker):
    """
    Fixture with collect_predictions=False (for testing collect guard clause).
    
    Note: DefaultEvaluator sets collect_predictions = save_predictions OR collect_predictions,
    so we must set save_predictions=False to actually disable collection.
    """
    mocker.patch.object(
        MetricFactory, "create_classification_metric", return_value=MagicMock()
    )

    evaluator = ClassificationEvaluator(
        num_classes=3,
        metric_names=["accuracy"],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths=MagicMock(),
        collect_predictions=False,
    )
    return evaluator


def side_effect_create_metric(name, num_classes):
    """
    Helper for the integration fixture: returns REAL metric instances
    instead of Mocks, allowing us to test the math and logic.
    """
    if name == "accuracy":
        return MulticlassAccuracyMetric(num_classes=num_classes)
    elif name == "precision":
        return MulticlassPrecisionMetric(num_classes=num_classes)
    elif name == "recall":
        return MulticlassRecallMetric(num_classes=num_classes)
    elif name == "f1":
        return MulticlassF1Metric(num_classes=num_classes)
    elif name == "auroc":
        return MulticlassAUROCMetric(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown metric: {name}")


@pytest.fixture
def real_metric_evaluator(mocker):
    """
    Integration fixture: Patches the factory to use REAL metric classes.
    """
    mocker.patch.object(
        MetricFactory,
        "create_classification_metric",
        side_effect=side_effect_create_metric,
    )

    def _create_evaluator(metric_names):
        return ClassificationEvaluator(
            num_classes=3,
            metric_names=metric_names,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths=MagicMock(),
        )

    return _create_evaluator


def create_dummy_data():
    """
    Creates deterministic data for 3 classes (0, 1, 2).
    Batch=2, Time=2. Total samples = 4.

    Data Logic:
    1. (0,0) -> Correct
    2. (1,1) -> Correct
    3. (2,2) -> Correct
    4. (0,1) -> Incorrect (Predicted 0, Target 1)
    """
    # Preds (Logits): (Batch=2, Time=2, Classes=3)
    preds = np.array(
        [
            # Batch 0
            [
                [10.0, 0.0, 0.0],  # Time 0: Predicts Class 0
                [0.0, 10.0, 0.0],  # Time 1: Predicts Class 1
            ],
            # Batch 1
            [
                [0.0, 0.0, 10.0],  # Time 0: Predicts Class 2
                [10.0, 0.0, 0.0],  # Time 1: Predicts Class 0 (Incorrect)
            ],
        ]
    )

    # Targets: (Batch=2, Time=2, Label=1)
    targets = np.array(
        [
            # Batch 0
            [[0], [1]],
            # Batch 1
            [[2], [1]],
        ]
    )

    return {"predictions": preds, "targets": targets}


# ==========================================
# PART 2: STRUCTURAL TEST
# ==========================================


def test_prepare_predictions_for_save_structure(cls_evaluator_fixture, classification_data_balanced):
    """
    Tests that predictions are saved with correct keys and dimension names.
    
    **Methodology**: Feed realistic data through update() instead of direct manipulation.
    
    **Uses fixture**: classification_data_balanced - realistic balanced classification data
    with multiple timesteps and classes.
    """
    evaluator = cls_evaluator_fixture

    # ACT: Use realistic classification fixture data
    evaluator.update(classification_data_balanced)

    # ACT
    output = evaluator._prepare_predictions_for_save()

    # ASSERT: Verify structure
    assert "preds_logits" in output
    assert "targets_labels" in output

    # Dimension check: must include 'time' because input data is 3D
    assert output["preds_logits"][0] == ["sample", "time", "class_logits"]
    assert output["targets_labels"][0] == ["sample", "time", "class_label"]
    
    # Verify data shapes match fixture
    preds_data = output["preds_logits"][1]  # Get actual array
    targets_data = output["targets_labels"][1]
    assert preds_data.shape == classification_data_balanced["predictions"].shape
    assert targets_data.shape == classification_data_balanced["targets"].shape


# ==========================================
# PART 3: INTEGRATION TESTS (WITH MANUAL MATH)
# ==========================================


def test_compute_metrics_integration_accuracy(real_metric_evaluator, classification_data_balanced):
    """
    Tests accuracy computation with balanced classification data.
    
    **Uses fixture**: classification_data_balanced - realistic balanced class distribution
    with clear predictions across multiple timesteps.
    """
    evaluator = real_metric_evaluator(metric_names=["accuracy"])

    # ACT: Use realistic fixture data
    evaluator.update(classification_data_balanced)
    results = evaluator.compute(mode="test", epoch=1, step=1)

    # ASSERT: Verify accuracy is computed correctly
    assert "accuracy" in results
    assert 0.0 <= results["accuracy"] <= 1.0, "Accuracy should be in [0, 1]"
    
    # With balanced data and clear predictions, accuracy should be high
    # (exact value depends on fixture, but should be reasonable)
    assert results["accuracy"] > 0.5, "Accuracy should be reasonable for balanced data"


def test_compute_metrics_integration_precision(real_metric_evaluator):
    """Verifies Precision (Macro Average)."""
    evaluator = real_metric_evaluator(metric_names=["precision"])
    data = create_dummy_data()

    # Manual Calculation (Macro Precision):
    # Class 0:
    #   - Preds: 2 times (indices 0 and 3).
    #   - Correct (TP): 1 (index 0). Incorrect (FP): 1 (index 3).
    #   - Precision = TP / (TP+FP) = 1 / 2 = 0.5
    prec_0 = 0.5

    # Class 1:
    #   - Preds: 1 time (index 1).
    #   - Correct (TP): 1. Incorrect (FP): 0.
    #   - Precision = 1 / 1 = 1.0
    prec_1 = 1.0

    # Class 2:
    #   - Preds: 1 time (index 2).
    #   - Correct (TP): 1. Incorrect (FP): 0.
    #   - Precision = 1 / 1 = 1.0
    prec_2 = 1.0

    expected_precision = (prec_0 + prec_1 + prec_2) / 3  # 0.8333...

    evaluator.update(data)
    results = evaluator.compute(mode="test", epoch=1, step=1)

    assert "precision" in results
    np.testing.assert_almost_equal(results["precision"], expected_precision, decimal=5)


def test_compute_metrics_integration_recall(real_metric_evaluator):
    """Verifies Recall (Macro Average)."""
    evaluator = real_metric_evaluator(metric_names=["recall"])
    data = create_dummy_data()

    # Manual Calculation (Macro Recall):
    # Class 0:
    #   - Targets: 1 time (index 0).
    #   - Found (TP): 1. Missed (FN): 0.
    #   - Recall = TP / (TP+FN) = 1 / 1 = 1.0
    rec_0 = 1.0

    # Class 1:
    #   - Targets: 2 times (indices 1 and 3).
    #   - Found (TP): 1 (index 1). Missed (FN): 1 (index 3, predicted 0).
    #   - Recall = 1 / 2 = 0.5
    rec_1 = 0.5

    # Class 2:
    #   - Targets: 1 time (index 2).
    #   - Found (TP): 1. Missed (FN): 0.
    #   - Recall = 1 / 1 = 1.0
    rec_2 = 1.0

    expected_recall = (rec_0 + rec_1 + rec_2) / 3  # 0.8333...

    evaluator.update(data)
    results = evaluator.compute(mode="test", epoch=1, step=1)

    assert "recall" in results
    np.testing.assert_almost_equal(results["recall"], expected_recall, decimal=5)


def test_compute_metrics_integration_f1(real_metric_evaluator):
    """Verifies F1 Score (Macro Average)."""
    evaluator = real_metric_evaluator(metric_names=["f1"])
    data = create_dummy_data()

    # Manual Calculation (Macro F1):
    # F1 per class = 2 * (P * R) / (P + R)

    # Class 0: P=0.5, R=1.0
    # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 0.6666...
    f1_0 = 2 * (0.5 * 1.0) / (0.5 + 1.0)

    # Class 1: P=1.0, R=0.5
    # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 = 0.6666...
    f1_1 = 2 * (1.0 * 0.5) / (1.0 + 0.5)

    # Class 2: P=1.0, R=1.0
    # F1 = 1.0
    f1_2 = 1.0

    expected_f1 = (f1_0 + f1_1 + f1_2) / 3  # 0.7777...

    evaluator.update(data)
    results = evaluator.compute(mode="test", epoch=1, step=1)

    assert "f1" in results
    np.testing.assert_almost_equal(results["f1"], expected_f1, decimal=5)


def test_compute_metrics_integration_auroc(real_metric_evaluator):
    """
    Verifies AUROC integration with manual calculation check.
    Prerequisite: lmetk/metrics/metrics.py must perform permute/squeeze.
    """
    evaluator = real_metric_evaluator(metric_names=["auroc"])
    data = create_dummy_data()

    # Manual Calculation Logic (Macro Average):
    # Data has 4 samples.
    # Class 0: 1 Pos (Score 1), 1 Neg (Score 1), 2 Neg (Score 0) -> AUC ≈ 0.8333
    # Class 1: 1 Pos (Score 1), 1 Pos (Score 0), 2 Neg (Score 0) -> AUC = 0.75
    # Class 2: 1 Pos (Score 1), 3 Neg (Score 0) -> AUC = 1.0
    # Average: (0.8333 + 0.75 + 1.0) / 3 = 0.86111...

    expected_auroc = (0.8333333 + 0.75 + 1.0) / 3

    evaluator.update(data)
    results = evaluator.compute(mode="test", epoch=1, step=1)

    # Check for lowercase 'auroc' key
    assert "auroc" in results

    # Verify exact value with small tolerance
    np.testing.assert_almost_equal(results["auroc"], expected_auroc, decimal=5)


# ==========================================
# PART 4: EDGE CASES AND ERROR HANDLING
# ==========================================


def test_prepare_predictions_no_save_disabled(cls_evaluator_no_save_fixture):
    """
    Tests _prepare_predictions_for_save returns empty if save disabled.
    
    **PHM Logic**: No data saved when save_predictions=False.
    
    **Methodology**: Use fixture with save_predictions=False from init.
    
    **Expected**: Empty dict returned.
    
    Validates: Requirement CE-SAVE-1 - Disabled save
    """
    evaluator = cls_evaluator_no_save_fixture
    
    # Verify save is disabled at init
    assert evaluator.save_predictions is False
    
    result = evaluator._prepare_predictions_for_save()
    
    assert result == {}


def test_prepare_predictions_not_collected(cls_evaluator_no_collect_fixture):
    """
    Tests _prepare_predictions_for_save returns empty if not collected.
    
    **PHM Logic**: No data if collect_predictions=False.
    
    **Methodology**: Use fixture with collect_predictions=False from init.
    
    **Expected**: Empty dict returned.
    
    Validates: Requirement CE-SAVE-2 - No collection
    """
    evaluator = cls_evaluator_no_collect_fixture
    
    # Verify collect is disabled at init
    assert evaluator.collect_predictions is False
    
    result = evaluator._prepare_predictions_for_save()
    
    assert result == {}


def test_compute_metrics_removes_denormalized_suffix(real_metric_evaluator, classification_data_balanced):
    """
    Tests _compute_metrics strips _denormalized suffix.
    
    **PHM Logic**: Classification doesn't use denormalized metrics.
    
    **Methodology**: Verify key format in results using realistic data.
    
    **Uses fixture**: classification_data_balanced - realistic balanced classification data.
    
    **Expected**: Keys without _denormalized suffix.
    
    Validates: Requirement CE-COMP-1 - Suffix removal
    """
    evaluator = real_metric_evaluator(metric_names=["accuracy"])
    
    # ACT: Use realistic fixture data
    evaluator.update(classification_data_balanced)
    results = evaluator._compute_metrics()
    
    # ASSERT: Should have base name without suffix
    assert "accuracy" in results
    assert "accuracy_denormalized" not in results
    assert "accuracy_normalized" not in results
    assert 0.0 <= results["accuracy"] <= 1.0


def test_compute_metrics_removes_normalized_suffix(real_metric_evaluator, classification_data_balanced):
    """
    Tests _compute_metrics strips _normalized suffix.
    
    **PHM Logic**: Classification results use base metric names.
    
    **Methodology**: Ensure _normalized suffix removed, use realistic data.
    
    **Uses fixture**: classification_data_balanced - realistic balanced classification data.
    
    **Expected**: Base metric name in results.
    
    Validates: Requirement CE-COMP-2 - Normalized suffix removal
    """
    evaluator = real_metric_evaluator(metric_names=["precision"])
    
    # ACT: Use realistic fixture data
    evaluator.update(classification_data_balanced)
    results = evaluator._compute_metrics()
    
    # ASSERT: Base metric name should be present
    assert "precision" in results
    assert "precision_normalized" not in results
    assert "precision_denormalized" not in results
    assert 0.0 <= results["precision"] <= 1.0


def test_compute_metrics_multiple_metrics(real_metric_evaluator, classification_data_balanced):
    """
    Tests _compute_metrics with multiple metrics.
    
    **PHM Logic**: Multiple metrics computed together for efficiency.
    
    **Methodology**: Request multiple metrics, use realistic data.
    
    **Uses fixture**: classification_data_balanced - realistic balanced classification data.
    
    **Expected**: All metrics in results.
    
    Validates: Requirement CE-COMP-3 - Multiple metrics
    """
    evaluator = real_metric_evaluator(metric_names=["accuracy", "f1"])
    
    # ACT: Use realistic fixture data
    evaluator.update(classification_data_balanced)
    results = evaluator._compute_metrics()
    
    # ASSERT: All requested metrics should be present
    assert "accuracy" in results
    assert "f1" in results
    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["f1"] <= 1.0


def test_init_basic_configuration(mocker):
    """
    Tests basic ClassificationEvaluator initialization.
    
    **PHM Logic**: Classification requires num_classes configuration.
    
    **Methodology**: Create evaluator and verify attributes.
    
    **Expected**: Attributes correctly set.
    
    Validates: Requirement CE-INIT-1 - Basic initialization
    """
    mocker.patch.object(
        MetricFactory, "create_classification_metric", return_value=MagicMock()
    )
    
    evaluator = ClassificationEvaluator(
        num_classes=5,
        metric_names=["accuracy"],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=True,
        paths={},
    )
    
    assert evaluator.num_classes == 5
    assert evaluator.task_type == "classification"
    assert evaluator.save_predictions is True


def test_update_with_batched_data(cls_evaluator_fixture, classification_data_balanced):
    """
    Tests update handles batched classification data.
    
    **PHM Logic**: Classification processes batches of samples.
    
    **Methodology**: Use fixture (collect_predictions=True by default),
    use reset() to clear collections, then feed realistic data through update().
    
    **Uses fixture**: classification_data_balanced - realistic balanced classification data.
    
    **Expected**: Data collected correctly.
    
    Validates: Requirement CE-UPD-1 - Batch data handling
    """
    evaluator = cls_evaluator_fixture
    
    # Verify fixture has collection enabled
    assert evaluator.collect_predictions is True
    
    # Use reset() to clear collections (not manual assignment)
    evaluator.reset()
    
    # ACT: Use realistic fixture data
    evaluator.update(classification_data_balanced)
    
    # ASSERT: Data collected correctly
    assert len(evaluator.preds) == 1
    np.testing.assert_array_equal(evaluator.preds[0], classification_data_balanced["predictions"])
    assert len(evaluator.targets) == 1
    np.testing.assert_array_equal(evaluator.targets[0], classification_data_balanced["targets"])


# ==========================================
# PART 5: ADDITIONAL COVERAGE TESTS
# ==========================================


def test_compute_metrics_handles_denormalized_suffix(mocker):
    """
    Tests _compute_metrics handles _denormalized suffix correctly.
    
    **PHM Logic**: Classification results may have _denormalized suffix 
    from parent class when scaling is enabled.
    
    **Methodology**: Use real metrics with scaling enabled to test actual integration.
    When scaling is enabled, parent adds _denormalized suffix, child strips it.
    
    **Expected**: Suffix stripped from results.
    
    Validates: Requirement CE-COMP-4 - Denormalized suffix handling
    """
    # Create evaluator with scaling enabled to get _denormalized suffix from parent
    mocker.patch.object(
        MetricFactory,
        "create_classification_metric",
        side_effect=side_effect_create_metric,
    )
    
    evaluator = ClassificationEvaluator(
        num_classes=3,
        metric_names=["accuracy"],
        inverse_transform=None,
        apply_inverse_scaling=True,
        save_predictions=False,
        paths={},
    )
    
    # Create data and update evaluator
    data = create_dummy_data()
    evaluator.update(data)
    
    # Call _compute_metrics which calls super()._compute_metrics()
    # The parent will add _denormalized suffix (because scaling is enabled),
    # then child strips it
    results = evaluator._compute_metrics()
    
    # Should have stripped suffix and returned base metric name
    assert "accuracy" in results
    assert "accuracy_denormalized" not in results
    assert "accuracy_normalized" not in results
    # Verify it's a valid accuracy value
    assert 0.0 <= results["accuracy"] <= 1.0


def test_compute_metrics_raises_on_unexpected_suffix(mocker):
    """
    Tests _compute_metrics raises ValueError for unexpected key format.
    
    **PHM Logic**: Only _normalized and _denormalized suffixes expected.
    
    **Methodology**: Manually inject unexpected key format into parent's return
    by patching at a lower level (the metrics dict processing).
    
    **Expected**: ValueError raised.
    
    Validates: Requirement CE-COMP-5 - Unexpected format error
    """
    mocker.patch.object(
        MetricFactory,
        "create_classification_metric",
        return_value=MagicMock()
    )
    
    evaluator = ClassificationEvaluator(
        num_classes=3,
        metric_names=["accuracy"],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
    )

    # Patch the parent's _compute_metrics to return unexpected key format
    # This tests the child's error handling for malformed parent output
    original_parent_compute = DefaultEvaluator._compute_metrics
    
    def mock_parent_compute(self):
        # Return unexpected format that child should reject
        return {"accuracy_unknown": 0.85}
    
    with patch.object(
        DefaultEvaluator, '_compute_metrics', mock_parent_compute
    ):
        with pytest.raises(ValueError, match="Unexpected metric key format"):
            evaluator._compute_metrics()


def test_compute_metrics_conflict_raises_assertion(mocker):
    """
    Tests _compute_metrics raises AssertionError on conflicting values.
    
    **PHM Logic**: Same metric with different values is a logic error.
    This can happen if both normalized and denormalized metrics exist
    but have different values (shouldn't happen in classification).
    
    **Methodology**: Patch parent to return same metric with different values.
    
    **Expected**: AssertionError raised.
    
    Validates: Requirement CE-COMP-6 - Conflict detection
    """
    mocker.patch.object(
        MetricFactory,
        "create_classification_metric",
        return_value=MagicMock()
    )
    
    evaluator = ClassificationEvaluator(
        num_classes=3,
        metric_names=["accuracy"],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
    )

    # Patch parent's _compute_metrics to return same metric with different values
    # This simulates a bug where normalized and denormalized metrics differ
    original_parent_compute = DefaultEvaluator._compute_metrics
    
    def mock_parent_compute(self):
        return {
            "accuracy_normalized": 0.85,
            "accuracy_denormalized": 0.90  # Different value - conflict!
        }
    
    with patch.object(
        DefaultEvaluator, '_compute_metrics', mock_parent_compute
    ):
        with pytest.raises(AssertionError, match="Conflict in metric values"):
            evaluator._compute_metrics()
