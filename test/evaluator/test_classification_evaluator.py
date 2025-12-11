import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# Adjust imports to match your project structure
from lmetk.evaluator.classification import ClassificationEvaluator
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
    """
    mocker.patch.object(
        MetricFactory, "create_classification_metric", return_value=MagicMock()
    )

    evaluator = ClassificationEvaluator(
        num_classes=3,
        metric_names=["accuracy"],
        save_predictions=True,
        collect_predictions=True,
        paths=MagicMock(),
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
            save_predictions=False,
            collect_predictions=True,
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


def test_prepare_predictions_for_save_structure(cls_evaluator_fixture):
    """
    Tests that predictions are saved with correct keys and dimension names.
    """
    evaluator = cls_evaluator_fixture

    # Populate dummy data (Batch=1, Time=2)
    evaluator.preds.append(np.zeros((1, 2, 3)))
    evaluator.targets.append(np.zeros((1, 2, 1)))

    # ACT
    output = evaluator._prepare_predictions_for_save()

    # ASSERT
    assert "preds_logits" in output
    assert "targets_labels" in output

    # Dimension check: must include 'time' because input data is 3D
    assert output["preds_logits"][0] == ["sample", "time", "class_logits"]
    assert output["targets_labels"][0] == ["sample", "time", "class_label"]


# ==========================================
# PART 3: INTEGRATION TESTS (WITH MANUAL MATH)
# ==========================================


def test_compute_metrics_integration_accuracy(real_metric_evaluator):
    evaluator = real_metric_evaluator(metric_names=["accuracy"])
    data = create_dummy_data()

    # Manual Calculation:
    # Total samples = 4
    # Correct predictions = 3 (0==0, 1==1, 2==2)
    # Incorrect = 1 (Pred 0 != Target 1)
    expected_accuracy = 3 / 4

    evaluator.update(data)
    results = evaluator.compute(mode="test", epoch=1, step=1)

    assert "accuracy" in results
    np.testing.assert_almost_equal(results["accuracy"], expected_accuracy)


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
