import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from lmetk.evaluator.classification import ClassificationEvaluator
from lmetk.metrics.metric_factory import MetricFactory


@pytest.fixture
def cls_evaluator_fixture(mocker):
    # Mock MetricFactory to allow instantiation
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


def test_prepare_predictions_for_save_structure(cls_evaluator_fixture):
    """Tests that predictions are saved with correct keys (logits/labels) and dimension names."""
    evaluator = cls_evaluator_fixture

    # Populate dummy data
    # Preds: (Batch=1, Time=2, Classes=3)
    evaluator.preds.append(np.zeros((1, 2, 3)))
    # Targets: (Batch=1, Time=2, Label=1)
    evaluator.targets.append(np.zeros((1, 2, 1)))

    # ACT
    output = evaluator._prepare_predictions_for_save()

    # ASSERT
    assert "preds_logits" in output
    assert "targets_labels" in output
    # Check dimension names
    assert output["preds_logits"][0] == ["sample", "time", "class_logits"]
    assert output["targets_labels"][0] == ["sample", "class_labels"]


def test_compute_metrics_renames_keys(cls_evaluator_fixture):
    """Tests that _normalized and _denormalized suffixes are removed."""
    evaluator = cls_evaluator_fixture

    # Mock the super()._compute_metrics output to simulate Base behavior
    dirty_results = {
        "accuracy_normalized_mean": 0.85,
        "f1_denormalized_mean": 0.80,
        "simple_metric_mean": 0.5,
    }

    # We mock the super() call by patching DefaultEvaluator._compute_metrics
    with patch(
        "lmetk.evaluator.default.DefaultEvaluator._compute_metrics",
        return_value=dirty_results,
    ):
        clean_results = evaluator._compute_metrics()

    # ASSERT
    assert "accuracy" in clean_results
    assert clean_results["accuracy"] == 0.85

    assert "f1" in clean_results
    assert clean_results["f1"] == 0.80

    assert "simple_metric" in clean_results
    assert clean_results["simple_metric"] == 0.5

    # Ensure old keys are gone
    assert "accuracy_normalized_mean" not in clean_results
