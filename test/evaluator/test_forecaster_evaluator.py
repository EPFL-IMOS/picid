import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from lmetk.evaluator.forecasting import ForecastingEvaluator
from lmetk.metrics.metric_factory import MetricFactory


@pytest.fixture
def forecasting_evaluator_fixture(mocker):
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    # We mock the super().update to verify what data it receives
    mocker.patch("lmetk.evaluator.default.DefaultEvaluator.update")

    return ForecastingEvaluator(
        target_dim_position=None,
        metric_names=["mse"],
        model_seq_len=10,
        model_label_len=5,
        model_pred_len=5,  # Total output length
        effective_pred_len=2,  # Only evaluate last 2 steps
        collect_predictions=True,
    )


def test_update_slices_effective_pred_len(forecasting_evaluator_fixture):
    """
    Tests that update() slices the last N steps based on effective_pred_len
    before passing to super().
    """
    evaluator = forecasting_evaluator_fixture

    # Input: (Batch=1, Time=5, Feat=1)
    # Values: 0, 1, 2, 3, 4
    preds = np.arange(5).reshape(1, 5, 1)
    targets = np.arange(5).reshape(1, 5, 1)

    model_out = {"predictions": preds, "targets": targets}

    # ACT
    evaluator.update(model_out)

    # ASSERT
    # We check what arguments were passed to DefaultEvaluator.update
    from lmetk.evaluator.default import DefaultEvaluator

    args, _ = DefaultEvaluator.update.call_args

    # FIX: args[0] is model_out. There is no args[1] because self is implicit/handled by mock.
    passed_model_out = args[0]

    # Expected: Last 2 steps -> [3, 4]
    expected_slice = np.array([3, 4]).reshape(1, 2, 1)

    np.testing.assert_array_equal(passed_model_out["predictions"], expected_slice)
    np.testing.assert_array_equal(passed_model_out["targets"], expected_slice)


def test_update_selects_target_dim(forecasting_evaluator_fixture):
    """
    Tests dimension selection for univariate tasks.
    """
    evaluator = forecasting_evaluator_fixture
    evaluator.task_mode = "univariate"
    evaluator.target_dim_position = 1  # Select index 1
    evaluator.effective_pred_len = None  # Disable time slicing for clarity

    # Input: (Batch=1, Time=1, Feat=3)
    # Feat 0: [10], Feat 1: [20], Feat 2: [30]
    preds = np.array([[[10, 20, 30]]])
    targets = np.array([[[10, 20, 30]]])

    model_out = {"predictions": preds, "targets": targets}

    # ACT
    evaluator.update(model_out)

    # ASSERT
    from lmetk.evaluator.default import DefaultEvaluator

    args, _ = DefaultEvaluator.update.call_args

    # FIX: args[0] is model_out
    passed_model_out = args[0]

    # Expected: Value from index 1 -> 20.
    # Shape becomes (Batch, Time) due to simple indexing in code
    expected = np.array([[20]])

    np.testing.assert_array_equal(passed_model_out["predictions"], expected)
