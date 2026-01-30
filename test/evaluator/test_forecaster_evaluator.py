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


def test_update_slices_effective_pred_len(forecasting_evaluator_fixture, forecasting_data_multi_step):
    """
    Tests that update() slices the last N steps based on effective_pred_len
    before passing to super().
    
    **Uses fixture**: forecasting_data_multi_step - realistic multi-step forecasting
    data with degradation curves over 24 timesteps.
    """
    evaluator = forecasting_evaluator_fixture
    # evaluator.effective_pred_len = 2 (from fixture)

    # ACT: Use realistic fixture data (has 24 timesteps)
    evaluator.update(forecasting_data_multi_step)

    # ASSERT
    # We check what arguments were passed to DefaultEvaluator.update
    from lmetk.evaluator.default import DefaultEvaluator

    args, _ = DefaultEvaluator.update.call_args
    passed_model_out = args[0]

    # Expected: Last 2 steps (effective_pred_len=2) from 24 timesteps
    # Should be timesteps 22 and 23 (indices -2 and -1)
    original_shape = forecasting_data_multi_step["predictions"].shape
    expected_slice_shape = (original_shape[0], 2, original_shape[2])
    
    assert passed_model_out["predictions"].shape == expected_slice_shape
    assert passed_model_out["targets"].shape == expected_slice_shape
    
    # Verify it's the last 2 timesteps
    original_preds = forecasting_data_multi_step["predictions"]
    expected_last_two = original_preds[:, -2:, :]
    np.testing.assert_array_equal(passed_model_out["predictions"], expected_last_two)
    
    # ADDITIONAL ASSERTIONS: Verify side effects
    assert DefaultEvaluator.update.called, "Parent update() should have been called"


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
    
    # ADDITIONAL ASSERTIONS: Verify side effects
    assert DefaultEvaluator.update.called, "Parent update() should have been called"


# =========================================================================
# === 3. Tests: Error Handling ===
# =========================================================================


def test_init_raises_without_model_pred_len(mocker):
    """
    Tests that init raises ValueError if model_pred_len is None.
    
    **PHM Logic**: model_pred_len is required for forecasting.
    
    **Methodology**: Create without model_pred_len.
    
    **Expected**: ValueError raised.
    
    Validates: Requirement FE-ERR-1 - Missing pred_len error
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    
    with pytest.raises(ValueError, match="model_pred_len.*must be set"):
        ForecastingEvaluator(
            target_dim_position=None,
            metric_names=["mse"],
            model_seq_len=10,
            model_label_len=5,
            model_pred_len=None,  # Missing!
        )


def test_init_raises_effective_greater_than_model_pred_len(mocker):
    """
    Tests that init raises ValueError if effective_pred_len > model_pred_len.
    
    **PHM Logic**: Cannot evaluate more steps than model produces.
    
    **Methodology**: Set effective_pred_len > model_pred_len.
    
    **Expected**: ValueError raised.
    
    Validates: Requirement FE-ERR-2 - Invalid effective_pred_len
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    
    with pytest.raises(ValueError, match="effective_pred_len.*cannot be greater"):
        ForecastingEvaluator(
            target_dim_position=None,
            metric_names=["mse"],
            model_seq_len=10,
            model_label_len=5,
            model_pred_len=5,
            effective_pred_len=10,  # > model_pred_len!
        )


def test_update_raises_shape_mismatch(mocker):
    """
    Tests that update raises ValueError for shape mismatch.
    
    **PHM Logic**: Predictions and targets must have matching shapes.
    
    **Methodology**: Pass mismatched prediction/target shapes.
    
    **Expected**: ValueError raised.
    
    Validates: Requirement FE-ERR-3 - Shape mismatch detection
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    mocker.patch("lmetk.evaluator.default.DefaultEvaluator.update")
    
    evaluator = ForecastingEvaluator(
        target_dim_position=None,
        metric_names=["mse"],
        model_seq_len=10,
        model_label_len=5,
        model_pred_len=5,
        effective_pred_len=None,  # No slicing
    )
    
    # Mismatched shapes
    preds = np.zeros((1, 5, 1))
    targets = np.zeros((1, 3, 1))  # Different time dimension!
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluator.update({"predictions": preds, "targets": targets})


# =========================================================================
# === 4. Tests: Task Mode Handling ===
# =========================================================================


def test_update_no_dimension_selection_multivariate(mocker, forecasting_data_multivariate):
    """
    Tests that multivariate mode doesn't select target dimension.
    
    **PHM Logic**: Multivariate keeps all dimensions.
    
    **Methodology**: Set task_mode != "univariate", use realistic multivariate data.
    
    **Uses fixture**: forecasting_data_multivariate - realistic multivariate forecasting
    data with 5 sensor channels.
    
    **Expected**: All dimensions preserved.
    
    Validates: Requirement FE-MODE-1 - Multivariate dimension handling
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    mocker.patch("lmetk.evaluator.default.DefaultEvaluator.update")
    
    evaluator = ForecastingEvaluator(
        target_dim_position=1,  # Would select dim if univariate
        metric_names=["mse"],
        model_seq_len=10,
        model_label_len=5,
        model_pred_len=forecasting_data_multivariate["predictions"].shape[1],
        task_mode="multivariate",  # Not univariate
    )
    
    # ACT: Use realistic multivariate fixture data
    evaluator.update(forecasting_data_multivariate)
    
    from lmetk.evaluator.default import DefaultEvaluator
    args, _ = DefaultEvaluator.update.call_args
    passed = args[0]
    
    # ASSERT: All features should be preserved (5 features from fixture)
    assert passed["predictions"].shape[-1] == forecasting_data_multivariate["predictions"].shape[-1]
    assert passed["targets"].shape[-1] == forecasting_data_multivariate["targets"].shape[-1]


def test_update_no_effective_pred_len_slicing(mocker, forecasting_data_multi_step):
    """
    Tests that effective_pred_len=None disables time slicing.
    
    **PHM Logic**: Some tasks evaluate full prediction horizon.
    
    **Methodology**: Set effective_pred_len=None, use realistic multi-step data.
    
    **Uses fixture**: forecasting_data_multi_step - realistic multi-step forecasting
    data with 24 timesteps.
    
    **Expected**: Full time dimension preserved.
    
    Validates: Requirement FE-MODE-2 - No slicing when disabled
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    mocker.patch("lmetk.evaluator.default.DefaultEvaluator.update")
    
    horizon = forecasting_data_multi_step["predictions"].shape[1]
    
    evaluator = ForecastingEvaluator(
        target_dim_position=None,
        metric_names=["mse"],
        model_seq_len=10,
        model_label_len=5,
        model_pred_len=horizon,
        effective_pred_len=None,  # No slicing
    )
    
    # ACT: Use realistic fixture data
    evaluator.update(forecasting_data_multi_step)
    
    from lmetk.evaluator.default import DefaultEvaluator
    args, _ = DefaultEvaluator.update.call_args
    passed = args[0]
    
    # ASSERT: Full horizon preserved (24 timesteps from fixture)
    assert passed["predictions"].shape[1] == horizon
    assert passed["targets"].shape[1] == horizon


def test_update_univariate_no_dim_selection_when_none(mocker):
    """
    Tests univariate mode without target_dim_position.
    
    **PHM Logic**: target_dim_position=None means use all dimensions.
    
    **Methodology**: Set univariate mode with target_dim_position=None.
    
    **Expected**: All dimensions preserved.
    
    Validates: Requirement FE-MODE-3 - Univariate without dim selection
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    mocker.patch("lmetk.evaluator.default.DefaultEvaluator.update")
    
    evaluator = ForecastingEvaluator(
        target_dim_position=None,  # No selection
        metric_names=["mse"],
        model_seq_len=10,
        model_label_len=5,
        model_pred_len=5,
        task_mode="univariate",
    )
    
    preds = np.ones((1, 5, 3))
    targets = np.ones((1, 5, 3))
    
    evaluator.update({"predictions": preds, "targets": targets})
    
    from lmetk.evaluator.default import DefaultEvaluator
    args, _ = DefaultEvaluator.update.call_args
    passed = args[0]
    
    # All dimensions preserved
    assert passed["predictions"].shape[-1] == 3
