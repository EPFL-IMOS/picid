import pytest
import numpy as np
from unittest.mock import MagicMock
from picid.evaluator.forecasting import ForecastingEvaluator
from picid.metrics.metric_factory import MetricFactory


@pytest.fixture
def forecasting_evaluator_fixture(mocker):
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    # We mock the super().update to verify what data it receives
    mocker.patch("picid.evaluator.default.DefaultEvaluator.update")

    return ForecastingEvaluator(
        target_dim_position=None,
        metric_names=["mse"],
        model_seq_len=10,
        model_label_len=5,
        effective_pred_len=2,  # Only evaluate last 2 steps
        model_pred_len=5,  # Total output length
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
    )


def test_update_slices_effective_pred_len(
    forecasting_evaluator_fixture, forecasting_data_multi_step
):
    """
    Tests that update() slices the last N steps based on effective_pred_len
    before passing to super().
    """
    evaluator = forecasting_evaluator_fixture

    # ACT
    evaluator.update(forecasting_data_multi_step)

    # ASSERT
    from picid.evaluator.default import DefaultEvaluator

    args, _ = DefaultEvaluator.update.call_args
    passed_model_out = args[0]

    original_shape = forecasting_data_multi_step["predictions"].shape
    expected_slice_shape = (original_shape[0], 2, original_shape[2])

    assert passed_model_out["predictions"].shape == expected_slice_shape
    assert passed_model_out["targets"].shape == expected_slice_shape

    original_preds = forecasting_data_multi_step["predictions"]
    expected_last_two = original_preds[:, -2:, :]
    np.testing.assert_array_equal(passed_model_out["predictions"], expected_last_two)
    assert DefaultEvaluator.update.called


def test_update_selects_target_dim(forecasting_evaluator_fixture):
    """
    Tests dimension selection for univariate tasks.
    """
    evaluator = forecasting_evaluator_fixture
    evaluator.task_mode = "univariate"
    evaluator.target_dim_position = 1
    evaluator.effective_pred_len = None

    # Input: (Batch=1, Time=1, Feat=3)
    preds = np.array([[[10, 20, 30]]])
    targets = np.array([[[10, 20, 30]]])
    model_out = {"predictions": preds, "targets": targets}

    # ACT
    evaluator.update(model_out)

    # ASSERT
    from picid.evaluator.default import DefaultEvaluator

    args, _ = DefaultEvaluator.update.call_args
    passed_model_out = args[0]

    # FIX: Expected is now 3D (1, 1, 1) because univariate slicing preserves dims for guardians
    expected = np.array([[[20]]])

    np.testing.assert_array_equal(passed_model_out["predictions"], expected)
    assert DefaultEvaluator.update.called


# =========================================================================
# === 3. Tests: Error Handling ===
# =========================================================================


def test_init_raises_without_model_pred_len(mocker):
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    with pytest.raises(ValueError, match="model_pred_len.*must be set"):
        ForecastingEvaluator(
            target_dim_position=None,
            metric_names=["mse"],
            model_seq_len=10,
            model_label_len=5,
            effective_pred_len=None,
            model_pred_len=None,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
        )


def test_init_raises_effective_greater_than_model_pred_len(mocker):
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    with pytest.raises(ValueError, match="effective_pred_len.*cannot be greater"):
        ForecastingEvaluator(
            target_dim_position=None,
            metric_names=["mse"],
            model_seq_len=10,
            model_label_len=5,
            effective_pred_len=10,
            model_pred_len=5,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
        )


def test_update_raises_shape_mismatch(mocker):
    """
    Tests that update raises AssertionError for shape mismatch (due to Guardians).
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    # Note: We do NOT mock DefaultEvaluator.update here because we want to hit the real guardians

    evaluator = ForecastingEvaluator(
        target_dim_position=None,
        metric_names=["mse"],
        model_seq_len=10,
        model_label_len=5,
        effective_pred_len=None,
        model_pred_len=5,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
    )

    # Mismatched time dimension
    preds = np.zeros((1, 5, 1))
    targets = np.zeros((1, 3, 1))

    # FIX: Catch AssertionError from the MetricManager/DefaultEvaluator guardians
    with pytest.raises(AssertionError, match="Shape mismatch"):
        evaluator.update({"predictions": preds, "targets": targets})


# =========================================================================
# === 4. Tests: Task Mode Handling ===
# =========================================================================


def test_update_no_dimension_selection_multivariate(
    mocker, forecasting_data_multivariate
):
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    mocker.patch("picid.evaluator.default.DefaultEvaluator.update")

    evaluator = ForecastingEvaluator(
        target_dim_position=1,
        metric_names=["mse"],
        model_seq_len=10,
        model_label_len=5,
        effective_pred_len=None,
        model_pred_len=forecasting_data_multivariate["predictions"].shape[1],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
        task_mode="multivariate",
    )

    evaluator.update(forecasting_data_multivariate)

    from picid.evaluator.default import DefaultEvaluator

    args, _ = DefaultEvaluator.update.call_args
    passed = args[0]

    assert (
        passed["predictions"].shape[-1]
        == forecasting_data_multivariate["predictions"].shape[-1]
    )
    assert (
        passed["targets"].shape[-1]
        == forecasting_data_multivariate["targets"].shape[-1]
    )


def test_update_multivariate_full_stack_no_mock(mocker, forecasting_data_multivariate):
    """Multivariate (N, T, C) must pass DefaultEvaluator validation and reach metrics."""
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    evaluator = ForecastingEvaluator(
        target_dim_position=1,
        metric_names=["mse"],
        model_seq_len=10,
        model_label_len=5,
        effective_pred_len=None,
        model_pred_len=forecasting_data_multivariate["predictions"].shape[1],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
        task_mode="multivariate",
    )

    evaluator.update(forecasting_data_multivariate)

    assert evaluator.metric_manager.metrics["mse"].update.called


def test_update_no_effective_pred_len_slicing(mocker, forecasting_data_multi_step):
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    mocker.patch("picid.evaluator.default.DefaultEvaluator.update")

    horizon = forecasting_data_multi_step["predictions"].shape[1]

    evaluator = ForecastingEvaluator(
        target_dim_position=None,
        metric_names=["mse"],
        model_seq_len=10,
        model_label_len=5,
        effective_pred_len=None,
        model_pred_len=horizon,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
    )

    evaluator.update(forecasting_data_multi_step)

    from picid.evaluator.default import DefaultEvaluator

    args, _ = DefaultEvaluator.update.call_args
    passed = args[0]

    assert passed["predictions"].shape[1] == horizon
    assert passed["targets"].shape[1] == horizon


def test_update_univariate_no_dim_selection_when_none(mocker):
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    mocker.patch("picid.evaluator.default.DefaultEvaluator.update")

    evaluator = ForecastingEvaluator(
        target_dim_position=None,
        metric_names=["mse"],
        model_seq_len=10,
        model_label_len=5,
        effective_pred_len=None,
        model_pred_len=5,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
        task_mode="univariate",
    )

    preds = np.ones((1, 5, 3))
    targets = np.ones((1, 5, 3))

    evaluator.update({"predictions": preds, "targets": targets})

    from picid.evaluator.default import DefaultEvaluator

    args, _ = DefaultEvaluator.update.call_args
    passed = args[0]

    assert passed["predictions"].shape[-1] == 3
