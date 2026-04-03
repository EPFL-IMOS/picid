import numpy as np
from unittest.mock import MagicMock, patch
from numpy.testing import assert_array_equal
from pathlib import Path
import pytest

# --- Adjust these imports to match your project structure ---
from picid.evaluator.default import DefaultEvaluator
from picid.evaluator.hooks.base import BaseEvalHook
from picid.evaluator.hooks.save_predictions import (
    SavePredictionsHook,
)  # ADDED: For testing decoupled logic
from picid.metrics.metrics import MSEMetric, RMSEMetric
from picid.metrics.metric_factory import MetricFactory

# =========================================================================
# === 1. MOCKS and FIXTURES for Dependencies ===
# =========================================================================


class _DoubleInverseTransform:
    """Real inverse hook used by ScalingWrapper tests (2x scale, legacy mock parity)."""

    def inverse_transform(self, data, metadata=None):
        if getattr(data, "predictions", None) is not None:
            arr = data.predictions
        else:
            arr = data.targets
        return arr * 2.0


class _HundredXInverseTransform:
    """Simple inverse hook that multiplies predictions/targets by 100."""

    def inverse_transform(self, data, metadata=None):
        if getattr(data, "predictions", None) is not None:
            arr = data.predictions
        else:
            arr = data.targets
        return arr * 100.0


class _RecordingComputeHook(BaseEvalHook):
    """Minimal hook that records compute-end payloads (no MagicMock chains)."""

    def __init__(self):
        self.compute_end_calls: list[dict] = []

    def on_compute_end(self, results, evaluator, mode, epoch, step):
        self.compute_end_calls.append(
            {
                "results": dict(results),
                "mode": mode,
                "epoch": epoch,
                "step": step,
            }
        )


@pytest.fixture
def mock_path_obj():
    """
    Creates a dedicated Mock object for Path operations, preventing filesystem access.
    """
    p = MagicMock(spec=Path)
    p.__truediv__.return_value = p  # Support path / "string"
    p.mkdir.return_value = None
    p.parent = MagicMock()
    p.parent.mkdir.return_value = None
    return p


# --- Evaluator Fixtures ---
@pytest.fixture
def default_evaluator_regression_fixture(mocker):
    """
    Provides a DefaultEvaluator configured with real metric classes (MSE, RMSE).
    """
    # Use real metric instances for proper accumulation/reset testing
    metrics = {"mse": MSEMetric(), "rmse": RMSEMetric()}

    # Patch MetricFactory to return our real instances
    mocker.patch.object(
        MetricFactory, "create_metric", lambda name, paths: metrics[name.lower()]
    )

    evaluator = DefaultEvaluator(
        metric_names=["mse", "rmse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
        remote_logger=None,
        collect_predictions=False,
    )

    return evaluator


@pytest.fixture
def default_evaluator_with_scaling_fixture(mocker):
    """
    DefaultEvaluator fixture with inverse scaling ENABLED.
    """

    # Create fresh metric instances for each test (avoid sharing state)
    def create_metric_factory(name, paths=None):
        if name.lower() == "mse":
            return MSEMetric()
        elif name.lower() == "rmse":
            return RMSEMetric()
        raise ValueError(f"Unknown metric: {name}")

    mocker.patch.object(
        MetricFactory, "create_metric", side_effect=create_metric_factory
    )

    evaluator = DefaultEvaluator(
        metric_names=["mse", "rmse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=_DoubleInverseTransform(),
        apply_inverse_scaling=True,
        save_predictions=True,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )

    return evaluator


@pytest.fixture
def default_evaluator_regression_fixture_saving(mocker):
    """
    DefaultEvaluator fixture with saving and collection enabled (no scaling).
    """
    metrics = {"mse": MSEMetric(), "rmse": RMSEMetric()}
    mocker.patch.object(
        MetricFactory, "create_metric", lambda name, paths: metrics[name.lower()]
    )

    mock_paths = MagicMock(eval_details="/mock/base")

    evaluator = DefaultEvaluator(
        metric_names=["mse", "rmse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=True,
        paths=mock_paths,
        remote_logger=None,
        collect_predictions=True,
    )

    return evaluator


# =========================================================================
# === 2. Helper Data and Functions ===
# =========================================================================

# Use 3D Arrays (N, 1, 1) to satisfy Shape Checks in Scaling/Metrics
P_NORM = np.array([[[1.0]], [[3.0]]])
T_NORM = np.array([[[2.0]], [[4.0]]])


def _update_evaluator(evaluator: DefaultEvaluator, predictions, targets):
    """Helper to simulate update call."""
    evaluator.update({"predictions": predictions, "targets": targets})


# =========================================================================
# === 3. Unit Tests ===
# =========================================================================


def test_evaluator_init_initializes_collections_and_secondary_metrics(mocker):
    """Tests if collections and secondary metric dicts are initialized correctly."""
    mocker.patch.object(MetricFactory, "create_metric", lambda name, paths: MSEMetric())

    # Case A: Scaling OFF
    evaluator_a = DefaultEvaluator(
        metric_names=["mse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )
    # FIX: Access via buffer.data
    assert isinstance(evaluator_a.buffer.data["preds"], list)
    # FIX: Access via metric_manager
    assert evaluator_a.metric_manager.normalized_metrics is None

    # Case B: Scaling ON
    evaluator_b = DefaultEvaluator(
        metric_names=["mse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=_DoubleInverseTransform(),
        apply_inverse_scaling=True,
        save_predictions=False,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )
    # FIX: Access via metric_manager
    assert evaluator_b.metric_manager.normalized_metrics is not None
    assert len(evaluator_b.metric_manager.normalized_metrics) == 1


def test_evaluator_reset_clears_all_collections_and_metrics(
    default_evaluator_with_scaling_fixture,
    regression_data_basic,
):
    """
    Tests reset clears all metrics and collected data.
    """
    evaluator = default_evaluator_with_scaling_fixture

    # Verify scaling is enabled and normalized_metrics exists
    assert evaluator.scaling_wrapper.apply_inverse is True
    # FIX: Check metric_manager
    assert evaluator.metric_manager.normalized_metrics is not None

    # Setup state with realistic data
    evaluator.update(regression_data_basic)
    assert evaluator.metric_manager.metrics["mse"].total_count > 0

    # ACT
    evaluator.reset()

    # ASSERT: Everything should be cleared
    # FIX: Access via buffer.data dictionary
    assert len(evaluator.buffer.data["preds"]) == 0
    assert len(evaluator.buffer.data["norm_preds"]) == 0
    assert len(evaluator.buffer.data["targets"]) == 0
    assert len(evaluator.buffer.data["norm_targets"]) == 0

    # FIX: Access via metric_manager
    assert evaluator.metric_manager.metrics["mse"].total_count == 0
    assert evaluator.metric_manager.normalized_metrics["mse"].total_count == 0


def test_update_collection_logic_scaling_off(
    default_evaluator_regression_fixture_saving, regression_data_basic
):
    """
    Tests data collection logic when scaling is OFF.
    """
    evaluator = default_evaluator_regression_fixture_saving

    # Verify scaling is OFF
    assert evaluator.scaling_wrapper.apply_inverse is False

    evaluator.reset()

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # ASSERT: Only self.buffer.data["preds"] holds data (no scaling)
    # FIX: Access via buffer.data
    assert len(evaluator.buffer.data["preds"]) > 0
    assert_array_equal(
        evaluator.buffer.data["preds"][0], regression_data_basic["predictions"]
    )
    assert len(evaluator.buffer.data["norm_preds"]) == 0


def test_update_collection_logic_scaling_on(
    default_evaluator_with_scaling_fixture, regression_data_basic
):
    """
    Tests data collection logic when scaling is ON.
    """
    evaluator = default_evaluator_with_scaling_fixture

    # Verify scaling is ON
    assert evaluator.scaling_wrapper.apply_inverse is True

    evaluator.reset()

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # ASSERT: Both normalized and denormalized data collected
    # FIX: Access via buffer.data

    # norm_preds holds NORMALIZED data
    assert len(evaluator.buffer.data["norm_preds"]) > 0
    assert_array_equal(
        evaluator.buffer.data["norm_preds"][0], regression_data_basic["predictions"]
    )

    # preds holds DENORMALIZED data (2x from mock)
    assert len(evaluator.buffer.data["preds"]) > 0
    expected_scaled = regression_data_basic["predictions"] * 2.0
    assert_array_equal(evaluator.buffer.data["preds"][0], expected_scaled)


def test_default_evaluator_dual_reporting_with_scaling(
    default_evaluator_with_scaling_fixture,
    regression_data_basic,
):
    """
    Tests Dual Reporting: Scaling is ON. Reports both NORMALIZED and DENORMALIZED.
    """
    evaluator = default_evaluator_with_scaling_fixture

    # Verify scaling is enabled
    assert evaluator.scaling_wrapper.apply_inverse is True
    # FIX: Access via metric_manager
    assert evaluator.metric_manager.normalized_metrics is not None

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # COMPUTE
    # FIX: _compute_metrics moved to metric_manager.compute
    results = evaluator.metric_manager.compute()

    # ASSERT: Both normalized and denormalized metrics should be present
    assert "mse_denormalized" in results
    assert "mse_normalized" in results

    # Denormalized should be larger (scaled by 2x, so error scales by 4x)
    assert results["mse_denormalized"] > results["mse_normalized"]

    # Both should be non-negative
    assert results["mse_denormalized"] >= 0.0
    assert results["mse_normalized"] >= 0.0


def test_default_evaluator_compute_no_scaling_correct_naming(
    default_evaluator_regression_fixture,
    regression_data_basic,
):
    """
    Tests that when scaling is OFF, the single result is correctly named as _normalized.
    """
    evaluator = default_evaluator_regression_fixture

    # Verify scaling is disabled (fixture default)
    assert evaluator.scaling_wrapper.apply_inverse is False

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # COMPUTE
    # FIX: _compute_metrics moved to metric_manager.compute
    results = evaluator.metric_manager.compute()

    # ASSERT: Should have _normalized suffix, not _denormalized
    assert "mse_denormalized" not in results
    assert "mse_normalized" in results
    assert results["mse_normalized"] >= 0.0


@patch("picid.evaluator.hooks.save_predictions.xr.Dataset")
def test_prepare_predictions_with_scaling(
    mock_xr_dataset, default_evaluator_with_scaling_fixture, regression_data_basic
):
    """
    Tests preparation correctly handles array concatenation and naming via Hook logic.
    """
    evaluator = default_evaluator_with_scaling_fixture

    # Verify scaling is enabled
    assert evaluator.scaling_wrapper.apply_inverse is True

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # ACT: Decoupled logic testing
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    dict_to_save = hook._format_xarray(evaluator.buffer.get_all(), evaluator)

    # ASSERT - preds should be 2x the original (from fixture mock)
    assert "preds" in dict_to_save
    assert "preds_normalized" in dict_to_save

    # Verify data shapes are correct (format is (dims, data))
    preds_data = dict_to_save["preds"][1]
    preds_norm_data = dict_to_save["preds_normalized"][1]
    assert preds_data.shape == regression_data_basic["predictions"].shape
    assert preds_norm_data.shape == regression_data_basic["predictions"].shape


@patch("picid.evaluator.hooks.save_predictions.xr.Dataset")
def test_save_predictions_to_file_orchestration(
    mock_xr_dataset, default_evaluator_regression_fixture_saving, mock_path_obj
):
    """
    Tests that the save orchestration calls the necessary methods via Hook compute end.
    """
    evaluator = default_evaluator_regression_fixture_saving
    mock_dataset_instance = MagicMock()
    mock_xr_dataset.return_value = mock_dataset_instance

    # FIX: Patch Path in the hook's module
    with patch(
        "picid.evaluator.hooks.save_predictions.Path", return_value=mock_path_obj
    ):
        evaluator.paths = MagicMock(eval_details="/mock/base")
        evaluator.update(
            {"predictions": np.array([[[1]]]), "targets": np.array([[[1]]])}
        )

        # ACT: Use the hook's entry point
        hook = SavePredictionsHook(dims=["sample", "time", "feature"])
        hook.on_compute_end({}, evaluator, mode="test_mode", epoch=1, step=1)

        # ASSERT
        mock_path_obj.mkdir.assert_called_with(parents=True, exist_ok=True)
        mock_dataset_instance.to_netcdf.assert_called_once()


# =========================================================================
# === 4. Additional Tests for Full Coverage ===
# =========================================================================


def test_update_classification_task(classification_data_balanced):
    """
    Tests update with classification task type.
    """
    evaluator = DefaultEvaluator(
        metric_names=["accuracy"],
        task_type="classification",
        num_classes=classification_data_balanced["predictions"].shape[-1],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )

    # Verify scaling is disabled
    assert evaluator.scaling_wrapper.apply_inverse is False

    # ACT: Use realistic classification fixture data
    evaluator.update(classification_data_balanced)

    # ASSERT: Targets should be collected (converted to int internally)
    # FIX: Access via buffer.data
    assert len(evaluator.buffer.data["preds"]) == 1
    assert len(evaluator.buffer.data["targets"]) == 1
    # Verify targets are integers (converted internally)
    assert evaluator.buffer.data["targets"][0].dtype in [np.int32, np.int64, int]

    acc = evaluator.metric_manager.compute()["accuracy"]
    assert 0.0 <= acc <= 1.0


def test_update_forecasting_task(forecasting_data_multi_step):
    """
    Tests update with forecasting task type.
    """
    evaluator = DefaultEvaluator(
        metric_names=["mse"],
        task_type="forecasting",
        num_classes=None,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=True,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )

    # ACT: Use realistic multi-step forecasting fixture data
    evaluator.update(forecasting_data_multi_step)

    # ASSERT: Data collected correctly
    # FIX: Access via buffer.data
    assert len(evaluator.buffer.data["preds"]) == 1
    assert (
        evaluator.buffer.data["preds"][0].shape
        == forecasting_data_multi_step["predictions"].shape
    )
    assert (
        evaluator.buffer.data["targets"][0].shape
        == forecasting_data_multi_step["targets"].shape
    )

    mse = evaluator.metric_manager.compute()["mse_normalized"]
    assert mse >= 0.0


def test_update_unsupported_task_raises_error():
    """
    Tests unsupported task types.
    """
    # FIX: Regex match updated to "Unsupported task:"
    with pytest.raises(AssertionError, match="Unsupported task:"):
        DefaultEvaluator(
            metric_names=["mse"],
            task_type="invalid_task",
            num_classes=None,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )


def test_log_plot_with_remote_logger():
    """
    Tests log_plot calls remote logger experiment.
    """
    mock_logger = MagicMock()

    evaluator = DefaultEvaluator(
        metric_names=["mse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
        remote_logger=mock_logger,
        collect_predictions=True,
    )

    mock_fig = MagicMock()

    evaluator.log_plot(mock_fig, "test_plot", "val", epoch=1, step=10)

    mock_logger.experiment.log.assert_called_once()


def test_log_plot_without_remote_logger():
    """
    Tests log_plot does nothing without remote logger.
    """
    evaluator = DefaultEvaluator(
        metric_names=["mse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )

    mock_fig = MagicMock()

    # Should not raise
    evaluator.log_plot(mock_fig, "test_plot", "val", epoch=1, step=10)


def test_prepare_predictions_classification_preserves_logits_shape_and_values():
    """
    Save-prediction formatting keeps classification logits intact for downstream consumers.
    """
    evaluator = DefaultEvaluator(
        metric_names=["accuracy"],
        task_type="classification",
        num_classes=3,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=True,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )

    # Feed logits through proper interface
    preds = np.array([[[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]])  # (batch, time, classes)
    targets = np.array([[[1], [0]]])  # (batch, time, 1)

    evaluator.update({"predictions": preds, "targets": targets})

    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    result = hook._format_xarray(evaluator.buffer.get_all(), evaluator)

    pred_dims, pred_values = result["preds"]
    target_dims, target_values = result["targets"]
    assert pred_dims == ["sample", "time", "feature"]
    assert target_dims == ["sample", "time", "feature_label"]
    np.testing.assert_array_equal(pred_values, preds)
    np.testing.assert_array_equal(target_values, targets)


def test_save_predictions_empty_dict(default_evaluator_regression_fixture_saving):
    """
    Tests SavePredictionsHook handles empty buffer gracefully.
    """
    evaluator = default_evaluator_regression_fixture_saving
    evaluator.buffer.clear()

    # Should not raise or create files
    hook = SavePredictionsHook()
    hook.on_compute_end({}, evaluator, mode="test", epoch=1, step=1)


def test_compute_invokes_hooks_and_returns_metric_values(
    default_evaluator_regression_fixture_saving,
):
    """
    Tests compute runs real metric aggregation and passes the same dict to hooks.
    """
    evaluator = default_evaluator_regression_fixture_saving
    hook = _RecordingComputeHook()
    evaluator.hooks = [hook]

    preds = np.array([[[1.0]], [[0.0]]], dtype=np.float32)
    tgts = np.array([[[2.0]], [[0.0]]], dtype=np.float32)
    evaluator.update({"predictions": preds, "targets": tgts})

    result = evaluator.compute(mode="test", epoch=1, step=10)

    # MSE = mean([(1-2)^2, (0-0)^2]) over 2 elements = 0.5
    assert result["mse_normalized"] == pytest.approx(0.5)
    assert len(hook.compute_end_calls) == 1
    assert hook.compute_end_calls[0]["results"]["mse_normalized"] == pytest.approx(0.5)
    assert hook.compute_end_calls[0]["mode"] == "test"
    assert hook.compute_end_calls[0]["epoch"] == 1
    assert hook.compute_end_calls[0]["step"] == 10


def test_default_evaluator_uses_real_scaling_wrapper_and_multiply_inverse(mocker):
    """
    End-to-end path: built-in ScalingWrapper + inverse transform (no wrapper override).
    """
    mocker.patch.object(MetricFactory, "create_metric", lambda name, paths: MSEMetric())

    evaluator = DefaultEvaluator(
        metric_names=["mse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=_HundredXInverseTransform(),
        apply_inverse_scaling=True,
        save_predictions=False,
        paths={},
        remote_logger=None,
        collect_predictions=False,
    )

    preds = np.array([[[0.5]], [[0.3]]], dtype=np.float32)
    targets = np.array([[[0.4]], [[0.2]]], dtype=np.float32)
    evaluator.update({"predictions": preds, "targets": targets})

    results = evaluator.metric_manager.compute()
    assert results["mse_denormalized"] > results["mse_normalized"]

    evaluator.reset()
    assert evaluator.metric_manager.metrics["mse"].total_count == 0
    assert evaluator.metric_manager.normalized_metrics["mse"].total_count == 0
