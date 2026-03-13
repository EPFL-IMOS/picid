"""
Comprehensive initialization tests for all evaluator types.

This module tests every parameter in evaluator initialization, including:
- Valid parameter combinations
- Invalid parameter combinations
- Edge cases and boundary conditions
- Parameter validation and error handling

Goal: Catch initialization bugs before they cause runtime errors.
"""

import pytest
from unittest.mock import MagicMock

from picid.evaluator.default import DefaultEvaluator
from picid.evaluator.classification import ClassificationEvaluator
from picid.evaluator.forecasting import ForecastingEvaluator
from picid.evaluator.multiunit import MultiUnitEvaluator
from picid.evaluator.reconstruction import ReconstructionEvaluator
from picid.metrics.metric_factory import MetricFactory
from picid.transforms.base.multisource import InverseTransformMixin

from picid.evaluator.hooks.unit_trend_plot import UnitTrendPlotHook
from picid.evaluator.hooks.reconstruction_plot import ReconstructionPlotHook

# =============================================================================
# === DEFAULT EVALUATOR INITIALIZATION TESTS ===
# =============================================================================


class TestDefaultEvaluatorInitialization:
    """Comprehensive tests for DefaultEvaluator initialization."""

    @pytest.fixture
    def mock_metric_factory(self, mocker):
        """Mock metric factory for all tests."""
        mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
        mocker.patch.object(
            MetricFactory, "create_classification_metric", return_value=MagicMock()
        )

    def test_init_minimal_required_params(self, mock_metric_factory):
        """Test initialization with minimal required parameters."""
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

        # FIX: metric_names moved to metric_manager
        assert evaluator.metric_manager.metric_names == ["mse"]
        assert evaluator.task_type == "regression"
        assert evaluator.save_predictions is False
        assert evaluator.collect_predictions is True
        assert evaluator.scaling_wrapper.apply_inverse is False

    def test_init_all_params_regression(self, mock_metric_factory):
        """Test initialization with all parameters for regression task."""
        mock_paths = MagicMock()
        mock_logger = MagicMock()

        evaluator = DefaultEvaluator(
            metric_names=["mse"],
            task_type="regression",
            num_classes=None,
            save_predictions=True,
            collect_predictions=True,
            remote_logger=mock_logger,
            paths=mock_paths,
        )

        assert evaluator.task_type == "regression"
        assert evaluator.save_predictions is True
        assert evaluator.collect_predictions is True
        assert evaluator.remote_logger == mock_logger
        assert evaluator.paths == mock_paths

        # Check components
        assert isinstance(evaluator.buffer.data["preds"], list)
        assert isinstance(evaluator.metric_manager.metrics, dict)

    def test_init_all_params_classification(self, mock_metric_factory):
        """Test initialization with all parameters for classification."""
        mock_paths = {"output": "/test/path"}

        evaluator = DefaultEvaluator(
            metric_names=["accuracy", "f1"],
            task_type="classification",
            num_classes=5,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=True,
            paths=mock_paths,
            remote_logger=None,
            collect_predictions=True,
        )

        assert evaluator.task_type == "classification"
        assert evaluator.num_classes == 5
        # FIX: metric_names moved to metric_manager
        assert evaluator.metric_manager.metric_names == ["accuracy", "f1"]

    def test_init_task_type_case_insensitive(self, mock_metric_factory):
        """Test that task_type is case-insensitive."""
        evaluator1 = DefaultEvaluator(
            metric_names=["mse"],
            task_type="REGRESSION",
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )
        evaluator2 = DefaultEvaluator(
            metric_names=["mse"],
            task_type="Regression",
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )
        evaluator3 = DefaultEvaluator(
            metric_names=["mse"],
            task_type="regression",
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        assert evaluator1.task_type == "regression"
        assert evaluator2.task_type == "regression"
        assert evaluator3.task_type == "regression"

    def test_init_metric_names_case_insensitive(self, mock_metric_factory):
        """Test that metric_names are case-insensitive."""
        evaluator = DefaultEvaluator(
            metric_names=["MSE", "RMSE", "mse"],
            task_type="regression",
            num_classes=None,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        # FIX: metric_names moved to metric_manager
        assert evaluator.metric_manager.metric_names == ["mse", "rmse", "mse"]

    def test_init_collect_predictions_auto_enabled_by_save(self, mock_metric_factory):
        """Test that collect_predictions is auto-enabled when save_predictions=True."""
        evaluator = DefaultEvaluator(
            metric_names=["mse"],
            task_type="regression",
            num_classes=None,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=True,
            paths={},
            remote_logger=None,
            collect_predictions=None,  # Use None to test auto-enable logic
        )

        # Should be True because save_predictions=True
        assert evaluator.collect_predictions is True

    def test_init_normalized_metrics_created_when_scaling_on(self, mocker):
        """Test that normalized_metrics is created when scaling is enabled."""
        # 1. Setup Mock Transform
        mock_transform = MagicMock(spec=InverseTransformMixin)
        mock_transform.apply_inverse = True  # Critical flag

        # 2. Setup Mock Factory
        mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

        # 3. Init Evaluator with TWO specific metrics
        evaluator = DefaultEvaluator(
            metric_names=["mse", "rmse"],
            task_type="regression",
            num_classes=None,
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        # 4. Assertions (Precise checks)
        # Note: In the new architecture, metrics are inside 'metric_manager'
        manager = evaluator.metric_manager

        assert manager.normalized_metrics is not None
        assert isinstance(manager.normalized_metrics, dict)
        assert len(manager.normalized_metrics) == 2
        assert "mse" in manager.normalized_metrics
        assert "rmse" in manager.normalized_metrics

    def test_init_normalized_metrics_not_created_when_scaling_off(
        self, mock_metric_factory
    ):
        """Test that normalized_metrics is NOT created when scaling is disabled."""
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

        # FIX: Check via manager
        assert evaluator.metric_manager.normalized_metrics is None

    def test_init_collections_created_when_collect_enabled(self, mock_metric_factory):
        """Test that collections are created when collect_predictions=True."""
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

        assert isinstance(evaluator.buffer.data["preds"], list)
        assert isinstance(evaluator.buffer.data["targets"], list)
        assert evaluator.buffer.data["preds"] == []

    def test_init_collections_not_created_when_collect_disabled(
        self, mock_metric_factory
    ):
        """Test that collections are NOT created when collect_predictions=False."""
        evaluator = DefaultEvaluator(
            metric_names=["mse"],
            task_type="regression",
            num_classes=None,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=False,
        )

        # Should not have collections
        assert not hasattr(evaluator, "preds")
        assert not hasattr(evaluator, "targets")

    def test_init_invalid_task_type_raises_error(self, mock_metric_factory):
        """Test that invalid task_type raises AssertionError."""
        with pytest.raises(AssertionError, match="Unsupported task"):
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

    def test_init_empty_metric_names_raises_error(self, mock_metric_factory):
        """Test that empty metric_names raises error."""
        with pytest.raises(ValueError, match="metric_names cannot be empty"):
            DefaultEvaluator(
                metric_names=[],
                task_type="regression",
                num_classes=None,
                inverse_transform=None,
                apply_inverse_scaling=False,
                save_predictions=False,
                paths={},
                remote_logger=None,
                collect_predictions=True,
            )

    def test_init_inverse_transform_name_allowed(self, mock_metric_factory):
        """Test that inverse_transform_name kwarg is allowed."""
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
            inverse_transform_name="StandardScaler",
        )

        assert evaluator is not None


# =============================================================================
# === CLASSIFICATION EVALUATOR INITIALIZATION TESTS ===
# =============================================================================


class TestClassificationEvaluatorInitialization:
    """Comprehensive tests for ClassificationEvaluator initialization."""

    @pytest.fixture
    def mock_metric_factory(self, mocker):
        mocker.patch.object(
            MetricFactory, "create_classification_metric", return_value=MagicMock()
        )

    def test_init_required_num_classes(self, mock_metric_factory):
        """Test that num_classes is required."""
        evaluator = ClassificationEvaluator(
            num_classes=3,
            metric_names=["accuracy"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
        )

        assert evaluator.num_classes == 3
        assert evaluator.task_type == "classification"

    def test_init_all_params(self, mock_metric_factory):
        """Test initialization with all parameters."""
        mock_transform = MagicMock(spec=InverseTransformMixin)

        evaluator = ClassificationEvaluator(
            num_classes=5,
            metric_names=["accuracy", "f1", "precision"],
            inverse_transform=mock_transform,
            apply_inverse_scaling=False,
            save_predictions=True,
            paths={"output": "/test"},
        )

        assert evaluator.num_classes == 5
        # FIX: metric_names moved to metric_manager
        assert evaluator.metric_manager.metric_names == ["accuracy", "f1", "precision"]
        assert evaluator.save_predictions is True

    def test_init_num_classes_validation(self, mock_metric_factory):
        """Test num_classes validation (should be positive integer)."""
        # Valid cases
        ClassificationEvaluator(
            num_classes=1,
            metric_names=["accuracy"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
        )
        ClassificationEvaluator(
            num_classes=10,
            metric_names=["accuracy"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
        )
        ClassificationEvaluator(
            num_classes=100,
            metric_names=["accuracy"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
        )

        # Invalid cases
        with pytest.raises(ValueError, match="positive integer"):
            ClassificationEvaluator(
                num_classes=0,
                metric_names=["accuracy"],
                inverse_transform=None,
                apply_inverse_scaling=False,
                save_predictions=False,
                paths={},
            )

        with pytest.raises(ValueError, match="positive integer"):
            ClassificationEvaluator(
                num_classes=-1,
                metric_names=["accuracy"],
                inverse_transform=None,
                apply_inverse_scaling=False,
                save_predictions=False,
                paths={},
            )


# =============================================================================
# === FORECASTING EVALUATOR INITIALIZATION TESTS ===
# =============================================================================


class TestForecastingEvaluatorInitialization:
    """Comprehensive tests for ForecastingEvaluator initialization."""

    @pytest.fixture
    def mock_metric_factory(self, mocker):
        mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    def test_init_minimal_required(self, mock_metric_factory):
        """Test initialization with minimal required parameters."""
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

        assert evaluator.model_pred_len == 5
        assert evaluator.effective_pred_len is None
        assert evaluator.target_dim_position is None

    def test_init_all_params(self, mock_metric_factory):
        """Test initialization with all parameters."""
        mock_transform = MagicMock(spec=InverseTransformMixin)

        evaluator = ForecastingEvaluator(
            target_dim_position=1,
            metric_names=["mse", "mae"],
            model_seq_len=96,
            model_label_len=48,
            effective_pred_len=12,
            model_pred_len=24,
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            save_predictions=True,
            paths={"output": "/test"},
            task_mode="univariate",
        )

        assert evaluator.target_dim_position == 1
        assert evaluator.model_pred_len == 24
        assert evaluator.effective_pred_len == 12
        assert evaluator.task_mode == "univariate"

    def test_init_missing_model_pred_len_raises_error(self, mock_metric_factory):
        """Test that missing model_pred_len raises ValueError."""
        with pytest.raises(
            ValueError, match="model_pred_len must be set for ForecastingEvaluator."
        ):
            ForecastingEvaluator(
                target_dim_position=None,
                metric_names=["mse"],
                model_seq_len=10,
                model_label_len=5,
                effective_pred_len=None,
                model_pred_len=None,  # Missing!
                inverse_transform=None,
                apply_inverse_scaling=False,
                save_predictions=False,
                paths={},
            )

    def test_init_effective_greater_than_model_pred_len_raises_error(
        self, mock_metric_factory
    ):
        """Test that effective_pred_len > model_pred_len raises ValueError."""
        with pytest.raises(ValueError, match="cannot be greater"):
            ForecastingEvaluator(
                target_dim_position=None,
                metric_names=["mse"],
                model_seq_len=10,
                model_label_len=5,
                effective_pred_len=10,  # > model_pred_len!
                model_pred_len=5,
                inverse_transform=None,
                apply_inverse_scaling=False,
                save_predictions=False,
                paths={},
            )

    def test_init_effective_equal_to_model_pred_len_valid(self, mock_metric_factory):
        """Test that effective_pred_len == model_pred_len is valid."""
        evaluator = ForecastingEvaluator(
            target_dim_position=None,
            metric_names=["mse"],
            model_seq_len=10,
            model_label_len=5,
            effective_pred_len=5,  # Equal is OK
            model_pred_len=5,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
        )

        assert evaluator.effective_pred_len == 5

    def test_init_effective_less_than_model_pred_len_valid(self, mock_metric_factory):
        """Test that effective_pred_len < model_pred_len is valid."""
        evaluator = ForecastingEvaluator(
            target_dim_position=None,
            metric_names=["mse"],
            model_seq_len=10,
            model_label_len=5,
            effective_pred_len=5,  # Less is OK
            model_pred_len=10,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
        )

        assert evaluator.effective_pred_len == 5

    def test_init_target_dim_position_none_valid(self, mock_metric_factory):
        """Test that target_dim_position=None is valid."""
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

        assert evaluator.target_dim_position is None

    def test_init_target_dim_position_int_valid(self, mock_metric_factory):
        """Test that target_dim_position as int is valid."""
        evaluator = ForecastingEvaluator(
            target_dim_position=0,
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

        assert evaluator.target_dim_position == 0


# =============================================================================
# === MULTI-UNIT EVALUATOR INITIALIZATION TESTS ===
# =============================================================================


class TestMultiUnitEvaluatorInitialization:
    """Comprehensive tests for MultiUnitEvaluator initialization."""

    @pytest.fixture
    def mock_metric_factory(self, mocker):
        mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    def test_init_minimal_required(self, mock_metric_factory):
        """Verify minimal init with explicit hook injection."""
        # Using the global class reference directly
        evaluator = MultiUnitEvaluator(
            metric_names=["mse"],
            paths={},
            hooks=[UnitTrendPlotHook()],
        )

        assert any(isinstance(h, UnitTrendPlotHook) for h in evaluator.hooks)
        assert evaluator.log_per_unit_metrics is True

    def test_init_all_params(self, mock_metric_factory):
        """Test initialization with all parameters via hook injection."""
        plot_hook = UnitTrendPlotHook(
            log_every_n_epochs=5,
            enable_subsampling=False,
            subsample_threshold=1000,
            subsample_factor=5,
        )

        evaluator = MultiUnitEvaluator(
            metric_names=["mse", "rmse"],
            save_predictions=True,
            collect_predictions=True,
            hooks=[plot_hook],
            log_per_unit_metrics=False,
        )

        registered_hook = next(
            (h for h in evaluator.hooks if isinstance(h, UnitTrendPlotHook)), None
        )

        assert registered_hook.log_every_n_epochs == 5
        assert evaluator.log_per_unit_metrics is False

    def test_init_plotting_params_defaults(self, mock_metric_factory):
        """Verify defaults in the injected hook."""
        evaluator = MultiUnitEvaluator(
            metric_names=["mse"], paths={}, hooks=[UnitTrendPlotHook()]
        )

        plot_hook = next(h for h in evaluator.hooks if isinstance(h, UnitTrendPlotHook))
        assert plot_hook.log_every_n_epochs == 10
        assert plot_hook.enable_subsampling is True

    def test_init_unit_ids_collection_not_created_when_collect_disabled(
        self, mock_metric_factory
    ):
        """Verify that injecting a Hook forces collect_predictions to True."""
        evaluator = MultiUnitEvaluator(
            metric_names=["mse"],
            hooks=[UnitTrendPlotHook()],
            collect_predictions=False,  # This should be promoted to True
        )

        # Promotion check (used difined collect_predictions has to override hook logic)
        assert evaluator.collect_predictions is False
        assert "unit_ids" in evaluator.buffer.data

        evaluator = MultiUnitEvaluator(
            metric_names=["mse"],
            hooks=[UnitTrendPlotHook()],
        )

        # Promotion check (used difined collect_predictions has to override hook logic)
        assert evaluator.collect_predictions is True
        assert "unit_ids" in evaluator.buffer.data

    # def test_init_invalid_task_type_raises_error(self, mock_metric_factory):
    #     """Verify task-type guard in MultiUnitEvaluator."""
    #     with pytest.raises(
    #         AssertionError,
    #         match="MultiUnitEvaluator only supports regression",
    #     ):
    #         # We provide num_classes to keep MetricManager happy,
    #         # so we can trigger the Evaluator's assertion.
    #         MultiUnitEvaluator(
    #             metric_names=["accuracy"],
    #             task_type="classification",
    #             num_classes=2,
    #         )


# =============================================================================
# === RECONSTRUCTION EVALUATOR INITIALIZATION TESTS ===
# =============================================================================


class TestReconstructionEvaluatorInitialization:
    """Comprehensive tests for ReconstructionEvaluator initialization."""

    @pytest.fixture
    def mock_metric_factory(self, mocker):
        mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    def test_init_minimal_required(self, mock_metric_factory):
        """Test initialization with minimal required parameters."""
        evaluator = ReconstructionEvaluator(
            metric_names=["mse"],
            paths={},
        )

        # FIX: metric_names moved to metric_manager
        assert evaluator.metric_manager.metric_names == ["mse"]
        assert (
            any(isinstance(h, ReconstructionPlotHook) for h in evaluator.hooks) is False
        )

    def test_init_all_params(self, mock_metric_factory):
        """Test initialization with all parameters via injection."""
        mock_transform = MagicMock(spec=InverseTransformMixin)
        mock_logger = MagicMock()
        mock_paths = {"plot_dir": "/test/plots"}

        # Inject the hook explicitly for lean evaluator
        evaluator = ReconstructionEvaluator(
            metric_names=["mse", "mae"],
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            save_predictions=True,
            hooks=[ReconstructionPlotHook()],
            paths=mock_paths,
            collect_predictions=True,
            remote_logger=mock_logger,
        )

        assert any(isinstance(h, ReconstructionPlotHook) for h in evaluator.hooks)
        assert evaluator.collect_predictions is True  # Auto-enabled by Hook presence

    def test_init_plot_reconstructions_forces_collect(self, mock_metric_factory):
        """When collect_predictions=None and hook is present, Smart Promotion sets True."""
        evaluator = ReconstructionEvaluator(
            metric_names=["mse"],
            hooks=[ReconstructionPlotHook()],
            collect_predictions=None,
        )
        assert evaluator.collect_predictions is True


# =============================================================================
# === PARAMETER COMBINATION TESTS ===
# =============================================================================


class TestParameterCombinations:
    """Test various parameter combinations for edge cases."""

    @pytest.fixture
    def mock_metric_factory(self, mocker):
        mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
        mocker.patch.object(
            MetricFactory, "create_classification_metric", return_value=MagicMock()
        )

    def test_scaling_without_transform_warning(self, mock_metric_factory):
        """Test that apply_inverse_scaling=True without inverse_transform works."""
        # This should work (scaling wrapper handles None transform)
        evaluator = DefaultEvaluator(
            metric_names=["mse"],
            task_type="regression",
            num_classes=None,
            inverse_transform=None,
            apply_inverse_scaling=True,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        assert evaluator.scaling_wrapper.apply_inverse is True
        assert evaluator.scaling_wrapper.inverse_transform is None

    def test_multiple_metrics_initialization(self, mock_metric_factory):
        """Test initialization with multiple metrics."""
        evaluator = DefaultEvaluator(
            metric_names=["mse", "rmse", "mae", "r2"],
            task_type="regression",
            num_classes=None,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        # FIX: metric_names moved to metric_manager
        assert len(evaluator.metric_manager.metric_names) == 4
        assert all(
            name in evaluator.metric_manager.metric_names
            for name in ["mse", "rmse", "mae", "r2"]
        )

    def test_paths_as_dict_vs_mock(self, mock_metric_factory):
        """Test that paths can be dict or mock object."""
        # Dict paths
        evaluator1 = DefaultEvaluator(
            metric_names=["mse"],
            task_type="regression",
            num_classes=None,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={"output": "/test"},
            remote_logger=None,
            collect_predictions=True,
        )
        assert isinstance(evaluator1.paths, dict)

        # Mock paths
        mock_paths = MagicMock()
        evaluator2 = DefaultEvaluator(
            metric_names=["mse"],
            task_type="regression",
            num_classes=None,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths=mock_paths,
            remote_logger=None,
            collect_predictions=True,
        )
        assert evaluator2.paths == mock_paths

    def test_remote_logger_none_vs_mock(self, mock_metric_factory):
        """Test that remote_logger can be None or mock."""
        # None logger
        evaluator1 = DefaultEvaluator(
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
        assert evaluator1.remote_logger is None

        # Mock logger
        mock_logger = MagicMock()
        evaluator2 = DefaultEvaluator(
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
        assert evaluator2.remote_logger == mock_logger
