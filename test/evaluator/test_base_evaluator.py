"""
Tests for picid.evaluator.base module.

This module contains tests for AbstractEvaluator, the base class for all
PHM evaluators. It handles initialization, scaling wrapper setup, and
defines the interface for concrete evaluators.

Coverage Target: 100%

PHM Context:
- AbstractEvaluator provides foundation for RUL, classification, and forecasting evaluators
- Handles inverse transform setup for denormalizing predictions
- Manages paths for saving evaluation results
"""

import numpy as np
from unittest.mock import MagicMock

from picid.evaluator.base import AbstractEvaluator
from picid.evaluator.scaling_wrapper import (
    ScalingWrapper,
    MultivariateTimeseriesScalingWrapper,
)
from picid.transforms.base.multisource import InverseTransformMixin


# =============================================================================
# === Concrete Implementation for Testing ===
# =============================================================================


class ConcreteEvaluator(AbstractEvaluator):
    """
    Concrete implementation of AbstractEvaluator for testing.

    Implements all abstract methods with minimal functionality
    to allow instantiation and testing of base class behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_called = False
        self._compute_called = False
        self._reset_called = False

    def update(self, model_out: dict):
        """Record update call."""
        self._update_called = True
        self._last_model_out = model_out

    def compute(self):
        """Record compute call and return dummy metrics."""
        self._compute_called = True
        return {"dummy_metric": 1.0}

    def reset(self):
        """Record reset call."""
        self._reset_called = True


# =============================================================================
# === Test Class for AbstractEvaluator ===
# =============================================================================


class TestAbstractEvaluatorInitialization:
    """Tests for AbstractEvaluator initialization.

    Validates correct setup of scaling wrappers, paths, and task modes.
    """

    def test_init_basic(self):
        """Test basic initialization without inverse transform.

        **PHM Logic**: Evaluators can work without inverse scaling
        for tasks that use normalized outputs directly.

        **Methodology**: Create evaluator without inverse_transform.

        **Expected**: ScalingWrapper created with apply_inverse=False.

        Validates: Requirement AE-INIT-1 - Basic initialization
        """
        evaluator = ConcreteEvaluator(
            paths={"output": "/test/path"},
            inverse_transform=None,
            apply_inverse_scaling=False,
            task_mode=False,
        )

        assert isinstance(evaluator.scaling_wrapper, ScalingWrapper)
        assert evaluator.scaling_wrapper.apply_inverse is False
        assert evaluator.paths == {"output": "/test/path"}

    def test_init_with_inverse_transform(self):
        """Test initialization with inverse transform enabled.

        **PHM Logic**: Most PHM tasks require inverse scaling to
        convert normalized predictions to physical units.

        **Methodology**: Create evaluator with mock inverse_transform.

        **Expected**: ScalingWrapper configured for inverse scaling.

        Validates: Requirement AE-INIT-2 - Inverse transform setup
        """
        mock_transform = MagicMock(spec=InverseTransformMixin)

        evaluator = ConcreteEvaluator(
            paths={"output": "/test/path"},
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode=False,
        )

        assert evaluator.scaling_wrapper.inverse_transform == mock_transform
        assert evaluator.scaling_wrapper.apply_inverse is True

    def test_init_multivariate_task_mode(self):
        """Test initialization with multivariate task mode.

        **PHM Logic**: Multivariate forecasting uses specialized
        scaling wrapper for multi-channel sensor data.

        **Methodology**: Create evaluator with task_mode="multivariate".

        **Expected**: MultivariateTimeseriesScalingWrapper created.

        Validates: Requirement AE-INIT-3 - Multivariate mode
        """
        mock_transform = MagicMock(spec=InverseTransformMixin)

        evaluator = ConcreteEvaluator(
            paths={"output": "/test/path"},
            inverse_transform=mock_transform,
            apply_inverse_scaling=True,
            task_mode="multivariate",
        )

        assert isinstance(
            evaluator.scaling_wrapper, MultivariateTimeseriesScalingWrapper
        )
        assert evaluator.task_mode == "multivariate"

    def test_init_non_multivariate_task_mode(self):
        """Test initialization with non-multivariate task modes.

        **PHM Logic**: Standard regression/forecasting uses basic wrapper.

        **Methodology**: Create evaluator with various task_modes.

        **Expected**: Standard ScalingWrapper created.

        Validates: Requirement AE-INIT-4 - Non-multivariate modes
        """
        for task_mode in [False, None, "regression", "forecasting"]:
            evaluator = ConcreteEvaluator(
                paths={},
                inverse_transform=None,
                apply_inverse_scaling=False,
                task_mode=task_mode,
            )

            assert isinstance(evaluator.scaling_wrapper, ScalingWrapper)
            assert evaluator.task_mode == task_mode

    def test_init_inverse_transform_name_allowed(self):
        """Test that inverse_transform_name kwarg is allowed.

        **PHM Logic**: inverse_transform_name is a special allowed kwarg
        for configuration/logging purposes.

        **Methodology**: Pass inverse_transform_name kwarg.

        **Expected**: No error raised.

        Validates: Requirement AE-INIT-6 - Allowed special kwargs
        """
        # Should not raise
        evaluator = ConcreteEvaluator(
            paths={},
            inverse_transform=None,
            apply_inverse_scaling=False,
            task_mode=False,
            inverse_transform_name="StandardScaler",
        )

        assert evaluator is not None


class TestAbstractEvaluatorRepr:
    """Tests for AbstractEvaluator __repr__ method.

    Validates string representation includes relevant attributes.
    """

    def test_repr_basic(self):
        """Test __repr__ returns informative string.

        **PHM Logic**: Repr helps debug evaluator configuration.

        **Methodology**: Create evaluator and call repr().

        **Expected**: String contains class name and simple attributes.

        Validates: Requirement AE-REPR-1 - Basic repr
        """
        evaluator = ConcreteEvaluator(
            paths={"output": "/test"},
            inverse_transform=None,
            apply_inverse_scaling=False,
            task_mode="regression",
        )

        repr_str = repr(evaluator)

        assert "ConcreteEvaluator" in repr_str
        assert "task_mode" in repr_str
        assert "regression" in repr_str

    def test_repr_excludes_complex_objects(self):
        """Test __repr__ excludes complex non-simple objects.

        **PHM Logic**: Complex objects (arrays, callables) not shown in repr.

        **Methodology**: Create evaluator with complex attributes.

        **Expected**: Complex objects not in repr string.

        Validates: Requirement AE-REPR-2 - Complex object exclusion
        """
        mock_transform = MagicMock(spec=InverseTransformMixin)

        evaluator = ConcreteEvaluator(
            paths={"output": "/test"},
            inverse_transform=mock_transform,
            apply_inverse_scaling=False,
            task_mode=False,
        )

        repr_str = repr(evaluator)

        # scaling_wrapper (complex object) should not appear
        assert "MagicMock" not in repr_str

    def test_repr_includes_simple_types(self):
        """Test __repr__ includes int, float, str, bool, None.

        **PHM Logic**: Simple configuration values shown for debugging.

        **Methodology**: Verify various simple types appear in repr.

        **Expected**: Simple attribute values visible.

        Validates: Requirement AE-REPR-3 - Simple type inclusion
        """
        evaluator = ConcreteEvaluator(
            paths={"output": "/test"},
            inverse_transform=None,
            apply_inverse_scaling=False,
            task_mode=False,
        )

        # Add some simple attributes manually for test
        evaluator.test_int = 42
        evaluator.test_float = 3.14
        evaluator.test_str = "hello"
        evaluator.test_bool = True
        evaluator.test_none = None

        repr_str = repr(evaluator)

        # All simple types should be included
        assert "test_int" in repr_str
        assert "42" in repr_str
        assert "test_float" in repr_str
        assert "test_str" in repr_str
        assert "test_bool" in repr_str
        assert "test_none" in repr_str


class TestAbstractEvaluatorAbstractMethods:
    """Tests verifying abstract method interface.

    Validates that abstract methods must be implemented by subclasses.
    """

    def test_abstract_methods_exist(self):
        """Test that AbstractEvaluator has required abstract methods.

        **PHM Logic**: Interface ensures consistent evaluator behavior.

        **Methodology**: Check abstract method definitions.

        **Expected**: update, compute, reset are abstract.

        Validates: Requirement AE-ABS-1 - Abstract method interface
        """
        abstract_methods = AbstractEvaluator.__abstractmethods__

        assert "update" in abstract_methods
        assert "compute" in abstract_methods
        assert "reset" in abstract_methods

    def test_concrete_update_callable(self):
        """Test that concrete update method is callable.

        **PHM Logic**: Update method processes model outputs.

        **Methodology**: Call update on concrete evaluator.

        **Expected**: Method executes without error.

        Validates: Requirement AE-ABS-2 - Update implementation
        """
        evaluator = ConcreteEvaluator(
            paths={},
            inverse_transform=None,
            apply_inverse_scaling=False,
            task_mode=False,
        )

        model_out = {"predictions": np.array([1.0]), "targets": np.array([1.0])}
        evaluator.update(model_out)

        assert evaluator._update_called is True
        assert evaluator._last_model_out == model_out

    def test_concrete_compute_callable(self):
        """Test that concrete compute method is callable.

        **PHM Logic**: Compute method returns metrics dictionary.

        **Methodology**: Call compute on concrete evaluator.

        **Expected**: Returns dict with metrics.

        Validates: Requirement AE-ABS-3 - Compute implementation
        """
        evaluator = ConcreteEvaluator(
            paths={},
            inverse_transform=None,
            apply_inverse_scaling=False,
            task_mode=False,
        )

        result = evaluator.compute()

        assert evaluator._compute_called is True
        assert isinstance(result, dict)
        assert "dummy_metric" in result

    def test_concrete_reset_callable(self):
        """Test that concrete reset method is callable.

        **PHM Logic**: Reset method clears accumulated state.

        **Methodology**: Call reset on concrete evaluator.

        **Expected**: Method executes without error.

        Validates: Requirement AE-ABS-4 - Reset implementation
        """
        evaluator = ConcreteEvaluator(
            paths={},
            inverse_transform=None,
            apply_inverse_scaling=False,
            task_mode=False,
        )

        evaluator.reset()

        assert evaluator._reset_called is True


class TestAbstractEvaluatorPathHandling:
    """Tests for path configuration in AbstractEvaluator.

    Validates correct handling of output paths for saving results.
    """

    def test_paths_stored_correctly(self):
        """Test that paths dict is stored correctly.

        **PHM Logic**: Paths dict specifies output locations.

        **Methodology**: Pass paths dict and verify storage.

        **Expected**: paths attribute equals input.

        Validates: Requirement AE-PATH-1 - Path storage
        """
        paths = {
            "output": "/results/run_001",
            "plots": "/results/run_001/plots",
            "eval_details": "/results/run_001/eval",
        }

        evaluator = ConcreteEvaluator(
            paths=paths,
            inverse_transform=None,
            apply_inverse_scaling=False,
            task_mode=False,
        )

        assert evaluator.paths == paths

    def test_paths_can_be_mock_object(self):
        """Test that paths can be a mock object (e.g., from Hydra).

        **PHM Logic**: Hydra configs may provide DictConfig/dataclass.

        **Methodology**: Pass MagicMock with path attributes.

        **Expected**: paths stored as-is.

        Validates: Requirement AE-PATH-2 - Mock/config path object
        """
        mock_paths = MagicMock()
        mock_paths.output = "/results/run_001"
        mock_paths.eval_details = "/results/run_001/eval"

        evaluator = ConcreteEvaluator(
            paths=mock_paths,
            inverse_transform=None,
            apply_inverse_scaling=False,
            task_mode=False,
        )

        assert evaluator.paths == mock_paths
        assert evaluator.paths.output == "/results/run_001"
