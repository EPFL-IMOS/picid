"""
Tests for picid.evaluator.hooks.base.BaseEvalHook.

Ref: docs/evaluators/index.md - AbstractEvaluator, hook lifecycle.
Validates: Abstract interface contract, on_update_end no-op, on_compute_end abstract.
"""

import pytest

from picid.evaluator.hooks.base import BaseEvalHook


# =============================================================================
# === Gold Standard: Concrete Hook for Testing Abstract Base ===
# =============================================================================


class _ConcreteEvalHook(BaseEvalHook):
    """
    Concrete implementation of BaseEvalHook for testing.

    Methodology: Implements on_compute_end to satisfy abstract contract.
    Expected physical outcome: Enables validation of base class behavior.
    Ref: docs/evaluators/index.md - hook on_compute_end primary entry point.
    """

    def __init__(self):
        self.compute_called = False
        self.update_called = False

    def on_update_end(self, batch, evaluator):
        """Override to track calls; base default is no-op."""
        self.update_called = True

    def on_compute_end(self, results, evaluator, mode, epoch, step):
        """Required abstract implementation."""
        self.compute_called = True


# =============================================================================
# === Tests ===
# =============================================================================


def test_base_eval_hook_cannot_be_instantiated_directly():
    """
    Validates that BaseEvalHook is abstract and cannot be instantiated.

    Methodology: Attempt to instantiate BaseEvalHook directly.
    Expected outcome: TypeError due to abstract on_compute_end.
    Ref: docs/evaluators/index.md - AbstractEvaluator, hook interface.
    """
    with pytest.raises(TypeError):
        BaseEvalHook()


def test_concrete_hook_implements_on_compute_end():
    """
    Validates that a concrete hook must implement on_compute_end.

    Methodology: Concrete subclass implements abstract method.
    Expected outcome: Hook can be instantiated and used.
    Ref: docs/evaluators/index.md - on_compute_end primary entry point.
    """
    hook = _ConcreteEvalHook()
    assert hasattr(hook, "on_compute_end")
    hook.on_compute_end({}, None, "val", 0, 0)
    assert hook.compute_called is True


def test_base_eval_hook_on_update_end_default_no_op():
    """
    Validates that base on_update_end is a no-op (does not raise).

    Methodology: Subclass that does NOT override on_update_end; call it.
    Expected outcome: No exception; default implementation does nothing.
    Ref: docs/evaluators/index.md - on_update_end for live streaming.
    """

    class _MinimalHook(BaseEvalHook):
        def on_compute_end(self, results, evaluator, mode, epoch, step):
            pass

    hook = _MinimalHook()
    # Base on_update_end has empty body - should not raise
    hook.on_update_end({"preds": [], "targets": []}, None)


def test_concrete_hook_on_update_end_can_be_overridden():
    """
    Validates that on_update_end can be overridden for custom behavior.

    Methodology: Subclass overrides on_update_end and tracks calls.
    Expected outcome: Override is invoked when hook receives batch.
    Ref: docs/evaluators/index.md - hook on_update_end after each batch.
    """
    hook = _ConcreteEvalHook()
    batch = {"preds": [[[0.5]]], "targets": [[[0.5]]]}
    evaluator = object()

    hook.on_update_end(batch, evaluator)

    assert hook.update_called is True


def test_concrete_hook_on_compute_end_receives_all_parameters():
    """
    Validates that on_compute_end receives results, evaluator, mode, epoch, step.

    Methodology: Concrete hook captures and asserts parameter presence.
    Expected outcome: All lifecycle parameters passed correctly.
    Ref: docs/evaluators/index.md - compute triggers on_compute_end.
    """

    class _CaptureHook(BaseEvalHook):
        def on_compute_end(self, results, evaluator, mode, epoch, step):
            self.results = results
            self.evaluator = evaluator
            self.mode = mode
            self.epoch = epoch
            self.step = step

    hook = _CaptureHook()
    results = {"mse_denormalized": 0.1}
    evaluator = object()

    hook.on_compute_end(results, evaluator, mode="test", epoch=5, step=100)

    assert hook.results == results
    assert hook.evaluator is evaluator
    assert hook.mode == "test"
    assert hook.epoch == 5
    assert hook.step == 100
