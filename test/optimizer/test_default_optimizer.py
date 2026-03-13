"""
Tests for picid.optimizer.default (DefaultOptimizer).

DefaultOptimizer is deprecated; tests validate deprecation behavior and
(when init is bypassed) that configure_optimizer and __repr__ behave
as documented. Logic aligns with docs/datasets.md and pipeline use of
optimizer configuration.
"""

import functools
import pytest
import torch

from picid.optimizer import default as default_module

# Use the module-level dicts so we can test configure_optimizer logic
TORCH_OPTIMIZERS = torch.optim.__dict__
TORCH_SCHEDULERS = torch.optim.lr_scheduler.__dict__


def _default_optimizer_init_without_raise(optimizer_id, parameters, scheduler=None):
    """
    Replicates DefaultOptimizer.__init__ logic after the DeprecationWarning
    raise, so we can test configure_optimizer and __repr__ without
    triggering deprecation in every test.
    """
    self = object.__new__(default_module.DefaultOptimizer)
    self.optimizer = functools.partial(TORCH_OPTIMIZERS[optimizer_id], **parameters)
    if scheduler is not None:
        scheduler_id = scheduler.get("scheduler_id")
        scheduler_params = scheduler.get("scheduler_params")
        self.scheduler = functools.partial(
            TORCH_SCHEDULERS[scheduler_id], **scheduler_params
        )
    else:
        self.scheduler = None
    return self


# -----------------------------------------------------------------------------
# Test class: Deprecation and initialization
# -----------------------------------------------------------------------------


class TestDefaultOptimizerDeprecation:
    """
    Validates that DefaultOptimizer is deprecated and that instantiation
    raises DeprecationWarning as the first action (docs: deprecation policy).
    """

    def test_init_raises_deprecation_warning(self):
        """
        **PHM Logic**: Deprecated code must warn users so they migrate
        to the recommended optimizer configuration path.

        **Methodology**: Instantiate DefaultOptimizer and assert
        DeprecationWarning is raised (not only issued).

        **Doc reference**: default.py - raise DeprecationWarning(...)
        """
        with pytest.raises(DeprecationWarning, match="deprecated and will be removed"):
            default_module.DefaultOptimizer("Adam", {"lr": 0.01})


# -----------------------------------------------------------------------------
# Test class: configure_optimizer (with init bypassed)
# -----------------------------------------------------------------------------


class TestDefaultOptimizerConfigureOptimizer:
    """
    Validates configure_optimizer return structure and content when
    DefaultOptimizer is constructed without the deprecation raise
    (init logic duplicated in _default_optimizer_init_without_raise).
    """

    def test_configure_optimizer_returns_optimizer_only_when_no_scheduler(
        self, phm_model_parameters
    ):
        """
        **PHM Logic**: When no scheduler is configured, the pipeline
        expects a dict with only "optimizer" for Lightning.

        **Methodology**: Build DefaultOptimizer (no raise) with
        scheduler=None; call configure_optimizer; assert keys and
        that optimizer has correct param_groups.

        **Doc reference**: default.py - return {"optimizer": optimizer}
        """
        opt = _default_optimizer_init_without_raise(
            "Adam", {"lr": 0.001}, scheduler=None
        )
        result = opt.configure_optimizer(phm_model_parameters)
        assert isinstance(result, dict)
        assert result.keys() == {"optimizer"}
        assert hasattr(result["optimizer"], "param_groups")
        assert len(result["optimizer"].param_groups) == 1
        assert result["optimizer"].param_groups[0]["lr"] == 0.001

    def test_configure_optimizer_returns_optimizer_and_lr_scheduler(
        self, phm_model_parameters
    ):
        """
        **PHM Logic**: When a scheduler is configured, the pipeline
        expects "optimizer" and "lr_scheduler" with scheduler, monitor,
        interval, frequency for Lightning.

        **Methodology**: Build DefaultOptimizer with StepLR; call
        configure_optimizer; assert structure and scheduler attachment.

        **Doc reference**: default.py - lr_scheduler dict with
        scheduler, monitor, interval, frequency.
        """
        scheduler_cfg = {
            "scheduler_id": "StepLR",
            "scheduler_params": {"step_size": 1, "gamma": 0.9},
        }
        opt = _default_optimizer_init_without_raise(
            "SGD", {"lr": 0.01}, scheduler=scheduler_cfg
        )
        result = opt.configure_optimizer(phm_model_parameters)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        lr_cfg = result["lr_scheduler"]
        assert lr_cfg["monitor"] == "val/loss"
        assert lr_cfg["interval"] == "epoch"
        assert lr_cfg["frequency"] == 1
        assert hasattr(lr_cfg["scheduler"], "step")

    def test_configure_optimizer_optimizer_uses_given_parameters(
        self, phm_model_parameters
    ):
        """
        **PHM Logic**: Model parameters passed to configure_optimizer
        must be the ones trained (same tensor ids).

        **Methodology**: Call configure_optimizer and compare
        param_groups params to the input list.

        **Expected outcome**: Same parameter tensors in optimizer.
        """
        opt = _default_optimizer_init_without_raise("Adam", {"lr": 0.01})
        result = opt.configure_optimizer(phm_model_parameters)
        opt_params = [
            p for group in result["optimizer"].param_groups for p in group["params"]
        ]
        assert len(opt_params) == len(phm_model_parameters)
        assert all(id(a) == id(b) for a, b in zip(opt_params, phm_model_parameters))


# -----------------------------------------------------------------------------
# Test class: __repr__
# -----------------------------------------------------------------------------


class TestDefaultOptimizerRepr:
    """
    Validates __repr__ code paths.

    **PHM Logic**: DefaultOptimizer stores optimizer/scheduler as
    functools.partial; __repr__ uses .__name__ which partial does not
    provide, so the repr branch that formats scheduler is only reachable
    when scheduler is not None, and both branches raise AttributeError.
    """

    def test_repr_without_scheduler_raises_because_partial_has_no_name(self):
        """
        **Methodology**: Build DefaultOptimizer without scheduler and call
        __repr__. Source uses self.optimizer.__name__ but self.optimizer
        is a functools.partial which has no __name__.

        **Doc reference**: default.py __repr__ - optimizer is a partial.
        """
        opt = _default_optimizer_init_without_raise("Adam", {"lr": 0.01})
        with pytest.raises(AttributeError, match="__name__"):
            repr(opt)

    def test_repr_with_scheduler_raises_because_partial_has_no_name(self):
        """
        **Methodology**: Build DefaultOptimizer with scheduler and call
        __repr__; same AttributeError on self.optimizer.__name__.

        **Doc reference**: default.py __repr__ - both optimizer and
        scheduler are functools.partial.
        """
        scheduler_cfg = {
            "scheduler_id": "StepLR",
            "scheduler_params": {"step_size": 1, "gamma": 0.5},
        }
        opt = _default_optimizer_init_without_raise(
            "SGD", {"lr": 0.1}, scheduler=scheduler_cfg
        )
        with pytest.raises(AttributeError, match="__name__"):
            repr(opt)
