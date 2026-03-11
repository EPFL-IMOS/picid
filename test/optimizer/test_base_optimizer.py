"""
Tests for picid.optimizer.base (AbstractOptimizer).

Validates the abstract optimizer interface required by the PHM pipeline
(docs/datasets.md, docs/how_to_use_configs.md). All concrete optimizers
must implement configure_optimizer(model_parameters) to supply Lightning
with optimizer and optional LR scheduler.
"""

import pytest
import torch

from picid.optimizer.base import AbstractOptimizer


# -----------------------------------------------------------------------------
# Gold-standard: concrete implementation for testing abstract interface
# Docs: base.py - AbstractOptimizer defines configure_optimizer(model_parameters)
# -----------------------------------------------------------------------------


class ConcreteOptimizer(AbstractOptimizer):
    """
    Minimal concrete implementation of AbstractOptimizer for testing.

    **PHM Logic**: Pipeline (TrainingLightningModule) calls configure_optimizer
    with model parameters; return value must be a dict with "optimizer" key
    and optionally "lr_scheduler" for Lightning's configure_optimizers().
    """

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def configure_optimizer(self, model_parameters: dict):
        return {
            "optimizer": torch.optim.SGD(model_parameters, lr=self.lr),
        }


# -----------------------------------------------------------------------------
# Test class: AbstractOptimizer interface
# -----------------------------------------------------------------------------


class TestAbstractOptimizerInterface:
    """
    Validates that AbstractOptimizer is abstract and that concrete
    implementations satisfy the interface required by the pipeline.
    """

    def test_cannot_instantiate_abstract_optimizer(self):
        """
        **PHM Logic**: Only concrete optimizer implementations should be
        instantiated; the base class must remain abstract.

        **Methodology**: Attempt to instantiate AbstractOptimizer and
        expect TypeError (abstract methods not implemented).

        **Doc reference**: base.py - AbstractOptimizer(ABC) with
        abstractmethod configure_optimizer.
        """
        with pytest.raises(TypeError):
            AbstractOptimizer()

    def test_concrete_optimizer_configure_returns_dict(self, phm_model_parameters):
        """
        **PHM Logic**: configure_optimizer must return a dict compatible
        with Lightning's configure_optimizers() (optimizer key required).

        **Methodology**: Use ConcreteOptimizer and assert return type and
        presence of "optimizer" key.

        **Doc reference**: pipeline/base.py TrainingLightningModule
        uses configure_optimizers() returning such a dict.
        """
        opt = ConcreteOptimizer(lr=0.001)
        result = opt.configure_optimizer(phm_model_parameters)
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert hasattr(result["optimizer"], "param_groups")

    def test_concrete_optimizer_parameters_used(self, phm_model_parameters):
        """
        **PHM Logic**: The passed model_parameters must be the ones
        actually used by the optimizer (no substitution or copy that
        would break training).

        **Methodology**: Configure optimizer and check that param_groups
        contain the same parameter tensors (by id) as the input list.

        **Expected outcome**: Optimizer holds references to the same
        tensors as phm_model_parameters.
        """
        opt = ConcreteOptimizer(lr=0.01)
        result = opt.configure_optimizer(phm_model_parameters)
        optimizer = result["optimizer"]
        opt_params = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        input_ids = {id(p) for p in phm_model_parameters}
        assert opt_params == input_ids
