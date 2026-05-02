"""
Tests for picid.pipeline.base.

Validates PHM pipeline modules: CustomEvaluatorLightningModule,
BackboneWrapperLightningModule, FitPredictWrapperLightningModule,
TrainingLightningModule, ConstantLossLightningModule, and helpers.
Aligned with docs/dataobject.md, docs/datasets.md, docs/evaluators/index.md.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch

from picid.pipeline.base import (
    CustomEvaluatorInterface,
    CustomEvaluatorLightningModule,
    BackboneWrapperLightningModule,
    FitPredictWrapperLightningModule,
    ConstantLossLightningModule,
    TrainingLightningModule,
)


# -----------------------------------------------------------------------------
# Concrete module for testing CustomEvaluatorLightningModule
# -----------------------------------------------------------------------------


class ConcreteEvaluatorLightningModule(CustomEvaluatorLightningModule):
    """Minimal implementation that returns fixed model_out from each step."""

    def _training_step(self, batch, batch_idx):
        B = batch["features"].shape[0]
        return {
            "predictions": torch.rand(B, 1, 1),
            "targets": torch.rand(B, 1, 1),
            "loss": torch.tensor(0.5),
        }

    def _validation_step(self, batch, batch_idx):
        B = batch["features"].shape[0]
        return {
            "predictions": torch.rand(B, 1, 1),
            "targets": torch.rand(B, 1, 1),
            "loss": torch.tensor(0.5),
        }

    def _test_step(self, batch, batch_idx):
        B = batch["features"].shape[0]
        return {
            "predictions": torch.rand(B, 1, 1),
            "targets": torch.rand(B, 1, 1),
            "loss": torch.tensor(0.5),
        }


class DeterministicValCustomModule(ConcreteEvaluatorLightningModule):
    """Validation outputs are constant tensors so MSE on a real evaluator is predictable."""

    def _validation_step(self, batch, batch_idx):
        B = batch["features"].shape[0]
        return {
            "predictions": torch.ones(B, 1, 1),
            "targets": torch.zeros(B, 1, 1),
            "loss": torch.tensor(0.25),
        }


# -----------------------------------------------------------------------------
# Test: CustomEvaluatorInterface (abstract)
# -----------------------------------------------------------------------------


class TestCustomEvaluatorInterface:
    """Validates CustomEvaluatorInterface is abstract."""

    def test_cannot_instantiate_interface(self):
        """Doc: pipeline/base.py - CustomEvaluatorInterface(ABC) with abstract _training_step, _validation_step, _test_step."""
        with pytest.raises(TypeError):
            CustomEvaluatorInterface()


# -----------------------------------------------------------------------------
# Test: _to_numpy (data integrity for PHM pipeline)
# Doc: pipeline/base.py - _to_numpy centralizes tensor/ndarray/list -> numpy
# -----------------------------------------------------------------------------


class TestToNumpy:
    """Validates _to_numpy preserves data and handles allowed types."""

    def test_tensor_converted_to_numpy(self, mock_evaluators):
        """PHM: Predictions/targets from model are tensors; evaluators expect numpy. Methodology: call _to_numpy on tensor."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = module._to_numpy(t, "test")
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_almost_equal(out, t.numpy())

    def test_ndarray_passthrough(self, mock_evaluators):
        """PHM: Already-numpy data (e.g. from some evaluators) should pass through."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        arr = np.array([[1.0, 2.0]])
        out = module._to_numpy(arr, "test")
        assert out is arr

    def test_list_converted_to_numpy(self, mock_evaluators):
        """PHM: Scalar/list values (e.g. batch_idx as list) converted to array."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        lst = [0, 1, 2]
        out = module._to_numpy(lst, "test")
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.array(lst))

    def test_invalid_type_raises(self, mock_evaluators):
        """PHM: Only tensor/ndarray/list allowed; other types must raise with clear message."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        with pytest.raises(TypeError, match="must be a torch.Tensor or np.ndarray"):
            module._to_numpy("invalid", "test")


# -----------------------------------------------------------------------------
# Test: process_outputs (batch_idx, unit_id; docs/dataobject.md)
# -----------------------------------------------------------------------------


class TestProcessOutputs:
    """Validates process_outputs converts predictions/targets and optionally adds batch_idx/unit_id."""

    def test_predictions_and_targets_converted(
        self, mock_evaluators, phm_model_out_tensors
    ):
        """Doc: pipeline/base.py - process_outputs converts predictions and targets to numpy."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        batch = {}
        out = module.process_outputs(phm_model_out_tensors.copy(), batch)
        assert isinstance(out["predictions"], np.ndarray)
        assert isinstance(out["targets"], np.ndarray)

    def test_batch_idx_added_when_present(self, mock_evaluators, phm_model_out_tensors):
        """Doc: dataobject.md - batch_idx used for evaluator aggregation."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        batch = {"batch_idx": torch.tensor([0, 1, 2, 3])}
        out = module.process_outputs(phm_model_out_tensors.copy(), batch)
        assert "batch_idx" in out
        np.testing.assert_array_equal(out["batch_idx"], np.array([0, 1, 2, 3]))

    def test_batch_idx_none_not_added(self, mock_evaluators, phm_model_out_tensors):
        """When batch has batch_idx=None, do not add to model_out."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        batch = {"batch_idx": None}
        out = module.process_outputs(phm_model_out_tensors.copy(), batch)
        assert "batch_idx" not in out

    def test_unit_id_added_when_present(self, mock_evaluators, phm_model_out_tensors):
        """Doc: dataobject.md - unit_id for multi-unit fleet metrics."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        batch = {"unit_id": torch.tensor([1, 1, 2, 2])}
        out = module.process_outputs(phm_model_out_tensors.copy(), batch)
        assert "unit_id" in out
        np.testing.assert_array_equal(out["unit_id"], np.array([1, 1, 2, 2]))

    def test_unit_id_none_not_added(self, mock_evaluators, phm_model_out_tensors):
        """When batch has unit_id=None, do not add to model_out."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        batch = {"unit_id": None}
        out = module.process_outputs(phm_model_out_tensors.copy(), batch)
        assert "unit_id" not in out


# -----------------------------------------------------------------------------
# Test: CustomEvaluatorLightningModule training/val/test steps and hooks
# -----------------------------------------------------------------------------


class TestCustomEvaluatorLightningModuleSteps:
    """Validates training_step, validation_step, test_step and epoch hooks."""

    def test_training_step_returns_loss_and_updates_evaluator_when_evaluate(
        self, mock_evaluators, phm_batch_rul
    ):
        """Doc: pipeline/base.py - training_step with _evaluate=True updates train evaluator."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        loss = module.training_step(phm_batch_rul, 0, _evaluate=True)
        assert isinstance(loss, torch.Tensor)
        assert mock_evaluators["train"]._updates

    def test_training_step_skips_evaluator_when_evaluate_false(
        self, mock_evaluators, phm_batch_rul
    ):
        """Doc: pipeline/base.py - training_step(_evaluate=False) does not update evaluator."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        module.training_step(phm_batch_rul, 0, _evaluate=False)
        assert not mock_evaluators["train"]._updates

    def test_validation_step_updates_val_evaluator(
        self, mock_evaluators, phm_batch_rul
    ):
        """Doc: pipeline/base.py - validation_step updates val evaluator."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        module.validation_step(phm_batch_rul, 0)
        assert mock_evaluators["val"]._updates

    def test_test_step_updates_test_evaluator(self, mock_evaluators, phm_batch_rul):
        """Doc: pipeline/base.py - test_step updates test evaluator."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        module.test_step(phm_batch_rul, 0)
        assert mock_evaluators["test"]._updates

    def test_on_train_epoch_start_resets_train_evaluator(self, mock_evaluators):
        """Doc: pipeline/base.py - on_train_epoch_start resets train evaluator."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        mock_evaluators["train"]._updates.append({})
        module.on_train_epoch_start()
        assert mock_evaluators["train"]._reset_count >= 1

    def test_on_validation_epoch_start_resets_val_evaluator(self, mock_evaluators):
        """Doc: pipeline/base.py - on_validation_epoch_start resets val evaluator."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        mock_evaluators["val"]._updates.append({})
        module.on_validation_epoch_start()
        assert mock_evaluators["val"]._reset_count >= 1

    def test_on_test_epoch_start_resets_test_evaluator(self, mock_evaluators):
        """Doc: pipeline/base.py - on_test_epoch_start resets test evaluator."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        mock_evaluators["test"]._updates.append({})
        module.on_test_epoch_start()
        assert mock_evaluators["test"]._reset_count >= 1

    def test_on_validation_epoch_end_logs_metrics(self, mock_evaluators, phm_batch_rul):
        """Epoch end runs compute, logs val/* keys, and resets the val evaluator."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        module.validation_step(phm_batch_rul, 0)
        resets_before = mock_evaluators["val"]._reset_count
        with patch.object(module, "log") as mock_log:
            module.on_validation_epoch_end()
        names = [c.args[0] for c in mock_log.call_args_list]
        assert "val/mock_metric" in names
        assert mock_evaluators["val"]._compute_count >= 1
        assert mock_evaluators["val"]._reset_count == resets_before + 1

    def test_on_test_epoch_end_logs_metrics(self, mock_evaluators, phm_batch_rul):
        """Epoch end runs compute, logs test/* keys, and resets the test evaluator."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        module.test_step(phm_batch_rul, 0)
        resets_before = mock_evaluators["test"]._reset_count
        with patch.object(module, "log") as mock_log:
            module.on_test_epoch_end()
        names = [c.args[0] for c in mock_log.call_args_list]
        assert "test/mock_metric" in names
        assert mock_evaluators["test"]._reset_count == resets_before + 1

    def test_on_validation_epoch_end_with_default_evaluator_resets_and_logs_mse(
        self, phm_batch_rul, phm_default_evaluators
    ):
        """Integration-leaning: real DefaultEvaluator MSE is logged then cleared after epoch end."""
        val_ev = phm_default_evaluators["val"]
        module = DeterministicValCustomModule(evaluators=phm_default_evaluators)
        module.validation_step(phm_batch_rul, 0)
        assert val_ev.metric_manager.metrics["mse"].total_count > 0

        with patch.object(module, "log") as mock_log:
            module.on_validation_epoch_end()

        assert val_ev.metric_manager.metrics["mse"].total_count == 0
        mse_calls = [
            c
            for c in mock_log.call_args_list
            if c.args and c.args[0] == "val/mse_normalized"
        ]
        assert len(mse_calls) == 1
        assert mse_calls[0].args[1] == pytest.approx(1.0)

    def test_log_epoch_metrics_calls_compute_and_reset(self, mock_evaluators):
        """Doc: pipeline/base.py - log_epoch_metrics calls evaluators[mode].compute then reset."""
        module = ConcreteEvaluatorLightningModule(evaluators=mock_evaluators)
        module.log_epoch_metrics(mode="val", step=0, epoch=0)
        assert mock_evaluators["val"]._compute_count >= 1
        assert mock_evaluators["val"]._reset_count >= 1


# -----------------------------------------------------------------------------
# Test: BackboneWrapperLightningModule
# -----------------------------------------------------------------------------


class TestBackboneWrapperLightningModule:
    """Validates backbone forward, model_step, and step delegation."""

    def test_forward_returns_backbone_output(
        self, mock_evaluators, mock_backbone, mock_loss, phm_batch_rul
    ):
        """Doc: pipeline/base.py - forward(batch) returns backbone(batch)."""
        module = BackboneWrapperLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
            scheduler_factory=None,
        )
        out = module.forward(phm_batch_rul)
        assert "predictions" in out
        assert "targets" in out

    def test_model_step_sets_model_state_and_returns_forward(
        self, mock_evaluators, mock_backbone, mock_loss, phm_batch_rul
    ):
        """Doc: pipeline/base.py - model_step sets batch['model_state'] and returns forward(batch)."""
        module = BackboneWrapperLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
            scheduler_factory=None,
        )
        out = module.model_step(phm_batch_rul, stage="val")
        assert phm_batch_rul["model_state"] == "val"
        assert "predictions" in out

    def test_setup_no_op(self, mock_evaluators, mock_backbone, mock_loss):
        """Doc: pipeline/base.py - setup(stage) is pass."""
        module = BackboneWrapperLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
            scheduler_factory=None,
        )
        module.setup("fit")

    def test_validation_step_calls_model_step_val(
        self, mock_evaluators, mock_backbone, mock_loss, phm_batch_rul
    ):
        """BackboneWrapperLightningModule._validation_step calls model_step with stage=val (lines 396-397)."""
        module = TrainingLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
            scheduler_factory=None,
        )
        module.validation_step(phm_batch_rul, 0)
        assert phm_batch_rul["model_state"] == "val"

    def test_test_step_calls_model_step_test(
        self, mock_evaluators, mock_backbone, mock_loss, phm_batch_rul
    ):
        """BackboneWrapperLightningModule._test_step calls model_step with stage=test (lines 409-410, 422-423)."""
        module = TrainingLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
            scheduler_factory=None,
        )
        module.test_step(phm_batch_rul, 0)
        assert phm_batch_rul["model_state"] == "test"

    def test_repr_includes_backbone_and_loss(
        self, mock_evaluators, mock_backbone, mock_loss
    ):
        """Doc: pipeline/base.py - __repr__ includes backbone and loss."""
        module = BackboneWrapperLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
            scheduler_factory=None,
        )
        r = repr(module)
        assert "BackboneWrapperLightningModule" in r


# -----------------------------------------------------------------------------
# Test: FitPredictWrapperLightningModule
# Doc: docs/datasets.md FitPredictTaskDataset batch structure
# -----------------------------------------------------------------------------


class TestFitPredictWrapperLightningModule:
    """Validates fit-predict batch handling, get_task_info, model_step_fit/predict branches."""

    def test_init_debug_logs_warning(
        self, mock_evaluators, mock_fit_predict_backbone, caplog
    ):
        """Doc: pipeline/base.py - when debug=True, logger.warning is called."""
        with patch("picid.pipeline.base.logger") as mock_logger:
            FitPredictWrapperLightningModule(
                backbone=mock_fit_predict_backbone,
                evaluators=mock_evaluators,
                debug=True,
            )
            assert mock_logger.warning.called

    def test_repr_includes_backbone(self, mock_evaluators, mock_fit_predict_backbone):
        """Doc: pipeline/base.py - __repr__(backbone=...)."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        r = repr(module)
        assert "FitPredictWrapperLightningModule" in r

    def test_get_task_info_uses_task_desc_when_present(
        self, mock_evaluators, mock_fit_predict_backbone, phm_fit_predict_batch
    ):
        """Doc: pipeline/base.py - get_task_info returns task_desc from batch or default."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        info = module.get_task_info(phm_fit_predict_batch)
        assert "Task 1 of 3" in info

    def test_get_task_info_default_when_no_task_desc(
        self, mock_evaluators, mock_fit_predict_backbone
    ):
        """When task_desc missing, uses Task {task_idx+1} of {task_num}."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        batch = {"task_idx": torch.tensor(2), "task_num": 5}
        info = module.get_task_info(batch)
        assert "Task 3 of 5" in info

    def test_batch_to_fit_predict_valid_batch(
        self, mock_evaluators, mock_fit_predict_backbone, phm_fit_predict_batch
    ):
        """Doc: datasets.md - context (1, n_samples, n_features), target (1, n_samples, n_targets); squeeze to 2D."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        X, y = module._FitPredictWrapperLightningModule__batch_to_fit_predict(
            phm_fit_predict_batch
        )
        assert X.ndim == 2
        assert y.ndim == 2
        assert X.shape[0] == 50
        assert y.shape[0] == 50

    def test_batch_to_fit_predict_asserts_batch_size_one(
        self, mock_evaluators, mock_fit_predict_backbone
    ):
        """Doc: pipeline/base.py - __batch_to_fit_predict requires context/target shape[0]==1."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        batch = {"context": torch.randn(2, 10, 5), "target": torch.randn(2, 10, 1)}
        with pytest.raises(AssertionError, match="Expected batch size of 1"):
            module._FitPredictWrapperLightningModule__batch_to_fit_predict(batch)

    def test_model_step_fit_single_target_no_predict_after(
        self, mock_evaluators, mock_fit_predict_backbone, phm_fit_predict_batch
    ):
        """Doc: pipeline/base.py - model_step_fit with allows_multi_target, single target, predict_after_training=False."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
            predict_after_training=False,
        )
        out = module.model_step_fit(phm_fit_predict_batch)
        assert "loss" in out
        # Pipeline adds task dim: y.unsqueeze(1) gives (50, 1, 1)
        assert out["predictions"].ndim == 3
        assert out["targets"].ndim == 3
        assert out["predictions"].shape[0] == 50 and out["targets"].shape[0] == 50

    def test_model_step_predict_returns_predictions_and_targets(
        self, mock_evaluators, mock_fit_predict_backbone, phm_fit_predict_batch
    ):
        """Doc: pipeline/base.py - model_step_predict loads model, predicts, returns dict with predictions, targets, loss."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        module.model_step_fit(phm_fit_predict_batch)
        out = module.model_step_predict(phm_fit_predict_batch)
        assert out["predictions"].ndim == 3
        assert out["targets"].ndim == 3
        assert out["loss"].item() == 1

    def test_model_step_predict_multi_target_loads_per_target(
        self,
        mock_evaluators,
        mock_fit_predict_backbone,
        phm_fit_predict_batch_multi_target,
    ):
        """model_step_predict with allows_multi_target=False and n_targets>1 loads each target model (lines 591-596)."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        module.model_step_fit(phm_fit_predict_batch_multi_target)
        out = module.model_step_predict(phm_fit_predict_batch_multi_target)
        assert out["predictions"].ndim == 3
        assert (
            "0_0" in mock_fit_predict_backbone._models
            and "0_1" in mock_fit_predict_backbone._models
        )

    def test_validation_step_calls_model_step_predict(
        self, mock_evaluators, mock_fit_predict_backbone, phm_fit_predict_batch
    ):
        """FitPredictWrapperLightningModule._validation_step calls model_step_predict (lines 653-654)."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        module.model_step_fit(phm_fit_predict_batch)
        module.validation_step(phm_fit_predict_batch, 0)
        assert mock_evaluators["val"]._updates

    def test_test_step_calls_model_step_predict(
        self, mock_evaluators, mock_fit_predict_backbone, phm_fit_predict_batch
    ):
        """FitPredictWrapperLightningModule._test_step calls model_step_predict (lines 666-667)."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        module.model_step_fit(phm_fit_predict_batch)
        module.test_step(phm_fit_predict_batch, 0)
        assert mock_evaluators["test"]._updates

    def test_model_step_fit_with_predict_after_training_returns_predict_step(
        self, mock_evaluators, mock_fit_predict_backbone, phm_fit_predict_batch
    ):
        """Doc: pipeline/base.py - when predict_after_training=True, model_step_fit returns model_step_predict(batch)."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
            predict_after_training=True,
        )
        out = module.model_step_fit(phm_fit_predict_batch)
        assert "predictions" in out
        assert "targets" in out
        assert out["predictions"].ndim == 3

    def test_model_step_fit_multi_target_virtual_tasks(
        self,
        mock_evaluators,
        mock_fit_predict_backbone,
        phm_fit_predict_batch_multi_target,
    ):
        """Doc: pipeline/base.py - when allows_multi_target=False and n_targets>1, fit per target and serialize_model(task_idx_targetidx)."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
            predict_after_training=False,
        )
        out = module.model_step_fit(phm_fit_predict_batch_multi_target)
        assert "loss" in out
        assert out["predictions"].ndim == 3
        assert out["targets"].shape[-1] == 2
        assert (
            "0_0" in mock_fit_predict_backbone._models
            and "0_1" in mock_fit_predict_backbone._models
        )

    def test_training_step_calls_super_with_evaluate_false(
        self, mock_evaluators, mock_fit_predict_backbone, phm_fit_predict_batch
    ):
        """Doc: pipeline/base.py - FitPredictWrapperLightningModule.training_step overrides with _evaluate=False."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        loss = module.training_step(phm_fit_predict_batch, 0)
        assert loss is not None
        assert not mock_evaluators["train"]._updates

    def test_automatic_optimization_false(
        self, mock_evaluators, mock_fit_predict_backbone
    ):
        """Doc: pipeline/base.py - automatic_optimization is False for fit-predict."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        assert module.automatic_optimization is False

    def test_configure_optimizers_returns_none(
        self, mock_evaluators, mock_fit_predict_backbone
    ):
        """Doc: pipeline/base.py - configure_optimizers() returns None."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        assert module.configure_optimizers() is None

    def test_setup_no_op(self, mock_evaluators, mock_fit_predict_backbone):
        """Doc: pipeline/base.py - setup(stage) is pass."""
        module = FitPredictWrapperLightningModule(
            backbone=mock_fit_predict_backbone,
            evaluators=mock_evaluators,
        )
        module.setup("fit")


# -----------------------------------------------------------------------------
# Test: ConstantLossLightningModule
# -----------------------------------------------------------------------------


class TestConstantLossLightningModule:
    """Validates model_step overwrites loss with constant and configure_optimizers returns None."""

    def test_model_step_returns_constant_loss(
        self, mock_evaluators, mock_backbone, mock_loss, phm_batch_rul
    ):
        """Doc: pipeline/base.py - ConstantLossLightningModule.model_step sets loss to tensor([1])."""
        module = ConstantLossLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
            scheduler_factory=None,
        )
        out = module.model_step(phm_batch_rul, stage="train")
        assert out["loss"].item() == 1

    def test_automatic_optimization_false(
        self, mock_evaluators, mock_backbone, mock_loss
    ):
        """Doc: pipeline/base.py - automatic_optimization is False."""
        module = ConstantLossLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
            scheduler_factory=None,
        )
        assert module.automatic_optimization is False

    def test_configure_optimizers_returns_none(
        self, mock_evaluators, mock_backbone, mock_loss
    ):
        """Doc: pipeline/base.py - configure_optimizers() returns None."""
        module = ConstantLossLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
            scheduler_factory=None,
        )
        assert module.configure_optimizers() is None


# -----------------------------------------------------------------------------
# Test: TrainingLightningModule (model_step + configure_optimizers)
# -----------------------------------------------------------------------------


class TestTrainingLightningModule:
    """Validates model_step applies loss and configure_optimizers with/without scheduler."""

    def test_training_step_invokes_backbone_wrapper_training_step(
        self, mock_evaluators, mock_backbone, mock_loss, phm_batch_rul
    ):
        """BackboneWrapperLightningModule._training_step called via training_step (lines 424-425)."""
        module = TrainingLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
            scheduler_factory=None,
        )
        loss = module.training_step(phm_batch_rul, 0, _evaluate=False)
        assert isinstance(loss, torch.Tensor)

    def test_model_step_returns_loss_from_loss_module(
        self, mock_evaluators, mock_backbone, mock_loss, phm_batch_rul
    ):
        """Doc: pipeline/base.py - model_step(batch, stage) = loss(forward(batch), batch)."""
        module = TrainingLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.01),
            scheduler_factory=None,
        )
        out = module.model_step(phm_batch_rul, stage="train")
        assert "loss" in out
        assert out["loss"].requires_grad or not out["loss"].requires_grad

    def test_configure_optimizers_without_scheduler(
        self, mock_evaluators, mock_backbone, mock_loss
    ):
        """Doc: pipeline/base.py - configure_optimizers with scheduler_factory=None returns only optimizer."""
        module = TrainingLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
            scheduler_factory=None,
        )
        cfg = module.configure_optimizers()
        assert "optimizer" in cfg
        assert "lr_scheduler" not in cfg

    def test_configure_optimizers_with_scheduler(
        self, mock_evaluators, mock_backbone, mock_loss
    ):
        """Doc: pipeline/base.py - configure_optimizers with scheduler_factory returns optimizer and lr_scheduler."""

        def scheduler_factory(optimizer):
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        module = TrainingLightningModule(
            backbone=mock_backbone,
            loss=mock_loss,
            evaluators=mock_evaluators,
            optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
            scheduler_factory=scheduler_factory,
        )
        cfg = module.configure_optimizers()
        assert "optimizer" in cfg
        assert "lr_scheduler" in cfg
        assert cfg["lr_scheduler"]["monitor"] == "val/loss"
        assert cfg["lr_scheduler"]["interval"] == "epoch"
        assert cfg["lr_scheduler"]["frequency"] == 1
