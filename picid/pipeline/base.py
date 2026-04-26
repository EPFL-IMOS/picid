from abc import ABC, abstractmethod
from typing import Any, Callable, Union, Dict, override

import numpy as np
import torch
from lightning import LightningModule

from picid.evaluator.base import AbstractEvaluator
from picid.model.adapters.base import AbstractFitPredictWrapper

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomEvaluatorInterface(ABC):
    """Interface for custom evaluators."""

    @abstractmethod
    def _training_step(self, batch, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Perform a single training step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing model predictions and auxiliary outputs.
        """
        pass

    @abstractmethod
    def _validation_step(self, batch, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Perform a single validation step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing model predictions and auxiliary outputs.
        """
        pass

    @abstractmethod
    def _test_step(self, batch, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Perform a single test step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing model predictions and auxiliary outputs.
        """
        pass


class CustomEvaluatorLightningModule(LightningModule, CustomEvaluatorInterface):
    def __init__(self, evaluators: dict[str, AbstractEvaluator]) -> None:
        super().__init__()

        self.evaluators = evaluators
        self.train_metrics_logged = False

        # TODO: This should be moved from here to somewhere else
        for name, evaluator in evaluators.items():
            self.effective_pred_len = getattr(evaluator, "effective_pred_len", None)

    def log_epoch_metrics(self, mode=None, step=None, epoch=None):
        """
        Log metrics for the requested evaluation split.

        Parameters
        ----------
        mode : str, optional
            The mode of the model, either "train", "val", or "test".
        step : int, optional
            Global step recorded for the log message.
        epoch : int, optional
            Current epoch recorded for the evaluator compute call.
        """
        metrics_dict = self.evaluators[mode].compute(mode=mode, epoch=epoch, step=step)

        for key in metrics_dict:
            self.log(
                f"{mode}/{key}",
                metrics_dict[key],
                prog_bar=True,
                on_step=False,
                # Needs to be true to update the callbacks (such as early stopping)
                on_epoch=True,
            )

        self.evaluators[mode].reset()

    def training_step(
        self, batch, batch_idx: int, _evaluate: bool = True
    ) -> torch.Tensor:
        """
        Perform a single training step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.
        _evaluate : bool, default=True
            Whether to update the evaluator during this step.

        Returns
        -------
        torch.Tensor
            A tensor of losses between model predictions and targets.
        """
        model_outs = self._training_step(batch, batch_idx)
        model_outs = self.process_outputs(model_out=model_outs, batch=batch)
        if _evaluate:
            self.evaluators["train"].update(model_outs)

        self.log(
            "train/loss",
            model_outs["loss"].detach().cpu(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=model_outs["targets"].shape[0],
        )

        # Return loss for backpropagation step
        return model_outs["loss"]

    def validation_step(self, batch, batch_idx: int) -> None:
        """
        Perform a single validation step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.
        """
        model_outs = self._validation_step(batch, batch_idx)
        model_outs = self.process_outputs(model_out=model_outs, batch=batch)
        self.evaluators["val"].update(model_outs)
        self.log(
            "val/loss",
            model_outs["loss"].detach().cpu(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=model_outs["targets"].shape[0],
        )

    def test_step(self, batch, batch_idx: int) -> None:
        """
        Perform a single test step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.
        """
        model_outs = self._test_step(batch, batch_idx)
        model_outs = self.process_outputs(model_out=model_outs, batch=batch)
        self.evaluators["test"].update(model_outs)
        self.log(
            "test/loss",
            model_outs["loss"].detach().cpu(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=model_outs["targets"].shape[0],
        )

    def on_validation_epoch_start(self) -> None:
        """
        Reset validation metrics at the start of a validation epoch.

        According to PyTorch Lightning, this hook is called at the beginning
        of the validation epoch.

        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        Note that the validation step is within the train epoch. Hence here we
        reset the validation evaluator before the validation loop starts.
        """
        # Log train metrics and reset evaluator
        self.evaluators["val"].reset()

    def on_train_epoch_start(self) -> None:
        """
        Reset train metrics at the start of a train epoch.

        This hook is used to reset the train metrics.
        """
        self.evaluators["train"].reset()

    def on_test_epoch_start(self) -> None:
        """
        Reset test metrics at the start of a test epoch.

        This hook is used to reset the test metrics.
        """
        self.evaluators["test"].reset()

    def on_train_epoch_end(self) -> None:
        """Called when a train epoch ends."""
        # Don't log train metrics due to compute error with no predictions

    def on_validation_epoch_end(self) -> None:
        """
        Log validation metrics at the end of a validation epoch.

        This hook is used to log the validation metrics.
        """
        # Log validation metrics and reset evaluator
        self.log_epoch_metrics(
            mode="val", step=self.global_step, epoch=self.current_epoch
        )

    def on_test_epoch_end(self) -> None:
        """
        Log test metrics at the end of a test epoch.

        This hook is used to log the test metrics.
        """
        self.log_epoch_metrics(
            mode="test", step=self.global_step, epoch=self.current_epoch
        )

    def _to_numpy(self, data: Union[torch.Tensor, np.ndarray], name: str) -> np.ndarray:
        """
        Convert a PyTorch tensor or NumPy array to a NumPy array.

        This helper function centralizes the conversion logic, detaching a tensor
        from the computation graph and moving it to the CPU.

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Input value to convert.
        name : str
            Human-readable name used in the error message.

        Returns
        -------
        np.ndarray
            A NumPy view or copy of the input data.
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, list):
            # Handle scalar values by converting them to a NumPy array
            return np.array(data)
        # Raise a more informative error
        raise TypeError(
            f"'{name}' must be a torch.Tensor or np.ndarray, but got {type(data).__name__}"
        )

    def process_outputs(
        self, model_out: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle model outputs by converting specified tensors to NumPy arrays.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output, including 'predictions' and 'targets'.
        batch : dict
            The input batch, containing 'batch_idx' and optionally 'unit_id'.

        Returns
        -------
        dict
            The updated model_out dictionary with values converted to NumPy arrays.
        """

        # if task=='rul':
        #     # We expect here the result always 2d for the regression task

        # Process required keys from model_out and batch
        model_out["predictions"] = self._to_numpy(
            model_out["predictions"], "predictions"
        )
        model_out["targets"] = self._to_numpy(model_out["targets"], "targets")

        if "batch_idx" in batch and batch["batch_idx"] is not None:
            model_out["batch_idx"] = self._to_numpy(batch["batch_idx"], "batch_idx")

        # Process optional 'unit_id' from the batch
        if "unit_id" in batch and batch["unit_id"] is not None:
            model_out["unit_id"] = self._to_numpy(batch["unit_id"], "unit_id")

        # Extend here if you want to pass smth from the batch to the evaluator

        return model_out


class BackboneWrapperLightningModule(CustomEvaluatorLightningModule):
    """
    Lightning module for standard backbone training.

    Parameters
    ----------
    backbone : torch.nn.Module
        The backbone model to train.
    loss : torch.nn.Module
        Loss module used during training.
    evaluators : dict[str, AbstractEvaluator]
        Mapping from split name to evaluator instance.
    optimizer_factory : Callable[..., torch.optim.Optimizer]
        Factory used to create the optimizer.
    scheduler_factory : Callable[..., torch.optim.lr_scheduler._LRScheduler]
        Factory used to create the scheduler.
    **kwargs
        Additional keyword arguments forwarded to the parent class.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        loss: torch.nn.Module,
        evaluators: dict[str, AbstractEvaluator],
        optimizer_factory: Callable[..., torch.optim.Optimizer],
        scheduler_factory: Callable[..., torch.optim.lr_scheduler._LRScheduler],
        **kwargs,
    ) -> None:
        super().__init__(evaluators=evaluators)

        # This line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["backbone", "readout", "feature_encoder"]
        )

        self.backbone = backbone
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.loss = loss

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backbone={self.backbone}, loss={self.loss}"

    def forward(self, batch) -> dict:
        """
        Perform a forward pass through the model ``self.backbone``.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output.
        """
        return self.backbone(batch)

    def model_step(self, batch, stage: str) -> dict:
        """
        Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        stage : str
            Current training stage, such as ``"train"`` or ``"val"``.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """

        # Allow batch object to know the phase of the training
        batch["model_state"] = stage
        model_out = self.forward(batch)
        return model_out

    def _training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            A tensor of losses between model predictions and targets.
        """
        model_out = self.model_step(batch, stage="train")
        return model_out

    def _validation_step(self, batch, batch_idx: int) -> None:
        """
        Perform a single validation step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """
        model_out = self.model_step(batch, stage="val")
        return model_out

    def _test_step(self, batch, batch_idx: int) -> None:
        """
        Perform a single test step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """
        model_out = self.model_step(batch, stage="test")
        return model_out

    def setup(self, stage: str) -> None:
        """
        Hook to call ``torch.compile``.

        Lightning hook that is called at the beginning of fit (train +
        validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust
        something about them. This hook is called on every process when using
        DDP.

        Parameters
        ----------
        stage : str
            Either "fit", "validate", "test", or "predict".
        """
        pass
        # if self.hparams.compile and stage == "fit":
        #     self.net = torch.compile(self.net)


class FitPredictWrapperLightningModule(CustomEvaluatorLightningModule):
    """
    Lightning module wrapper for fit-predict style backbones.

    Parameters
    ----------
    backbone : AbstractFitPredictWrapper
        The wrapped fit-predict model.
    evaluators : dict[str, AbstractEvaluator]
        Mapping from split name to evaluator instance.
    predict_after_training : bool, default=False
        Whether to run prediction immediately after fitting.
    debug : bool, default=False
        Enable verbose debug logging.
    **kwargs
        Additional keyword arguments forwarded to the parent class.
    """

    def __init__(
        self,
        backbone: AbstractFitPredictWrapper,
        evaluators: dict[str, AbstractEvaluator],
        predict_after_training: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(evaluators=evaluators)
        self.predict_after_training = predict_after_training
        self.debug = debug
        if self.debug:
            logger.warning("---------------------------------------------------")
            logger.warning("Debug mode is ON. Using only small subsets of data.")
            logger.warning("---------------------------------------------------")

        # This line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["backbone"])
        self.backbone = backbone

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backbone={self.backbone})"

    def __batch_to_fit_predict(self, batch):
        X = batch["context"]
        y = batch["target"]
        assert (
            X.shape[0] == 1
        ), f"Expected batch size of 1 because we can only process one task per iteration, got {X.shape[0]}"
        assert (
            y.shape[0] == 1
        ), f"Expected batch size of 1 because we can only process one task per iteration, got {y.shape[0]}"
        X = X.squeeze(0)
        y = y.squeeze(0)

        assert X.ndim == 2, f"X must be 2-dimensional, got {X.ndim} dimensions"
        assert y.ndim == 2, f"y must be 2-dimensional, got {y.ndim} dimensions"

        return X, y

    def get_task_info(self, batch) -> str:
        task_idx = int(batch["task_idx"].item())
        task_num = int(batch.get("task_num", -1))
        task_desc = batch.get("task_desc", f"Task {task_idx + 1} of {task_num}")
        return f"{task_desc}"

    def model_step_fit(self, batch) -> dict:
        """
        Perform a single model step that fits the wrapped model.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """

        X, y = self.__batch_to_fit_predict(batch)

        logger.info(f"Fitting: {self.get_task_info(batch)}")
        logger.info(
            f"Fitting model {self.backbone} on data with shapes X: {X.shape}, y: {y.shape}"
        )

        allows_multi_target = self.backbone.allows_multi_target
        n_targets = y.shape[1]

        if not allows_multi_target and n_targets > 1:
            # Reshape y to be 1D if the model does not support multi-target
            # such that we create a "virtual task" for each target dimension.

            logger.warning(
                f"Model {self.backbone} does not support multi-target, "
                "but y has {n_targets} target dimensions. "
                f"Reshaping y to be 1D and creating a "
                "virtual task for each target dimension."
            )
            for target_idx in range(n_targets):
                logger.info(f"Fitting target dimension {target_idx + 1} of {n_targets}")
                self.backbone.fit(X, y[:, target_idx].unsqueeze(1))
                model_id = f"{batch["task_idx"].item()}_{target_idx}"
                self.backbone.serialize_model(model_id)
        else:
            self.backbone.fit(X, y)
            model_id = str(batch["task_idx"].item())
            self.backbone.serialize_model(model_id)

        if self.predict_after_training:
            return self.model_step_predict(batch)
        else:
            # Add a dummy loss
            model_out = {}
            model_out["loss"] = torch.tensor([1])
            # Add task dimension back to predictions and targets
            model_out["predictions"] = y.unsqueeze(1)
            model_out["targets"] = y.unsqueeze(1)
            return model_out

    def model_step_predict(self, batch) -> dict:
        """
        Perform a single model step that loads and predicts with the wrapped model.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """

        X, y = self.__batch_to_fit_predict(batch)

        allows_multi_target = self.backbone.allows_multi_target
        n_targets = y.shape[1]

        if not allows_multi_target and n_targets > 1:
            # Reshape y to be 1D if the model does not support multi-target
            # such that we create a "virtual task" for each target dimension.
            for target_idx in range(n_targets):
                logger.info(
                    f"Predicting target dimension {target_idx + 1} of {n_targets}"
                )
                self.backbone.load_model(f"{batch["task_idx"].item()}_{target_idx}")
                outputs = self.backbone.predict(X)
        else:
            self.backbone.load_model(int(batch["task_idx"].item()))
        outputs = self.backbone.predict(X)

        assert isinstance(
            outputs, torch.Tensor
        ), f"outputs must be a torch.Tensor, got {type(outputs)}"

        assert (
            outputs.ndim == 2
        ), f"outputs must be 2-dimensional, got {outputs.ndim} dimensions"

        model_out = {
            # Add task dimension back to predictions and targets
            "predictions": outputs.unsqueeze(1),
            "targets": y.unsqueeze(1),
        }

        # Add a dummy loss
        model_out["loss"] = torch.tensor([1])
        return model_out

    @override
    def training_step(self, batch, batch_idx, _evaluate: bool = True):
        # Don't evaluate during training step because we only record dummy values.
        # This is to make fit / predict work with lightning.
        return super().training_step(batch, batch_idx, _evaluate=False)

    def _training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            A tensor of losses between model predictions and targets.
        """
        model_out = self.model_step_fit(batch)
        return model_out

    def _validation_step(self, batch, batch_idx: int) -> None:
        """
        Perform a single validation step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """
        model_out = self.model_step_predict(batch)
        return model_out

    def _test_step(self, batch, batch_idx: int) -> None:
        """
        Perform a single test step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """
        model_out = self.model_step_predict(batch)
        return model_out

    def setup(self, stage: str) -> None:
        pass

    @property
    def automatic_optimization(self):
        return False

    def configure_optimizers(self):
        return None


class ConstantLossLightningModule(BackboneWrapperLightningModule):
    def __init__(
        self,
        backbone,
        loss,
        evaluators,
        optimizer_factory: Callable[..., torch.optim.Optimizer] = None,
        scheduler_factory: Callable[..., torch.optim.lr_scheduler._LRScheduler] = None,
        **kwargs,
    ):
        super().__init__(
            backbone,
            loss,
            evaluators,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            **kwargs,
        )

    def model_step(self, batch, stage: str) -> dict:
        model_out = super().model_step(batch, stage)
        # This is a dummy loss to make the training work with lightning.
        model_out["loss"] = torch.tensor([1])
        return model_out

    @property
    def automatic_optimization(self):
        return False

    def configure_optimizers(self):
        return None


class TrainingLightningModule(BackboneWrapperLightningModule):
    def __init__(
        self,
        backbone,
        loss,
        evaluators,
        optimizer_factory: Callable[..., torch.optim.Optimizer],
        scheduler_factory: Callable[..., torch.optim.lr_scheduler._LRScheduler],
        **kwargs,
    ):
        super().__init__(
            backbone, loss, evaluators, optimizer_factory, scheduler_factory, **kwargs
        )

    # TODO: I should not rewrite the whole model_step
    def model_step(self, batch, stage: str) -> dict:
        """
        Perform a single model step on a batch of data and compute the loss.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        stage : str
            Current training stage.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """

        # Allow batch object to know the phase of the training
        batch["model_state"] = stage
        model_out = self.forward(batch)
        model_out = self.loss(model_out=model_out, batch=batch)
        return model_out

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizers and learning-rate schedulers.

        Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or
        similar you might have multiple.

        Returns
        -------
        dict
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.

        Examples
        --------
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer_factory(list(self.backbone.parameters()))

        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        else:
            return {"optimizer": optimizer}
