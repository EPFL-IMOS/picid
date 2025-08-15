from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from lightning import LightningModule

from lmetk.evaluator.base import AbstractEvaluator
from lmetk.model.wrappers.base import FitPredictWrapper


class CustomEvaluatorInterface(ABC):
    """Interface for custom evaluators."""

    @abstractmethod
    def _training_step(self, batch, batch_idx: int) -> dict[str, torch.Tensor]:
        """Perform a single training step on a batch of data.

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
        """Perform a single validation step on a batch of data.

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
        """Perform a single test step on a batch of data.

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

        # Make sure effective prediction lenght of the model is greater of equal to evaluator effective prediction lenght
        # Determine the effective prediction length, defaulting to 0 if not set
        for name, evaluator in evaluators.items():
            self.effective_pred_len = getattr(evaluator, "effective_pred_len", None)
            if self.effective_pred_len:
                assert self.backbone.kwargs.pred_len >= self.effective_pred_len, (
                    f"Not possible to have self.evaluator.effective_pred_len: "
                    + f"{self.effective_pred_len} greater than model pred_len: {self.backbone.kwargs.pred_len}"
                )

    def log_epoch_metrics(self, mode=None):
        """Log metrics.

        Parameters
        ----------
        mode : str, optional
            The mode of the model, either "train", "val", or "test" (default: None).
        """
        metrics_dict = self.evaluators[mode].compute()

        for key in metrics_dict:
            self.log(
                f"{mode}/{key}",
                metrics_dict[key],
                prog_bar=True,
                on_step=False,
            )

        self.evaluators[mode].reset()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data.

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
        model_outs = self._training_step(batch, batch_idx)
        self.evaluators["train"].update(model_outs)
        self.log(
            "train/loss",
            model_outs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

        # Return loss for backpropagation step
        return model_outs["loss"]

    def validation_step(self, batch, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.
        """
        model_outs = self._validation_step(batch, batch_idx)
        self.evaluators["val"].update(model_outs)
        self.log(
            "val/loss",
            model_outs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

    def test_step(self, batch, batch_idx: int) -> None:
        """Perform a single test step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.
        """
        model_outs = self._test_step(batch, batch_idx)
        self.evaluators["test"].update(model_outs)
        self.log(
            "test/loss",
            model_outs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

    def on_validation_epoch_start(self) -> None:
        """Hook called when a validation epoch begins.

        According pytorch lightning documentation this hook is called at the
        beginning of the validation epoch.

        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        Note that the validation step is within the train epoch. Hence here we have to log the train metrics
        before we reset the evaluator to start the validation loop.
        """
        # Log train metrics and reset evaluator
        self.evaluators["val"].reset()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a train epoch begins.

        This hook is used to reset the train metrics.
        """
        self.evaluators["train"].reset()

    def on_test_epoch_start(self) -> None:
        """Lightning hook that is called when a test epoch begins.

        This hook is used to reset the test metrics.
        """
        self.evaluators["test"].reset()

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a train epoch ends.

        This hook is used to log the train metrics.
        """
        # Log train metrics and reset evaluator
        self.log_epoch_metrics(mode="train")

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends.

        This hook is used to log the validation metrics.
        """
        # Log validation metrics and reset evaluator
        self.log_epoch_metrics(mode="val")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends.

        This hook is used to log the test metrics.
        """
        self.log_epoch_metrics(mode="test")

    def process_outputs(self, model_out: dict, batch) -> dict:
        """Handle model outputs.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        model_out["predictions"] = model_out["predictions"].detach().cpu().numpy()
        model_out["targets"] = model_out["targets"].detach().cpu().numpy()
        return model_out


class BackboneWrapperLightningModule(CustomEvaluatorLightningModule):
    """A `LightningModule` to define a network.

    Parameters
    ----------
    backbone : torch.nn.Module
        The backbone model to train.
    backbone_wrapper : torch.nn.Module
        The backbone wrapper class.
    readout : torch.nn.Module
        The readout class.
    loss : torch.nn.Module
        The loss class.
    feature_encoder : torch.nn.Module, optional
        The feature encoder (default: None).
    evaluator : Any, optional
        The evaluator class (default: None).
    optimizer : Any, optional
        The optimizer class (default: None).
    **kwargs : Any
        Additional keyword arguments.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        # backbone_wrapper: torch.nn.Module,
        loss: torch.nn.Module,
        evaluators: dict[str, AbstractEvaluator],
        optimizer: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(evaluators=evaluators)

        # This line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["backbone", "readout", "feature_encoder"]
        )

        self.backbone = backbone
        self.optimizer = optimizer
        self.loss = loss

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backbone={self.backbone}, loss={self.loss}"

    def forward(self, batch) -> dict:
        """Perform a forward pass through the model `self.backbone`.

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
        """Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """

        # Allow batch object to know the phase of the training
        batch["model_state"] = stage
        model_out = self.forward(batch)
        model_out = self.process_outputs(model_out=model_out, batch=batch)
        return model_out

    def _training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data.

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
        """Perform a single validation step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.
        """
        model_out = self.model_step(batch, stage="val")
        return model_out

    def _test_step(self, batch, batch_idx: int) -> None:
        """Perform a single test step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.
        """
        model_out = self.model_step(batch, stage="test")
        return model_out

    def setup(self, stage: str) -> None:
        """Hook to call torch.compile.

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
    """A `LightningModule` to evaluate fit-predict (sklearn style models).

    Parameters
    ----------
    backbone : torch.nn.Module
        The backbone model to train.
    backbone_wrapper : torch.nn.Module
        The backbone wrapper class.
    readout : torch.nn.Module
        The readout class.
    loss : torch.nn.Module
        The loss class.
    feature_encoder : torch.nn.Module, optional
        The feature encoder (default: None).
    evaluator : Any, optional
        The evaluator class (default: None).
    optimizer : Any, optional
        The optimizer class (default: None).
    **kwargs : Any
        Additional keyword arguments.
    """

    def __init__(
        self,
        backbone: FitPredictWrapper,
        evaluators: dict[str, AbstractEvaluator],
        **kwargs,
    ) -> None:
        super().__init__(evaluators=evaluators)

        # This line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["backbone"])
        self.backbone = backbone

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backbone={self.backbone})"

    def model_step_fit(self, batch) -> dict:
        """Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """

        #

        X, y = self.backbone.batch_to_sklearn(batch)
        self.backbone.fit(X, y)
        model_out = {}
        model_out["loss"] = torch.tensor([1])
        return model_out

    def model_step_predict(self, batch) -> dict:
        """Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """
        model_out = {}
        model_out["loss"] = torch.tensor([1])

        # Update evaluator with current batch
        n_sequence, n_features = (
            batch["target"]["y"].shape[1],
            batch["target"]["y"].shape[2],
        )
        model_out = {
            "predictions": rearrange(
                torch.Tensor(outputs),
                inverse_pattern,
                sequence=n_sequence,
                features=n_features,
            ),
            "targets": batch["target"]["y"],
        }

        return model_out

    def _training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data.

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
        """Perform a single validation step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.
        """
        model_out = self.model_step_predict(batch)
        return model_out

    def _test_step(self, batch, batch_idx: int) -> None:
        """Perform a single test step on a batch of data.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.
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


class CustomTimePFNLightningModule(CustomEvaluatorLightningModule):

    def process_outputs(self, model_out: dict, batch) -> dict:
        """Handle model outputs.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        # TODO: (Raffael)  this is a hack to make sure the model outputs are in the right format
        # Determine the effective prediction length, defaulting to 0 if not set
        effective_pred_len = getattr(self.evaluator, "effective_pred_len", None)
        model_out["predictions"] = (
            model_out["predictions"][:, :effective_pred_len, :].detach().cpu().numpy()
        )
        model_out["targets"] = (
            model_out["targets"][:, :effective_pred_len, :].detach().cpu().numpy()
        )
        return model_out

    def __init__(self, backbone, loss, evaluators, optimizer=None, **kwargs):
        super().__init__(backbone, loss, evaluators, optimizer, **kwargs)


class ConstantLossLightningModule(BackboneWrapperLightningModule):

    def __init__(self, backbone, loss, evaluators, optimizer=None, **kwargs):
        super().__init__(backbone, loss, evaluators, optimizer, **kwargs)

    def model_step(self, batch, stage: str) -> dict:
        model_out = super().model_step(batch, stage)
        # TODO: (Raffael)  what is going on here?
        model_out["loss"] = torch.tensor([1])
        return model_out

    @property
    def automatic_optimization(self):
        return False

    def configure_optimizers(self):
        return None
