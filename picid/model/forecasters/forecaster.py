from abc import abstractmethod
from typing import Callable, Tuple

from einops import rearrange
import torch

from picid.evaluator.base import AbstractEvaluator
from picid.model.definitions import (
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
)
from picid.model.forecasters.linear_model.linear_ar import LinearModel
from picid.model.utils.revin import RevIN, SeriesDecomposition
from picid.pipeline.base import CustomEvaluatorLightningModule


class Forecaster(CustomEvaluatorLightningModule):
    def __init__(
        self,
        optimizer_factory: Callable[..., torch.optim.Optimizer],
        scheduler_factory: Callable[..., torch.optim.lr_scheduler._LRScheduler],
        d_x: int,
        d_yc: int,
        d_yt: int,
        task_type: str,
        loss: str = "mse",
        linear_window: int = 0,
        linear_shared_weights: bool = False,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        evaluators: dict[str, AbstractEvaluator] = {},
        # catch all parameters
        verbose: int = True,
        metadata: dict = {},
    ):
        super().__init__(evaluators=evaluators)
        self._inv_scaler = lambda x: x
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.time_masked_idx = None
        self.null_value = None
        self.loss = loss
        self.task_type = task_type
        self.classification_criterion = torch.nn.CrossEntropyLoss(reduction="none")

        # Define supported tasks based on your new categories
        self.supported_types = ALL_TASKS

        if task_type not in self.supported_types:
            raise ValueError(f"Task {task_type} not supported for Forecaster.")
        assert task_type in ALL_TASKS, f"Unrecognized task type: {task_type}"

        if linear_window:
            self.linear_model = LinearModel(
                linear_window, shared_weights=linear_shared_weights, d_yt=d_yt
            )
        else:
            self.linear_model = lambda x, *args, **kwargs: 0.0

        self.use_revin = use_revin
        if use_revin:
            assert d_yc == d_yt, "TODO: figure out exo case for revin"
            self.revin = RevIN(num_features=d_yc)
        else:
            self.revin = lambda x, *args, **kwargs: x

        self.use_seasonal_decomp = use_seasonal_decomp
        if use_seasonal_decomp:
            self.seasonal_decomp = SeriesDecomposition(kernel_size=25)
        else:
            self.seasonal_decomp = lambda x: (x, x.clone())

        self.d_x = d_x
        self.d_yc = d_yc
        self.d_yt = d_yt
        self.save_hyperparameters()

    def set_null_value(self, val: float) -> None:
        self.null_value = val

    def set_inv_scaler(self, scaler) -> None:
        self._inv_scaler = scaler

    def set_scaler(self, scaler) -> None:
        self._scaler = scaler

    @property
    @abstractmethod
    def train_step_forward_kwargs(self):
        # can just return an empty dict if no kwargs are needed
        pass

    @property
    @abstractmethod
    def eval_step_forward_kwargs(self):
        # can just return an empty dict if no kwargs are needed
        pass

    def loss_fn(
        self, true: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        true = torch.nan_to_num(true)

        if self.loss == "mse":
            loss = (mask * (true - preds)).square().sum() / max(mask.sum(), 1)
        elif self.loss == "mae":
            loss = torch.abs(mask * (true - preds)).sum() / max(mask.sum(), 1)
        elif self.loss == "smape":
            num = 2.0 * abs(preds - true)
            den = abs(preds.detach()) + abs(true) + 1e-5
            loss = 100.0 * (mask * (num / den)).sum() / max(mask.sum(), 1)
        else:
            raise ValueError(f"Unrecognized Loss Function : {self.loss}")
        return loss

    def forecasting_loss(
        self, outputs: torch.Tensor, y_t: torch.Tensor, time_mask: int, feat_mask=None
    ) -> Tuple[torch.Tensor]:
        if self.null_value is not None:
            null_mask_mat = y_t != self.null_value
        else:
            null_mask_mat = torch.ones_like(y_t)

        # genuine NaN failsafe
        null_mask_mat *= ~torch.isnan(y_t)

        time_mask_mat = torch.ones_like(y_t)
        if time_mask is not None:
            time_mask_mat[:, time_mask:] = False

        feat_mask_mat = torch.ones_like(y_t)
        if feat_mask is not None:
            feat_mask_mat = torch.zeros_like(y_t)
            feat_mask_mat[:, :, feat_mask] = True

        full_mask = time_mask_mat * null_mask_mat * feat_mask_mat
        forecasting_loss = self.loss_fn(y_t, outputs, full_mask)
        return forecasting_loss, full_mask

    def classification_loss(
        self, outputs: torch.Tensor, y_t: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked cross-entropy for batched sequence classification.

        Parameters
        ----------
        outputs : torch.Tensor
            ``(B, T, C)`` unbounded logits.
        y_t : torch.Tensor
            ``(B, T)`` integer class labels.
        mask : torch.Tensor or None
            ``(B, T)`` binary mask with ``1`` for valid positions.

        Returns
        -------
        torch.Tensor
            Scalar masked mean loss.
        """

        b, t, c = outputs.shape
        # flatten
        outputs_flat = outputs.reshape(-1, c)
        y_flat = y_t.reshape(-1).long()
        if mask is None:
            mask_flat = torch.ones(outputs_flat.shape[0], device=outputs_flat.device)
        else:
            mask_flat = mask.reshape(-1).float()

        loss = self.classification_criterion(outputs_flat, y_flat)
        # Avoid division-by-zero when every position is masked out.
        masked_loss = (loss * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)
        return masked_loss

    def compute_loss(
        self,
        batch: Tuple[torch.Tensor],
        time_mask: int = None,
        forward_kwargs: dict = {},
    ) -> Tuple[torch.Tensor]:
        x_c, y_c, x_t, y_t = batch

        outputs, *_ = self(x_c, y_c, x_t, y_t, **forward_kwargs)

        if self.task_type in CLASSIFICATION_TASKS:
            assert time_mask is None, "time masking not supported for classification"
            loss = self.classification_loss(outputs=outputs, y_t=y_t, mask=None)
            return loss, outputs, None
        else:
            loss, mask = self.forecasting_loss(
                outputs=outputs, y_t=y_t, time_mask=time_mask
            )
            return loss, outputs, mask

    def predict(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        sample_preds: bool = False,
    ) -> torch.Tensor:
        og_device = y_c.device
        # move to model device
        x_c = x_c.to(self.device).float()
        x_t = x_t.to(self.device).float()
        # move y_c to cpu if it isn't already there, scale, and then move back to the model device
        y_c = torch.from_numpy(self._scaler(y_c.cpu().numpy())).to(self.device).float()

        # create dummy y_t of zeros
        y_t = (
            torch.zeros((x_t.shape[0], x_t.shape[1], self.d_yt)).to(self.device).float()
        )

        with torch.no_grad():
            # gradient-free prediction
            normalized_preds, *_ = self.forward(
                x_c, y_c, x_t, y_t, **self.eval_step_forward_kwargs
            )

        # preds --> cpu --> inverse scale to original units --> original device of y_c
        preds = (
            torch.from_numpy(self._inv_scaler(normalized_preds.cpu().numpy()))
            .to(og_device)
            .float()
        )
        return preds

    @abstractmethod
    def forward_model_pass(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        **forward_kwargs,
    ) -> Tuple[torch.Tensor]:
        return NotImplemented

    def nan_to_num(self, *inps):
        return tuple(torch.nan_to_num(i) for i in inps)

    def forward(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        **forward_kwargs,
    ) -> Tuple[torch.Tensor]:
        if y_t is not None:
            (y_t,) = self.nan_to_num(y_t)

        if x_t is not None:
            (x_t,) = self.nan_to_num(x_t)

        if x_c is not None:
            (x_c,) = self.nan_to_num(x_c)

        if y_c is not None:
            (y_c,) = self.nan_to_num(y_c)

        if y_c is not None:
            _, pred_len, d_yt = y_t.shape
            # does nothing if use_revin = False
            y_c = self.revin(y_c, mode="norm")
            # both are the original if use_seasonal_decomp = False
            seasonal_yc, trend_yc = self.seasonal_decomp(y_c)
        else:
            seasonal_yc, trend_yc = None, 0

        preds, *extra = self.forward_model_pass(
            x_c, seasonal_yc, x_t, y_t, **forward_kwargs
        )

        if self.task_type not in CLASSIFICATION_TASKS:
            assert preds.shape == y_t.shape, f"{preds.shape} != {y_t.shape}"
        else:
            assert preds.shape == (
                y_t.shape[0],
                y_t.shape[1],
                self.d_yt,
            ), f"{preds.shape} != {(y_t.shape[0], y_t.shape[1], self.d_yt)}"

        if y_c is not None:
            baseline = self.linear_model(trend_yc, pred_len=pred_len, d_yt=d_yt)
            output = self.revin(preds + baseline, mode="denorm")
        else:
            output = preds

        if extra:
            return (output,) + tuple(extra)
        return (output,)

    def step(
        self, batch: Tuple[torch.Tensor], train: bool = False
    ) -> dict[str, torch.Tensor]:
        kwargs = (
            self.train_step_forward_kwargs if train else self.eval_step_forward_kwargs
        )
        time_mask = self.time_masked_idx if train else None

        loss, outputs, mask = self.compute_loss(
            batch=batch,
            time_mask=time_mask,
            forward_kwargs=kwargs,
        )
        *_, y_t = batch

        # Here we squeeze 2 because this class handles multiple tasks
        # and some tasks produce just one feature,
        # which should go into the 2D evaluator,
        return {
            "loss": loss.mean(),
            # This is needed because evaluator works over the numpy arrays
            "predictions": outputs,
            "mask": mask,
            "targets": y_t,
        }

    def _map_batch_to_task(self, batch):
        # x_c, y_c, x_t, y_t,
        if self.task_type == "forecasting":
            batch = (
                batch["context"]["features_seq_x"],
                batch["target"]["target_seq_x"],
                batch["context"]["features_seq_y"],
                batch["target"]["target_seq_y"],
            )

        elif self.task_type == "state_forecasting":
            batch = (
                None,
                batch["features_seq_x"],  # y_c
                None,
                batch["features_seq_y"],  # y_t
            )

        elif self.task_type in REGRESSION_TASKS or (
            self.task_type in CLASSIFICATION_TASKS
        ):
            f = batch["features"]
            assert f.dim() == 3, "features must be 3D (B, T, D)"
            t = batch[self.task_type]
            if t.dim() == 2:
                t = rearrange(t, "b f -> b 1 f")
            if t.dim() == 1:
                t = rearrange(t, "b -> b 1 1")

            batch = (
                None,
                None,
                f,  # x_t
                t,  # y_t
            )
        else:
            raise ValueError(f"Unrecognized task type: {self.task_type}")

        return batch

    # target (B, F)
    def _training_step(self, batch, batch_idx):
        batch = self._map_batch_to_task(batch)
        outs = self.step(batch, train=True)
        return outs

    def _validation_step(self, batch, batch_idx):
        batch = self._map_batch_to_task(batch)
        outs = self.step(batch, train=False)
        return outs

    def _test_step(self, batch, batch_idx):
        batch = self._map_batch_to_task(batch)
        outs = self.step(batch, train=False)
        return outs

    def predict_step(self, batch, batch_idx):
        return self(*batch, **self.eval_step_forward_kwargs)

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(params=self.parameters())

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
            raise NotImplementedError(
                "Check this branch if it is really working as intended and remove this line."
            )
            return optimizer


class TransformerForecaster(Forecaster):
    def __init__(self, scheduler_Loss_monitor="forecast_loss", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scheduler_Loss_monitor = scheduler_Loss_monitor
        self.validation_step_outputs = []

    def _validation_step(self, batch, batch_idx):
        outs = super()._validation_step(batch, batch_idx)
        self.validation_step_outputs.append(outs)
        return outs

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        outs = self.validation_step_outputs
        total = 0
        count = 0
        for dict_ in outs:
            if self.scheduler_Loss_monitor in dict_:
                total += dict_[self.scheduler_Loss_monitor].mean()
                count += 1
            else:
                raise ValueError(
                    f"Could not find {self.scheduler_Loss_monitor} in the validation outputs"
                )

        avg_val_loss = total / count
        self.scheduler.step(avg_val_loss, is_end_epoch=True)
        self.validation_step_outputs.clear()

    def training_step(self, batch, batch_idx):
        outs = super().training_step(batch, batch_idx)
        self.scheduler.step()
        return outs

    def configure_optimizers(self):
        self.optimizer = self.optimizer_factory(params=self.parameters())
        assert self.scheduler_factory is not None, "Scheduler factory must be provided"
        self.scheduler = self.scheduler_factory(optimizer=self.optimizer)

        # Were not returning the scheduler here as it is stepped manually
        # in the training/validation steps
        return [self.optimizer]
