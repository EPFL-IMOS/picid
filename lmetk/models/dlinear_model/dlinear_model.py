import random
from typing import Tuple

from omegaconf import OmegaConf
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.dlinear_model.DLinear import Model

import models as stf


class DLinear_Forecaster(stf.Forecaster):
    def __init__(
        self,
        d_x: int = 6,
        d_yc: int = 1,
        d_yt: int = 1,
        ts_in: int = 1,
        ts_out: int = 1,
        include_x_t: bool = True,
        individual: bool = False,
        # training
        l2_coeff: float = 0,
        loss: str = "mse",
        linear_window: int = 0,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        linear_shared_weights: bool = False,
        warmup_steps: float = 1000,
        decay_factor: float = 0.8,
        init_lr: float = 1e-10,
        base_lr: float = 5e-4,
        patience: int = 3,
    ):
        super().__init__(
            d_x=d_x,
            d_yc=d_yc,
            d_yt=d_yt,
            l2_coeff=l2_coeff,
            loss=loss,
            linear_window=linear_window,
            use_revin=use_revin,
            use_seasonal_decomp=use_seasonal_decomp,
            linear_shared_weights=linear_shared_weights,
        )

        configs = OmegaConf.create(
            {
                "task_name": "long_term_forecast",
                "seq_len": ts_in + ts_out if include_x_t else ts_in,
                # "label_len": ts_in, # not used?
                "pred_len": ts_out,
                "individual": individual,
                "enc_in": d_yc + d_x,
            }
        )

        self.model = Model(configs)

        self.init_lr = init_lr
        self.base_lr = base_lr
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.target_mask = None
        self.include_x_t = include_x_t

    @property
    def train_step_forward_kwargs(self):
        return {"output_attn": False}

    @property
    def eval_step_forward_kwargs(self):
        return {"output_attn": False}

    def forward_model_pass(
        self, x_c, y_c, x_t, y_t, output_attn=False, target_mask=None
    ):
        if len(y_c.shape) == 2:
            y_c = y_c.unsqueeze(-1)
        if len(y_t.shape) == 2:
            y_t = y_t.unsqueeze(-1)

        if self.include_x_t:
            batch_x = torch.cat([y_c, x_c], dim=2)
            batch_y = torch.cat([torch.zeros_like(y_t), x_t], dim=2)
            outputs = self.model(x=torch.cat([batch_x, batch_y], dim=1))
        else:
            batch_x = torch.cat([y_c, x_c], dim=2)
            outputs = self.model(x=batch_x)

        outputs = outputs[:, :, : y_t.shape[2]]

        return (outputs,)

    def step(self, batch: Tuple[torch.Tensor], train: bool):
        kwargs = (
            self.train_step_forward_kwargs if train else self.eval_step_forward_kwargs
        )

        if self.target_mask is not None:
            kwargs["target_mask"] = self.target_mask

        time_mask = self.time_masked_idx if train else None

        loss_dict = self.compute_loss(
            batch=batch,
            time_mask=time_mask,
            forward_kwargs=kwargs,
        )

        forecast_out = loss_dict["forecast_out"]
        forecast_mask = loss_dict["forecast_mask"]
        *_, y_t = batch

        stats = self._compute_stats(forecast_out, y_t, forecast_mask)

        stats["forecast_loss"] = loss_dict["forecast_loss"]
        stats["loss"] = loss_dict["forecast_loss"]
        return stats

    def compute_loss(self, batch, time_mask=None, forward_kwargs={}):
        target_mask = forward_kwargs.get("target_mask", None)

        x_c, y_c, x_t, y_t = batch
        (forecast_out,) = self(x_c, y_c, x_t, y_t, **forward_kwargs)

        forecast_loss, forecast_mask = self.forecasting_loss(
            outputs=forecast_out, y_t=y_t, time_mask=time_mask, feat_mask=target_mask
        )

        return {
            "forecast_loss": forecast_loss,
            "forecast_out": forecast_out,
            "forecast_mask": forecast_mask,
        }

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.base_lr,
            weight_decay=self.l2_coeff,
        )
        self.scheduler = stf.lr_scheduler.WarmupReduceLROnPlateau(
            self.optimizer,
            init_lr=self.init_lr,
            peak_lr=self.base_lr,
            warmup_steps=self.warmup_steps,
            patience=self.patience,
            factor=self.decay_factor,
        )
        return [self.optimizer], [self.scheduler]

    @classmethod
    def add_cli(self, parser):
        super().add_cli(parser)
