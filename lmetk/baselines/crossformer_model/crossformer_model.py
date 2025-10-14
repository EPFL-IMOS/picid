import random
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from lmetk.baselines.definitions import REGRESSION_TASKS
from lmetk.baselines.forecaster import TransformerForecaster
from lmetk.lr_scheduler.warmup_reduce_lr_on_plateau_scheduler import (
    WarmupReduceLROnPlateauScheduler,
)

from .cross_former import Crossformer


class Crossformer_Forecaster(TransformerForecaster):
    def __init__(
        self,
        d_x: int,
        d_yc: int,
        d_yt: int,
        ts_in: int,
        ts_out: int,
        task_type: str,
        dropout: float = 0.1,
        e_layers: int = 3,
        d_ff: int = 64,
        d_model: int = 64,
        seg_len: int = 24,
        win_size: int = 4,
        factor: int = 10,
        n_heads: int = 8,
        baseline: bool = False,
        decoder_embedding: str = "random",
        # training
        l2_coeff: float = 0,
        loss: str = "mse",
        linear_window: int = 0,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        linear_shared_weights: bool = False,
        device: str = "cuda",
        warmup_steps: float = 1000,
        decay_factor: float = 0.8,
        init_lr: float = 1e-10,
        base_lr: float = 5e-4,
        patience: int = 3,
        d_x_mask=None,
        mask_y_c: bool = False,
        **kwargs,
    ):
        super().__init__(
            scheduler_Loss_monitor="forecast_loss",
            d_x=d_x,
            d_yc=d_yc,
            d_yt=d_yt,
            l2_coeff=l2_coeff,
            loss=loss,
            linear_window=linear_window,
            use_revin=use_revin,
            use_seasonal_decomp=use_seasonal_decomp,
            linear_shared_weights=linear_shared_weights,
            evaluators=kwargs.get("evaluators", None),
            task_type=task_type,
        )

        no_y_as_input = task_type in REGRESSION_TASKS

        configs = {
            "data_dim": (
                d_yc + d_x
                if (decoder_embedding == "DSW" and not no_y_as_input)
                else (d_x if no_y_as_input else d_yc)
            ),
            "in_len": ts_in,
            "out_len": ts_out,
            "seg_len": seg_len,
            "win_size": win_size,
            "factor": factor,
            "d_model": d_model,
            "d_ff": d_ff,
            "n_heads": n_heads,
            "e_layers": e_layers,
            "dropout": dropout,
            "baseline": baseline,
            "device": device,
            "decoder_embedding": decoder_embedding,
            # "d_x_embedding_router": d_x_embedding_router
        }

        self.crossformer = Crossformer(**configs)

        n_parameters = sum(
            p.numel() for p in self.crossformer.parameters() if p.requires_grad
        )
        print("number of crossformer parameters: %d" % n_parameters)

        self.init_lr = init_lr
        self.base_lr = base_lr
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.target_mask = None
        self.decoder_embedding = decoder_embedding
        self.out_data_dim = d_yt
        self.mask_y_c = mask_y_c

        if d_x_mask is not None:
            assert self.decoder_embedding == "DSW"
            self.mask_tt = torch.tensor(d_x_mask) == 0
            self.mask_t = torch.tensor(d_x_mask) == 1
            self.d_x_mask = d_x_mask
        else:
            self.d_x_mask = None
        # self.teacher_forcing_prob = teacher_forcing_prob

        assert d_yt <= d_yc, "d_yt should be less than or equal to d_yc"

        if self.decoder_embedding == "DSW":
            assert not self.task_type in [
                "rul",
                "ahrul",
                "soc",
            ], "DSW decoder embedding not supported for task_type in ['rul', 'ahrul', 'soc']"

    @property
    def train_step_forward_kwargs(self):
        return {"output_attn": False}

    @property
    def eval_step_forward_kwargs(self):
        return {"output_attn": False}

    def forward_model_pass(
        self, x_c, y_c, x_t, y_t, output_attn=False, target_mask=None
    ):
        # set data to [batch, length, dim] format
        if y_c is not None:
            if len(y_c.shape) == 2:
                y_c = y_c.unsqueeze(-1)
            if self.mask_y_c:
                y_c = torch.zeros_like(y_c)

        if len(y_t.shape) == 2:
            y_t = y_t.unsqueeze(-1)

        # best we can do is zero out the target y and put the full sequence as context to encoder.
        # enc_y = torch.concat((y_c, torch.zeros_like(y_t)), dim=1)
        # enc_x = torch.concat((x_c, x_t), dim=1)
        # enc = torch.concat((enc_x, enc_y), dim=2)

        if self.decoder_embedding == "DSW":

            if self.d_x_mask is not None:
                x_t[:, :, self.mask_tt] = 0

            batch_x = torch.cat([x_c, y_c], dim=2)
            batch_y = torch.cat([x_t, torch.zeros_like(y_t)], dim=2)

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            outputs = self.crossformer(x_seq=batch_x, y_seq=batch_y)
            forecast_out = outputs[..., -self.out_data_dim :]
        elif self.decoder_embedding == "random":

            if self.task_type in REGRESSION_TASKS:
                x_seq = x_t
                outputs = self.crossformer(x_seq=x_seq)
                forecast_out = outputs[..., -self.out_data_dim :]
            else:
                x_seq = y_c
                forecast_out = self.crossformer(x_seq=x_seq)

        return (forecast_out,)

    def step(self, batch: Tuple[torch.Tensor], train: bool):
        kwargs = (
            self.train_step_forward_kwargs if train else self.eval_step_forward_kwargs
        )

        if self.target_mask is not None:
            kwargs["target_mask"] = self.target_mask

        time_mask = self.time_masked_idx if train else None

        # compute all loss values
        loss_dict = self.compute_loss(
            batch=batch,
            time_mask=time_mask,
            forward_kwargs=kwargs,
        )

        forecast_out = loss_dict["forecast_out"]
        forecast_mask = loss_dict["forecast_mask"]
        *_, y_t = batch

        stats = {
            "predictions": forecast_out.squeeze(2).detach().cpu().numpy(),
            "targets": y_t.squeeze(2).detach().cpu().numpy(),
            "mask": forecast_mask.squeeze(2),
        }

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
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.init_lr,
            # weight_decay=self.l2_coeff,
        )
        self.scheduler = WarmupReduceLROnPlateauScheduler(
            self.optimizer,
            init_lr=self.init_lr,
            peak_lr=self.base_lr,
            warmup_steps=self.warmup_steps,
            patience=self.patience,
            factor=self.decay_factor,
        )

        # Were not returning the scheduler here as it is stepped manually
        # in the training/validation steps
        return [self.optimizer]
