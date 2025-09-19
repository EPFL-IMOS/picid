import random
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .cross_former import Crossformer
from lmetk.baselines.forecaster import Forecaster


class Crossformer_Forecaster(Forecaster):
    def __init__(
        self,
        d_x: int = 6,
        d_yc: int = 1,
        d_yt: int = 1,
        ts_in: int = 1,
        ts_out: int = 1,
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

        configs = {
            "data_dim": d_yc + d_x if decoder_embedding == "DSW" else d_yc,
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

        if d_x_mask is not None:
            assert self.decoder_embedding == "DSW"
            self.mask_tt = torch.tensor(d_x_mask) == 0
            self.mask_t = torch.tensor(d_x_mask) == 1
            self.d_x_mask = d_x_mask
        else:
            self.d_x_mask = None
        # self.teacher_forcing_prob = teacher_forcing_prob

        assert d_yt <= d_yc, "d_yt should be less than or equal to d_yc"

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
        if len(y_c.shape) == 2:
            y_c = y_c.unsqueeze(-1)
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
            forecast_out = self.crossformer(x_seq=y_c)

        # needed for what?

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

        # compute prediction accuracy stats for logging
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
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.base_lr,
            # weight_decay=self.l2_coeff,
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
