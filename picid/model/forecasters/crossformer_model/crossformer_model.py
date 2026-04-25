import logging
from typing import Callable
from typing import Tuple

import torch

from picid.model.definitions import CLASSIFICATION_TASKS, REGRESSION_TASKS
from picid.model.forecasters.forecaster import TransformerForecaster

from .cross_former import Crossformer

logger = logging.getLogger(__name__)


class Crossformer_Forecaster(TransformerForecaster):
    def __init__(
        self,
        optimizer_factory: Callable[..., torch.optim.Optimizer],
        scheduler_factory: Callable[..., torch.optim.lr_scheduler._LRScheduler],
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
        loss: str = "mse",
        linear_window: int = 0,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        linear_shared_weights: bool = False,
        device: str = "cuda",
        d_x_mask=None,
        mask_y_c: bool = False,
        **kwargs,
    ):
        super().__init__(
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            scheduler_Loss_monitor="forecast_loss",
            d_x=d_x,
            d_yc=d_yc,
            d_yt=d_yt,
            loss=loss,
            linear_window=linear_window,
            use_revin=use_revin,
            use_seasonal_decomp=use_seasonal_decomp,
            linear_shared_weights=linear_shared_weights,
            evaluators=kwargs.get("evaluators", None),
            task_type=task_type,
        )

        no_y_as_input = task_type in REGRESSION_TASKS + CLASSIFICATION_TASKS

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
        logger.info("number of crossformer parameters: %d" % n_parameters)

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

        if self.task_type not in REGRESSION_TASKS + CLASSIFICATION_TASKS:
            assert d_yt <= d_yc, "d_yt should be less than or equal to d_yc"

        if self.decoder_embedding == "DSW":
            not_supported = CLASSIFICATION_TASKS + REGRESSION_TASKS
            assert self.task_type not in [
                "rul",
                "ahrul",
                "soc",
            ], f"DSW decoder embedding not supported for task_type in {not_supported}"

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
            if self.task_type in REGRESSION_TASKS + CLASSIFICATION_TASKS:
                x_seq = x_t
                outputs = self.crossformer(x_seq=x_seq)
                # TODO: read paper and see if this can be improved
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
            "predictions": forecast_out.detach().cpu().numpy(),
            "targets": y_t.detach().cpu().numpy(),
            "mask": forecast_mask,
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
