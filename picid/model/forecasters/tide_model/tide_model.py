from typing import Callable, Tuple

import torch
from omegaconf import OmegaConf

from picid.model.definitions import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    STATE_FORECASTING_TASKS,
)
from picid.model.forecasters.forecaster import TransformerForecaster

from .tide import Model as TiDE


class TiDE_Forecaster(TransformerForecaster):
    """
    Wrap the TiDE forecaster with the project training interface.

    Parameters
    ----------
    optimizer_factory : Callable[..., torch.optim.Optimizer]
        Factory for optimizer instantiation.
    scheduler_factory : Callable[..., torch.optim.lr_scheduler._LRScheduler]
        Factory for learning-rate scheduler instantiation.
    d_x : int
        Exogenous input dimension.
    d_yc : int
        Context target dimension.
    d_yt : int
        Forecast target dimension.
    ts_in : int
        Input sequence length.
    ts_out : int
        Output sequence length.
    task_type : str
        Task name used to select the TiDE configuration.
    dropout : float, default=0.1
        Dropout probability.
    e_layers : int, default=3
        Number of encoder layers.
    d_layers : int, default=3
        Number of decoder layers.
    d_ff : int, default=64
        Feed-forward hidden dimension.
    freq : str, default="h"
        Time frequency used for feature encoding.
    use_x_dim : bool, default=False
        Whether to derive the feature dimension from ``d_x``.
    include_x_t : bool, default=True
        Whether to include target-side exogenous features.
    loss : str, default="mse"
        Loss function name.
    linear_window : int, default=0
        Linear window size.
    d_model : int, default=64
        Model hidden dimension.
    use_revin : bool, default=False
        Whether to enable RevIN normalization.
    use_seasonal_decomp : bool, default=False
        Whether to enable seasonal decomposition.
    linear_shared_weights : bool, default=False
        Whether to share linear layer weights.
    time_emb_dim : int, default=0
        Time embedding dimension.
    transformer_args : dict | None, optional
        Extra transformer arguments.
    d_x_mask : list[int] | None, optional
        Mask for exogenous dimensions.
    mask_y_c : bool, default=False
        Whether to mask context targets.
    **kwargs
        Extra compatibility keywords forwarded to the parent class.
    """

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
        d_layers: int = 3,
        d_ff: int = 64,
        freq: str = "h",
        use_x_dim: bool = False,
        include_x_t: bool = True,
        # training
        loss: str = "mse",
        linear_window: int = 0,
        d_model: int = 64,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        linear_shared_weights: bool = False,
        time_emb_dim: int = 0,
        # this model could support teacher forcing
        # teacher_forcing_prob: float = 0.5,
        transformer_args: dict = None,
        d_x_mask: list[int] = None,
        mask_y_c: bool = False,
        **kwargs,
    ):
        """
        Initialize the TiDE forecaster wrapper.

        Parameters
        ----------
        optimizer_factory : Callable[..., torch.optim.Optimizer]
            Factory for optimizer instantiation.
        scheduler_factory : Callable[..., torch.optim.lr_scheduler._LRScheduler]
            Factory for learning rate scheduler.
        d_x : int
            Exogenous input dimension.
        d_yc : int
            Context target dimension.
        d_yt : int
            Forecast target dimension.
        ts_in : int
            Input sequence length.
        ts_out : int
            Output (prediction) length.
        task_type : str
            Task name used to select the TiDE configuration.
        dropout : float, default=0.1
            Dropout probability.
        e_layers : int, default=3
            Number of encoder layers.
        d_layers : int, default=3
            Number of decoder layers.
        d_ff : int, default=64
            Feed-forward hidden dimension.
        freq : str, default="h"
            Time frequency for feature encoding.
        use_x_dim : bool, default=False
            Use ``d_x`` as the feature dimension.
        include_x_t : bool, default=True
            Include exogenous features in the encoder.
        loss : str, default="mse"
            Loss function name.
        linear_window : int, default=0
            Linear window size.
        d_model : int, default=64
            Model hidden dimension.
        use_revin : bool, default=False
            Use RevIN normalization.
        use_seasonal_decomp : bool, default=False
            Use seasonal decomposition.
        linear_shared_weights : bool, default=False
            Share linear layer weights.
        time_emb_dim : int, default=0
            Time embedding dimension.
        transformer_args : dict | None, optional
            Extra transformer arguments.
        d_x_mask : list[int] | None, optional
            Mask for exogenous dimensions.
        mask_y_c : bool, default=False
            Mask context targets.
        **kwargs
            Extra compatibility keywords forwarded to the parent class.
        """
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

        configs = {
            "seq_len": ts_in,
            "pred_len": ts_out,
            "freq": freq,
            "feature_encode_dim": time_emb_dim,
            "dropout": dropout,
            "d_model": d_model,
            "e_layers": e_layers,
            "d_layers": d_layers,
            "d_ff": d_ff,
            "feature_dim": d_x if use_x_dim else None,
        }

        # This might be confusing at first, but the reason why we keep
        # task names aligned with the original codebase is
        # to make it easier to merge future changes from upstream.
        if task_type in CLASSIFICATION_TASKS:
            extra_configs = {
                "task_name": "classification",
                "num_class": d_yt,
                "c_out": 1,
            }
        elif task_type in REGRESSION_TASKS:
            extra_configs = {
                "task_name": "regression",
                "c_out": 1,
            }
        elif task_type in STATE_FORECASTING_TASKS:
            extra_configs = {
                "task_name": "state_forecasting",
                "c_out": d_yt,
            }
        else:
            extra_configs = {
                "task_name": "long_term_forecast",
                "c_out": d_yt,
            }

        configs = OmegaConf.create(
            {
                **configs,
                **extra_configs,
            }
        )

        self.model = TiDE(configs)

        self.target_mask = None
        self.transformer_args = transformer_args
        self.include_x_t = include_x_t
        self.mask_y_c = mask_y_c
        self.configs = configs
        # self.teacher_forcing_prob = teacher_forcing_prob

        if d_x_mask is not None:
            assert self.include_x_t is True
            self.mask_tt = torch.tensor(d_x_mask) == 0
            self.mask_t = torch.tensor(d_x_mask) == 1
            self.d_x_mask = d_x_mask
        else:
            self.d_x_mask = None

    @property
    def train_step_forward_kwargs(self):
        return {"output_attn": False}

    @property
    def eval_step_forward_kwargs(self):
        return {"output_attn": False}

    def forward_model_pass(
        self, x_c, y_c, x_t, y_t, output_attn=False, target_mask=None
    ):
        """
        Run a forward pass through the TiDE model.

        Parameters
        ----------
        x_c : torch.Tensor or None
            Context exogenous features; None for regression/state_forecasting.
        y_c : torch.Tensor or None
            Context targets; None for regression/state_forecasting.
        x_t : torch.Tensor
            Target exogenous features; shape (batch, ts_in, d_x).
        y_t : torch.Tensor
            Target values; shape (batch, ts_out, d_yt).
        output_attn : bool, optional
            Unused (TiDE has no attention).
        target_mask : torch.Tensor or None, optional
            Optional mask for target features.

        Returns
        -------
        tuple of torch.Tensor
            Single-element tuple (outputs,) with shape (batch, pred_len, c_out)
            or (batch, 1, num_class) for classification.
        """
        # set data to [batch, length, dim] format
        if y_c is not None:
            if len(y_c.shape) == 2:
                y_c = y_c.unsqueeze(-1)
            if self.mask_y_c:
                y_c = torch.zeros_like(y_c)

        if len(y_t.shape) == 2:
            y_t = y_t.unsqueeze(-1)

        if not self.include_x_t:
            x_t = torch.zeros_like(x_t, device=self.device)
            assert (
                self.d_x_mask is None
            ), "d_x_mask should be None if include_x_t is False"
        else:
            if self.d_x_mask is not None:
                x_t[:, :, self.mask_tt] = 0

        # In the regresssion case, we show x_t as additional features
        # to the encoder TODO: now we show a small slice or regression targets
        # twice. Maybe we should remove this. segment (in tide forward):
        #   batch_y_mark = torch.concat(
        #         [x_mark_enc, batch_y_mark[:, -self.pred_len :, :]], dim=1
        #     )
        if self.task_type in REGRESSION_TASKS + CLASSIFICATION_TASKS:
            # We only support this case so far
            assert x_c is None, "x_c should be None for regression tasks"
            assert y_c is None, "y_c should be None for regression tasks"
            # For now we assume that we have no past context in regression tasks
            # technically we could support it here.
            # We also assume that we have no historical target,
            # if we had, we would call it "contextual forecasting"
            x_mark_enc = x_t
            batch_y_mark = x_t
            x_enc = torch.zeros(
                (x_t.shape[0], x_t.shape[1], self.configs.c_out), device=x_t.device
            )

        elif self.task_type == "state_forecasting":
            # In state forecasting, we have historical target values
            # but no context, as we want to learn the state transition dynamics
            assert x_c is None, "x_c should be None for state forecasting tasks"
            assert x_t is None, "x_t should be None for state forecasting tasks"
            x_mark_enc = None
            batch_y_mark = None
            x_enc = y_c
        else:
            # contextual forecasting, we use the full information
            x_mark_enc = x_c
            batch_y_mark = x_t
            x_enc = y_c

        outputs = self.model(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            batch_y_mark=batch_y_mark,
            x_dec=None,
        )

        return (outputs,)

    def step(self, batch: Tuple[torch.Tensor], train: bool):
        """
        Run one training or evaluation step.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            (x_c, y_c, x_t, y_t).
        train : bool
            If True, training mode; else evaluation.

        Returns
        -------
        dict
            Stats with keys: predictions, targets, mask, forecast_loss, loss.
        """
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
        """
        Compute forecasting loss for a batch.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            (x_c, y_c, x_t, y_t).
        time_mask : torch.Tensor or None, optional
            Optional mask over time steps.
        forward_kwargs : dict, optional
            Kwargs passed to forward (e.g. target_mask).

        Returns
        -------
        dict
            Keys: forecast_loss, forecast_out, forecast_mask.
        """
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
