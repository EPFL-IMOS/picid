from typing import Callable, Tuple

import torch
from omegaconf import OmegaConf

from picid.model.definitions import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    STATE_FORECASTING_TASKS,
)
from picid.model.forecasters.forecaster import TransformerForecaster

from .thuml_patchtst import Model as PatchTST


class PatchTST_Forecaster(TransformerForecaster):
    """
    PatchTST forecaster for time-series forecasting, regression, and classification.

    Wraps the PatchTST model (thuml_patchtst.Model) with training logic,
    loss computation, and step handling. Supports long-term forecasting,
    regression, and classification task types.

    Parameters
    ----------
    optimizer_factory : Callable[..., torch.optim.Optimizer]
        Factory returning a torch optimizer.
    scheduler_factory : Callable[..., torch.optim.lr_scheduler._LRScheduler]
        Factory returning a learning rate scheduler.
    d_x : int
        Dimension of exogenous input features.
    d_yc : int
        Dimension of context target variables.
    d_yt : int
        Dimension of target variables (output).
    ts_in : int
        Input sequence length.
    ts_out : int
        Output sequence length (prediction horizon).
    task_type : str
        One of long_term_forecast, regression, classification, etc.
    dropout : float, optional
        Dropout probability. Default is 0.1.
    patch_len : int, optional
        Patch length for PatchTST embedding. Default is 16.
    stride : int, optional
        Stride for PatchTST embedding. Default is 8.
    e_layers : int, optional
        Number of encoder layers. Default is 3.
    d_layers : int, optional
        Number of decoder layers. Default is 3.
    d_ff : int, optional
        Feed-forward dimension. Default is 64.
    n_heads : int, optional
        Number of attention heads. Default is 4.
    activation : str, optional
        Activation function name. Default is ``"gelu"``.
    factor : int, optional
        Attention factor. Default is 3.
    use_x_dim : bool, optional
        Whether to use exogenous features. Default is False.
    include_x_t : bool, optional
        Whether to include future exogenous in input. Default is True.
    loss : str, optional
        Loss function name. Default is ``"mse"``.
    linear_window : int, optional
        Linear window size. Default is 0.
    d_model : int, optional
        Model dimension. Default is 64.
    use_revin : bool, optional
        Whether to use RevIN normalization. Default is False.
    use_seasonal_decomp : bool, optional
        Whether to use seasonal decomposition. Default is False.
    linear_shared_weights : bool, optional
        Whether to share linear weights. Default is False.
    transformer_args : dict, optional
        Extra transformer arguments. Default is None.
    d_x_mask : list of int, optional
        Mask for exogenous dimensions. Default is None.
    mask_y_c : bool, optional
        Whether to mask context targets. Default is False.
    **kwargs
        Passed to the parent forecaster, for example ``evaluators``.
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
        patch_len: int = 16,
        stride: int = 8,
        e_layers: int = 3,
        d_layers: int = 3,
        d_ff: int = 64,
        n_heads: int = 4,
        activation: str = "gelu",
        factor: int = 3,
        use_x_dim: bool = False,
        include_x_t: bool = True,
        # training
        loss: str = "mse",
        linear_window: int = 0,
        d_model: int = 64,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        linear_shared_weights: bool = False,
        # this model could support teacher forcing
        # teacher_forcing_prob: float = 0.5,
        transformer_args: dict = None,
        d_x_mask: list[int] = None,
        mask_y_c: bool = False,
        **kwargs,
    ):
        """
        Initialize PatchTST forecaster.

        Parameters
        ----------
        optimizer_factory : callable
            Factory returning a torch optimizer.
        scheduler_factory : callable
            Factory returning a learning rate scheduler.
        d_x : int
            Dimension of exogenous input features.
        d_yc : int
            Dimension of context target variables.
        d_yt : int
            Dimension of target variables (output).
        ts_in : int
            Input sequence length.
        ts_out : int
            Output sequence length (prediction horizon).
        task_type : str
            One of long_term_forecast, regression, classification, etc.
        dropout : float, optional
            Dropout probability. Default is 0.1.
        patch_len : int, optional
            Patch length for PatchTST embedding. Default is 16.
        stride : int, optional
            Stride for PatchTST embedding. Default is 8.
        e_layers : int, optional
            Number of encoder layers. Default is 3.
        d_layers : int, optional
            Number of decoder layers. Default is 3.
        d_ff : int, optional
            Feed-forward dimension. Default is 64.
        n_heads : int, optional
            Number of attention heads. Default is 4.
        activation : str, optional
            Activation function name. Default is "gelu".
        factor : int, optional
            Attention factor. Default is 3.
        use_x_dim : bool, optional
            Whether to use exogenous features. Default is False.
        include_x_t : bool, optional
            Whether to include future exogenous in input. Default is True.
        loss : str, optional
            Loss function name. Default is "mse".
        linear_window : int, optional
            Linear window size. Default is 0.
        d_model : int, optional
            Model dimension. Default is 64.
        use_revin : bool, optional
            Whether to use RevIN normalization. Default is False.
        use_seasonal_decomp : bool, optional
            Whether to use seasonal decomposition. Default is False.
        linear_shared_weights : bool, optional
            Whether to share linear weights. Default is False.
        transformer_args : dict, optional
            Extra transformer arguments. Default is None.
        d_x_mask : list of int, optional
            Mask for exogenous dimensions. Default is None.
        mask_y_c : bool, optional
            Whether to mask context targets. Default is False.
        **kwargs
            Passed to parent (e.g. evaluators).
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
            "seq_len": ts_in + ts_out if include_x_t else ts_in,
            # "label_len": ts_in, # not used?
            "pred_len": ts_out,
            # "freq": freq,
            # "feature_encode_dim": time_emb_dim,
            "dropout": dropout,
            "d_model": d_model,
            "e_layers": e_layers,
            "d_layers": d_layers,
            "c_out": d_yt,
            "d_ff": d_ff,
            "feature_dim": d_yc + d_x if use_x_dim else None,
            "factor": factor,
            "n_heads": n_heads,
            "activation": activation,
        }

        if task_type not in CLASSIFICATION_TASKS + REGRESSION_TASKS:
            extra_configs = {
                "task_name": "long_term_forecast",
                "enc_in": 1,
            }

        elif task_type in REGRESSION_TASKS:
            extra_configs = {
                "task_name": "regression",
                "num_reg_targets": 1,
                "enc_in": d_yc + d_x,
            }

        elif task_type in CLASSIFICATION_TASKS:
            extra_configs = {
                "task_name": "classification",
                "num_class": d_yt,
                "enc_in": d_yc + d_x,
            }

        configs = OmegaConf.create(
            {
                **configs,
                **extra_configs,
            }
        )

        self.model = PatchTST(configs, patch_len=patch_len, stride=stride)

        self.target_mask = None
        self.transformer_args = transformer_args
        self.include_x_t = include_x_t
        # self.teacher_forcing_prob = teacher_forcing_prob
        if d_x_mask is not None:
            assert self.include_x_t is True
            self.mask_tt = torch.tensor(d_x_mask) == 0
            self.mask_t = torch.tensor(d_x_mask) == 1
            self.d_x_mask = d_x_mask
        else:
            self.d_x_mask = None
        self.mask_y_c = mask_y_c

        if (
            self.task_type
            in REGRESSION_TASKS + STATE_FORECASTING_TASKS + CLASSIFICATION_TASKS
        ):
            # We only support this case so far
            assert (
                self.include_x_t is False
            ), "include_x_t must be False for regression and classification tasks"

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
        Run a single forward pass through the PatchTST model.

        Parameters
        ----------
        x_c : torch.Tensor or None
            Context exogenous input [B, L, d_x]. None for regression/classification.
        y_c : torch.Tensor or None
            Context target input [B, L, d_yc]. None for regression/classification.
        x_t : torch.Tensor or None
            Target exogenous input [B, L, d_x]. For regression/classification, used as encoder input.
        y_t : torch.Tensor
            Target values [B, L, d_yt].
        output_attn : bool, optional
            Whether to output attention weights. Default is False.
        target_mask : torch.Tensor or None, optional
            Feature mask for loss. Default is None.

        Returns
        -------
        tuple of torch.Tensor
            Single-element tuple (outputs,) where outputs has shape [B, L, D]
            for forecasting or [B, 1, D] for regression/classification.
        """
        # set data to [batch, length, dim] format
        if y_c is not None:
            if len(y_c.shape) == 2:
                y_c = y_c.unsqueeze(-1)
            if self.mask_y_c:
                y_c = torch.zeros_like(y_c)

        # In the regresssion case, we show x_t as additional features
        # to the encoder.
        if self.task_type in REGRESSION_TASKS + CLASSIFICATION_TASKS:
            # We only support this case so far
            assert x_c is None, "x_c should be None for regression tasks"
            assert y_c is None, "y_c should be None for regression tasks"
            x_c = x_t
            x_t = None

        if self.include_x_t:
            if self.d_x_mask is not None:
                x_t[:, :, self.mask_tt] = 0

            batch_x = torch.cat([y_c, x_c], dim=2)
            batch_y = torch.cat([torch.zeros_like(y_t), x_t], dim=2)

            # PatchTST only uses x_enc
            outputs = self.model(
                x_enc=torch.cat([batch_x, batch_y], dim=1),
                x_mark_enc=None,
                x_dec=None,
                x_mark_dec=None,
            )
        else:
            if x_t is not None:
                x_t = torch.zeros_like(x_t, device=self.device)

            if x_c is not None and y_c is not None:
                batch_x = torch.cat([y_c, x_c], dim=2)
            else:
                # Take one or the other if only one of them is not none.
                batch_x = y_c if y_c is not None else x_c

            # PatchTST only uses x_enc, the other tensors are unused
            outputs = self.model(
                x_enc=batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None
            )

        if self.task_type in REGRESSION_TASKS + CLASSIFICATION_TASKS:
            # 2d tensor [B, num_class] -> [B, 1, num_class]
            outputs = outputs.unsqueeze(1)
        else:
            # outputs are 3d tensor [B, L, D]
            outputs = outputs[:, :, : y_t.shape[2]]

        return (outputs,)

    def step(self, batch: Tuple[torch.Tensor], train: bool):
        """
        Execute one training or evaluation step.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            (x_c, y_c, x_t, y_t) batch tensors.
        train : bool
            If True, run in training mode; else evaluation mode.

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
        Compute forecast loss from a batch.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            (x_c, y_c, x_t, y_t) batch tensors.
        time_mask : torch.Tensor or None, optional
            Mask over time steps for loss. Default is None.
        forward_kwargs : dict, optional
            Extra kwargs passed to forward (e.g. target_mask). Default is {}.

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
