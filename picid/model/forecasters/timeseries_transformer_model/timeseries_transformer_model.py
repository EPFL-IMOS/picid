from typing import Callable, Tuple

import torch

from picid.model.definitions import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    STATE_FORECASTING_TASKS,
)
from picid.model.forecasters.forecaster import TransformerForecaster
from picid.model.forecasters.spacetimeformer_model.nn.time2vec import Time2Vec

from .timeseries_transformer import Model as Transformer


class Timeseries_Transformer_Forecaster(TransformerForecaster):
    """
    Forecaster wrapping a vanilla Transformer for forecasting, regression, and classification.

    Uses decoder-only mode for regression/classification/state_forecasting tasks,
    and full encoder-decoder for forecasting. Supports ContextEmbedding (embed_type 5/6)
    with d_x_embedding_router for separating time features from context.

    Parameters
    ----------
    optimizer_factory : callable
        Factory returning an optimizer given model parameters.
    scheduler_factory : callable
        Factory returning a learning rate scheduler given optimizer.
    d_x : int
        Dimension of context features (including time features if n_timefeatures > 0).
    d_yc : int
        Dimension of context target channels.
    d_yt : int
        Dimension of target channels.
    task_type : str
        One of "forecasting", "rul", "classification", "state_forecasting", etc.
    n_timefeatures : int, optional
        Number of time features in d_x; used to build d_x_embedding_router. Defaults to 0.
    loss : str, optional
        Loss name. Defaults to "mse".
    linear_window : int, optional
        Linear window size. Defaults to 0.
    use_revin : bool, optional
        Whether to use RevIN. Defaults to False.
    use_seasonal_decomp : bool, optional
        Whether to use seasonal decomposition. Defaults to False.
    linear_shared_weights : bool, optional
        Whether to share linear weights. Defaults to False.
    time_emb_dim : int, optional
        Time embedding dimension. Defaults to 0.
    teacher_forcing_prob : float, optional
        Teacher forcing probability. Defaults to 0.5.
    transformer_args : dict, optional
        Config passed to the underlying Transformer Model.
    device : str, optional
        Device string. Defaults to "cuda".
    d_x_mask : list of int, optional
        Mask for d_x dimensions. Defaults to None.
    mask_y_c : bool, optional
        Whether to mask y_c. Defaults to False.
    **kwargs
        Additional arguments (e.g. evaluators) passed to TransformerForecaster.
    """

    def __init__(
        self,
        optimizer_factory: Callable[..., torch.optim.Optimizer],
        scheduler_factory: Callable[..., torch.optim.lr_scheduler._LRScheduler],
        d_x: int,
        d_yc: int,
        d_yt: int,
        task_type: str,
        # d_x contains timefeatures if n_timefeatures > 0
        # these can be rerouted via d_x_embedding_router
        # to a seperate additive embedding.
        n_timefeatures: int = 0,
        loss: str = "mse",
        linear_window: int = 0,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        linear_shared_weights: bool = False,
        time_emb_dim: int = 0,
        teacher_forcing_prob: float = 0.5,
        transformer_args: dict = None,
        device: str = "cuda",
        d_x_mask: list[int] = None,
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

        if self.d_x > 0:
            self.t2v = Time2Vec(input_dim=d_x, embed_dim=time_emb_dim * d_x)

        _time_dim = time_emb_dim * d_x if time_emb_dim > 0 else d_x

        self.n_timefeatures = n_timefeatures
        if self.n_timefeatures > 0:
            transformer_args["d_x_embedding_router"] = torch.tensor(
                [0] * (d_x - n_timefeatures) + [1] * n_timefeatures
            )
        else:
            transformer_args["d_x_embedding_router"] = None

        is_decoder_only = (
            self.task_type
            in REGRESSION_TASKS + CLASSIFICATION_TASKS + STATE_FORECASTING_TASKS
        )

        self.transformer = Transformer(transformer_args, decoder_only=is_decoder_only)
        self.target_mask = None
        self.transformer_args = transformer_args
        self.mask_y_c = mask_y_c
        # self.teacher_forcing_prob = teacher_forcing_prob

        if d_x_mask is not None:
            self.mask_tt = torch.tensor(d_x_mask) == 0
            self.mask_t = torch.tensor(d_x_mask) == 1
            self.d_x_mask = d_x_mask
        else:
            self.d_x_mask = None

    @property
    def train_step_forward_kwargs(self):
        """
        Forward kwargs for training.

        Returns
        -------
        dict
            Forward keyword arguments with attention output disabled.
        """
        return {"output_attn": False}

    @property
    def eval_step_forward_kwargs(self):
        """
        Forward kwargs for evaluation.

        Returns
        -------
        dict
            Forward keyword arguments with attention output disabled.
        """
        return {"output_attn": False}

    def forward_model_pass(
        self, x_c, y_c, x_t, y_t, output_attn=False, target_mask=None
    ):
        """
        Run a single forward pass through the transformer.

        Parameters
        ----------
        x_c : torch.Tensor or None
            Context features (B, L_ctx, d_x). None for regression/classification.
        y_c : torch.Tensor or None
            Context targets (B, L_ctx, d_yc). None for regression/classification.
        x_t : torch.Tensor
            Target features (B, L_tgt, d_x).
        y_t : torch.Tensor
            Target values (B, L_tgt, d_yt).
        output_attn : bool, optional
            Whether to return attention weights. Defaults to False.
        target_mask : torch.Tensor, optional
            Mask for target features. Defaults to None.

        Returns
        -------
        tuple of torch.Tensor
            (output,) where output has shape (B, pred_len, d_yt).
        """
        # set data to [batch, length, dim] format
        if y_c is not None:
            if len(y_c.shape) == 2:
                y_c = y_c.unsqueeze(-1)
            if self.mask_y_c:
                y_c = torch.zeros_like(y_c)
        if len(y_t.shape) == 2:
            y_t = y_t.unsqueeze(-1)

        # TODO: rows 182-184 are potentially dead code. Reason: Duplicate of lines 176-177;
        # when y_c is not None we already zero it above; when y_c is None (regression/classification)
        # we assert before reaching this. Additional Note: Remove or consolidate with mask_y_c block above.
        if self.mask_y_c:
            y_c = torch.zeros_like(y_c)

        # best we can do is zero out the target y and put the full sequence as context to encoder.
        # enc_y = torch.concat((y_c, torch.zeros_like(y_t)), dim=1)
        # enc_x = torch.concat((x_c, x_t), dim=1)
        # enc = torch.concat((enc_x, enc_y), dim=2)
        if self.d_x_mask is not None:
            x_t[:, :, self.mask_tt] = 0

        if self.task_type in REGRESSION_TASKS + CLASSIFICATION_TASKS:
            # We only support this case so far
            assert x_c is None, "x_c should be None for regression tasks"
            assert y_c is None, "y_c should be None for regression tasks"
            x_c = x_t
            y_c = torch.zeros(
                (x_t.shape[0], x_t.shape[1], self.d_yt), device=x_c.device
            )

            batch_y, batch_y_mark = (
                y_c.to(self.device),
                x_c.to(self.device),
            )
            output = self.transformer(None, None, batch_y, batch_y_mark)

        elif self.task_type in STATE_FORECASTING_TASKS:
            batch_x = y_c.to(self.device)
            output = self.transformer(None, None, batch_x, None)
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = (
                y_c.to(self.device),
                y_t.to(self.device),
                x_c.to(self.device),
                x_t.to(self.device),
            )

            # decoder input (taken from dlinear https://github.com/vivva/DLinear/blob/main/exp/exp_main.py)
            dec_inp = torch.zeros_like(
                batch_y[:, -self.transformer_args.pred_len :, :]
            ).float()

            dec_inp = (
                torch.cat(
                    [batch_y[:, : self.transformer_args.label_len, :], dec_inp], dim=1
                )
                .float()
                .to(self.device)
            )

            output = self.transformer(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return (output,)

    def step(self, batch: Tuple[torch.Tensor], train: bool):
        """
        Execute one training or evaluation step.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            (x_c, y_c, x_t, y_t).
        train : bool
            If True, training step; else evaluation step.

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
        Compute forecast loss for a batch.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            (x_c, y_c, x_t, y_t).
        time_mask : torch.Tensor, optional
            Mask for time dimension. Defaults to None.
        forward_kwargs : dict, optional
            Kwargs passed to forward (e.g. output_attn, target_mask). Defaults to {}.

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
