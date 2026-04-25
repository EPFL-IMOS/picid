from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torchmetrics

from picid.model.definitions import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    STATE_FORECASTING_TASKS,
)
from picid.model.forecasters.forecaster import TransformerForecaster

from .nn import Spacetimeformer


def find_ckpt(path):
    """
    Find first checkpoint file in ``path/checkpoints`` (excluding ``last``).

    Parameters
    ----------
    path : pathlib.Path
        Root directory containing a checkpoints subdirectory.

    Returns
    -------
    pathlib.Path
        Path to first .ckpt file found, excluding stem 'last'.
    """
    path = path / "checkpoints"
    files = [
        x
        for x in path.iterdir()
        if x.is_file() and x.suffix == ".ckpt" and x.stem != "last"
    ]
    return files[0]


class Spacetimeformer_Forecaster(TransformerForecaster):
    """
    Spacetimeformer forecaster for regression, classification, and state forecasting.

    Supports decoder-only mode (regression/classification) and encoder-decoder mode
    (forecasting). Uses spatio-temporal or temporal embeddings with optional
    Performer attention.

    Parameters
    ----------
    optimizer_factory : Callable[..., torch.optim.Optimizer]
        Factory used to create the optimizer.
    scheduler_factory : Callable[..., torch.optim.lr_scheduler._LRScheduler]
        Factory used to create the learning-rate scheduler.
    d_yc : int
        Context target dimension.
    d_yt : int
        Target output dimension.
    d_x : int
        Covariate dimension.
    task_type : str
        Task family handled by the forecaster.
    n_timefeatures : int, default=0
        Number of time features in ``d_x``.
    max_seq_len : int | None, default=None
        Maximum sequence length.
    start_token_len : int, default=64
        Number of decoder start tokens.
    attn_factor : int, default=5
        Attention factor for sparse attention.
    d_model : int, default=200
        Model width.
    d_queries_keys : int, default=50
        Query/key width.
    d_values : int, default=50
        Value width.
    n_heads : int, default=4
        Number of attention heads.
    e_layers : int, default=2
        Number of encoder layers.
    d_layers : int, default=2
        Number of decoder layers.
    d_ff : int, default=800
        Feed-forward width.
    dropout_emb : float, default=0.1
        Embedding dropout.
    dropout_qkv : float, default=0.0
        QKV dropout.
    dropout_ff : float, default=0.2
        Feed-forward dropout.
    dropout_attn_out : float, default=0.0
        Attention-output dropout.
    dropout_attn_matrix : float, default=0.0
        Attention-matrix dropout.
    pos_emb_type : str, default="abs"
        Positional embedding type.
    timetable_emb_type : str | None, default=None
        Timetable embedding type.
    time_emb_type : str, default="t2v"
        Time embedding type.
    variable_emb_type : str | None, default=None
        Variable embedding type.
    inverted_encoder : bool, default=False
        Whether to use an inverted encoder.
    global_self_attn : str, default="performer"
        Global self-attention type.
    local_self_attn : str, default="performer"
        Local self-attention type.
    global_cross_attn : str, default="performer"
        Global cross-attention type.
    local_cross_attn : str, default="performer"
        Local cross-attention type.
    performer_kernel : str, default="relu"
        Performer kernel name.
    embed_method : str, default="spatio-temporal"
        Embedding method name.
    attention_method : str, default="non-causual"
        Attention strategy name.
    performer_relu : bool, default=True
        Whether to use ReLU features in Performer.
    performer_redraw_interval : int, default=1000
        Performer projection redraw interval.
    attn_time_windows : int, default=1
        Number of attention time windows.
    use_shifted_time_windows : bool, default=True
        Whether to use shifted time windows.
    activation : str, default="gelu"
        Activation function name.
    head_f : str, default="linear"
        Head function name.
    norm : str, default="batch"
        Normalization strategy.
    use_final_norm : bool, default=True
        Whether to apply final normalization.
    initial_downsample_convs : int, default=0
        Number of initial downsampling convolutions.
    intermediate_downsample_convs : int, default=0
        Number of intermediate downsampling convolutions.
    loss : str, default="mse"
        Loss name.
    class_loss_imp : float, default=1e-3
        Weight for classification loss.
    recon_loss_imp : float, default=0
        Weight for reconstruction loss.
    time_emb_dim : int, default=6
        Time embedding dimension.
    null_value : float | None, default=None
        Null value used for masking.
    pad_value : float | None, default=None
        Padding value.
    linear_window : int, default=0
        Linear window size.
    linear_shared_weights : bool, default=False
        Whether the linear head shares weights.
    use_revin : bool, default=False
        Whether to use RevIN.
    use_seasonal_decomp : bool, default=False
        Whether to use seasonal decomposition.
    use_val_enc : bool, default=True
        Whether to use value embeddings in the encoder.
    use_val_dec : bool, default=True
        Whether to use value embeddings in the decoder.
    use_tt_enc : bool, default=True
        Whether to use timetable embeddings in the encoder.
    use_tt_dec : bool, default=True
        Whether to use timetable embeddings in the decoder.
    use_time : bool, default=True
        Whether to use time embeddings.
    use_space : bool, default=True
        Whether to use space embeddings.
    use_given : bool, default=True
        Whether to use given embeddings.
    recon_mask_skip_all : float, default=1.0
        Reconstruction-mask skip probability.
    recon_mask_max_seq_len : int, default=5
        Maximum sequence length for reconstruction masking.
    recon_mask_drop_seq : float, default=0.1
        Sequence-drop probability.
    recon_mask_drop_standard : float, default=0.2
        Standard-drop probability.
    recon_mask_drop_full : float, default=0.05
        Full-drop probability.
    verbose : bool, default=True
        Whether to enable verbose logging.
    target_mask : Any, default=None
        Optional target mask for multi-output training.
    mask_y_c : bool, default=False
        Whether to zero out the context targets.
    **kwargs
        Additional keyword arguments forwarded to the parent class.
    """

    def __init__(
        self,
        optimizer_factory: Callable[..., torch.optim.Optimizer],
        scheduler_factory: Callable[..., torch.optim.lr_scheduler._LRScheduler],
        d_yc: int,
        d_yt: int,
        d_x: int,
        task_type: str,
        # d_x contains timefeatures if n_timefeatures > 0
        # these can be rerouted via d_x_embedding_router
        # to a seperate additive embedding.
        n_timefeatures: int = 0,
        max_seq_len: int = None,
        start_token_len: int = 64,
        attn_factor: int = 5,
        d_model: int = 200,
        d_queries_keys=50,
        d_values=50,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 2,
        d_ff: int = 800,
        dropout_emb: float = 0.1,
        dropout_qkv: float = 0.0,
        dropout_ff: float = 0.2,
        dropout_attn_out: float = 0.0,
        dropout_attn_matrix: float = 0.0,
        pos_emb_type: str = "abs",
        timetable_emb_type: Optional[str] = None,
        time_emb_type: str = "t2v",
        variable_emb_type: Optional[str] = None,
        inverted_encoder: bool = False,  # iTransformer style encoder
        global_self_attn: str = "performer",
        local_self_attn: str = "performer",
        global_cross_attn: str = "performer",
        local_cross_attn: str = "performer",
        performer_kernel: str = "relu",
        embed_method: str = "spatio-temporal",
        attention_method: str = "non-causual",
        # d_x_embedding_router: Optional[torch.Tensor] = None,
        performer_relu: bool = True,
        performer_redraw_interval: int = 1000,
        attn_time_windows: int = 1,
        use_shifted_time_windows: bool = True,
        activation: str = "gelu",
        head_f: str = "linear",
        norm: str = "batch",
        use_final_norm: bool = True,
        initial_downsample_convs: int = 0,
        intermediate_downsample_convs: int = 0,
        loss: str = "mse",
        class_loss_imp: float = 1e-3,
        recon_loss_imp: float = 0,
        time_emb_dim: int = 6,
        null_value: float = None,
        pad_value: float = None,
        linear_window: int = 0,
        linear_shared_weights: bool = False,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        use_val_enc: bool = True,
        use_val_dec: bool = True,
        use_tt_enc: bool = True,
        use_tt_dec: bool = True,
        use_time: bool = True,
        use_space: bool = True,
        use_given: bool = True,
        recon_mask_skip_all: float = 1.0,
        recon_mask_max_seq_len: int = 5,
        recon_mask_drop_seq: float = 0.1,
        recon_mask_drop_standard: float = 0.2,
        recon_mask_drop_full: float = 0.05,
        verbose=True,
        target_mask=None,
        mask_y_c: bool = False,
        **kwargs,
    ):
        """
        Initialize the Spacetimeformer forecaster.

        Parameters
        ----------
        optimizer_factory : Callable[..., torch.optim.Optimizer]
            Factory for creating the optimizer.
        scheduler_factory : Callable[..., torch.optim.lr_scheduler._LRScheduler]
            Factory for creating the learning rate scheduler.
        d_yc : int
            Context target dimension.
        d_yt : int
            Target output dimension.
        d_x : int
            Covariate dimension (0 for state forecasting).
        task_type : str
            One of regression, classification, or state_forecasting.
        n_timefeatures : int, optional
            Number of time features in d_x.
        max_seq_len : int, optional
            Maximum sequence length.
        start_token_len : int, optional
            Length of start tokens for encoder-decoder (0 for decoder-only).
        attn_factor : int, optional
            Attention factor for ProbSparse attention.
        d_model : int, optional
            Model dimension.
        d_queries_keys : int, optional
            Query/key dimension.
        d_values : int, optional
            Value dimension.
        n_heads : int, optional
            Number of attention heads.
        e_layers : int, optional
            Number of encoder layers.
        d_layers : int, optional
            Number of decoder layers.
        d_ff : int, optional
            Feed-forward dimension.
        dropout_emb : float, optional
            Embedding dropout.
        dropout_qkv : float, optional
            QKV dropout.
        dropout_ff : float, optional
            Feed-forward dropout.
        dropout_attn_out : float, optional
            Attention output dropout.
        dropout_attn_matrix : float, optional
            Attention matrix dropout.
        pos_emb_type : str, optional
            Positional embedding type.
        timetable_emb_type : str, optional
            Timetable embedding type.
        time_emb_type : str, optional
            Time embedding type.
        variable_emb_type : str, optional
            Variable embedding type.
        inverted_encoder : bool, optional
            Use iTransformer-style inverted encoder.
        global_self_attn : str, optional
            Global self-attention type.
        local_self_attn : str, optional
            Local self-attention type.
        global_cross_attn : str, optional
            Global cross-attention type.
        local_cross_attn : str, optional
            Local cross-attention type.
        performer_kernel : str, optional
            Performer attention kernel.
        embed_method : str, optional
            Embedding method (spatio-temporal or temporal).
        attention_method : str, optional
            Attention method (causal or non-causal).
        performer_relu : bool, optional
            Use ReLU in Performer.
        performer_redraw_interval : int, optional
            Performer projection redraw interval.
        attn_time_windows : int, optional
            Number of attention time windows.
        use_shifted_time_windows : bool, optional
            Use shifted time windows.
        activation : str, optional
            Activation function.
        head_f : str, optional
            Head function type.
        norm : str, optional
            Normalization type.
        use_final_norm : bool, optional
            Use final normalization.
        initial_downsample_convs : int, optional
            Initial downsampling convolutions.
        intermediate_downsample_convs : int, optional
            Intermediate downsampling convolutions.
        loss : str, optional
            Loss function.
        class_loss_imp : float, optional
            Classification loss importance.
        recon_loss_imp : float, optional
            Reconstruction loss importance.
        time_emb_dim : int, optional
            Time embedding dimension.
        null_value : float, optional
            Null value for masking.
        pad_value : float, optional
            Padding value.
        linear_window : int, optional
            Linear window size.
        linear_shared_weights : bool, optional
            Share linear weights.
        use_revin : bool, optional
            Use RevIN normalization.
        use_seasonal_decomp : bool, optional
            Use seasonal decomposition.
        use_val_enc : bool, optional
            Use value embedding in encoder.
        use_val_dec : bool, optional
            Use value embedding in decoder.
        use_tt_enc : bool, optional
            Use timetable embedding in encoder.
        use_tt_dec : bool, optional
            Use timetable embedding in decoder.
        use_time : bool, optional
            Use time embedding.
        use_space : bool, optional
            Use space embedding.
        use_given : bool, optional
            Use given embedding.
        recon_mask_skip_all : float, optional
            Reconstruction mask skip probability.
        recon_mask_max_seq_len : int, optional
            Max sequence length for reconstruction mask.
        recon_mask_drop_seq : float, optional
            Sequence drop probability.
        recon_mask_drop_standard : float, optional
            Standard drop probability.
        recon_mask_drop_full : float, optional
            Full drop probability.
        verbose : bool, optional
            Verbosity.
        target_mask : optional
            Target mask for multi-output.
        mask_y_c : bool, optional
            Mask context targets.
        **kwargs
            Passed to parent (e.g. evaluators).
        """
        if task_type in STATE_FORECASTING_TASKS:
            # currently we populate d_x as zero vector for state forecasting
            # this would need to be changes if we want to include timefeatures
            assert d_x == 0, "for state forecasting d_x must be 0"

            # in state forecasting we do not have covariates
            # however, we still have toembed a zero vector because
            # the embedding adds positional and variable embeddings
            d_x = 1

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

        decoder_only = self.task_type in REGRESSION_TASKS + CLASSIFICATION_TASKS
        if decoder_only:
            start_token_len = 0

        self.torchmetrics_version = torchmetrics.__version__

        if start_token_len < 0:
            timetable_emb_enc_type = timetable_emb_type
            timetable_emb_dec_type = None
        else:
            timetable_emb_enc_type = timetable_emb_type
            timetable_emb_dec_type = timetable_emb_type

        self.n_timefeatures = n_timefeatures
        if self.n_timefeatures > 0:
            dxr = torch.tensor([0] * (d_x - n_timefeatures) + [1] * n_timefeatures).to(
                self.device
            )
            self.d_x_embedding_router = dxr
        else:
            self.d_x_embedding_router = None

        self.spacetimeformer = Spacetimeformer(
            d_yc=d_yc,
            d_yt=d_yt,
            d_x=d_x,
            start_token_len=start_token_len if start_token_len >= 0 else 0,
            attn_factor=attn_factor,
            d_model=d_model,
            d_queries_keys=d_queries_keys,
            d_values=d_values,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            initial_downsample_convs=initial_downsample_convs,
            intermediate_downsample_convs=intermediate_downsample_convs,
            dropout_emb=dropout_emb,
            dropout_attn_out=dropout_attn_out,
            dropout_attn_matrix=dropout_attn_matrix,
            dropout_qkv=dropout_qkv,
            dropout_ff=dropout_ff,
            pos_emb_type=pos_emb_type,
            timetable_emb_enc_type=timetable_emb_enc_type,
            timetable_emb_dec_type=timetable_emb_dec_type,
            inverted_encoder=inverted_encoder,
            time_emb_type=time_emb_type,
            variable_emb_type=variable_emb_type,
            global_self_attn=global_self_attn,
            local_self_attn=local_self_attn,
            global_cross_attn=global_cross_attn,
            local_cross_attn=local_cross_attn,
            activation=activation,
            head_f=head_f,
            norm=norm,
            use_final_norm=use_final_norm,
            embed_method=embed_method,
            attention_method=attention_method,
            d_x_embedding_router=self.d_x_embedding_router,
            performer_attn_kernel=performer_kernel,
            performer_redraw_interval=performer_redraw_interval,
            attn_time_windows=attn_time_windows,
            use_shifted_time_windows=use_shifted_time_windows,
            time_emb_dim=time_emb_dim,
            verbose=True,
            null_value=null_value,
            pad_value=pad_value,
            max_seq_len=max_seq_len,
            use_val_enc=use_val_enc,
            use_val_dec=use_val_dec,
            use_tt_dec=use_tt_dec,
            use_tt_enc=use_tt_enc,
            use_time=use_time,
            use_space=use_space,
            use_given=use_given,
            recon_mask_skip_all=recon_mask_skip_all,
            recon_mask_max_seq_len=recon_mask_max_seq_len,
            recon_mask_drop_seq=recon_mask_drop_seq,
            recon_mask_drop_standard=recon_mask_drop_standard,
            recon_mask_drop_full=recon_mask_drop_full,
            decoder_only=decoder_only,
        )

        self.target_mask = target_mask
        self.start_token_len = start_token_len
        self.embed_method = embed_method
        self.class_loss_imp = class_loss_imp if not decoder_only else 0.0
        self.recon_loss_imp = recon_loss_imp
        self.set_null_value(null_value)
        self.pad_value = pad_value
        self.mask_y_c = mask_y_c
        self.decoder_only = decoder_only

        self.save_hyperparameters()

        self.validation_step_outputs = []

    @property
    def train_step_forward_kwargs(self):
        kwargs = {"output_attn": False}
        if self.target_mask is not None:
            kwargs["target_mask"] = self.target_mask
        return kwargs

    @property
    def eval_step_forward_kwargs(self):
        kwargs = {"output_attn": False}
        if self.target_mask is not None:
            kwargs["target_mask"] = self.target_mask
        return kwargs

    def step(self, batch: Tuple[torch.Tensor], train: bool) -> dict[str, torch.Tensor]:
        """
        Execute one training or evaluation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor]
            (x_c, y_c, x_t, y_t) batch tuple.
        train : bool
            Whether in training mode.

        Returns
        -------
        dict
            Stats with keys: predictions, targets, mask, forecast_loss, class_loss,
            recon_loss, loss, acc.
        """
        kwargs = (
            self.train_step_forward_kwargs if train else self.eval_step_forward_kwargs
        )

        time_mask = self.time_masked_idx if train else None

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
        stats["class_loss"] = loss_dict["class_loss"]
        stats["recon_loss"] = loss_dict["recon_loss"]

        stats["loss"] = (
            loss_dict["forecast_loss"]
            + self.class_loss_imp * loss_dict["class_loss"]
            + self.recon_loss_imp * loss_dict["recon_loss"]
        )
        stats["acc"] = loss_dict["acc"]
        return stats

    def classification_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Compute cross-entropy classification loss and accuracy.

        Parameters
        ----------
        logits : torch.Tensor
            Model logits (B*T, num_classes).
        labels : torch.Tensor
            Integer class labels.

        Returns
        -------
        Tuple[torch.Tensor]
            (class_loss, accuracy).
        """
        labels = labels.view(-1).to(logits.device)
        d_y = labels.max() + 1

        logits = logits.view(-1, d_y)

        class_loss = F.cross_entropy(logits, labels)

        if self.torchmetrics_version == "0.9.3":
            acc = torchmetrics.functional.accuracy(
                torch.softmax(logits, dim=1),
                labels,
            )
        else:
            if d_y == 1:
                acc = torchmetrics.functional.accuracy(
                    torch.softmax(logits, dim=1).squeeze(1), labels, task="binary"
                )
            else:
                acc = torchmetrics.functional.accuracy(
                    torch.softmax(logits, dim=1),
                    labels,
                    task="multiclass",
                    num_classes=d_y.item(),
                )
        return class_loss, acc

    def compute_loss(self, batch, time_mask=None, forward_kwargs={}):
        """
        Compute forecast, reconstruction, and optional classification loss.

        Parameters
        ----------
        batch : Tuple
            (x_c, y_c, x_t, y_t) batch.
        time_mask : optional
            Time mask for forecasting loss.
        forward_kwargs : dict, optional
            Kwargs passed to forward (e.g. target_mask).

        Returns
        -------
        dict
            Forecast, reconstruction, and classification loss outputs.
        """
        x_c, y_c, x_t, y_t = batch

        target_mask = forward_kwargs.get("target_mask", None)

        forecast_out, recon_out, (logits, labels) = self(
            x_c, y_c, x_t, y_t, **forward_kwargs
        )

        if target_mask is not None:
            y_t = y_t[:, :, target_mask]

        forecast_loss, forecast_mask = self.forecasting_loss(
            outputs=forecast_out, y_t=y_t, time_mask=time_mask
        )

        if self.recon_loss_imp > 0:
            recon_loss, recon_mask = self.forecasting_loss(
                outputs=recon_out, y_t=y_c, time_mask=None
            )
        else:
            recon_loss, recon_mask = -1.0, 0.0

        if self.embed_method == "spatio-temporal" and self.class_loss_imp > 0:
            class_loss, acc = self.classification_loss(logits=logits, labels=labels)
        else:
            class_loss, acc = 0.0, -1.0

        return {
            "forecast_loss": forecast_loss,
            "class_loss": class_loss,
            "acc": acc,
            "forecast_out": forecast_out,
            "forecast_mask": forecast_mask,
            "recon_out": recon_out,
            "recon_loss": recon_loss,
            "recon_mask": recon_mask,
        }

    def nan_to_num(self, *inps):
        return inps

    def forward_model_pass(
        self, x_c, y_c, x_t, y_t, output_attn=False, target_mask=None
    ):
        """
        Run the forward pass through the Spacetimeformer model.

        Parameters
        ----------
        x_c : torch.Tensor or None
            Context covariates (B, L_c, d_x).
        y_c : torch.Tensor or None
            Context targets (B, L_c, d_yc).
        x_t : torch.Tensor or None
            Target covariates (B, L_t, d_x).
        y_t : torch.Tensor
            Target values (B, L_t, d_yt).
        output_attn : bool, optional
            Return attention weights.
        target_mask : optional
            Mask for multi-output targets.

        Returns
        -------
        Tuple
            (forecast_output, recon_output, (logits, labels)).
            If output_attn: adds attention as 4th element.
        """
        # set data to [batch, length, dim] format
        if y_c is not None:
            if len(y_c.shape) == 2:
                y_c = y_c.unsqueeze(-1)
            if self.mask_y_c:
                y_c = torch.zeros_like(y_c)

        if len(y_t.shape) == 2:
            y_t = y_t.unsqueeze(-1)

        if self.task_type in STATE_FORECASTING_TASKS:
            # in state forecasting we do not have covariates
            # however, we still have toembed a zero vector because
            # the embedding adds positional and variable embeddings
            x_c = torch.zeros(y_c.shape[0], y_c.shape[1], 1).to(y_c.device)
            x_t = torch.zeros(y_t.shape[0], y_t.shape[1], 1).to(y_t.device)

        enc_x = x_c
        enc_y = y_c
        dec_x = x_t

        if target_mask is not None:
            if self.decoder_only:
                assert (
                    x_t.shape[1] == y_t.shape[1]
                ), "in decoder only mode, x_t and y_t must have the same time dimension"
            dec_y = y_t.clone()
            dec_y[:, :, target_mask] = 0.0
        else:
            if self.decoder_only:
                assert (
                    y_t.shape[1] <= x_t.shape[1]
                ), "in decoder only mode, y_t must have a smaller or equal time dimension than x_t"

                # in this case we do not have an encoder, therefore
                # time dimensions of x_t and y_t need to match
                dec_y = torch.zeros(
                    (x_t.shape[0], x_t.shape[1], self.d_yt), device=x_t.device
                )
            else:
                dec_y = torch.zeros_like(y_t).to(self.device)

        if self.start_token_len > 0:
            dec_y = torch.cat((y_c[:, -self.start_token_len :, :], dec_y), dim=1).to(
                self.device
            )
            dec_x = torch.cat((x_c[:, -self.start_token_len :, :], dec_x), dim=1)
        if self.start_token_len < 0:
            enc_y = torch.cat((enc_y, dec_y[:, self.start_token_len :, :]), dim=1).to(
                self.device
            )
            enc_x = torch.cat((enc_x, dec_x[:, self.start_token_len :, :]), dim=1)

            dec_x = dec_x[:, :, self.d_x_embedding_router == 1]

        (
            forecast_output,
            recon_output,
            (logits, labels),
            attn,
            (enc_latent, dec_latent),
        ) = self.spacetimeformer(
            enc_x=enc_x,
            enc_y=enc_y,
            dec_x=dec_x,
            dec_y=dec_y,
            output_attention=output_attn,
        )

        if self.decoder_only:
            # TODO: find a better solution for this
            forecast_output = forecast_output[:, -y_t.shape[1] :, :]

        if target_mask is not None:
            forecast_output = forecast_output[:, :, target_mask]

        if output_attn:
            return forecast_output, recon_output, (logits, labels), attn
        return forecast_output, recon_output, (logits, labels)
