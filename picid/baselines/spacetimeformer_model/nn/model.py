from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from .extra_layers import ConvBlock, Normalization, FoldForPred
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .attn import (
    FullAttention,
    ProbAttention,
    AttentionLayer,
    PerformerAttention,
)
from .embed import Embedding
from .data_dropout import ReconstructionDropout
from ..utils.masking import TriangularCausalMask

# from models.spacetimeformer_model.utils.masking import TriangularCausalMask


class Spacetimeformer(nn.Module):
    """
    Encoder-decoder or decoder-only spatio-temporal transformer.

    Supports forecasting (encoder-decoder) and regression/classification
    (decoder-only). Uses Performer, full, or ProbSparse attention.

    Parameters
    ----------
    d_yc : int, default=1
        Context target dimension.
    d_yt : int, default=1
        Target output dimension.
    d_x : int, default=4
        Covariate dimension.
    max_seq_len : int, optional
        Maximum sequence length.
    attn_factor : int, default=5
        ProbSparse attention factor.
    d_model : int, default=200
        Model dimension.
    d_queries_keys : int, default=30
        Query/key dimension.
    d_values : int, default=30
        Value dimension.
    n_heads : int, default=8
        Number of attention heads.
    e_layers : int, default=2
        Number of encoder layers.
    d_layers : int, default=3
        Number of decoder layers.
    d_ff : int, default=800
        Feed-forward dimension.
    start_token_len : int, default=0
        Start token length.
    time_emb_dim : int, default=6
        Time embedding dimension.
    dropout_emb : float, default=0.1
        Embedding dropout.
    dropout_attn_matrix : float, default=0.0
        Attention matrix dropout.
    dropout_attn_out : float, default=0.0
        Attention output dropout.
    dropout_ff : float, default=0.2
        Feed-forward dropout.
    dropout_qkv : float, default=0.0
        QKV dropout.
    pos_emb_type : str, default="abs"
        Positional embedding type.
    timetable_emb_enc_type : str or None, default="abs"
        Encoder timetable embedding type.
    timetable_emb_dec_type : str or None, default="abs"
        Decoder timetable embedding type.
    time_emb_type : str, default="t2v"
        Time embedding type.
    inverted_encoder : bool, default=False
        Use an inverted encoder layout.
    variable_emb_type : str or None, default=None
        Variable embedding type.
    global_self_attn : str, default="performer"
        Global self-attention type.
    local_self_attn : str, default="performer"
        Local self-attention type.
    global_cross_attn : str, default="performer"
        Global cross-attention type.
    local_cross_attn : str, default="performer"
        Local cross-attention type.
    performer_attn_kernel : str, default="relu"
        Performer kernel name.
    performer_redraw_interval : int, default=1000
        Performer redraw interval.
    attn_time_windows : int, default=1
        Number of attention time windows.
    use_shifted_time_windows : bool, default=True
        Use shifted time windows.
    embed_method : str, default="spatio-temporal"
        Embedding method.
    attention_method : str, default="non-causual"
        Attention method.
    d_x_embedding_router : torch.Tensor or None, optional
        Router for d_x embedding.
    activation : str, default="gelu"
        Activation function.
    head_f : str, default="linear"
        Head function type.
    norm : str, default="batch"
        Normalization type.
    use_final_norm : bool, default=True
        Use final normalization.
    initial_downsample_convs : int, default=0
        Initial downsampling convolutions.
    intermediate_downsample_convs : int, default=0
        Intermediate downsampling convolutions.
    null_value : float or None, optional
        Null value for masking.
    pad_value : float or None, optional
        Padding value.
    out_dim : int or None, optional
        Output dimension.
    use_val_enc : bool, default=True
        Use value embedding in encoder.
    use_val_dec : bool, default=True
        Use value embedding in decoder.
    use_tt_enc : bool, default=True
        Use timetable in encoder.
    use_tt_dec : bool, default=True
        Use timetable in decoder.
    use_time : bool, default=True
        Use time embedding.
    use_space : bool, default=True
        Use space embedding.
    use_given : bool, default=True
        Use given embedding.
    recon_mask_skip_all : float, default=1.0
        Reconstruction mask skip probability.
    recon_mask_max_seq_len : int, default=5
        Max sequence length for reconstruction masking.
    recon_mask_drop_seq : float, default=0.1
        Sequence drop probability.
    recon_mask_drop_standard : float, default=0.2
        Standard drop probability.
    recon_mask_drop_full : float, default=0.05
        Full drop probability.
    decoder_only : bool, default=False
        Decoder-only mode.
    verbose : bool, default=True
        Verbosity flag.
    """

    def __init__(
        self,
        d_yc: int = 1,
        d_yt: int = 1,
        d_x: int = 4,
        max_seq_len: int = None,
        attn_factor: int = 5,
        d_model: int = 200,
        d_queries_keys: int = 30,
        d_values: int = 30,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 3,
        d_ff: int = 800,
        start_token_len: int = 0,
        time_emb_dim: int = 6,
        dropout_emb: float = 0.1,
        dropout_attn_matrix: float = 0.0,
        dropout_attn_out: float = 0.0,
        dropout_ff: float = 0.2,
        dropout_qkv: float = 0.0,
        pos_emb_type: str = "abs",
        timetable_emb_enc_type: Optional[str] = "abs",
        timetable_emb_dec_type: Optional[str] = "abs",
        time_emb_type: str = "t2v",
        inverted_encoder: bool = False,
        variable_emb_type: Optional[str] = None,
        global_self_attn: str = "performer",
        local_self_attn: str = "performer",
        global_cross_attn: str = "performer",
        local_cross_attn: str = "performer",
        performer_attn_kernel: str = "relu",
        performer_redraw_interval: int = 1000,
        attn_time_windows: int = 1,
        use_shifted_time_windows: bool = True,
        embed_method: str = "spatio-temporal",
        attention_method: str = "non-causual",
        d_x_embedding_router: Optional[torch.Tensor] = None,
        activation: str = "gelu",
        head_f: str = "linear",
        norm: str = "batch",
        use_final_norm: bool = True,
        initial_downsample_convs: int = 0,
        intermediate_downsample_convs: int = 0,
        null_value: float = None,
        pad_value: float = None,
        out_dim: int = None,
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
        decoder_only: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize Spacetimeformer.

        Parameters
        ----------
        d_yc : int
            Context target dimension.
        d_yt : int
            Target output dimension.
        d_x : int
            Covariate dimension.
        max_seq_len : int, optional
            Maximum sequence length.
        attn_factor : int
            ProbSparse attention factor.
        d_model : int
            Model dimension.
        d_queries_keys : int
            Query/key dimension.
        d_values : int
            Value dimension.
        n_heads : int
            Number of attention heads.
        e_layers : int
            Number of encoder layers.
        d_layers : int
            Number of decoder layers.
        d_ff : int
            Feed-forward dimension.
        start_token_len : int
            Start token length (0 for decoder-only).
        time_emb_dim : int
            Time embedding dimension.
        dropout_emb : float
            Embedding dropout.
        dropout_attn_matrix : float
            Attention matrix dropout.
        dropout_attn_out : float
            Attention output dropout.
        dropout_ff : float
            Feed-forward dropout.
        dropout_qkv : float
            QKV dropout.
        pos_emb_type : str
            Positional embedding type.
        timetable_emb_enc_type : str, optional
            Encoder timetable embedding type.
        timetable_emb_dec_type : str, optional
            Decoder timetable embedding type.
        time_emb_type : str
            Time embedding type.
        inverted_encoder : bool
            Use inverted (iTransformer-style) encoder.
        variable_emb_type : str, optional
            Variable embedding type.
        global_self_attn : str
            Global self-attention type.
        local_self_attn : str
            Local self-attention type.
        global_cross_attn : str
            Global cross-attention type.
        local_cross_attn : str
            Local cross-attention type.
        performer_attn_kernel : str
            Performer kernel type.
        performer_redraw_interval : int
            Performer redraw interval.
        attn_time_windows : int
            Number of attention time windows.
        use_shifted_time_windows : bool
            Use shifted time windows.
        embed_method : str
            Embedding method.
        attention_method : str
            Attention method.
        d_x_embedding_router : torch.Tensor, optional
            Router for d_x embedding.
        activation : str
            Activation function.
        head_f : str
            Head function type.
        norm : str
            Normalization type.
        use_final_norm : bool
            Use final normalization.
        initial_downsample_convs : int
            Initial downsampling convolutions.
        intermediate_downsample_convs : int
            Intermediate downsampling convolutions.
        null_value : float, optional
            Null value for masking.
        pad_value : float, optional
            Padding value.
        out_dim : int, optional
            Output dimension.
        use_val_enc : bool
            Use value embedding in encoder.
        use_val_dec : bool
            Use value embedding in decoder.
        use_tt_enc : bool
            Use timetable in encoder.
        use_tt_dec : bool
            Use timetable in decoder.
        use_time : bool
            Use time embedding.
        use_space : bool
            Use space embedding.
        use_given : bool
            Use given embedding.
        recon_mask_skip_all : float
            Reconstruction mask skip probability.
        recon_mask_max_seq_len : int
            Max seq len for recon mask.
        recon_mask_drop_seq : float
            Sequence drop probability.
        recon_mask_drop_standard : float
            Standard drop probability.
        recon_mask_drop_full : float
            Full drop probability.
        decoder_only : bool
            Decoder-only mode (no encoder).
        verbose : bool
            Verbosity.
        """
        super().__init__()
        if e_layers:
            assert intermediate_downsample_convs <= e_layers - 1
        if embed_method == "temporal":
            assert (
                local_self_attn == "none"
            ), "local attention not compatible with Temporal-only embedding"
            assert (
                local_cross_attn == "none"
            ), "Local Attention not compatible with Temporal-only embedding"
            split_length_into = 1
        else:
            split_length_into = d_yc

        self.pad_value = pad_value
        self.embed_method = embed_method
        self.d_yt = d_yt
        self.d_yc = d_yc
        self.start_token_len = start_token_len
        self.attention_method = attention_method
        self.decoder_only = decoder_only
        self.inverted_encoder = inverted_encoder

        recon_dropout = ReconstructionDropout(
            drop_full_timesteps=recon_mask_drop_full,
            drop_standard=recon_mask_drop_standard,
            drop_seq=recon_mask_drop_seq,
            drop_max_seq_len=recon_mask_max_seq_len,
            skip_all_drop=recon_mask_skip_all,
        )

        if d_x_embedding_router is not None:
            assert (
                len(d_x_embedding_router) == d_x
            ), "Embedding router must have same length as d_x"

        if not self.decoder_only:
            if inverted_encoder:
                self.enc_embedding = Embedding(
                    c_in=max_seq_len, d_model=d_model, dropout=dropout_emb
                )
            else:
                self.enc_embedding = Embedding(
                    d_y=d_yc,
                    d_x=d_x,
                    d_model=d_model,
                    time_emb_dim=time_emb_dim,
                    downsample_convs=initial_downsample_convs,
                    method=embed_method,
                    null_value=null_value,
                    pad_value=pad_value,
                    start_token_len=start_token_len,
                    is_encoder=True,
                    position_emb=pos_emb_type,
                    time_emb=time_emb_type,
                    timetable_emb=timetable_emb_enc_type,
                    variable_emb=variable_emb_type,
                    embedding_router=d_x_embedding_router,
                    max_seq_len=max_seq_len,
                    data_dropout=recon_dropout,
                    use_val=use_val_enc,
                    use_timetable=use_tt_enc,
                    use_time=use_time,
                    use_space=use_space,
                    use_given=use_given,
                )

        self.dec_embedding = Embedding(
            d_y=d_yt,
            d_x=d_x,
            d_model=d_model,
            time_emb_dim=time_emb_dim,
            downsample_convs=initial_downsample_convs,
            method=embed_method,
            null_value=null_value,
            pad_value=pad_value,
            start_token_len=start_token_len,
            is_encoder=False,
            position_emb=pos_emb_type,
            time_emb=time_emb_type,
            timetable_emb=timetable_emb_dec_type,
            variable_emb=variable_emb_type,
            embedding_router=d_x_embedding_router,
            max_seq_len=max_seq_len,
            data_dropout=None,
            use_val=use_val_dec,
            use_timetable=use_tt_dec,
            use_time=use_time,
            use_space=use_space,
            use_given=use_given,
        )

        attn_kwargs = {
            "d_model": d_model,
            "n_heads": n_heads,
            "d_qk": d_queries_keys,
            "d_v": d_values,
            "dropout_qkv": dropout_qkv,
            "dropout_attn_matrix": dropout_attn_matrix,
            "attn_factor": attn_factor,
            "performer_attn_kernel": performer_attn_kernel,
            "performer_redraw_interval": performer_redraw_interval,
        }

        if not self.decoder_only:
            self.encoder = Encoder(
                attn_layers=[
                    EncoderLayer(
                        global_attention=self._attn_switch(
                            global_self_attn,
                            **attn_kwargs,
                        ),
                        local_attention=self._attn_switch(
                            local_self_attn,
                            **attn_kwargs,
                        ),
                        d_model=d_model,
                        d_yc=d_yc if embed_method == "spatio-temporal" else 1,
                        time_windows=attn_time_windows,
                        time_window_offset=(
                            2 if use_shifted_time_windows and (l % 2 == 1) else 0
                        ),
                        d_ff=d_ff,
                        dropout_ff=dropout_ff,
                        dropout_attn_out=dropout_attn_out,
                        activation=activation,
                        norm=norm,
                    )
                    for l in range(e_layers)  # noqa: E741
                ],
                conv_layers=[
                    ConvBlock(split_length_into=split_length_into, d_model=d_model)
                    for l in range(intermediate_downsample_convs)  # noqa: E741
                ],
                norm_layer=(
                    Normalization(norm, d_model=d_model) if use_final_norm else None
                ),
                emb_dropout=dropout_emb,
            )

        if self.decoder_only:
            local_cross_attn = "none"
            global_cross_attn = "none"

        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    global_self_attention=self._attn_switch(
                        global_self_attn,
                        **attn_kwargs,
                    ),
                    local_self_attention=self._attn_switch(
                        local_self_attn,
                        **attn_kwargs,
                    ),
                    global_cross_attention=self._attn_switch(
                        global_cross_attn,
                        **attn_kwargs,
                    ),
                    local_cross_attention=self._attn_switch(
                        local_cross_attn,
                        **attn_kwargs,
                    ),
                    d_model=d_model,
                    time_windows=attn_time_windows,
                    time_window_offset=(
                        2 if use_shifted_time_windows and (l % 2 == 1) else 0
                    ),
                    d_ff=d_ff,
                    d_yt=d_yt if embed_method == "spatio-temporal" else 1,
                    d_yc=d_yc if embed_method == "spatio-temporal" else 1,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                )
                for l in range(d_layers)  # noqa: E741
            ],
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb,
        )

        if not out_dim:
            out_dim = 1 if self.embed_method == "spatio-temporal" else d_yt
            recon_dim = 1 if self.embed_method == "spatio-temporal" else d_yc

        self.classifier = nn.Linear(d_model, d_yc, bias=True)

        if head_f == "linear":
            self.forecaster = nn.Linear(d_model, out_dim, bias=True)
            self.reconstructor = nn.Linear(d_model, recon_dim, bias=True)
        else:
            raise ValueError(f"Unrecognized head_f code '{head_f}'")

    def forward(
        self,
        enc_x,
        enc_y,
        dec_x,
        dec_y,
        output_attention=False,
    ):
        """
        Forward pass.

        Parameters
        ----------
        enc_x : torch.Tensor or None
            Encoder covariates (B, L_enc, d_x).
        enc_y : torch.Tensor or None
            Encoder targets (B, L_enc, d_yc).
        dec_x : torch.Tensor
            Decoder covariates (B, L_dec, d_x).
        dec_y : torch.Tensor
            Decoder targets (B, L_dec, d_yt).
        output_attention : bool, optional
            Return attention weights.

        Returns
        -------
        Tuple
            (forecast_out, recon_out, (classifier_out, enc_var_idxs),
             attention_weights, (enc_out, dec_out)).
        """
        if self.inverted_encoder:
            enc_y = torch.cat([enc_y, dec_y], dim=1)
            enc_x = torch.cat([enc_x, dec_x], dim=1)

        if not self.decoder_only:
            enc_vt_emb, enc_s_emb, enc_var_idxs, enc_mask_seq = self.enc_embedding(
                y=enc_y, x=enc_x
            )

            enc_out, enc_self_attns = self.encoder(
                val_time_emb=enc_vt_emb,
                space_emb=enc_s_emb,
                self_mask_seq=enc_mask_seq,
                output_attn=output_attention,
            )

            if enc_mask_seq is not None:
                enc_dec_mask_seq = enc_mask_seq.clone()
            else:
                enc_dec_mask_seq = enc_mask_seq

        else:
            enc_var_idxs = None
            enc_out = None
            enc_dec_mask_seq = None
            enc_self_attns = None

        dec_vt_emb, dec_s_emb, _, dec_mask_seq = self.dec_embedding(y=dec_y, x=dec_x)

        if self.attention_method == "causual":
            b, l, f = dec_y.shape  # noqa: E741
            dec_mask_seq = TriangularCausalMask(b, l, device=dec_y.device).mask.squeeze(
                1
            )

        dec_out, dec_cross_attns = self.decoder(
            val_time_emb=dec_vt_emb,
            space_emb=dec_s_emb,
            cross=enc_out,
            self_mask_seq=dec_mask_seq,
            cross_mask_seq=enc_dec_mask_seq,
            output_cross_attn=output_attention,
        )

        forecast_out = self.forecaster(dec_out)

        if not self.decoder_only:
            recon_out = self.reconstructor(enc_out)
        else:
            recon_out = None

        if self.embed_method == "spatio-temporal":
            forecast_out = FoldForPred(forecast_out, dy=self.d_yt)
            if not self.decoder_only:
                recon_out = FoldForPred(recon_out, dy=self.d_yc)
        forecast_out = forecast_out[:, self.start_token_len :, :]

        if enc_var_idxs is not None:
            classifier_enc_out = self.classifier(enc_out.detach())
        else:
            classifier_enc_out, enc_var_idxs = None, None

        return (
            forecast_out,
            recon_out,
            (classifier_enc_out, enc_var_idxs),
            (
                (enc_self_attns, dec_cross_attns)
                if not self.decoder_only
                else (dec_cross_attns)
            ),
            (enc_out, dec_out),
        )

    def _attn_switch(
        self,
        attn_str: str,
        *,
        d_model: int,
        n_heads: int,
        d_qk: int,
        d_v: int,
        dropout_qkv: float,
        dropout_attn_matrix: float,
        attn_factor: int,
        performer_attn_kernel: str,
        performer_redraw_interval: int,
    ):
        """
        Return attention layer for given type (full, prob, performer, none).

        Parameters
        ----------
        attn_str : str
            One of 'full', 'prob', 'performer', 'none'.
        d_model : int
            Model dimension.
        n_heads : int
            Number of heads.
        d_qk : int
            Query/key dimension.
        d_v : int
            Value dimension.
        dropout_qkv : float
            QKV dropout.
        dropout_attn_matrix : float
            Attention matrix dropout.
        attn_factor : int
            ProbSparse factor.
        performer_attn_kernel : str
            Performer kernel.
        performer_redraw_interval : int
            Performer redraw interval.

        Returns
        -------
        AttentionLayer or None
            Attention layer or None if attn_str == 'none'.

        Raises
        ------
        ValueError
            If attn_str is not recognized.
        """
        if attn_str == "full":
            # standard full (n^2) attention
            Attn = AttentionLayer(
                attention=partial(FullAttention, attention_dropout=dropout_attn_matrix),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                mix=False,
                dropout_qkv=dropout_qkv,
            )
        elif attn_str == "prob":
            # Informer-style ProbSparse cross attention
            Attn = AttentionLayer(
                attention=partial(
                    ProbAttention,
                    factor=attn_factor,
                    attention_dropout=dropout_attn_matrix,
                ),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                mix=False,
                dropout_qkv=dropout_qkv,
            )
        elif attn_str == "performer":
            # Performer Linear Attention
            Attn = AttentionLayer(
                attention=partial(
                    PerformerAttention,
                    dim_heads=d_qk,
                    kernel=performer_attn_kernel,
                    feature_redraw_interval=performer_redraw_interval,
                ),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                mix=False,
                dropout_qkv=dropout_qkv,
            )
        elif attn_str == "none":
            Attn = None
        else:
            raise ValueError(f"Unrecognized attention str code '{attn_str}'")
        return Attn
