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
                self.enc_embedding = IEmbedding(
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
