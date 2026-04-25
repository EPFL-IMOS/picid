import torch
import torch.nn as nn
from picid.model.forecasters.shared_layers.Embed_DLinear import (
    ContextEmbedding,
    ContextEmbedding_wo_context,
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_pos_temp,
    DataEmbedding_wo_temp,
)
from picid.model.forecasters.shared_layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
)
from picid.model.forecasters.shared_layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
)


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity.

    Supports encoder-decoder or decoder-only modes. Embedding type is selected
    via configs.embed_type: 0=DataEmbedding, 1=DataEmbedding, 2=DataEmbedding_wo_pos,
    3=DataEmbedding_wo_temp, 4=DataEmbedding_wo_pos_temp, 5=ContextEmbedding,
    6=ContextEmbedding (encoder) + ContextEmbedding_wo_context (decoder).

    Parameters
    ----------
    configs : object
        Configuration with pred_len, output_attention, d_model, enc_in, dec_in,
        enc_context_in, dec_context_in, embed_type, embed, freq, dropout,
        n_heads, d_ff, e_layers, d_layers, activation, factor, c_out,
        d_x_embedding_router (optional).
    decoder_only : bool, optional
        If True, use decoder-only mode (no encoder). Defaults to False.

    Attributes
    ----------
    pred_len : int
        Prediction length.
    d_model : int
        Model dimension.
    decoder_only : bool
        Whether the model is decoder-only.
    """

    def __init__(self, configs, decoder_only: bool = False):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.decoder_only = decoder_only
        self.d_model = configs.d_model

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.dec_embedding = DataEmbedding(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.dec_embedding = DataEmbedding(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.dec_embedding = DataEmbedding_wo_pos(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.dec_embedding = DataEmbedding_wo_temp(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.dec_embedding = DataEmbedding_wo_pos_temp(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
        elif configs.embed_type == 5:
            self.enc_embedding = ContextEmbedding(
                configs.enc_in,
                configs.enc_context_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
                configs.d_x_embedding_router,
            )
            self.dec_embedding = ContextEmbedding(
                configs.dec_in,
                configs.dec_context_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
                configs.d_x_embedding_router,
            )
        elif configs.embed_type == 6:
            self.enc_embedding = ContextEmbedding(
                configs.enc_in,
                configs.enc_context_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
                configs.d_x_embedding_router,
            )
            self.dec_embedding = ContextEmbedding_wo_context(
                configs.dec_in,
                configs.dec_context_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
                configs.d_x_embedding_router,
            )

        if not decoder_only:
            # Encoder
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                False,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=configs.output_attention,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.e_layers)  # noqa: E741
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
            )
        else:
            self.enc_embedding = None

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)  # noqa: E741
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        """
        Forward pass.

        Parameters
        ----------
        x_enc : torch.Tensor or None
            Encoder input of shape (B, L_enc, enc_in). None if decoder_only.
        x_mark_enc : torch.Tensor or None
            Encoder temporal/context features (B, L_enc, D). None if decoder_only.
        x_dec : torch.Tensor
            Decoder input of shape (B, L_dec, dec_in).
        x_mark_dec : torch.Tensor or None
            Decoder temporal/context features (B, L_dec, D). None for some tasks.
        enc_self_mask : torch.Tensor, optional
            Encoder self-attention mask.
        dec_self_mask : torch.Tensor, optional
            Decoder self-attention mask.
        dec_enc_mask : torch.Tensor, optional
            Decoder-encoder cross-attention mask.

        Returns
        -------
        torch.Tensor or tuple
            If output_attention and not decoder_only: (pred, attns).
            Else: pred of shape (B, pred_len, c_out).
        """
        if not self.decoder_only:
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        else:
            enc_out = torch.zeros((x_dec.shape[0], x_dec.shape[1], self.d_model)).to(
                x_dec.device
            )

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )

        if self.output_attention and not self.decoder_only:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
