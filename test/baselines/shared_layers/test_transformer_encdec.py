"""Tests for Transformer_EncDec (Encoder, EncoderLayer, Decoder, DecoderLayer)."""

import torch
from picid.baselines.shared_layers.Transformer_EncDec import (
    Encoder,
    EncoderLayer,
    Decoder,
    DecoderLayer,
)
from picid.baselines.shared_layers.SelfAttention_Family import FullAttention, AttentionLayer


def _encoder_layer(d_model=16, d_ff=32, n_heads=2):
    attn = AttentionLayer(
        FullAttention(False, factor=3, attention_dropout=0.1, output_attention=False),
        d_model,
        n_heads,
    )
    return EncoderLayer(attn, d_model, d_ff, dropout=0.1, activation="gelu")


def test_encoder_forward():
    enc = Encoder([_encoder_layer()], norm_layer=torch.nn.LayerNorm(16))
    x = torch.randn(2, 10, 16)
    out, attns = enc(x)
    assert out.shape == x.shape
    assert len(attns) == 1


def test_decoder_forward():
    dec_layer = DecoderLayer(
        AttentionLayer(
            FullAttention(True, factor=3, attention_dropout=0.1, output_attention=False),
            16,
            2,
        ),
        AttentionLayer(
            FullAttention(False, factor=3, attention_dropout=0.1, output_attention=False),
            16,
            2,
        ),
        16,
        32,
        dropout=0.1,
        activation="gelu",
    )
    dec = Decoder([dec_layer], norm_layer=torch.nn.LayerNorm(16))
    x = torch.randn(2, 10, 16)
    enc_out = torch.randn(2, 10, 16)
    out = dec(x, enc_out)
    assert out.shape == x.shape
