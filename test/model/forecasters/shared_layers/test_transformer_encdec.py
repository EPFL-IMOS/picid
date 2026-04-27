"""Tests for shared-layer Transformer encoder/decoder blocks."""

import torch
from picid.model.forecasters.shared_layers.Transformer_EncDec import (
    ConvLayer,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
)
from picid.model.forecasters.shared_layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
)


def _encoder_layer(d_model=16, d_ff=32, n_heads=2, output_attention=False):
    attn = AttentionLayer(
        FullAttention(
            False, factor=3, attention_dropout=0.0, output_attention=output_attention
        ),
        d_model,
        n_heads,
    )
    return EncoderLayer(attn, d_model, d_ff, dropout=0.0, activation="gelu")


def test_conv_layer_downsamples_sequence_length():
    layer = ConvLayer(c_in=4)
    x = torch.randn(2, 8, 4)

    out = layer(x)

    assert out.shape == (2, 5, 4)


def test_encoder_forward():
    enc = Encoder([_encoder_layer()], norm_layer=torch.nn.LayerNorm(16))
    x = torch.randn(2, 10, 16)
    out, attns = enc(x)
    assert out.shape == x.shape
    assert len(attns) == 1


def test_encoder_with_conv_layer_path_collects_attention_from_each_stage():
    enc = Encoder(
        [_encoder_layer(output_attention=True), _encoder_layer(output_attention=True)],
        conv_layers=[ConvLayer(16)],
        norm_layer=torch.nn.LayerNorm(16),
    )
    x = torch.randn(2, 10, 16)

    out, attns = enc(x)

    assert out.shape == (2, 6, 16)
    assert len(attns) == 2
    assert all(attn is not None for attn in attns)


def test_decoder_forward():
    dec_layer = DecoderLayer(
        AttentionLayer(
            FullAttention(
                True, factor=3, attention_dropout=0.0, output_attention=False
            ),
            16,
            2,
        ),
        AttentionLayer(
            FullAttention(
                False, factor=3, attention_dropout=0.0, output_attention=False
            ),
            16,
            2,
        ),
        16,
        32,
        dropout=0.0,
        activation="gelu",
    )
    dec = Decoder([dec_layer], norm_layer=torch.nn.LayerNorm(16))
    x = torch.randn(2, 10, 16)
    enc_out = torch.randn(2, 10, 16)
    out = dec(x, enc_out)
    assert out.shape == x.shape


def test_decoder_applies_optional_projection():
    dec_layer = DecoderLayer(
        AttentionLayer(
            FullAttention(
                True, factor=3, attention_dropout=0.0, output_attention=False
            ),
            16,
            2,
        ),
        AttentionLayer(
            FullAttention(
                False, factor=3, attention_dropout=0.0, output_attention=False
            ),
            16,
            2,
        ),
        16,
        32,
        dropout=0.0,
    )
    dec = Decoder(
        [dec_layer],
        norm_layer=torch.nn.LayerNorm(16),
        projection=torch.nn.Linear(16, 3),
    )
    x = torch.randn(2, 10, 16)
    enc_out = torch.randn(2, 10, 16)

    out = dec(x, enc_out)

    assert out.shape == (2, 10, 3)
