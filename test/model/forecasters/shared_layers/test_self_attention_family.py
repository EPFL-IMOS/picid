"""Tests for shared attention primitives used by baseline forecasters."""

import torch

from picid.model.forecasters.shared_layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
    ProbAttention,
)


def test_full_attention_returns_attention_weights_when_requested():
    attention = FullAttention(
        mask_flag=False,
        attention_dropout=0.0,
        output_attention=True,
    )
    queries = torch.randn(2, 4, 2, 8)
    keys = torch.randn(2, 4, 2, 8)
    values = torch.randn(2, 4, 2, 8)

    output, attn = attention(queries, keys, values, attn_mask=None)

    assert output.shape == values.shape
    assert attn is not None
    assert attn.shape == (2, 2, 4, 4)


def test_full_attention_applies_causal_mask():
    attention = FullAttention(
        mask_flag=True,
        attention_dropout=0.0,
        output_attention=True,
    )
    queries = torch.ones(1, 3, 1, 2)
    keys = torch.ones(1, 3, 1, 2)
    values = torch.arange(6, dtype=torch.float32).view(1, 3, 1, 2)

    output, attn = attention(queries, keys, values, attn_mask=None)

    assert output.shape == values.shape
    assert attn is not None
    assert torch.equal(attn[0, 0].triu(diagonal=1), torch.zeros(3, 3))


def test_prob_attention_returns_attention_matrix_when_requested():
    torch.manual_seed(0)
    attention = ProbAttention(
        mask_flag=False,
        factor=1,
        attention_dropout=0.0,
        output_attention=True,
    )
    queries = torch.randn(2, 4, 2, 8)
    keys = torch.randn(2, 4, 2, 8)
    values = torch.randn(2, 4, 2, 8)

    output, attn = attention(queries, keys, values, attn_mask=None)

    assert output.shape == (2, 2, 4, 8)
    assert attn is not None
    assert attn.shape == (2, 2, 4, 4)


def test_prob_attention_respects_masked_self_attention_path():
    torch.manual_seed(0)
    attention = ProbAttention(
        mask_flag=True,
        factor=1,
        attention_dropout=0.0,
        output_attention=False,
    )
    queries = torch.randn(1, 4, 1, 4)
    keys = torch.randn(1, 4, 1, 4)
    values = torch.randn(1, 4, 1, 4)

    output, attn = attention(queries, keys, values, attn_mask=None)

    assert output.shape == (1, 1, 4, 4)
    assert attn is None


def test_attention_layer_projects_queries_and_returns_attention():
    torch.manual_seed(0)
    layer = AttentionLayer(
        FullAttention(mask_flag=False, attention_dropout=0.0, output_attention=True),
        d_model=8,
        n_heads=2,
    )
    queries = torch.randn(2, 5, 8)
    keys = torch.randn(2, 5, 8)
    values = torch.randn(2, 5, 8)

    output, attn = layer(queries, keys, values, attn_mask=None)

    assert output.shape == queries.shape
    assert attn is not None
    assert attn.shape == (2, 2, 5, 5)
