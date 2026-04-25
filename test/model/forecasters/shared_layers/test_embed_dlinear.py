"""Tests for embedding helpers shared with DLinear-style baselines."""

import torch

from picid.model.forecasters.shared_layers.Embed_DLinear import (
    ContextEmbedding,
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_pos_temp,
    DataEmbedding_wo_temp,
    FixedEmbedding,
    PositionalEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
)


def test_positional_embedding_returns_expected_slice():
    embedding = PositionalEmbedding(d_model=8, max_len=32)
    x = torch.randn(2, 5, 8)

    out = embedding(x)

    assert out.shape == (1, 5, 8)


def test_fixed_embedding_is_frozen_and_returns_expected_shape():
    embedding = FixedEmbedding(c_in=10, d_model=6)
    x = torch.tensor([[0, 1, 2], [3, 4, 5]])

    out = embedding(x)

    assert out.shape == (2, 3, 6)
    assert embedding.emb.weight.requires_grad is False


def test_temporal_embedding_supports_minute_frequency():
    embedding = TemporalEmbedding(d_model=8, embed_type="fixed", freq="t")
    x_mark = torch.tensor(
        [[[1, 2, 3, 4, 0], [2, 3, 4, 5, 1]]],
        dtype=torch.long,
    )

    out = embedding(x_mark)

    assert out.shape == (1, 2, 8)


def test_time_feature_embedding_projects_feature_dimension():
    embedding = TimeFeatureEmbedding(d_model=8, freq="h")
    x_mark = torch.randn(2, 6, 4)

    out = embedding(x_mark)

    assert out.shape == (2, 6, 8)


def test_data_embedding_uses_temporal_embeddings_for_fixed_and_timef_modes():
    x = torch.randn(2, 6, 3)
    fixed_marks = torch.tensor(
        [
            [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7],
                [5, 6, 0, 8],
                [6, 7, 1, 9],
            ],
            [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7],
                [5, 6, 0, 8],
                [6, 7, 1, 9],
            ],
        ],
        dtype=torch.long,
    )
    timef_marks = torch.randn(2, 6, 4)

    fixed_embedding = DataEmbedding(
        c_in=3, d_model=8, embed_type="fixed", freq="h", dropout=0.0
    )
    timef_embedding = DataEmbedding(
        c_in=3, d_model=8, embed_type="timeF", freq="h", dropout=0.0
    )

    fixed_out = fixed_embedding(x, fixed_marks)
    timef_out = timef_embedding(x, timef_marks)

    assert fixed_out.shape == (2, 6, 8)
    assert timef_out.shape == (2, 6, 8)


def test_context_embedding_handles_router_and_context_only_paths():
    x = torch.randn(2, 5, 3)
    context_only_marks = torch.randn(2, 5, 2)
    routed_marks = torch.cat(
        [
            torch.randn(2, 5, 2),
            torch.tensor(
                [
                    [
                        [1, 2, 3, 4],
                        [2, 3, 4, 5],
                        [3, 4, 5, 6],
                        [4, 5, 6, 7],
                        [5, 6, 0, 8],
                    ],
                    [
                        [1, 2, 3, 4],
                        [2, 3, 4, 5],
                        [3, 4, 5, 6],
                        [4, 5, 6, 7],
                        [5, 6, 0, 8],
                    ],
                ],
                dtype=torch.float32,
            ),
        ],
        dim=-1,
    )
    router = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.bool)

    context_only = ContextEmbedding(
        c_in=3,
        c_context=2,
        d_model=8,
        embed_type="timeF",
        freq="h",
        dropout=0.0,
        d_x_embedding_router=None,
    )
    routed = ContextEmbedding(
        c_in=3,
        c_context=2,
        d_model=8,
        embed_type="timeF",
        freq="h",
        dropout=0.0,
        d_x_embedding_router=router,
    )

    context_only_out = context_only(x, context_only_marks)
    routed_out = routed(x, routed_marks)

    assert context_only_out.shape == (2, 5, 8)
    assert routed_out.shape == (2, 5, 8)


def test_ablation_embeddings_preserve_batch_sequence_and_model_dims():
    x = torch.randn(2, 4, 3)
    timef_marks = torch.randn(2, 4, 4)

    wo_pos = DataEmbedding_wo_pos(
        c_in=3, d_model=8, embed_type="timeF", freq="h", dropout=0.0
    )
    wo_pos_temp = DataEmbedding_wo_pos_temp(
        c_in=3, d_model=8, embed_type="timeF", freq="h", dropout=0.0
    )
    wo_temp = DataEmbedding_wo_temp(
        c_in=3, d_model=8, embed_type="timeF", freq="h", dropout=0.0
    )

    assert wo_pos(x, timef_marks).shape == (2, 4, 8)
    assert wo_pos_temp(x, timef_marks).shape == (2, 4, 8)
    assert wo_temp(x, timef_marks).shape == (2, 4, 8)
