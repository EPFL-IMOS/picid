"""Tests for shared-layer PatchTST embeddings."""

import torch

from picid.model.forecasters.shared_layers.Embed_PatchTST import (
    DataEmbedding,
    DataEmbedding_inverted,
    DataEmbedding_wo_pos,
    PatchEmbedding,
    PositionalEmbedding,
)


def test_positional_embedding():
    pe = PositionalEmbedding(d_model=16, max_len=100)
    x = torch.randn(2, 50, 16)
    out = pe(x)
    assert out.shape == (1, 50, 16)


def test_data_embedding_handles_optional_temporal_marks():
    torch.manual_seed(0)
    embedding = DataEmbedding(c_in=2, d_model=8, embed_type="timeF", dropout=0.0)
    x = torch.randn(2, 6, 2)
    x_mark = torch.randn(2, 6, 4)

    without_marks = embedding(x, x_mark=None)
    with_marks = embedding(x, x_mark=x_mark)

    assert without_marks.shape == (2, 6, 8)
    assert with_marks.shape == (2, 6, 8)
    assert not torch.allclose(without_marks, with_marks)


def test_data_embedding_inverted_supports_extra_mark_features():
    embedding = DataEmbedding_inverted(c_in=6, d_model=8, dropout=0.0)
    x = torch.randn(2, 6, 3)
    x_mark = torch.randn(2, 6, 2)

    without_marks = embedding(x, x_mark=None)
    with_marks = embedding(x, x_mark=x_mark)

    assert without_marks.shape == (2, 3, 8)
    assert with_marks.shape == (2, 5, 8)


def test_data_embedding_without_positional_term_supports_both_paths():
    embedding = DataEmbedding_wo_pos(
        c_in=2, d_model=8, embed_type="fixed", freq="h", dropout=0.0
    )
    x = torch.randn(2, 6, 2)
    x_mark = torch.tensor(
        [
            [
                [1, 1, 1, 1],
                [1, 2, 2, 2],
                [1, 3, 3, 3],
                [1, 4, 4, 4],
                [1, 5, 5, 5],
                [1, 6, 6, 6],
            ],
            [
                [2, 1, 1, 1],
                [2, 2, 2, 2],
                [2, 3, 3, 3],
                [2, 4, 4, 4],
                [2, 5, 5, 5],
                [2, 6, 6, 6],
            ],
        ],
        dtype=torch.long,
    )

    without_marks = embedding(x, x_mark=None)
    with_marks = embedding(x, x_mark=x_mark)

    assert without_marks.shape == (2, 6, 8)
    assert with_marks.shape == (2, 6, 8)
    assert not torch.allclose(without_marks, with_marks)


def test_patch_embedding():
    pe = PatchEmbedding(d_model=16, patch_len=4, stride=2, padding=2, dropout=0)
    x = torch.randn(2, 4, 8)  # B, nvars, seq
    out, n_vars = pe(x)

    assert n_vars == 4
    assert out.shape == (8, 4, 16)
