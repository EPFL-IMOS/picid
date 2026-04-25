"""Tests for spacetimeformer Embedding layer."""

import torch

from picid.model.forecasters.spacetimeformer_model.nn.embed import Embedding


class TestEmbeddingTemporal:
    """Tests for temporal embedding method."""

    def test_temporal_embed_basic(self):
        emb = Embedding(
            d_y=2,
            d_x=4,
            d_model=16,
            method="temporal",
            max_seq_len=64,
            position_emb="abs",
            time_emb="t2v",
            timetable_emb=None,
            is_encoder=True,
        )
        B, L = 2, 16
        y = torch.randn(B, L, 2)
        x = torch.randn(B, L, 4)
        result = emb(x, y)
        assert len(result) == 4
        out, space_emb, var_idxs, mask = result
        assert out.shape[0] == B
        assert out.ndim == 3
        assert var_idxs is None

    def test_temporal_embed_position_t2v(self):
        emb = Embedding(
            d_y=1,
            d_x=2,
            d_model=16,
            method="temporal",
            max_seq_len=64,
            position_emb="t2v",
            time_emb="t2v",
            timetable_emb=None,
            is_encoder=True,
        )
        B, L = 2, 16
        y = torch.randn(B, L, 1)
        x = torch.randn(B, L, 2)
        out, _, _, _ = emb(x, y)
        assert out.shape[0] == B

    def test_temporal_embed_use_val_false(self):
        emb = Embedding(
            d_y=2,
            d_x=4,
            d_model=16,
            method="temporal",
            max_seq_len=64,
            position_emb="abs",
            timetable_emb=None,
            use_val=False,
            is_encoder=True,
        )
        B, L = 2, 16
        y = torch.randn(B, L, 2)
        x = torch.randn(B, L, 4)
        out, _, _, _ = emb(x, y)
        assert out.shape[0] == B

    def test_temporal_embed_use_time_false(self):
        emb = Embedding(
            d_y=2,
            d_x=4,
            d_model=16,
            method="temporal",
            max_seq_len=64,
            position_emb="abs",
            timetable_emb=None,
            use_time=False,
            is_encoder=True,
        )
        B, L = 2, 16
        y = torch.randn(B, L, 2)
        x = torch.randn(B, L, 4)
        out, _, _, _ = emb(x, y)
        assert out.shape[0] == B

    def test_temporal_embed_decoder_use_given(self):
        emb = Embedding(
            d_y=2,
            d_x=4,
            d_model=16,
            method="temporal",
            max_seq_len=64,
            position_emb="abs",
            timetable_emb=None,
            is_encoder=False,
            start_token_len=4,
            use_given=True,
        )
        B, L = 2, 16
        y = torch.randn(B, L, 2)
        x = torch.randn(B, L, 4)
        out, _, _, _ = emb(x, y)
        assert out.shape[0] == B


class TestEmbeddingSpatioTemporal:
    """Tests for spatio-temporal embedding method."""

    def test_spatio_temporal_embed_basic(self):
        emb = Embedding(
            d_y=4,
            d_x=2,
            d_model=16,
            method="spatio-temporal",
            max_seq_len=64,
            position_emb="abs",
            time_emb="t2v",
            timetable_emb=None,
            is_encoder=True,
        )
        B, L, dy = 2, 16, 4
        y = torch.randn(B, L, dy)
        x = torch.randn(B, L, 2)
        out, space_emb, var_idxs, mask = emb(x, y)
        assert out.shape[0] == B
        assert var_idxs is not None

    def test_spatio_temporal_embed_use_space_false(self):
        emb = Embedding(
            d_y=4,
            d_x=2,
            d_model=16,
            method="spatio-temporal",
            max_seq_len=64,
            position_emb="abs",
            timetable_emb=None,
            use_space=False,
            is_encoder=True,
        )
        B, L = 2, 16
        y = torch.randn(B, L, 4)
        x = torch.randn(B, L, 2)
        out, _, _, _ = emb(x, y)
        assert out.shape[0] == B

    def test_spatio_temporal_embed_use_given_false(self):
        emb = Embedding(
            d_y=4,
            d_x=2,
            d_model=16,
            method="spatio-temporal",
            max_seq_len=64,
            position_emb="abs",
            timetable_emb=None,
            use_given=False,
            is_encoder=True,
        )
        B, L = 2, 16
        y = torch.randn(B, L, 4)
        x = torch.randn(B, L, 2)
        out, _, _, _ = emb(x, y)
        assert out.shape[0] == B

    def test_spatio_temporal_embed_pad_value(self):
        emb = Embedding(
            d_y=4,
            d_x=2,
            d_model=16,
            method="spatio-temporal",
            max_seq_len=64,
            position_emb="abs",
            timetable_emb=None,
            pad_value=0.0,
            is_encoder=True,
        )
        B, L = 2, 16
        y = torch.randn(B, L, 4)
        x = torch.randn(B, L, 2)
        out, _, _, mask = emb(x, y)
        assert out.shape[0] == B

    def test_spatio_temporal_embed_with_router_and_linear_components(self):
        emb = Embedding(
            d_y=3,
            d_x=4,
            d_model=16,
            method="spatio-temporal",
            max_seq_len=64,
            position_emb="abs",
            time_emb="linear",
            timetable_emb="linear",
            variable_emb="linear",
            embedding_router=torch.tensor([0, 0, 1, 1]),
            use_timetable=False,
            null_value=0.0,
            downsample_convs=1,
            is_encoder=True,
        )
        B, L = 2, 8
        y = torch.randn(B, L, 3)
        y[:, 0, 0] = 0.0
        x = torch.randn(B, L, 4)
        out, _, var_idxs, mask = emb(x, y)

        assert out.shape[0] == B
        assert out.shape[1] < L * 3
        assert var_idxs is not None
        assert mask is None or mask.shape[0] == B

    def test_spatio_temporal_embed_router_without_timetable_embedding(self):
        emb = Embedding(
            d_y=2,
            d_x=4,
            d_model=8,
            method="spatio-temporal",
            max_seq_len=32,
            position_emb="abs",
            time_emb="linear",
            timetable_emb=None,
            embedding_router=torch.tensor([0, 0, 1, 1]),
            is_encoder=True,
        )
        assert emb.timetable_emb is None
        assert emb.embedding_router is not None


class TestEmbeddingInverted:
    """Tests for inverted embedding method (rearranges b l n -> b n l)."""

    def test_inverted_embed_basic(self):
        emb = Embedding(
            d_y=16,
            d_x=16,
            d_model=16,
            method="inverted",
            time_emb="t2v",
            timetable_emb=None,
            variable_emb=None,
            position_emb="t2v",
            is_encoder=True,
        )
        B, L, dy = 2, 16, 4
        y = torch.randn(B, L, dy)
        x = torch.randn(B, L, dy)
        out, space_emb, _, _ = emb(x, y)
        assert out.shape[0] == B

    def test_temporal_embed_with_variable_time2vec_projection(self):
        emb = Embedding(
            d_y=2,
            d_x=2,
            d_model=8,
            method="temporal",
            max_seq_len=32,
            position_emb="abs",
            time_emb="linear",
            timetable_emb=None,
            variable_emb="t2v",
            is_encoder=True,
        )
        y = torch.randn(2, 6, 2)
        x = torch.randn(2, 6, 2)
        out, _, _, _ = emb(x, y)
        assert out.shape[0] == 2
