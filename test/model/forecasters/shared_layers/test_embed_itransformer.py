"""Tests for picid.model.forecasters.shared_layers.Embed_iTransformer."""

import pytest
import torch

from picid.model.forecasters.shared_layers.Embed_iTransformer import (
    ContextEmbedding,
    DataEmbedding,
    DataEmbedding_inverted,
    FixedEmbedding,
    PositionalEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
    TokenEmbedding,
)

B, T, D_MODEL = 2, 10, 16


# ---------------------------------------------------------------------------
# PositionalEmbedding
# ---------------------------------------------------------------------------


class TestPositionalEmbedding:
    def test_output_shape(self):
        emb = PositionalEmbedding(d_model=D_MODEL)
        x = torch.randn(B, T, D_MODEL)
        out = emb(x)
        assert out.shape == (1, T, D_MODEL)

    def test_output_independent_of_batch_values(self):
        """PE depends only on sequence length, not on x values."""
        emb = PositionalEmbedding(d_model=D_MODEL)
        x1, x2 = torch.randn(B, T, D_MODEL), torch.zeros(B, T, D_MODEL)
        torch.testing.assert_close(emb(x1), emb(x2))

    def test_different_sequence_lengths(self):
        emb = PositionalEmbedding(d_model=D_MODEL)
        for length in [1, 5, T]:
            out = emb(torch.randn(1, length, D_MODEL))
            assert out.shape == (1, length, D_MODEL)


# ---------------------------------------------------------------------------
# TokenEmbedding
# ---------------------------------------------------------------------------


class TestTokenEmbedding:
    def test_output_shape(self):
        emb = TokenEmbedding(c_in=4, d_model=D_MODEL)
        out = emb(torch.randn(B, T, 4))
        assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# FixedEmbedding
# ---------------------------------------------------------------------------


class TestFixedEmbedding:
    def test_output_shape(self):
        emb = FixedEmbedding(c_in=8, d_model=D_MODEL)
        # indices must be in [0, c_in)
        x = torch.randint(0, 8, (B, T))
        out = emb(x)
        assert out.shape == (B, T, D_MODEL)

    def test_weights_not_updated_grad(self):
        emb = FixedEmbedding(c_in=8, d_model=D_MODEL)
        assert emb.emb.weight.requires_grad is False


# ---------------------------------------------------------------------------
# TemporalEmbedding
# ---------------------------------------------------------------------------


def _make_temporal_input(batch, time, n_cols, max_vals):
    """Build a long-integer input tensor for TemporalEmbedding."""
    parts = [torch.randint(0, mv, (batch, time)).unsqueeze(-1) for mv in max_vals]
    return torch.cat(parts, dim=-1).long()


class TestTemporalEmbedding:
    def test_forward_fixed_no_minute(self):
        """freq='h' → no minute_embed, 4 columns needed."""
        emb = TemporalEmbedding(d_model=D_MODEL, embed_type="fixed", freq="h")
        # x columns: month(0..12), day(0..31), weekday(0..6), hour(0..23)
        x = _make_temporal_input(B, T, 4, [13, 32, 7, 24])
        out = emb(x)
        assert out.shape == (B, T, D_MODEL)

    def test_forward_fixed_with_minute(self):
        """freq='t' → minute_embed created and used."""
        emb = TemporalEmbedding(d_model=D_MODEL, embed_type="fixed", freq="t")
        # x columns: month, day, weekday, hour, minute(0..3)
        x = _make_temporal_input(B, T, 5, [13, 32, 7, 24, 4])
        out = emb(x)
        assert out.shape == (B, T, D_MODEL)
        assert hasattr(emb, "minute_embed")

    def test_forward_learnable_embedding(self):
        """embed_type != 'fixed' → nn.Embedding used instead of FixedEmbedding."""
        emb = TemporalEmbedding(d_model=D_MODEL, embed_type="learned", freq="h")
        x = _make_temporal_input(B, T, 4, [13, 32, 7, 24])
        out = emb(x)
        assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# TimeFeatureEmbedding
# ---------------------------------------------------------------------------


class TestTimeFeatureEmbedding:
    @pytest.mark.parametrize("freq,d_inp", [("h", 4), ("t", 5), ("s", 6), ("m", 1)])
    def test_output_shape_per_freq(self, freq, d_inp):
        emb = TimeFeatureEmbedding(d_model=D_MODEL, embed_type="timeF", freq=freq)
        x = torch.randn(B, T, d_inp)
        assert emb(x).shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# ContextEmbedding  (suppresses its WARNING print via capsys)
# ---------------------------------------------------------------------------


class TestContextEmbedding:
    def test_forward_shape_with_timef(self, capsys):
        """timeF + 4 time cols + 3 context cols → [B, T, D_MODEL]."""
        # D_C=7: last 4 are time features, first 3 are context
        c_in, c_context, D_C = 8, 3, 7
        emb = ContextEmbedding(
            c_in=c_in,
            c_context=c_context,
            d_model=D_MODEL,
            embed_type="timeF",
            freq="h",
        )
        x = torch.randn(B, T, c_in)
        x_mark = torch.randn(B, T, D_C)
        out = emb(x, x_mark)
        assert out.shape == (B, T, D_MODEL)

    def test_forward_shape_with_fixed(self, capsys):
        """embed_type='fixed' routes through TemporalEmbedding.

        ContextEmbedding.forward does ``time_features.flip(2)`` before passing
        to TemporalEmbedding, so the input order must be the *reversed* of
        [month, day, weekday, hour] → i.e. [hour, weekday, day, month].
        """
        c_in, c_context = 8, 3
        emb = ContextEmbedding(
            c_in=c_in,
            c_context=c_context,
            d_model=D_MODEL,
            embed_type="fixed",
            freq="h",
        )
        x = torch.randn(B, T, c_in)
        # last 4 cols: [hour(0-23), weekday(0-6), day(0-31), month(0-12)]
        # after flip → [month, day, weekday, hour] which TemporalEmbedding expects
        context_part = torch.randn(B, T, 3)
        time_part = _make_temporal_input(B, T, 4, [24, 7, 32, 13]).float()
        x_mark = torch.cat([context_part, time_part], dim=-1)
        out = emb(x, x_mark)
        assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# DataEmbedding
# ---------------------------------------------------------------------------


class TestDataEmbedding:
    def test_forward_with_x_mark_timef(self):
        emb = DataEmbedding(c_in=4, d_model=D_MODEL, embed_type="timeF", freq="h")
        x = torch.randn(B, T, 4)
        x_mark = torch.randn(B, T, 4)
        out = emb(x, x_mark)
        assert out.shape == (B, T, D_MODEL)

    def test_forward_with_x_mark_fixed(self):
        emb = DataEmbedding(c_in=4, d_model=D_MODEL, embed_type="fixed", freq="h")
        x = torch.randn(B, T, 4)
        # TemporalEmbedding expects long integer indices
        x_mark = _make_temporal_input(B, T, 4, [13, 32, 7, 24]).float()
        out = emb(x, x_mark)
        assert out.shape == (B, T, D_MODEL)

    def test_forward_without_x_mark(self):
        """x_mark=None → only value + position embeddings used."""
        emb = DataEmbedding(c_in=4, d_model=D_MODEL)
        out = emb(torch.randn(B, T, 4), x_mark=None)
        assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# DataEmbedding_inverted
# ---------------------------------------------------------------------------


class TestDataEmbeddingInverted:
    def test_forward_without_x_mark(self):
        """Inverted embedding: c_in = sequence length (time steps)."""
        c_in = T  # maps T→d_model for each variate
        n_var = 4
        emb = DataEmbedding_inverted(c_in=c_in, d_model=D_MODEL)
        x = torch.randn(B, T, n_var)
        out = emb(x, x_mark=None)
        # output: [B, n_var, D_MODEL]
        assert out.shape == (B, n_var, D_MODEL)

    def test_forward_with_x_mark(self):
        """With x_mark: variates and mark columns concatenated before linear."""
        c_in = T
        n_var = 4
        d_mark = 2
        emb = DataEmbedding_inverted(c_in=c_in, d_model=D_MODEL)
        x = torch.randn(B, T, n_var)
        x_mark = torch.randn(B, T, d_mark)
        out = emb(x, x_mark)
        # output: [B, n_var + d_mark, D_MODEL]
        assert out.shape == (B, n_var + d_mark, D_MODEL)
