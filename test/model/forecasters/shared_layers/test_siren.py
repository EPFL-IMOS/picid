"""Tests for picid.model.forecasters.shared_layers.siren."""

import pytest
import torch
import torch.nn as nn

from picid.model.forecasters.shared_layers.siren import (
    Modulator,
    Sine,
    Siren,
    SirenNet,
    SirenWrapper,
    cast_tuple,
    exists,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_exists_none_returns_false(self):
        assert exists(None) is False

    def test_exists_zero_returns_true(self):
        assert exists(0) is True

    def test_exists_empty_string_returns_true(self):
        assert exists("") is True

    def test_cast_tuple_with_tuple_unchanged(self):
        t = (1, 2, 3)
        assert cast_tuple(t, repeat=5) is t

    def test_cast_tuple_non_tuple_repeated(self):
        assert cast_tuple(7, repeat=3) == (7, 7, 7)

    def test_cast_tuple_none_repeated(self):
        assert cast_tuple(None, repeat=2) == (None, None)


# ---------------------------------------------------------------------------
# Sine activation
# ---------------------------------------------------------------------------


class TestSine:
    def test_forward_shape_preserved(self):
        sine = Sine(w0=2.0)
        x = torch.randn(3, 4)
        out = sine(x)
        assert out.shape == x.shape

    def test_forward_zero_input(self):
        sine = Sine(w0=1.0)
        assert sine(torch.zeros(1)).item() == pytest.approx(0.0)

    def test_w0_scaling(self):
        """sin(w0 * x) ≠ sin(x) for w0 ≠ 1."""
        x = torch.tensor([1.0])
        assert Sine(w0=2.0)(x).item() != pytest.approx(Sine(w0=1.0)(x).item())


# ---------------------------------------------------------------------------
# Siren layer
# ---------------------------------------------------------------------------


class TestSiren:
    def test_forward_with_bias(self):
        layer = Siren(dim_in=4, dim_out=8, use_bias=True)
        x = torch.randn(3, 4)
        assert layer(x).shape == (3, 8)

    def test_forward_without_bias(self):
        layer = Siren(dim_in=4, dim_out=8, use_bias=False)
        assert layer.bias is None
        assert layer(torch.randn(3, 4)).shape == (3, 8)

    def test_is_first_init(self):
        layer = Siren(dim_in=8, dim_out=4, is_first=True)
        assert layer.is_first is True

    def test_non_first_init(self):
        layer = Siren(dim_in=8, dim_out=4, is_first=False)
        assert layer.is_first is False

    def test_custom_activation(self):
        layer = Siren(dim_in=4, dim_out=4, activation=nn.Tanh())
        assert layer(torch.randn(2, 4)).shape == (2, 4)

    def test_dropout_zero_deterministic(self):
        layer = Siren(dim_in=4, dim_out=4, dropout=0.0)
        layer.eval()
        x = torch.randn(2, 4)
        torch.testing.assert_close(layer(x), layer(x))

    def test_bias_uniform_initialised(self):
        """Bias values should be non-zero after init (uniform ≠ 0 almost surely)."""
        layer = Siren(dim_in=16, dim_out=8, use_bias=True, is_first=False)
        assert layer.bias is not None


# ---------------------------------------------------------------------------
# SirenNet
# ---------------------------------------------------------------------------


class TestSirenNet:
    def test_forward_without_mods(self):
        net = SirenNet(dim_in=2, dim_hidden=8, dim_out=3, num_layers=2)
        out = net(torch.randn(4, 2))
        assert out.shape == (4, 3)

    def test_forward_with_mods(self):
        net = SirenNet(dim_in=2, dim_hidden=8, dim_out=3, num_layers=2)
        mods = (torch.randn(8), torch.randn(8))
        out = net(torch.randn(4, 2), mods=mods)
        assert out.shape == (4, 3)

    def test_final_activation_identity_by_default(self):
        net = SirenNet(dim_in=2, dim_hidden=8, dim_out=3, num_layers=2)
        assert isinstance(net.last_layer.activation, nn.Identity)

    def test_custom_final_activation_relu(self):
        net = SirenNet(
            dim_in=2,
            dim_hidden=8,
            dim_out=3,
            num_layers=2,
            final_activation=nn.ReLU(),
        )
        out = net(torch.randn(4, 2))
        assert (out >= 0).all()

    def test_num_layers_creates_correct_count(self):
        net = SirenNet(dim_in=2, dim_hidden=8, dim_out=3, num_layers=4)
        assert len(net.layers) == 4

    def test_w0_initial_applied_to_first_layer(self):
        """First layer uses w0_initial; others use w0."""
        net = SirenNet(
            dim_in=2, dim_hidden=8, dim_out=3, num_layers=2, w0=1.0, w0_initial=30.0
        )
        # first layer's activation should have w0=30.0
        assert net.layers[0].activation.w0 == pytest.approx(30.0)
        assert net.layers[1].activation.w0 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Modulator
# ---------------------------------------------------------------------------


class TestModulator:
    def test_forward_returns_tuple_of_hiddens(self):
        mod = Modulator(dim_in=4, dim_hidden=6, num_layers=3)
        result = mod(torch.randn(4))
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_hidden_shapes(self):
        mod = Modulator(dim_in=4, dim_hidden=6, num_layers=2)
        hiddens = mod(torch.randn(4))
        assert hiddens[0].shape == (6,)
        assert hiddens[1].shape == (6,)

    def test_single_layer(self):
        mod = Modulator(dim_in=4, dim_hidden=8, num_layers=1)
        result = mod(torch.randn(4))
        assert len(result) == 1
        assert result[0].shape == (8,)


# ---------------------------------------------------------------------------
# SirenWrapper
# ---------------------------------------------------------------------------


class TestSirenWrapper:
    def _net(self):
        return SirenNet(dim_in=2, dim_hidden=8, dim_out=3, num_layers=2)

    def test_forward_without_latent_returns_image_tensor(self):
        wrapper = SirenWrapper(self._net(), image_width=4, image_height=4)
        out = wrapper()
        assert out.shape == (1, 3, 4, 4)

    def test_forward_with_latent_returns_image_tensor(self):
        wrapper = SirenWrapper(self._net(), image_width=4, image_height=4, latent_dim=6)
        out = wrapper(latent=torch.randn(6))
        assert out.shape == (1, 3, 4, 4)

    def test_forward_with_img_and_latent_returns_scalar_loss(self):
        wrapper = SirenWrapper(self._net(), image_width=4, image_height=4, latent_dim=6)
        img = torch.randn(1, 3, 4, 4)
        loss = wrapper(img=img, latent=torch.randn(6))
        assert loss.shape == ()  # scalar MSE loss

    def test_wrong_net_type_raises_assertion(self):
        with pytest.raises(AssertionError):
            SirenWrapper(nn.Linear(2, 3), image_width=4, image_height=4)

    def test_modulate_without_latent_raises(self):
        """latent_dim set but latent not supplied → AssertionError."""
        wrapper = SirenWrapper(self._net(), image_width=4, image_height=4, latent_dim=6)
        with pytest.raises(AssertionError):
            wrapper()  # modulator exists but latent=None

    def test_no_modulator_with_latent_raises(self):
        """No latent_dim set but latent supplied → AssertionError."""
        wrapper = SirenWrapper(self._net(), image_width=4, image_height=4)
        with pytest.raises(AssertionError):
            wrapper(latent=torch.randn(6))

    def test_grid_registered_as_buffer(self):
        wrapper = SirenWrapper(self._net(), image_width=4, image_height=4)
        assert hasattr(wrapper, "grid")
        assert wrapper.grid.shape == (16, 2)  # 4*4 pixels, 2D coords
