import pytest
import torch

from picid.model.utils.magnitude_max_pooling import magnitude_max_pooling_1d


def test_magnitude_max_pooling_output_shape():
    B, N, L = 2, 3, 9
    pool_size, stride = 2, 1
    input_tensor = torch.randn(B, N, L)

    output = magnitude_max_pooling_1d(input_tensor, pool_size, stride)

    out_length = (L - pool_size) // stride + 1
    assert output.shape == (B, N, out_length)
    assert output.shape == (2, 3, 8)


def test_magnitude_max_pooling_selects_max_abs():
    """Each output value should be the element with max |value| in its window."""
    x = torch.tensor([[[1.0, -3.0, 2.0, 4.0, -1.0]]])  # B=1, N=1, L=5
    pool_size, stride = 2, 1

    out = magnitude_max_pooling_1d(x, pool_size, stride)

    # Window [1,-3] -> max abs is -3
    assert out[0, 0, 0].item() == pytest.approx(-3.0)
    # Window [-3,2] -> max abs is -3
    assert out[0, 0, 1].item() == pytest.approx(-3.0)
    # Window [2,4] -> max abs is 4
    assert out[0, 0, 2].item() == pytest.approx(4.0)
    # Window [4,-1] -> max abs is 4
    assert out[0, 0, 3].item() == pytest.approx(4.0)


def test_magnitude_max_pooling_stride_greater_than_one():
    B, N, L = 1, 2, 9
    pool_size, stride = 3, 2
    input_tensor = torch.randn(B, N, L)

    output = magnitude_max_pooling_1d(input_tensor, pool_size, stride)

    out_length = (L - pool_size) // stride + 1
    assert output.shape == (B, N, out_length)
    assert output.shape == (1, 2, 4)
