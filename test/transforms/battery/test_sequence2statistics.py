import numpy as np
import pytest

from picid.data.data_objects import NamedTransformInput
from picid.transforms.battery.sequence2statistics import Sequence2Statistics


def test_sequence2statistics_2d_input_reshaped_to_4d():
    """2D input gets unsqueezed to 4D (N_cycles=1, N_win=1, win_len, channels)."""
    transform = Sequence2Statistics()

    # 2D: (win_len=5, channels=3) -> becomes (1, 1, 5, 3)
    features = np.random.randn(5, 3)
    data = NamedTransformInput(features=features)
    metadata = {}

    result = transform.transform_data(data, metadata)

    assert result.ndim == 1
    # mean (3,) + std (3,) concatenated -> (6,)
    assert result.shape[0] == 6


def test_sequence2statistics_4d_input():
    """4D input (N_cycles, N_win, win_len, channels) produces correct shape."""
    transform = Sequence2Statistics()

    # (N_cycles=1, N_win=1, win_len=3, channels=2)
    features = np.array([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])
    data = NamedTransformInput(features=features)
    metadata = {}

    result = transform.transform_data(data, metadata)

    expected_mean = np.array([3.0, 4.0])  # mean over axis (1,2) -> per channel
    expected_std = np.array([np.std([1, 3, 5]), np.std([2, 4, 6])])
    expected = np.concatenate([expected_mean, expected_std])
    np.testing.assert_allclose(result, expected)


def test_sequence2statistics_with_padding_value():
    """Padding value is ignored in mean/std computation."""
    transform = Sequence2Statistics(padding_value=-999.0)

    # 4D: (1, 1, 3, 2), -999 as padding
    features = np.array([[[[1.0, 2.0], [-999.0, -999.0], [3.0, 4.0]]]])
    data = NamedTransformInput(features=features)
    metadata = {}

    result = transform.transform_data(data, metadata)

    # Mean of [1,3] and [2,4] = [2, 3]; std of [1,3] and [2,4]
    expected_mean = np.array([2.0, 3.0])
    expected_std = np.array([1.0, 1.0])
    expected = np.concatenate([expected_mean, expected_std])
    np.testing.assert_allclose(result, expected)


def test_sequence2statistics_requires_single_key():
    with pytest.raises(AssertionError, match="exactly one entry"):
        transform = Sequence2Statistics()
        data = NamedTransformInput(
            features=np.array([[[1.0]]]),
            target=np.array([[[1.0]]]),
        )
        transform.transform_data(data, {})


def test_sequence2statistics_fit_data_no_op():
    """fit_data is stateless and does nothing."""
    transform = Sequence2Statistics()
    data = NamedTransformInput(features=np.array([[[1.0, 2.0]]]))
    transform.fit_data(data, {})
