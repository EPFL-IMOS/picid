import numpy as np
import pytest

from lmetk.transforms.base_transforms.time_statistics import TimeStatsTransform


def test_transform_data_with_hankel_svd():
    """Test that hankel_svd statistic is computed correctly."""
    t = TimeStatsTransform(stats_to_compute=["hankel_svd"], hankel_window_size=10)

    # Simple sine wave signal
    n = 128
    signal = np.sin(np.linspace(0, 4 * np.pi, n))
    data = {"features": signal.reshape(-1, 1)}

    result = t.transform_data(data, metadata={})

    assert result.shape == (10,)
    for i in range(10):
        assert np.isfinite(result[i])
        assert result[i] >= 0  # SVD value should be non-negative


def test_transform_data_mean():
    """Test that hankel_svd statistic is computed correctly."""
    t = TimeStatsTransform(stats_to_compute=["mean"])

    # Simple sine wave signal
    n = 128
    signal = np.sin(np.linspace(0, 4 * np.pi, n))
    data = {"features": signal.reshape(-1, 1)}

    result = t.transform_data(data, metadata={})

    assert result.shape == (1,)
    assert np.isfinite(result[0])


def test_transform_data_mean_and_hankel_svd():
    """Test that both mean and hankel_svd statistics are computed correctly."""
    t = TimeStatsTransform(
        stats_to_compute=["mean", "hankel_svd"], hankel_window_size=10
    )

    # Simple sine wave signal
    n = 128
    signal = np.sin(np.linspace(0, 4 * np.pi, n))
    data = {"features": signal.reshape(-1, 1)}

    result = t.transform_data(data, metadata={})

    assert result.shape == (11,)  # 1 mean + 10 hankel_svd values
    for i in range(11):
        assert np.isfinite(result[i])
