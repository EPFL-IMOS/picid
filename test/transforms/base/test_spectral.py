import numpy as np
import pytest
from numpy.testing import assert_array_equal
from lmetk.transforms.base_transforms.spectral import SpectralStatsTransform


def test_validate_input_raises_on_non_2d_array():
    t = SpectralStatsTransform(stats_to_compute=["mean"])
    one_d = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError) as exc:
        t._validate_input(one_d, "features")
    assert "2D" in str(exc.value)


def test_validate_input_raises_on_infinite_values():
    t = SpectralStatsTransform(stats_to_compute=["mean"])
    arr = np.array([[1.0, np.inf], [2.0, 3.0]])
    with pytest.raises(ValueError) as exc:
        t._validate_input(arr, "features")
    assert "Infinite" in str(exc.value)


def test_validate_input_raises_on_too_short_signal_length():
    t = SpectralStatsTransform(stats_to_compute=["mean"])
    # Rows == 1 -> signal length along rows is too short for FFT
    short_arr = np.ones((1, 3))
    with pytest.raises(ValueError) as exc:
        t._validate_input(short_arr, "features")
    assert "too short" in str(exc.value)


def test_validate_input_returns_array_for_valid_input():
    t = SpectralStatsTransform(stats_to_compute=["mean", "maximum"])
    valid = np.array([[1.0, 2.0], [3.0, 4.0]])
    returned = t._validate_input(valid, "features")
    # Should return the same array content
    assert_array_equal(returned, valid)


def test_transform_data_produces_feature_vector_and_handles_nans():
    # Two signals (columns), two stats -> output length 4
    stats = ["mean", "maximum"]
    t = SpectralStatsTransform(stats_to_compute=stats)
    data = {
        # Column 0: valid signal [1,3,5]
        # Column 1: contains NaN -> stats for this signal should be NaN
        "features": np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    }

    result = t.transform_data(data, metadata={})
    assert result.shape == (len(stats) * 2,)

    # First two values correspond to column 0 and should be finite
    assert not np.isnan(result[0])
    assert not np.isnan(result[1])

    # Last two values correspond to column 1 (contains NaN) and should be NaN
    assert np.isnan(result[2])
    assert np.isnan(result[3])


def test_transform_data_with_named_transform_input():
    """Test transform_data with a proper NamedTransformInput dict."""
    stats = ["mean", "maximum", "variance"]
    t = SpectralStatsTransform(stats_to_compute=stats)

    # Create a NamedTransformInput (dict with "features" key)
    signal_data = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    )

    data = {"features": signal_data}  # NamedTransformInput
    metadata = {}

    result = t.transform_data(data, metadata)

    # Should have shape (1, n_signals * n_stats) = (1, 3 * 3) = (1, 9)
    assert result.shape == (9,)
    assert result.ndim == 1

    # All values should be finite (no NaNs in input)
    assert np.all(np.isfinite(result))


def test_transform_data_with_spectral_entropy():
    """Test that spectral_entropy statistic is computed correctly."""
    t = SpectralStatsTransform(stats_to_compute=["spectral_entropy"])

    # Simple sine wave signal
    n = 128
    signal = np.sin(np.linspace(0, 4 * np.pi, n))
    data = {"features": signal.reshape(-1, 1)}

    result = t.transform_data(data, metadata={})

    assert result.shape == (1,)
    assert np.isfinite(result[0])
    assert result[0] >= 0  # Entropy should be non-negative


def test_transform_data_with_permutation_entropy():
    """Test that permutation_entropy statistic is computed correctly."""
    t = SpectralStatsTransform(
        stats_to_compute=["permutation_entropy"], pe_dim=3, pe_tau=1
    )

    # Signal long enough for permutation entropy
    signal = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    data = {"features": signal.reshape(-1, 1)}

    result = t.transform_data(data, metadata={})

    assert result.shape == (1,)
    assert np.isfinite(result[0])
    assert result[0] >= 0


def test_transform_data_multiple_signals_multiple_stats():
    """Test with multiple signals and multiple statistics."""
    stats = ["mean", "maximum", "minimum"]
    t = SpectralStatsTransform(stats_to_compute=stats)

    # 3 signals (columns), 5 samples
    signal_data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ]
    )

    data = {"features": signal_data}
    result = t.transform_data(data, metadata={})

    # Shape should be (1, 3 signals * 3 stats) = (1, 9)
    assert result.shape == (9,)
    assert np.all(np.isfinite(result))


def test_transform_data_raises_on_empty_data():
    """Test that transform_data raises ValueError on empty dict."""
    t = SpectralStatsTransform(stats_to_compute=["mean"])

    with pytest.raises(ValueError) as exc:
        t.transform_data({}, metadata={})
    assert "No data provided" in str(exc.value)
