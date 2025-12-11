import pytest
import numpy as np
from lmetk.transforms.base_transforms.stfft import STFTTransform


class TestFrequencyFeaturesNp:
    """Test suite for frequency_features_np static method."""

    @pytest.fixture
    def sample_spectrum(self):
        """Create a sample magnitude spectrum for testing."""
        K, T = 128, 50  # 128 frequency bins, 50 time frames
        freqs = np.linspace(0, 1000, K)  # Frequencies from 0 to 1000 Hz
        S = np.random.rand(K, T) * 100  # Random magnitude spectrum
        return S, freqs

    def test_frequency_features_output_shape(self, sample_spectrum):
        """Test that all output features have correct shape (T,)."""
        S, freqs = sample_spectrum
        features = STFTTransform.frequency_features_np(S, freqs)

        _, T = S.shape
        for feature_name, feature_array in features.items():
            assert feature_array.shape == (
                T,
            ), f"{feature_name} shape {feature_array.shape} != expected ({T},)"

    def test_frequency_features_all_keys_present(self, sample_spectrum):
        """Test that all expected feature keys are present."""
        S, freqs = sample_spectrum
        features = STFTTransform.frequency_features_np(S, freqs)

        expected_keys = {
            "F12",
            "F13",
            "F14",
            "F15",
            "F16",
            "F17",
            "F18",
            "F19",
            "F20",
            "F21",
            "F22",
            "F23",
            # "F24",
        }
        assert (
            set(features.keys()) == expected_keys
        ), f"Missing or extra keys. Expected {expected_keys}, got {set(features.keys())}"

    def test_frequency_features_no_nan_values(self, sample_spectrum):
        """Test that output features contain no NaN values."""
        S, freqs = sample_spectrum
        features = STFTTransform.frequency_features_np(S, freqs)

        for feature_name, feature_array in features.items():
            assert not np.any(
                np.isnan(feature_array)
            ), f"{feature_name} contains NaN values"

    def test_frequency_features_no_inf_values(self, sample_spectrum):
        """Test that output features contain no infinite values."""
        S, freqs = sample_spectrum
        features = STFTTransform.frequency_features_np(S, freqs)

        for feature_name, feature_array in features.items():
            assert not np.any(
                np.isinf(feature_array)
            ), f"{feature_name} contains infinite values"

    def test_frequency_features_with_zero_spectrum(self):
        """Test behavior with zero spectrum (edge case)."""
        K, T = 64, 30
        freqs = np.linspace(0, 500, K)
        S = np.zeros((K, T))

        features = STFTTransform.frequency_features_np(S, freqs)

        # Should not raise an error and should have valid shape
        assert features["F12"].shape == (T,)
        # F12 should be zero for zero spectrum
        assert np.allclose(features["F12"], 0)

    def test_frequency_features_with_single_frequency(self):
        """Test with single frequency bin."""
        K, T = 1, 20
        freqs = np.array([100])
        S = np.random.rand(K, T) * 50

        features = STFTTransform.frequency_features_np(S, freqs)

        # F12 should equal the single spectrum value
        assert np.allclose(features["F12"], S[0])

    def test_frequency_features_with_single_time_frame(self):
        """Test with single time frame."""
        K, T = 100, 1
        freqs = np.linspace(0, 1000, K)
        S = np.random.rand(K, T) * 100

        features = STFTTransform.frequency_features_np(S, freqs)

        assert features["F12"].shape == (1,)
        for feature_name in features:
            assert not np.any(np.isnan(features[feature_name]))

    def test_frequency_features_f16_spectral_centroid(self):
        """Test F16 (spectral centroid) computation."""
        K, T = 100, 10
        freqs = np.linspace(0, 1000, K)
        S = np.ones((K, T)) * 100

        features = STFTTransform.frequency_features_np(S, freqs)

        # For uniform spectrum, centroid should be near middle frequency
        expected_centroid = np.mean(freqs)
        assert np.allclose(features["F16"], expected_centroid, rtol=0.1)

    def test_frequency_features_f13_rms_spectrum(self):
        """Test F13 (RMS of spectrum) computation."""
        K, T = 50, 15
        freqs = np.linspace(0, 500, K)
        S = np.random.rand(K, T) * 100

        features = STFTTransform.frequency_features_np(S, freqs)

        F12 = features["F12"]
        F13_expected = np.sqrt(np.mean((S - F12) ** 2, axis=0))

        assert np.allclose(features["F13"], F13_expected)

    def test_frequency_features_with_constant_spectrum(self):
        """Test with constant spectrum across all frequencies."""
        K, T = 100, 20
        freqs = np.linspace(0, 1000, K)
        constant_value = 50
        S = np.full((K, T), constant_value)

        features = STFTTransform.frequency_features_np(S, freqs)

        # F12 should equal constant value
        assert np.allclose(features["F12"], constant_value)
        # F13 (RMS) should be 0 for constant spectrum
        assert np.allclose(features["F13"], 0)

    def test_frequency_features_numerical_stability(self):
        """Test numerical stability with very small values."""
        K, T = 50, 25
        freqs = np.linspace(0, 500, K)
        S = np.random.rand(K, T) * 1e-10  # Very small values

        features = STFTTransform.frequency_features_np(S, freqs)

        # Should not produce NaN or Inf due to epsilon handling
        for feature_array in features.values():
            assert not np.any(np.isnan(feature_array))
            assert not np.any(np.isinf(feature_array))

    def test_frequency_features_return_type(self, sample_spectrum):
        """Test that return type is a dictionary."""
        S, freqs = sample_spectrum
        features = STFTTransform.frequency_features_np(S, freqs)

        assert isinstance(features, dict)

    def test_frequency_features_with_increasing_spectrum(self):
        """Test with linearly increasing spectrum across frequencies."""
        K, T = 100, 10
        freqs = np.linspace(0, 1000, K)
        S = np.tile(np.linspace(0, 100, K), (T, 1)).T

        features = STFTTransform.frequency_features_np(S, freqs)

        # F12 should increase gradually
        assert np.all(np.diff(features["F12"]) >= 0)
        # F16 should be biased towards higher frequencies
        assert np.all(features["F16"] > np.mean(freqs) / 2)
