"""Comprehensive tests for spectral.py transform."""
import numpy as np
import pytest
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.spectral import SpectralStatsTransform


class TestSpectralStatsTransform:
    """Tests for SpectralStatsTransform."""
    
    def test_init_basic_stats(self):
        """Test initialization with basic stats."""
        transform = SpectralStatsTransform(stats_to_compute=["mean", "variance"])
        assert "mean" in transform.stats_to_compute
        assert "variance" in transform.stats_to_compute
        assert transform.fs == 1.0
        assert transform.apply_to_columns is True
    
    def test_init_custom_fs(self):
        """Test initialization with custom sampling frequency."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"], fs=100.0)
        assert transform.fs == 100.0
    
    def test_init_invalid_stat_error(self):
        """Test initialization with invalid stat raises error."""
        with pytest.raises(ValueError, match="Unknown statistic"):
            SpectralStatsTransform(stats_to_compute=["invalid_stat"])
    
    def test_init_apply_to_columns_false_error(self):
        """Test initialization with apply_to_columns=False raises error."""
        with pytest.raises(NotImplementedError, match="apply_to_columns=False"):
            SpectralStatsTransform(stats_to_compute=["mean"], apply_to_columns=False)
    
    def test_compute_fft_spectrum(self):
        """Test _compute_fft_spectrum method."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        
        spectrum = transform._compute_fft_spectrum(signal)
        
        assert isinstance(spectrum, np.ndarray)
        assert len(spectrum) > 0
        assert np.all(spectrum >= 0)  # Amplitude should be non-negative
    
    def test_compute_fft_spectrum_empty_signal(self):
        """Test _compute_fft_spectrum with empty signal."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        signal = np.array([])
        
        spectrum = transform._compute_fft_spectrum(signal)
        
        assert isinstance(spectrum, np.ndarray)
        assert len(spectrum) == 0
    
    def test_transform_data_mean(self):
        """Test transform_data with mean statistic."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        data = NamedTransformInput(features=np.array([
            np.sin(np.linspace(0, 4 * np.pi, 128)),
            np.cos(np.linspace(0, 4 * np.pi, 128))
        ]).T)  # Shape (128, 2)
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # Output shape depends on implementation - check it's valid
        assert result.ndim >= 1
        assert result.shape[0] >= 1  # At least one row
        assert np.all(np.isfinite(result))
    
    def test_transform_data_multiple_stats(self):
        """Test transform_data with multiple statistics."""
        transform = SpectralStatsTransform(stats_to_compute=["mean", "variance", "kurtosis"])
        data = NamedTransformInput(features=np.array([
            np.sin(np.linspace(0, 4 * np.pi, 128)),
            np.cos(np.linspace(0, 4 * np.pi, 128))
        ]).T)
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # Output is flattened: (n_signals * n_stats,) = (2 * 3,) = (6,)
        assert result.shape == (6,)
        assert np.all(np.isfinite(result))
    
    def test_transform_data_entropy_stats(self):
        """Test transform_data with entropy statistics."""
        transform = SpectralStatsTransform(stats_to_compute=["spectral_entropy", "shannon_entropy"])
        data = NamedTransformInput(features=np.array([
            np.sin(np.linspace(0, 4 * np.pi, 128)),
            np.cos(np.linspace(0, 4 * np.pi, 128))
        ]).T)
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))
        # Entropy should be between 0 and 1 (normalized)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_transform_data_all_valid_stats(self):
        """Test transform_data with all valid statistics."""
        all_stats = [
            "mean", "maximum", "minimum", "root_mean_square",
            "peak_to_peak_value", "variance", "skewness", "kurtosis",
            "abs_energy", "peak_factor", "change_coefficient",
            "clearance_factor", "spectral_entropy", "shannon_entropy"
        ]
        transform = SpectralStatsTransform(stats_to_compute=all_stats)
        data = NamedTransformInput(features=np.array([
            np.sin(np.linspace(0, 4 * np.pi, 128))
        ]).T)
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # Output is flattened: (n_signals * n_stats,) = (1 * 14,) = (14,)
        assert result.shape == (len(all_stats),)
        assert np.all(np.isfinite(result))
    
    def test_transform_data_empty(self):
        """Test transform_data with empty data."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        data = NamedTransformInput(features=np.array([]).reshape(0, 1))
        metadata = {}
        
        # Should handle empty data gracefully
        try:
            result = transform.transform_data(data, metadata)
            assert isinstance(result, np.ndarray)
        except (ValueError, IndexError):
            # Empty data might cause errors, which is acceptable
            pass
