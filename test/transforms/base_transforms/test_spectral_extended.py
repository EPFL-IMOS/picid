"""Extended tests for spectral.py to reach 90% coverage."""
import numpy as np
import pytest
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.spectral import SpectralStatsTransform


class TestSpectralStatsTransformExtended:
    """Extended tests for SpectralStatsTransform to improve coverage."""
    
    def test_compute_stat_empty_spectrum(self):
        """Test _compute_stat with empty spectrum returns NaN."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        empty_spectrum = np.array([])
        
        result = transform._compute_stat(empty_spectrum, "mean")
        
        assert np.isnan(result)
    
    def test_compute_stat_maximum(self):
        """Test _compute_stat with maximum."""
        transform = SpectralStatsTransform(stats_to_compute=["maximum"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = transform._compute_stat(spectrum, "maximum")
        assert result == 5.0
    
    def test_compute_stat_minimum(self):
        """Test _compute_stat with minimum."""
        transform = SpectralStatsTransform(stats_to_compute=["minimum"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = transform._compute_stat(spectrum, "minimum")
        assert result == 1.0
    
    def test_compute_stat_root_mean_square(self):
        """Test _compute_stat with root_mean_square."""
        transform = SpectralStatsTransform(stats_to_compute=["root_mean_square"])
        spectrum = np.array([1.0, 2.0, 3.0])
        
        result = transform._compute_stat(spectrum, "root_mean_square")
        expected = np.sqrt(np.mean(spectrum**2))
        assert abs(result - expected) < 1e-10
    
    def test_compute_stat_peak_to_peak_value(self):
        """Test _compute_stat with peak_to_peak_value."""
        transform = SpectralStatsTransform(stats_to_compute=["peak_to_peak_value"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = transform._compute_stat(spectrum, "peak_to_peak_value")
        assert result == 4.0
    
    def test_compute_stat_variance(self):
        """Test _compute_stat with variance."""
        transform = SpectralStatsTransform(stats_to_compute=["variance"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = transform._compute_stat(spectrum, "variance")
        expected = np.var(spectrum)
        assert abs(result - expected) < 1e-10
    
    def test_compute_stat_skewness(self):
        """Test _compute_stat with skewness."""
        transform = SpectralStatsTransform(stats_to_compute=["skewness"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = transform._compute_stat(spectrum, "skewness")
        assert np.isfinite(result)
    
    def test_compute_stat_kurtosis(self):
        """Test _compute_stat with kurtosis."""
        transform = SpectralStatsTransform(stats_to_compute=["kurtosis"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = transform._compute_stat(spectrum, "kurtosis")
        assert np.isfinite(result)
    
    def test_compute_stat_abs_energy(self):
        """Test _compute_stat with abs_energy."""
        transform = SpectralStatsTransform(stats_to_compute=["abs_energy"])
        spectrum = np.array([1.0, 2.0, 3.0])
        
        result = transform._compute_stat(spectrum, "abs_energy")
        expected = np.sum(spectrum**2)
        assert abs(result - expected) < 1e-10
    
    def test_compute_stat_peak_factor(self):
        """Test _compute_stat with peak_factor."""
        transform = SpectralStatsTransform(stats_to_compute=["peak_factor"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = transform._compute_stat(spectrum, "peak_factor")
        assert result >= 0
        assert np.isfinite(result)
    
    def test_compute_stat_change_coefficient(self):
        """Test _compute_stat with change_coefficient."""
        transform = SpectralStatsTransform(stats_to_compute=["change_coefficient"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = transform._compute_stat(spectrum, "change_coefficient")
        assert result >= 0
        assert np.isfinite(result)
    
    def test_compute_stat_change_coefficient_single_value(self):
        """Test _compute_stat with change_coefficient and single value."""
        transform = SpectralStatsTransform(stats_to_compute=["change_coefficient"])
        spectrum = np.array([1.0])
        
        result = transform._compute_stat(spectrum, "change_coefficient")
        assert result == 0.0
    
    def test_compute_stat_clearance_factor(self):
        """Test _compute_stat with clearance_factor."""
        transform = SpectralStatsTransform(stats_to_compute=["clearance_factor"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = transform._compute_stat(spectrum, "clearance_factor")
        assert result >= 0
        assert np.isfinite(result)
    
    def test_compute_spectral_entropy(self):
        """Test _compute_spectral_entropy."""
        transform = SpectralStatsTransform(stats_to_compute=["spectral_entropy"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = transform._compute_spectral_entropy(spectrum)
        # Spectral entropy can be > 1 if not normalized, or normalized to [0,1]
        # Just check it's finite and non-negative
        assert result >= 0
        assert np.isfinite(result)
    
    def test_compute_permutation_entropy(self):
        """Test _compute_permutation_entropy."""
        transform = SpectralStatsTransform(
            stats_to_compute=["permutation_entropy"],
            pe_dim=3,
            pe_tau=1
        )
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        result = transform._compute_permutation_entropy(spectrum, dim=3, tau=1)
        assert result >= 0
        assert np.isfinite(result)
    
    def test_validate_input_not_2d_error(self):
        """Test _validate_input with non-2D array raises error."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        arr_1d = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="must be a 2D array"):
            transform._validate_input(arr_1d, "test")
    
    def test_validate_input_inf_error(self):
        """Test _validate_input with infinite values raises error."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        arr = np.array([[1.0, 2.0], [np.inf, 4.0]])
        
        with pytest.raises(ValueError, match="Infinite values"):
            transform._validate_input(arr, "test")
    
    def test_validate_input_short_signal_error(self):
        """Test _validate_input with signal too short raises error."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        arr = np.array([[1.0, 2.0]])  # Only 1 row
        
        with pytest.raises(ValueError, match="too short"):
            transform._validate_input(arr, "test")
    
    def test_transform_data_single_signal(self):
        """Test transform_data with single signal."""
        transform = SpectralStatsTransform(stats_to_compute=["mean", "variance"])
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 1
        assert np.all(np.isfinite(result))
    
    def test_transform_data_permutation_entropy(self):
        """Test transform_data with permutation_entropy."""
        transform = SpectralStatsTransform(
            stats_to_compute=["permutation_entropy"],
            pe_dim=3,
            pe_tau=1
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))
