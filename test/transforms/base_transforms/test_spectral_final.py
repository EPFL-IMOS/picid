"""Final tests for spectral.py to reach 90% coverage."""
import numpy as np
import pytest
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.spectral import SpectralStatsTransform


class TestSpectralStatsTransformFinal:
    """Final tests to reach 90% coverage for SpectralStatsTransform."""
    
    def test_transform_data_with_nans(self):
        """Test transform_data with NaN values in signal."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        signal[10:20] = np.nan  # Add NaNs
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        # Should log warning and return NaN stats
        with pytest.raises(ValueError, match="is NaN"):
            transform.transform_data(data, metadata)
    
    def test_get_feature_names(self):
        """Test get_feature_names method."""
        transform = SpectralStatsTransform(stats_to_compute=["mean", "variance"])
        input_keys = ["features"]
        input_shapes = {"features": (128, 3)}  # 128 rows, 3 columns
        
        feature_names = transform.get_feature_names(input_keys, input_shapes)
        
        assert len(feature_names) == 6  # 3 signals * 2 stats
        assert all("spec_" in name for name in feature_names)
        assert all("col" in name for name in feature_names)
    
    def test_get_feature_names_empty_keys(self):
        """Test get_feature_names with empty input_keys."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        feature_names = transform.get_feature_names([], {})
        
        assert feature_names == []
    
    def test_get_feature_names_missing_shape_error(self):
        """Test get_feature_names with missing shape raises error."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        input_keys = ["features"]
        input_shapes = {}  # Missing shape
        
        with pytest.raises(KeyError, match="not provided"):
            transform.get_feature_names(input_keys, input_shapes)
    
    def test_compute_permutation_entropy_short_signal(self):
        """Test _compute_permutation_entropy with short signal."""
        transform = SpectralStatsTransform(
            stats_to_compute=["permutation_entropy"],
            pe_dim=5,
            pe_tau=1
        )
        short_signal = np.array([1.0, 2.0])  # Too short
        
        result = transform._compute_permutation_entropy(short_signal, dim=5, tau=1)
        
        assert np.isnan(result)
    
    def test_compute_stat_unknown_stat_error(self):
        """Test _compute_stat with unknown stat raises error."""
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        spectrum = np.array([1.0, 2.0, 3.0])
        
        # This shouldn't happen due to __init__ validation, but test the path
        with pytest.raises(ValueError, match="Unknown statistic handler"):
            transform._compute_stat(spectrum, "unknown_stat")
