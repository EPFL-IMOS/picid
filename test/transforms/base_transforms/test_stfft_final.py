"""Final tests for stfft.py to reach 90% coverage."""
import numpy as np
import pytest
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.stfft import STFTTransform


class TestSTFTTransformFinal:
    """Final tests to reach 90% coverage for STFTTransform."""
    
    def test_validate_input_not_2d_error(self):
        """Test _validate_input with non-2D array raises error."""
        transform = STFTTransform(win_len=32, hop=8)
        arr_1d = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="must be a 2D array"):
            transform._validate_input(arr_1d, "test")
    
    def test_validate_input_nan_error(self):
        """Test _validate_input with NaN values raises error."""
        transform = STFTTransform(win_len=32, hop=8)
        arr = np.array([[1.0, 2.0], [np.nan, 4.0]])
        
        with pytest.raises(ValueError, match="NaN values"):
            transform._validate_input(arr, "test")
    
    def test_validate_input_inf_error(self):
        """Test _validate_input with infinite values raises error."""
        transform = STFTTransform(win_len=32, hop=8)
        arr = np.array([[1.0, 2.0], [np.inf, 4.0]])
        
        with pytest.raises(ValueError, match="Infinite values"):
            transform._validate_input(arr, "test")
    
    def test_validate_input_short_signal_error(self):
        """Test _validate_input with signal shorter than window raises error."""
        transform = STFTTransform(win_len=32, hop=8)
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])  # Only 2 rows < win_len 32
        
        with pytest.raises(ValueError, match="shorter than window length"):
            transform._validate_input(arr, "test")
    
    def test_apply_stft_to_signal_complex_output(self):
        """Test _apply_stft_to_signal with complex output."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            output_format="complex"
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        
        result = transform._apply_stft_to_signal(signal)
        
        assert isinstance(result, np.ndarray)
        # Complex output is stacked [real, imag]
        assert result.ndim >= 2
    
    def test_get_feature_names(self):
        """Test get_feature_names method."""
        transform = STFTTransform(win_len=32, hop=8)
        input_keys = ["features"]
        input_shapes = {"features": (128, 2)}  # 128 rows, 2 columns
        
        feature_names = transform.get_feature_names(input_keys, input_shapes)
        
        assert len(feature_names) > 0
        assert all("freq" in name for name in feature_names)
    
    def test_get_feature_names_complex(self):
        """Test get_feature_names with complex output format."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            output_format="complex"
        )
        input_keys = ["features"]
        input_shapes = {"features": (128, 1)}
        
        feature_names = transform.get_feature_names(input_keys, input_shapes)
        
        assert len(feature_names) > 0
        assert any("real" in name for name in feature_names)
        assert any("imag" in name for name in feature_names)
    
    def test_get_time_frequency_info(self):
        """Test get_time_frequency_info method."""
        transform = STFTTransform(win_len=32, hop=8, fs=100.0)
        
        info = transform.get_time_frequency_info()
        
        assert isinstance(info, dict)
        assert "n_freq_bins" in info
        assert "freq_bins" in info
        assert "delta_f" in info
        assert "delta_t" in info
        assert "hop_samples" in info
        assert "window_length" in info
