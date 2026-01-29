"""Comprehensive tests for stfft.py transform."""
import numpy as np
import pytest
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.stfft import STFTTransform


class TestSTFTTransform:
    """Tests for STFTTransform."""
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        transform = STFTTransform()
        assert transform.win_len == 256
        assert transform.hop == 64  # win_len // 4
        assert transform.fs == 1.0
        assert transform.fft_mode == "onesided"
        assert transform.output_format == "magnitude"
        assert transform.apply_to_columns is True
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        transform = STFTTransform(
            win_len=128,
            hop=32,
            fs=100.0,
            fft_mode="twosided",
            output_format="power"
        )
        assert transform.win_len == 128
        assert transform.hop == 32
        assert transform.fs == 100.0
        assert transform.fft_mode == "twosided"
        assert transform.output_format == "power"
    
    def test_init_invalid_fft_mode_error(self):
        """Test initialization with invalid fft_mode raises error."""
        with pytest.raises(ValueError, match="fft_mode must be one of"):
            STFTTransform(fft_mode="invalid")
    
    def test_init_invalid_scale_to_error(self):
        """Test initialization with invalid scale_to raises error."""
        with pytest.raises(ValueError, match="scale_to must be one of"):
            STFTTransform(scale_to="invalid")
    
    def test_init_invalid_output_format_error(self):
        """Test initialization with invalid output_format raises error."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            STFTTransform(output_format="invalid")
    
    def test_fit_data(self):
        """Test fit_data does nothing."""
        transform = STFTTransform()
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        # Should not raise
        transform.fit_data(data, metadata)
    
    def test_transform_data_magnitude(self):
        """Test transform_data with magnitude output."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            output_format="magnitude"
        )
        # Create a simple signal
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2  # Should be (time, frequency)
        assert np.all(result >= 0)  # Magnitude should be non-negative
    
    def test_transform_data_power(self):
        """Test transform_data with power output."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            output_format="power"
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)  # Power should be non-negative
    
    def test_transform_data_phase(self):
        """Test transform_data with phase output."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            output_format="phase"
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # Phase should be between -pi and pi
        assert np.all(result >= -np.pi)
        assert np.all(result <= np.pi)
    
    def test_transform_data_log_power(self):
        """Test transform_data with log_power output."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            output_format="log_power"
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))
    
    def test_transform_data_multiple_signals(self):
        """Test transform_data with multiple signals."""
        transform = STFTTransform(win_len=32, hop=8)
        signals = np.array([
            np.sin(np.linspace(0, 4 * np.pi, 128)),
            np.cos(np.linspace(0, 4 * np.pi, 128))
        ]).T  # Shape (128, 2)
        data = NamedTransformInput(features=signals)
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
    
    def test_transform_data_twosided_fft(self):
        """Test transform_data with twosided FFT mode."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            fft_mode="twosided"
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # Twosided should have more frequency bins
        assert result.shape[1] > 16  # More than onesided
    
    def test_transform_data_with_subbands(self):
        """Test transform_data with subbands."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            subbands=[(0.0, 0.1), (0.1, 0.2)]
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # Should extract subbands
        assert result.ndim >= 2
    
    def test_fit_multi_source(self):
        """Test fit_multi_source."""
        transform = STFTTransform(win_len=32, hop=8)
        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]])),
            NamedTransformInput(features=np.array([[5.0, 6.0], [7.0, 8.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}
        
        # Should not raise
        transform.fit_multi_source(data_segments, metadata)
