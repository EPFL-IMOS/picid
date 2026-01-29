"""Extended tests for stfft.py to reach 90% coverage."""
import numpy as np
import pytest
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.stfft import STFTTransform


class TestSTFTTransformExtended:
    """Extended tests for STFTTransform to improve coverage."""
    
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
    
    def test_transform_data_complex(self):
        """Test transform_data with complex output."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            output_format="complex"
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # Complex output is stacked [real, imag], so it's real-valued but represents complex
        assert result.ndim >= 2
    
    def test_transform_data_centered_fft(self):
        """Test transform_data with centered FFT mode."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            fft_mode="centered"
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
    
    def test_transform_data_onesided2x_fft(self):
        """Test transform_data with onesided2X FFT mode."""
        # onesided2X requires scale_to to be set
        transform = STFTTransform(
            win_len=32,
            hop=8,
            fft_mode="onesided2X",
            scale_to="magnitude"
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
    
    def test_transform_data_scale_to_magnitude(self):
        """Test transform_data with scale_to='magnitude'."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            scale_to="magnitude"
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)  # Magnitude should be non-negative
    
    def test_transform_data_scale_to_psd(self):
        """Test transform_data with scale_to='psd'."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            scale_to="psd"
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)  # PSD should be non-negative
    
    def test_transform_data_custom_mfft(self):
        """Test transform_data with custom mfft."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            mfft=64  # Zero-padding to 64
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # mfft affects frequency resolution
        assert result.ndim >= 2
    
    def test_transform_data_custom_window_type(self):
        """Test transform_data with custom window type."""
        transform = STFTTransform(
            win_len=32,
            hop=8,
            window_type="hamming"
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
    
    def test_transform_data_with_subbands_filtering(self):
        """Test transform_data with subbands filtering."""
        transform = STFTTransform(
            win_len=64,
            hop=16,
            fs=100.0,  # 100 Hz sampling
            subbands=[(0.0, 10.0), (10.0, 20.0)]  # Frequency bands in Hz
        )
        # Create signal with multiple frequencies
        t = np.linspace(0, 1, 200)
        signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 15 * t)
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # Should extract subbands
        assert result.ndim >= 2
