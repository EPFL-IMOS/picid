"""Comprehensive tests for stfft.py transform.

This file consolidates all tests for STFTTransform from multiple test files
to ensure complete coverage of picid.transforms.base_transforms.stfft.
"""

import numpy as np
import pytest
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base_transforms.stfft import STFTTransform


class TestSTFTTransform:
    """Comprehensive tests for STFTTransform."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    def test_init_defaults(self):
        """Test initialization with defaults.

        **Assumption**: STFTTransform should initialize with sensible defaults:
        win_len=256 (window length), hop=win_len//4 (hop size), fs=1.0 (sampling frequency),
        fft_mode="onesided" (one-sided FFT), output_format="magnitude" (magnitude spectrum),
        and apply_to_columns=True (process each column independently).

        **Action**: Create an STFTTransform with no arguments (using defaults).

        **Expected Result**: All default values should be set correctly. This validates
        that the transform can be created with sensible defaults, which is essential for
        ease of use and quick prototyping.
        """
        transform = STFTTransform()
        assert transform.win_len == 256
        assert transform.hop == 64  # win_len // 4
        assert transform.fs == 1.0
        assert transform.fft_mode == "onesided"
        assert transform.output_format == "magnitude"
        assert transform.apply_to_columns is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters.

        **Assumption**: STFTTransform should accept custom parameters for window length,
        hop size, sampling frequency, FFT mode, and output format. This allows fine-tuning
        the transform for specific use cases.

        **Action**: Create an STFTTransform with custom parameters: win_len=128, hop=32,
        fs=100.0, fft_mode="twosided", output_format="power".

        **Expected Result**: All custom parameters should be stored correctly. This validates
        that the transform can be configured for different scenarios, which is essential for
        adapting to different signal characteristics and analysis requirements.
        """
        transform = STFTTransform(
            win_len=128, hop=32, fs=100.0, fft_mode="twosided", output_format="power"
        )
        assert transform.win_len == 128
        assert transform.hop == 32
        assert transform.fs == 100.0
        assert transform.fft_mode == "twosided"
        assert transform.output_format == "power"

    def test_init_invalid_fft_mode_error(self):
        """Test initialization with invalid fft_mode raises error.

        **Assumption**: STFTTransform should validate that fft_mode is one of the supported
        values ("onesided", "twosided", "centered", "onesided2X"). If an invalid mode is
        provided, it should raise a ValueError with a descriptive error message.

        **Action**: Attempt to create an STFTTransform with fft_mode="invalid".

        **Expected Result**: The initialization should raise a ValueError with a message
        containing "fft_mode must be one of". This validates that input validation works
        correctly, preventing configuration errors.
        """
        with pytest.raises(ValueError, match="fft_mode must be one of"):
            STFTTransform(fft_mode="invalid")

    def test_init_invalid_scale_to_error(self):
        """Test initialization with invalid scale_to raises error.

        **Assumption**: STFTTransform should validate that scale_to is one of the supported
        values (e.g., "magnitude", "psd"). If an invalid value is provided, it should raise
        a ValueError.

        **Action**: Attempt to create an STFTTransform with scale_to="invalid".

        **Expected Result**: The initialization should raise a ValueError with a message
        containing "scale_to must be one of". This validates that input validation works
        correctly.
        """
        with pytest.raises(ValueError, match="scale_to must be one of"):
            STFTTransform(scale_to="invalid")

    def test_init_invalid_output_format_error(self):
        """Test initialization with invalid output_format raises error.

        **Assumption**: STFTTransform should validate that output_format is one of the
        supported values ("magnitude", "power", "phase", "complex", "log_power"). If an
        invalid format is provided, it should raise a ValueError.

        **Action**: Attempt to create an STFTTransform with output_format="invalid".

        **Expected Result**: The initialization should raise a ValueError with a message
        containing "output_format must be one of". This validates that input validation
        works correctly.
        """
        with pytest.raises(ValueError, match="output_format must be one of"):
            STFTTransform(output_format="invalid")

    # ========================================================================
    # FIT TESTS
    # ========================================================================

    def test_fit_data(self):
        """Test fit_data does nothing.

        **Assumption**: STFTTransform's fit_data method should be a no-op (doesn't need
        to learn parameters from data). It should complete without errors.

        **Action**: Create an STFTTransform and call fit_data with sample data.

        **Expected Result**: The method should complete without raising any errors. This
        validates that the fit interface works correctly, which is important for compatibility
        with the framework's fit/transform pattern.
        """
        transform = STFTTransform()
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}

        # Should not raise
        transform.fit_data(data, metadata)

    def test_fit_multi_source(self):
        """STFTTransform is stateless; fit_multi_source raises NotImplementedError."""
        transform = STFTTransform(win_len=32, hop=8)
        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]])),
            NamedTransformInput(features=np.array([[5.0, 6.0], [7.0, 8.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}
        with pytest.raises(
            NotImplementedError, match="stateless|does not support fitting"
        ):
            transform.fit_multi_source(data_segments, metadata)

    # ========================================================================
    # INPUT VALIDATION TESTS
    # ========================================================================

    def test_validate_input_not_2d_error(self):
        """Test _validate_input with non-2D array raises error.

        **Assumption**: STFTTransform._validate_input should validate that input arrays
        are 2D (samples × signals). If a 1D array is provided, it should raise a ValueError
        with a descriptive error message.

        **Action**: Create an STFTTransform and call _validate_input with a 1D array.

        **Expected Result**: The call should raise a ValueError with a message containing
        "must be a 2D array". This validates that input validation works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8)
        arr_1d = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="must be a 2D array"):
            transform._validate_input(arr_1d, "test")

    def test_validate_input_nan_error(self):
        """Test _validate_input with NaN values raises error.

        **Assumption**: STFTTransform._validate_input should validate that input arrays
        don't contain NaN values, as these can cause issues in FFT computations. If NaN
        values are found, it should raise a ValueError.

        **Action**: Create an STFTTransform and call _validate_input with a 2D array
        containing np.nan.

        **Expected Result**: The call should raise a ValueError with a message containing
        "NaN values". This validates that input validation works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8)
        arr = np.array([[1.0, 2.0], [np.nan, 4.0]])

        with pytest.raises(ValueError, match="NaN values"):
            transform._validate_input(arr, "test")

    def test_validate_input_inf_error(self):
        """Test _validate_input with infinite values raises error.

        **Assumption**: STFTTransform._validate_input should validate that input arrays
        don't contain infinite values, as these can cause issues in FFT computations.
        If infinite values are found, it should raise a ValueError.

        **Action**: Create an STFTTransform and call _validate_input with a 2D array
        containing np.inf.

        **Expected Result**: The call should raise a ValueError with a message containing
        "Infinite values". This validates that input validation works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8)
        arr = np.array([[1.0, 2.0], [np.inf, 4.0]])

        with pytest.raises(ValueError, match="Infinite values"):
            transform._validate_input(arr, "test")

    def test_validate_input_short_signal_error(self):
        """Test _validate_input with signal shorter than window raises error.

        **Assumption**: STFTTransform should validate that input signals have at least
        as many samples as the window length. If a signal is shorter than the window,
        STFT cannot be computed (there aren't enough samples to fill even one window),
        so the transform should raise a ValueError with a descriptive message.

        **Action**: Create an STFTTransform with win_len=32 and provide input data with
        only 2 rows (much shorter than the 32-sample window). Attempt to validate the input.

        **Expected Result**: The validation should raise a ValueError with a message
        containing "shorter than window length". This validates that input validation
        works correctly, which is essential for catching configuration errors early and
        preventing runtime failures when signals are too short for the specified window size.
        """
        transform = STFTTransform(win_len=32, hop=8)
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])  # Only 2 rows < win_len 32

        with pytest.raises(ValueError, match="shorter than window length"):
            transform._validate_input(arr, "test")

    # ========================================================================
    # TRANSFORM DATA TESTS - OUTPUT FORMATS
    # ========================================================================

    def test_transform_data_magnitude(self):
        """Test transform_data with magnitude output.

        **Assumption**: STFTTransform should compute the Short-Time Fourier Transform
        and return the magnitude spectrum. The magnitude represents the amplitude of
        each frequency component at each time window, providing a time-frequency
        representation of the signal. Magnitude values are always non-negative.

        **Action**: Create an STFTTransform with win_len=32, hop=8, and output_format="magnitude".
        Provide a sinusoidal signal (128 samples) and apply the transform.

        **Expected Result**: The result should be a numpy array with at least 2 dimensions
        (time windows x frequency bins), and all values should be non-negative (magnitude
        is always >= 0). This validates that STFT magnitude computation works correctly,
        which is essential for time-frequency analysis, feature extraction, and
        spectrogram generation in signal processing applications.
        """
        transform = STFTTransform(win_len=32, hop=8, output_format="magnitude")
        # Create a simple signal
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2  # Should be (time, frequency)
        assert np.all(result >= 0)  # Magnitude should be non-negative

    def test_transform_data_power(self):
        """Test transform_data with power output.

        **Assumption**: STFTTransform should support output_format="power" to return
        the power spectrum (magnitude squared), which represents the energy at each
        frequency component. Power values are always non-negative.

        **Action**: Create an STFTTransform with output_format="power" and apply it
        to a sinusoidal signal.

        **Expected Result**: The result should be a numpy array with all values
        non-negative (power is always >= 0). This validates that power spectrum
        computation works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8, output_format="power")
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)  # Power should be non-negative

    def test_transform_data_phase(self):
        """Test transform_data with phase output.

        **Assumption**: STFTTransform should support output_format="phase" to extract the
        phase spectrum (argument of complex STFT coefficients) instead of magnitude. Phase
        information is important for signal reconstruction and can capture phase relationships
        between frequency components. Phase values are typically in the range [-π, π].

        **Action**: Create an STFTTransform with output_format="phase", win_len=32, hop=8.
        Provide a sinusoidal signal and apply the transform to extract phase information.

        **Expected Result**: The result should be a numpy array with phase values, and all
        values should be in the range [-π, π]. This validates that phase extraction works
        correctly, which is essential for applications requiring phase information (e.g.,
        signal reconstruction, phase-based feature extraction, or when both magnitude and
        phase are needed for complete signal representation).
        """
        transform = STFTTransform(win_len=32, hop=8, output_format="phase")
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        # Phase should be between -pi and pi
        assert np.all(result >= -np.pi)
        assert np.all(result <= np.pi)

    def test_transform_data_log_power(self):
        """Test transform_data with log_power output.

        **Assumption**: STFTTransform should support output_format="log_power" to return
        the logarithm of the power spectrum, which is useful for visualizing wide dynamic
        ranges and for certain machine learning applications. Log power values should be
        finite (no NaN or Inf).

        **Action**: Create an STFTTransform with output_format="log_power" and apply it
        to a sinusoidal signal.

        **Expected Result**: The result should be a numpy array with all values finite.
        This validates that log power computation works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8, output_format="log_power")
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))

    def test_transform_data_complex(self):
        """Test transform_data with complex output.

        **Assumption**: STFTTransform should support output_format="complex" to return
        the complex STFT coefficients. Since numpy arrays can't directly represent complex
        values in the framework, the complex output is typically stacked as [real, imag]
        or returned in a format that preserves both real and imaginary parts.

        **Action**: Create an STFTTransform with output_format="complex" and apply it
        to a sinusoidal signal.

        **Expected Result**: The result should be a numpy array with at least 2 dimensions.
        Complex output is stacked [real, imag], so it's real-valued but represents complex
        data. This validates that complex output works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8, output_format="complex")
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        # Complex output is stacked [real, imag], so it's real-valued but represents complex
        assert result.ndim >= 2

    def test_apply_stft_to_signal_complex_output(self):
        """Test _apply_stft_to_signal with complex output.

        **Assumption**: STFTTransform._apply_stft_to_signal should compute STFT on a
        single signal and return the result in the specified output format. For complex
        output, it should return both real and imaginary parts.

        **Action**: Create an STFTTransform with output_format="complex" and call
        _apply_stft_to_signal with a sinusoidal signal.

        **Expected Result**: The result should be a numpy array with at least 2 dimensions.
        Complex output is stacked [real, imag]. This validates that the internal STFT
        computation works correctly for complex output.
        """
        transform = STFTTransform(win_len=32, hop=8, output_format="complex")
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))

        result = transform._apply_stft_to_signal(signal)

        assert isinstance(result, np.ndarray)
        # Complex output is stacked [real, imag]
        assert result.ndim >= 2

    # ========================================================================
    # TRANSFORM DATA TESTS - FFT MODES
    # ========================================================================

    def test_transform_data_twosided_fft(self):
        """Test transform_data with twosided FFT mode.

        **Assumption**: STFTTransform should support fft_mode="twosided" to compute
        a two-sided FFT, which includes both positive and negative frequencies. This
        provides more complete frequency information but requires more storage.

        **Action**: Create an STFTTransform with fft_mode="twosided" and apply it to
        a sinusoidal signal.

        **Expected Result**: The result should be a numpy array with more frequency bins
        than onesided mode (twosided has more frequency bins). This validates that
        twosided FFT mode works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8, fft_mode="twosided")
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        # Twosided should have more frequency bins
        assert result.shape[1] > 16  # More than onesided

    def test_transform_data_centered_fft(self):
        """Test transform_data with centered FFT mode.

        **Assumption**: STFTTransform should support fft_mode="centered" to compute
        a centered FFT, where zero frequency is in the middle. This is useful for
        visualization and certain analysis tasks.

        **Action**: Create an STFTTransform with fft_mode="centered" and apply it
        to a sinusoidal signal.

        **Expected Result**: The result should be a numpy array with at least 2 dimensions.
        This validates that centered FFT mode works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8, fft_mode="centered")
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2

    def test_transform_data_onesided2x_fft(self):
        """Test transform_data with onesided2X FFT mode.

        **Assumption**: STFTTransform should support fft_mode="onesided2X" to compute
        a one-sided FFT with doubled frequency resolution. This requires scale_to to be
        set to a valid value (e.g., "magnitude").

        **Action**: Create an STFTTransform with fft_mode="onesided2X" and scale_to="magnitude".
        Apply it to a sinusoidal signal.

        **Expected Result**: The result should be a numpy array with at least 2 dimensions.
        This validates that onesided2X FFT mode works correctly.
        """
        # onesided2X requires scale_to to be set
        transform = STFTTransform(
            win_len=32, hop=8, fft_mode="onesided2X", scale_to="magnitude"
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2

    # ========================================================================
    # TRANSFORM DATA TESTS - SCALING
    # ========================================================================

    def test_transform_data_scale_to_magnitude(self):
        """Test transform_data with scale_to='magnitude'.

        **Assumption**: STFTTransform should support scale_to="magnitude" to scale
        the output to magnitude units. This ensures consistent scaling across different
        FFT modes and window types.

        **Action**: Create an STFTTransform with scale_to="magnitude" and apply it
        to a sinusoidal signal.

        **Expected Result**: The result should be a numpy array with all values
        non-negative (magnitude is always >= 0). This validates that magnitude
        scaling works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8, scale_to="magnitude")
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)  # Magnitude should be non-negative

    def test_transform_data_scale_to_psd(self):
        """Test transform_data with scale_to='psd'.

        **Assumption**: STFTTransform should support scale_to="psd" to scale the output
        to Power Spectral Density (PSD) units. PSD represents power per unit frequency
        and is useful for power analysis.

        **Action**: Create an STFTTransform with scale_to="psd" and apply it to a
        sinusoidal signal.

        **Expected Result**: The result should be a numpy array with all values
        non-negative (PSD is always >= 0). This validates that PSD scaling works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8, scale_to="psd")
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)  # PSD should be non-negative

    # ========================================================================
    # TRANSFORM DATA TESTS - ADVANCED OPTIONS
    # ========================================================================

    def test_transform_data_custom_mfft(self):
        """Test transform_data with custom mfft.

        **Assumption**: STFTTransform should support mfft parameter to specify the
        FFT size (with zero-padding if mfft > win_len). This allows controlling
        frequency resolution independently of window length.

        **Action**: Create an STFTTransform with mfft=64 (zero-padding to 64) and
        apply it to a sinusoidal signal.

        **Expected Result**: The result should be a numpy array with at least 2
        dimensions. mfft affects frequency resolution (more bins with larger mfft).
        This validates that custom FFT size works correctly.
        """
        transform = STFTTransform(
            win_len=32,
            hop=8,
            mfft=64,  # Zero-padding to 64
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        # mfft affects frequency resolution
        assert result.ndim >= 2

    def test_transform_data_custom_window_type(self):
        """Test transform_data with custom window type.

        **Assumption**: STFTTransform should support different window types (e.g.,
        "hamming", "hanning", "blackman") for windowing the signal before FFT.
        Different windows have different frequency response characteristics.

        **Action**: Create an STFTTransform with window_type="hamming" and apply it
        to a sinusoidal signal.

        **Expected Result**: The result should be a numpy array with at least 2
        dimensions. This validates that custom window types work correctly.
        """
        transform = STFTTransform(win_len=32, hop=8, window_type="hamming")
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2

    def test_transform_data_multiple_signals(self):
        """Test transform_data with multiple signals.

        **Assumption**: STFTTransform should process multiple signals (columns)
        independently when apply_to_columns=True. Each signal should be transformed
        separately, and the results should be concatenated or organized appropriately.

        **Action**: Create an STFTTransform and provide input data with 2 signals
        (columns). Apply the transform.

        **Expected Result**: The result should be a numpy array with at least 2
        dimensions. This validates that multi-signal processing works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8)
        signals = np.array(
            [
                np.sin(np.linspace(0, 4 * np.pi, 128)),
                np.cos(np.linspace(0, 4 * np.pi, 128)),
            ]
        ).T  # Shape (128, 2)
        data = NamedTransformInput(features=signals)
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2

    def test_transform_data_with_subbands(self):
        """Test transform_data with subbands.

        **Assumption**: STFTTransform should support subbands parameter to extract
        specific frequency bands from the STFT result. This is useful for focusing
        on particular frequency ranges of interest.

        **Action**: Create an STFTTransform with subbands=[(0.0, 0.1), (0.1, 0.2)]
        (normalized frequency bands) and apply it to a sinusoidal signal.

        **Expected Result**: The result should be a numpy array with at least 2
        dimensions, representing the extracted subbands. This validates that subband
        extraction works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8, subbands=[(0.0, 0.1), (0.1, 0.2)])
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        # Should extract subbands
        assert result.ndim >= 2

    def test_transform_data_with_subbands_filtering(self):
        """Test transform_data with subbands filtering (frequency bands in Hz).

        **Assumption**: STFTTransform should support subbands specified in Hz (when
        fs is provided) to extract specific frequency bands. This allows working with
        physical frequencies rather than normalized frequencies.

        **Action**: Create an STFTTransform with fs=100.0 and subbands=[(0.0, 10.0), (10.0, 20.0)]
        (frequency bands in Hz). Provide a signal with multiple frequency components
        and apply the transform.

        **Expected Result**: The result should be a numpy array with at least 2
        dimensions, representing the extracted frequency subbands. This validates
        that frequency-band extraction works correctly.
        """
        transform = STFTTransform(
            win_len=64,
            hop=16,
            fs=100.0,  # 100 Hz sampling
            subbands=[(0.0, 10.0), (10.0, 20.0)],  # Frequency bands in Hz
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

    # ========================================================================
    # FEATURE NAMES TESTS
    # ========================================================================

    def test_get_feature_names(self):
        """Test get_feature_names method.

        **Assumption**: STFTTransform.get_feature_names should generate descriptive
        feature names for the STFT output. The names should indicate that they are
        frequency features ("freq") and include information about time windows and
        frequency bins.

        **Action**: Create an STFTTransform and call get_feature_names with input_keys=["features"]
        and input_shapes indicating 2 columns (signals) and 128 rows.

        **Expected Result**: The result should be a list of feature names, and all
        names should contain "freq" to indicate they are frequency features. This
        validates that feature naming works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8)
        input_keys = ["features"]
        input_shapes = {"features": (128, 2)}  # 128 rows, 2 columns

        feature_names = transform.get_feature_names(input_keys, input_shapes)

        assert len(feature_names) > 0
        assert all("freq" in name for name in feature_names)

    def test_get_feature_names_complex(self):
        """Test get_feature_names with complex output format.

        **Assumption**: STFTTransform.get_feature_names should generate appropriate
        feature names for complex output format, including both "real" and "imag"
        components in the names.

        **Action**: Create an STFTTransform with output_format="complex" and call
        get_feature_names.

        **Expected Result**: The result should be a list of feature names containing
        both "real" and "imag" to indicate real and imaginary parts. This validates
        that complex output feature naming works correctly.
        """
        transform = STFTTransform(win_len=32, hop=8, output_format="complex")
        input_keys = ["features"]
        input_shapes = {"features": (128, 1)}

        feature_names = transform.get_feature_names(input_keys, input_shapes)

        assert len(feature_names) > 0
        assert any("real" in name for name in feature_names)
        assert any("imag" in name for name in feature_names)

    def test_get_time_frequency_info(self):
        """Test get_time_frequency_info method.

        **Assumption**: STFTTransform.get_time_frequency_info should return a dictionary
        containing information about the time-frequency representation, including number
        of frequency bins, frequency bin values, frequency resolution (delta_f), time
        resolution (delta_t), hop size in samples, and window length.

        **Action**: Create an STFTTransform with win_len=32, hop=8, and fs=100.0. Call
        get_time_frequency_info().

        **Expected Result**: The result should be a dictionary containing keys like
        "n_freq_bins", "freq_bins", "delta_f", "delta_t", "hop_samples", and "window_length".
        This validates that time-frequency information is correctly computed and provided,
        which is essential for understanding the transform's output dimensions and interpreting
        the results.
        """
        transform = STFTTransform(win_len=32, hop=8, fs=100.0)

        info = transform.get_time_frequency_info()

        assert isinstance(info, dict)
        assert "n_freq_bins" in info
        assert "freq_bins" in info
        assert "delta_f" in info
        assert "delta_t" in info
        assert "hop_samples" in info
        assert "window_length" in info


class TestFrequencyFeaturesNp:
    """Tests for frequency_features_np static method."""

    @pytest.fixture
    def sample_spectrum(self):
        """Create a sample magnitude spectrum for testing.

        **Assumption**: This fixture creates a sample magnitude spectrum S with shape
        (K, T) where K=128 is the number of frequency bins and T=50 is the number of
        time frames. It also creates a frequency array freqs with K values from 0 to
        1000 Hz. This provides realistic test data for frequency feature extraction.

        **Action**: Generate a random magnitude spectrum S and a linearly spaced frequency
        array freqs.

        **Expected Result**: Returns a tuple (S, freqs) where S has shape (128, 50) and
        freqs has length 128. This provides test data for frequency feature extraction tests.
        """
        K, T = 128, 50  # 128 frequency bins, 50 time frames
        freqs = np.linspace(0, 1000, K)  # Frequencies from 0 to 1000 Hz
        S = np.random.rand(K, T) * 100  # Random magnitude spectrum
        return S, freqs

    def test_frequency_features_output_shape(self, sample_spectrum):
        """Test that all output features have correct shape (T,).

        **Assumption**: STFTTransform.frequency_features_np should extract frequency-domain
        features from a magnitude spectrum S with shape (K, T) where K is the number of
        frequency bins and T is the number of time frames. Each feature should have shape
        (T,), representing the feature value at each time frame.

        **Action**: Call frequency_features_np with a sample spectrum S (shape K×T) and
        frequency array. Extract all frequency features and verify their shapes.

        **Expected Result**: All feature arrays should have shape (T,), matching the number
        of time frames in the input spectrum. This validates that frequency feature extraction
        works correctly, which is essential for converting time-frequency representations
        into feature vectors for machine learning models.
        """
        S, freqs = sample_spectrum
        features = STFTTransform.frequency_features_np(S, freqs)

        _, T = S.shape
        for feature_name, feature_array in features.items():
            assert feature_array.shape == (
                T,
            ), f"{feature_name} shape {feature_array.shape} != expected ({T},)"

    def test_frequency_features_all_keys_present(self, sample_spectrum):
        """Test that all expected feature keys are present.

        **Assumption**: STFTTransform.frequency_features_np should return a dictionary
        containing all expected frequency feature keys (F12, F13, F14, etc.). These
        keys represent different frequency-domain features extracted from the spectrum.

        **Action**: Call frequency_features_np with a sample spectrum and verify that
        all expected keys are present in the result.

        **Expected Result**: The result should be a dictionary containing all expected
        keys (F12-F23). This validates that all frequency features are computed and
        returned correctly.
        """
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
        """Test that output features contain no NaN values.

        **Assumption**: STFTTransform.frequency_features_np should compute all frequency
        features without producing NaN values, even for edge cases. This ensures the
        features are valid and usable for downstream processing.

        **Action**: Call frequency_features_np with a sample spectrum and check all
        feature arrays for NaN values.

        **Expected Result**: No feature array should contain NaN values. This validates
        that numerical stability is maintained during feature computation.
        """
        S, freqs = sample_spectrum
        features = STFTTransform.frequency_features_np(S, freqs)

        for feature_name, feature_array in features.items():
            assert not np.any(
                np.isnan(feature_array)
            ), f"{feature_name} contains NaN values"

    def test_frequency_features_no_inf_values(self, sample_spectrum):
        """Test that output features contain no infinite values.

        **Assumption**: STFTTransform.frequency_features_np should compute all frequency
        features without producing infinite values, ensuring numerical stability.

        **Action**: Call frequency_features_np with a sample spectrum and check all
        feature arrays for infinite values.

        **Expected Result**: No feature array should contain infinite values. This validates
        that numerical stability is maintained during feature computation.
        """
        S, freqs = sample_spectrum
        features = STFTTransform.frequency_features_np(S, freqs)

        for feature_name, feature_array in features.items():
            assert not np.any(
                np.isinf(feature_array)
            ), f"{feature_name} contains infinite values"

    def test_frequency_features_with_zero_spectrum(self):
        """Test behavior with zero spectrum (edge case).

        **Assumption**: STFTTransform.frequency_features_np should handle edge cases like
        zero spectrum gracefully, returning valid (typically zero) feature values rather
        than NaN or Inf.

        **Action**: Call frequency_features_np with a zero spectrum (all values are 0).

        **Expected Result**: The result should have valid shape and F12 (mean spectrum)
        should be zero. This validates that edge cases are handled correctly.
        """
        K, T = 64, 30
        freqs = np.linspace(0, 500, K)
        S = np.zeros((K, T))

        features = STFTTransform.frequency_features_np(S, freqs)

        # Should not raise an error and should have valid shape
        assert features["F12"].shape == (T,)
        # F12 should be zero for zero spectrum
        assert np.allclose(features["F12"], 0)

    def test_frequency_features_with_single_frequency(self):
        """Test with single frequency bin.

        **Assumption**: STFTTransform.frequency_features_np should handle edge cases like
        single frequency bin correctly, returning valid feature values.

        **Action**: Call frequency_features_np with a spectrum containing only 1 frequency
        bin and 20 time frames.

        **Expected Result**: F12 (mean spectrum) should equal the single spectrum value.
        This validates that edge cases are handled correctly.
        """
        K, T = 1, 20
        freqs = np.array([100])
        S = np.random.rand(K, T) * 50

        features = STFTTransform.frequency_features_np(S, freqs)

        # F12 should equal the single spectrum value
        assert np.allclose(features["F12"], S[0])

    def test_frequency_features_with_single_time_frame(self):
        """Test with single time frame.

        **Assumption**: STFTTransform.frequency_features_np should handle edge cases like
        single time frame correctly, returning valid feature values with shape (1,).

        **Action**: Call frequency_features_np with a spectrum containing 100 frequency
        bins but only 1 time frame.

        **Expected Result**: All feature arrays should have shape (1,) and contain no
        NaN values. This validates that edge cases are handled correctly.
        """
        K, T = 100, 1
        freqs = np.linspace(0, 1000, K)
        S = np.random.rand(K, T) * 100

        features = STFTTransform.frequency_features_np(S, freqs)

        assert features["F12"].shape == (1,)
        for feature_name in features:
            assert not np.any(np.isnan(features[feature_name]))

    def test_frequency_features_f16_spectral_centroid(self):
        """Test F16 (spectral centroid) computation.

        **Assumption**: STFTTransform.frequency_features_np should compute F16 (spectral
        centroid), which represents the "center of mass" of the spectrum. For a uniform
        spectrum, the centroid should be near the middle frequency.

        **Action**: Call frequency_features_np with a uniform spectrum (all values equal)
        and verify F16 (spectral centroid) computation.

        **Expected Result**: F16 should be close to the mean frequency (middle of the
        frequency range). This validates that spectral centroid computation works correctly.
        """
        K, T = 100, 10
        freqs = np.linspace(0, 1000, K)
        S = np.ones((K, T)) * 100

        features = STFTTransform.frequency_features_np(S, freqs)

        # For uniform spectrum, centroid should be near middle frequency
        expected_centroid = np.mean(freqs)
        assert np.allclose(features["F16"], expected_centroid, rtol=0.1)

    def test_frequency_features_f13_rms_spectrum(self):
        """Test F13 (RMS of spectrum) computation.

        **Assumption**: STFTTransform.frequency_features_np should compute F13 (RMS of
        spectrum), which measures the root mean square deviation from the mean spectrum.
        This provides a measure of spectral variability.

        **Action**: Call frequency_features_np with a random spectrum and verify F13
        computation matches the expected RMS calculation.

        **Expected Result**: F13 should equal sqrt(mean((S - F12)^2)) where F12 is the
        mean spectrum. This validates that RMS computation works correctly.
        """
        K, T = 50, 15
        freqs = np.linspace(0, 500, K)
        S = np.random.rand(K, T) * 100

        features = STFTTransform.frequency_features_np(S, freqs)

        F12 = features["F12"]
        F13_expected = np.sqrt(np.mean((S - F12) ** 2, axis=0))

        assert np.allclose(features["F13"], F13_expected)

    def test_frequency_features_with_constant_spectrum(self):
        """Test with constant spectrum across all frequencies.

        **Assumption**: STFTTransform.frequency_features_np should handle edge cases like
        constant spectrum correctly. For a constant spectrum, F12 (mean) should equal the
        constant value, and F13 (RMS) should be 0 (no deviation from mean).

        **Action**: Call frequency_features_np with a constant spectrum (all values equal
        to 50).

        **Expected Result**: F12 should equal the constant value (50), and F13 (RMS)
        should be 0 (no deviation). This validates that edge cases are handled correctly.
        """
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
        """Test numerical stability with very small values.

        **Assumption**: STFTTransform.frequency_features_np should maintain numerical
        stability even with very small spectrum values, avoiding NaN or Inf due to division
        by zero or other numerical issues. Epsilon handling should prevent these problems.

        **Action**: Call frequency_features_np with a spectrum containing very small values
        (1e-10).

        **Expected Result**: All feature arrays should contain no NaN or Inf values. This
        validates that numerical stability is maintained, which is essential for robust
        feature extraction.
        """
        K, T = 50, 25
        freqs = np.linspace(0, 500, K)
        S = np.random.rand(K, T) * 1e-10  # Very small values

        features = STFTTransform.frequency_features_np(S, freqs)

        # Should not produce NaN or Inf due to epsilon handling
        for feature_array in features.values():
            assert not np.any(np.isnan(feature_array))
            assert not np.any(np.isinf(feature_array))

    def test_frequency_features_return_type(self, sample_spectrum):
        """Test that return type is a dictionary.

        **Assumption**: STFTTransform.frequency_features_np should return a dictionary
        mapping feature names (strings) to feature arrays (numpy arrays).

        **Action**: Call frequency_features_np with a sample spectrum and verify the
        return type.

        **Expected Result**: The result should be a dictionary. This validates that
        the return type is correct.
        """
        S, freqs = sample_spectrum
        features = STFTTransform.frequency_features_np(S, freqs)

        assert isinstance(features, dict)

    def test_frequency_features_with_increasing_spectrum(self):
        """Test with linearly increasing spectrum across frequencies.

        **Assumption**: STFTTransform.frequency_features_np should handle various spectrum
        shapes correctly. For a linearly increasing spectrum, F12 (mean) should increase
        gradually, and F16 (spectral centroid) should be biased towards higher frequencies.

        **Action**: Call frequency_features_np with a spectrum that increases linearly
        across frequencies.

        **Expected Result**: F12 should increase gradually, and F16 should be biased towards
        higher frequencies (greater than mean/2). This validates that the features correctly
        capture spectrum characteristics.
        """
        K, T = 100, 10
        freqs = np.linspace(0, 1000, K)
        S = np.tile(np.linspace(0, 100, K), (T, 1)).T

        features = STFTTransform.frequency_features_np(S, freqs)

        # F12 should increase gradually
        assert np.all(np.diff(features["F12"]) >= 0)
        # F16 should be biased towards higher frequencies
        assert np.all(features["F16"] > np.mean(freqs) / 2)
