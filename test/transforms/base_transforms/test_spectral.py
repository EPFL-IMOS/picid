"""Comprehensive tests for spectral.py transform.

This file consolidates all tests for SpectralStatsTransform from multiple test files
to ensure complete coverage of picid.transforms.base_transforms.spectral.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base_transforms.spectral import SpectralStatsTransform


class TestSpectralStatsTransform:
    """Comprehensive tests for SpectralStatsTransform."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    def test_init_basic_stats(self):
        """Test initialization with basic stats.

        **Assumption**: SpectralStatsTransform should accept stats_to_compute (list of
        statistic names to compute on FFT spectra), fs (sampling frequency), and
        apply_to_columns (whether to process each column independently). These parameters
        configure how spectral features are extracted from time-series signals.

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["mean", "variance"].

        **Expected Result**: The transform should be created successfully with both "mean"
        and "variance" in stats_to_compute, fs=1.0 (default), and apply_to_columns=True
        (default). This validates that the transform can be configured for spectral feature
        extraction, which is essential for frequency-domain analysis of time-series data.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean", "variance"])
        assert "mean" in transform.stats_to_compute
        assert "variance" in transform.stats_to_compute
        assert transform.fs == 1.0
        assert transform.apply_to_columns is True

    def test_init_custom_fs(self):
        """Test initialization with custom sampling frequency.

        **Assumption**: SpectralStatsTransform should accept a custom sampling frequency
        (fs) parameter, which is used for frequency-domain calculations. This allows the
        transform to work with signals sampled at different rates.

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["mean"] and
        fs=100.0 (100 Hz sampling frequency).

        **Expected Result**: The transform should be created successfully with fs=100.0.
        This validates that custom sampling frequencies can be configured, which is
        essential for working with signals sampled at different rates.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"], fs=100.0)
        assert transform.fs == 100.0

    def test_init_invalid_stat_error(self):
        """Test initialization with invalid stat raises error.

        **Assumption**: SpectralStatsTransform should validate that all statistics in
        stats_to_compute are supported. If an invalid statistic name is provided, it
        should raise a ValueError with a descriptive error message, preventing silent
        failures and helping users identify configuration errors.

        **Action**: Attempt to create a SpectralStatsTransform with stats_to_compute
        containing an invalid statistic name "invalid_stat".

        **Expected Result**: The initialization should raise a ValueError with a message
        containing "Unknown statistic". This validates that input validation works correctly,
        which is essential for catching configuration errors early.
        """
        with pytest.raises(ValueError, match="Unknown statistic"):
            SpectralStatsTransform(stats_to_compute=["invalid_stat"])

    def test_init_apply_to_columns_false_error(self):
        """Test initialization with apply_to_columns=False raises error.

        **Assumption**: SpectralStatsTransform currently only supports apply_to_columns=True
        (processing each column independently). If apply_to_columns=False is specified,
        it should raise a NotImplementedError, as this functionality is not yet implemented.

        **Action**: Attempt to create a SpectralStatsTransform with apply_to_columns=False.

        **Expected Result**: The initialization should raise a NotImplementedError with a
        message containing "apply_to_columns=False". This validates that unsupported
        functionality is properly documented and raises appropriate errors.
        """
        with pytest.raises(NotImplementedError, match="apply_to_columns=False"):
            SpectralStatsTransform(stats_to_compute=["mean"], apply_to_columns=False)

    # ========================================================================
    # FFT SPECTRUM COMPUTATION TESTS
    # ========================================================================

    def test_compute_fft_spectrum(self):
        """Test _compute_fft_spectrum method.

        **Assumption**: SpectralStatsTransform._compute_fft_spectrum should compute the
        FFT spectrum (magnitude) of a signal, returning the amplitude spectrum as a numpy
        array. The spectrum should contain non-negative values representing the magnitude
        at each frequency bin.

        **Action**: Create a SpectralStatsTransform and call _compute_fft_spectrum with
        a sinusoidal signal (128 samples).

        **Expected Result**: The result should be a numpy array with length > 0, and all
        values should be non-negative (amplitude is always non-negative). This validates
        that FFT spectrum computation works correctly, which is the foundation for all
        spectral feature extraction.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))

        spectrum = transform._compute_fft_spectrum(signal)

        assert isinstance(spectrum, np.ndarray)
        assert len(spectrum) > 0
        assert np.all(spectrum >= 0)  # Amplitude should be non-negative

    def test_compute_fft_spectrum_empty_signal(self):
        """Test _compute_fft_spectrum with empty signal.

        **Assumption**: SpectralStatsTransform._compute_fft_spectrum should handle edge
        cases like empty signals gracefully, returning an empty spectrum array.

        **Action**: Create a SpectralStatsTransform and call _compute_fft_spectrum with
        an empty signal array.

        **Expected Result**: The result should be a numpy array with length 0. This validates
        that edge cases are handled correctly, preventing crashes when dealing with empty
        or invalid input signals.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        signal = np.array([])

        spectrum = transform._compute_fft_spectrum(signal)

        assert isinstance(spectrum, np.ndarray)
        assert len(spectrum) == 0

    # ========================================================================
    # STATISTIC COMPUTATION TESTS
    # ========================================================================

    def test_compute_stat_empty_spectrum(self):
        """Test _compute_stat with empty spectrum returns NaN.

        **Assumption**: SpectralStatsTransform._compute_stat should handle edge cases like
        empty spectra gracefully, returning NaN when statistics cannot be computed (e.g.,
        mean of empty array is undefined).

        **Action**: Create a SpectralStatsTransform and call _compute_stat with an empty
        spectrum array and statistic "mean".

        **Expected Result**: The result should be NaN. This validates that edge cases are
        handled correctly, preventing crashes when dealing with empty spectra.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        empty_spectrum = np.array([])

        result = transform._compute_stat(empty_spectrum, "mean")

        assert np.isnan(result)

    def test_compute_stat_mean(self):
        """Test _compute_stat with mean statistic.

        **Assumption**: SpectralStatsTransform._compute_stat should compute the mean of
        the spectrum values, providing a measure of average spectral magnitude.

        **Action**: Create a SpectralStatsTransform and call _compute_stat with a spectrum
        array and statistic "mean".

        **Expected Result**: The result should be the mean of the spectrum values. This
        validates that mean computation works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(spectrum, "mean")
        expected = np.mean(spectrum)
        assert abs(result - expected) < 1e-10

    def test_compute_stat_maximum(self):
        """Test _compute_stat with maximum.

        **Assumption**: SpectralStatsTransform should compute the maximum value in the
        FFT spectrum, which represents the peak magnitude across all frequency bins.
        This statistic is useful for identifying dominant frequency components or detecting
        strong periodic patterns in the signal.

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["maximum"].
        Provide a spectrum array with values [1.0, 2.0, 3.0, 4.0, 5.0] and compute the
        maximum statistic.

        **Expected Result**: The result should be 5.0 (the maximum value in the spectrum).
        This validates that maximum computation works correctly, which is essential for
        spectral feature extraction and identifying peak frequencies in time-series signals.
        """
        transform = SpectralStatsTransform(stats_to_compute=["maximum"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(spectrum, "maximum")
        assert result == 5.0

    def test_compute_stat_minimum(self):
        """Test _compute_stat with minimum.

        **Assumption**: SpectralStatsTransform._compute_stat should compute the minimum
        value in the spectrum, which represents the smallest magnitude across all frequency
        bins.

        **Action**: Create a SpectralStatsTransform and call _compute_stat with a spectrum
        array and statistic "minimum".

        **Expected Result**: The result should be the minimum value in the spectrum. This
        validates that minimum computation works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["minimum"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(spectrum, "minimum")
        assert result == 1.0

    def test_compute_stat_root_mean_square(self):
        """Test _compute_stat with root_mean_square.

        **Assumption**: SpectralStatsTransform._compute_stat should compute the root mean
        square (RMS) of the spectrum values, providing a measure of overall spectral energy.

        **Action**: Create a SpectralStatsTransform and call _compute_stat with a spectrum
        array and statistic "root_mean_square".

        **Expected Result**: The result should be the RMS of the spectrum values (sqrt(mean(spectrum^2))).
        This validates that RMS computation works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["root_mean_square"])
        spectrum = np.array([1.0, 2.0, 3.0])

        result = transform._compute_stat(spectrum, "root_mean_square")
        expected = np.sqrt(np.mean(spectrum**2))
        assert abs(result - expected) < 1e-10

    def test_compute_stat_peak_to_peak_value(self):
        """Test _compute_stat with peak_to_peak_value.

        **Assumption**: SpectralStatsTransform._compute_stat should compute the peak-to-peak
        value (difference between maximum and minimum) of the spectrum, providing a measure
        of spectral range.

        **Action**: Create a SpectralStatsTransform and call _compute_stat with a spectrum
        array and statistic "peak_to_peak_value".

        **Expected Result**: The result should be the difference between maximum and minimum
        values in the spectrum. This validates that peak-to-peak computation works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["peak_to_peak_value"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(spectrum, "peak_to_peak_value")
        assert result == 4.0

    def test_compute_stat_variance(self):
        """Test _compute_stat with variance.

        **Assumption**: SpectralStatsTransform._compute_stat should compute the variance
        of the spectrum values, providing a measure of spectral variability.

        **Action**: Create a SpectralStatsTransform and call _compute_stat with a spectrum
        array and statistic "variance".

        **Expected Result**: The result should be the variance of the spectrum values.
        This validates that variance computation works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["variance"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(spectrum, "variance")
        expected = np.var(spectrum)
        assert abs(result - expected) < 1e-10

    def test_compute_stat_skewness(self):
        """Test _compute_stat with skewness.

        **Assumption**: SpectralStatsTransform._compute_stat should compute the skewness
        of the spectrum values, providing a measure of spectral asymmetry.

        **Action**: Create a SpectralStatsTransform and call _compute_stat with a spectrum
        array and statistic "skewness".

        **Expected Result**: The result should be a finite value representing the skewness
        of the spectrum. This validates that skewness computation works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["skewness"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(spectrum, "skewness")
        assert np.isfinite(result)

    def test_compute_stat_kurtosis(self):
        """Test _compute_stat with kurtosis.

        **Assumption**: SpectralStatsTransform._compute_stat should compute the kurtosis
        of the spectrum values, providing a measure of spectral tail heaviness.

        **Action**: Create a SpectralStatsTransform and call _compute_stat with a spectrum
        array and statistic "kurtosis".

        **Expected Result**: The result should be a finite value representing the kurtosis
        of the spectrum. This validates that kurtosis computation works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["kurtosis"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(spectrum, "kurtosis")
        assert np.isfinite(result)

    def test_compute_stat_abs_energy(self):
        """Test _compute_stat with abs_energy.

        **Assumption**: SpectralStatsTransform._compute_stat should compute the absolute
        energy (sum of squared values) of the spectrum, providing a measure of total spectral
        energy.

        **Action**: Create a SpectralStatsTransform and call _compute_stat with a spectrum
        array and statistic "abs_energy".

        **Expected Result**: The result should be the sum of squared spectrum values.
        This validates that absolute energy computation works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["abs_energy"])
        spectrum = np.array([1.0, 2.0, 3.0])

        result = transform._compute_stat(spectrum, "abs_energy")
        expected = np.sum(spectrum**2)
        assert abs(result - expected) < 1e-10

    def test_compute_stat_peak_factor(self):
        """Test _compute_stat with peak_factor.

        **Assumption**: SpectralStatsTransform._compute_stat should compute the peak factor
        (ratio of peak to RMS) of the spectrum, providing a measure of signal peakiness.

        **Action**: Create a SpectralStatsTransform and call _compute_stat with a spectrum
        array and statistic "peak_factor".

        **Expected Result**: The result should be a non-negative finite value. This validates
        that peak factor computation works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["peak_factor"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(spectrum, "peak_factor")
        assert result >= 0
        assert np.isfinite(result)

    def test_compute_stat_change_coefficient(self):
        """Test _compute_stat with change_coefficient.

        **Assumption**: SpectralStatsTransform._compute_stat should compute the change
        coefficient (measure of variability) of the spectrum.

        **Action**: Create a SpectralStatsTransform and call _compute_stat with a spectrum
        array and statistic "change_coefficient".

        **Expected Result**: The result should be a non-negative finite value. This validates
        that change coefficient computation works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["change_coefficient"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(spectrum, "change_coefficient")
        assert result >= 0
        assert np.isfinite(result)

    def test_compute_stat_change_coefficient_single_value(self):
        """Test _compute_stat with change_coefficient and single value.

        **Assumption**: SpectralStatsTransform._compute_stat should handle edge cases like
        single-value spectra when computing change coefficient, returning 0.0 (no change
        in a single value).

        **Action**: Create a SpectralStatsTransform and call _compute_stat with a spectrum
        containing a single value and statistic "change_coefficient".

        **Expected Result**: The result should be 0.0 (no change in a single value). This
        validates that edge cases are handled correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["change_coefficient"])
        spectrum = np.array([1.0])

        result = transform._compute_stat(spectrum, "change_coefficient")
        assert result == 0.0

    def test_compute_stat_clearance_factor(self):
        """Test _compute_stat with clearance_factor.

        **Assumption**: SpectralStatsTransform._compute_stat should compute the clearance
        factor (measure related to peak values) of the spectrum.

        **Action**: Create a SpectralStatsTransform and call _compute_stat with a spectrum
        array and statistic "clearance_factor".

        **Expected Result**: The result should be a non-negative finite value. This validates
        that clearance factor computation works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["clearance_factor"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(spectrum, "clearance_factor")
        assert result >= 0
        assert np.isfinite(result)

    def test_compute_spectral_entropy(self):
        """Test _compute_spectral_entropy.

        **Assumption**: SpectralStatsTransform._compute_spectral_entropy should compute
        the spectral entropy of the spectrum, which measures the randomness or complexity
        of the frequency distribution. Higher entropy indicates more uniform energy
        distribution across frequencies.

        **Action**: Create a SpectralStatsTransform and call _compute_spectral_entropy
        with a spectrum array.

        **Expected Result**: The result should be a non-negative finite value. This validates
        that spectral entropy computation works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["spectral_entropy"])
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_spectral_entropy(spectrum)
        # Spectral entropy can be > 1 if not normalized, or normalized to [0,1]
        # Just check it's finite and non-negative
        assert result >= 0
        assert np.isfinite(result)

    def test_compute_permutation_entropy(self):
        """Test _compute_permutation_entropy.

        **Assumption**: SpectralStatsTransform._compute_permutation_entropy should compute
        the permutation entropy of the spectrum, which measures the complexity based on
        ordinal patterns. This requires parameters pe_dim (embedding dimension) and pe_tau
        (time delay).

        **Action**: Create a SpectralStatsTransform with permutation_entropy in stats_to_compute
        and pe_dim=3, pe_tau=1. Call _compute_permutation_entropy with a spectrum array.

        **Expected Result**: The result should be a non-negative finite value. This validates
        that permutation entropy computation works correctly.
        """
        transform = SpectralStatsTransform(
            stats_to_compute=["permutation_entropy"], pe_dim=3, pe_tau=1
        )
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        result = transform._compute_permutation_entropy(spectrum, dim=3, tau=1)
        assert result >= 0
        assert np.isfinite(result)

    def test_compute_permutation_entropy_short_signal(self):
        """Test _compute_permutation_entropy with short signal.

        **Assumption**: SpectralStatsTransform._compute_permutation_entropy should handle
        edge cases like signals that are too short for the embedding dimension, returning
        NaN when permutation entropy cannot be computed.

        **Action**: Create a SpectralStatsTransform with pe_dim=5 and call _compute_permutation_entropy
        with a very short signal (only 2 values).

        **Expected Result**: The result should be NaN (signal too short for embedding dimension 5).
        This validates that edge cases are handled correctly.
        """
        transform = SpectralStatsTransform(
            stats_to_compute=["permutation_entropy"], pe_dim=5, pe_tau=1
        )
        short_signal = np.array([1.0, 2.0])  # Too short

        result = transform._compute_permutation_entropy(short_signal, dim=5, tau=1)

        assert np.isnan(result)

    def test_compute_stat_unknown_stat_error(self):
        """Test _compute_stat with unknown stat raises error.

        **Assumption**: SpectralStatsTransform._compute_stat should validate that the
        statistic name is supported. If an unknown statistic is provided (which shouldn't
        happen due to __init__ validation, but could occur if the method is called directly),
        it should raise a ValueError.

        **Action**: Create a SpectralStatsTransform and attempt to call _compute_stat with
        an unknown statistic name "unknown_stat".

        **Expected Result**: The call should raise a ValueError with a message containing
        "Unknown statistic handler". This validates that error handling works correctly
        even for edge cases that shouldn't normally occur.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        spectrum = np.array([1.0, 2.0, 3.0])

        # This shouldn't happen due to __init__ validation, but test the path
        with pytest.raises(ValueError, match="Unknown statistic handler"):
            transform._compute_stat(spectrum, "unknown_stat")

    # ========================================================================
    # INPUT VALIDATION TESTS
    # ========================================================================

    def test_validate_input_not_2d_error(self):
        """Test _validate_input with non-2D array raises error.

        **Assumption**: SpectralStatsTransform._validate_input should validate that input
        arrays are 2D (samples × signals). If a 1D array is provided, it should raise a
        ValueError with a descriptive error message.

        **Action**: Create a SpectralStatsTransform and call _validate_input with a 1D array.

        **Expected Result**: The call should raise a ValueError with a message containing
        "must be a 2D array". This validates that input validation works correctly, preventing
        errors downstream when invalid data shapes are provided.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        arr_1d = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="must be a 2D array"):
            transform._validate_input(arr_1d, "test")

    def test_validate_input_raises_on_non_2d_array(self):
        """Test _validate_input raises error on non-2D array (alternative test).

        **Assumption**: Same as test_validate_input_not_2d_error - validates input shape
        checking.

        **Action**: Create a SpectralStatsTransform and call _validate_input with a 1D array.

        **Expected Result**: The call should raise a ValueError with a message containing "2D".
        This validates that input validation works correctly.
        """
        t = SpectralStatsTransform(stats_to_compute=["mean"])
        one_d = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError) as exc:
            t._validate_input(one_d, "features")
        assert "2D" in str(exc.value)

    def test_validate_input_inf_error(self):
        """Test _validate_input with infinite values raises error.

        **Assumption**: SpectralStatsTransform._validate_input should validate that input
        arrays don't contain infinite values, as these can cause issues in FFT and statistical
        computations. If infinite values are found, it should raise a ValueError.

        **Action**: Create a SpectralStatsTransform and call _validate_input with a 2D array
        containing np.inf.

        **Expected Result**: The call should raise a ValueError with a message containing
        "Infinite values". This validates that input validation works correctly, preventing
        errors downstream when invalid data values are provided.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        arr = np.array([[1.0, 2.0], [np.inf, 4.0]])

        with pytest.raises(ValueError, match="Infinite values"):
            transform._validate_input(arr, "test")

    def test_validate_input_raises_on_infinite_values(self):
        """Test _validate_input raises error on infinite values (alternative test).

        **Assumption**: Same as test_validate_input_inf_error - validates infinite value
        checking.

        **Action**: Create a SpectralStatsTransform and call _validate_input with a 2D array
        containing np.inf.

        **Expected Result**: The call should raise a ValueError with a message containing "Infinite".
        This validates that input validation works correctly.
        """
        t = SpectralStatsTransform(stats_to_compute=["mean"])
        arr = np.array([[1.0, np.inf], [2.0, 3.0]])
        with pytest.raises(ValueError) as exc:
            t._validate_input(arr, "features")
        assert "Infinite" in str(exc.value)

    def test_validate_input_short_signal_error(self):
        """Test _validate_input with signal too short raises error.

        **Assumption**: SpectralStatsTransform._validate_input should validate that signals
        are long enough for FFT computation (typically need at least a few samples). If
        a signal is too short (e.g., only 1 row), it should raise a ValueError.

        **Action**: Create a SpectralStatsTransform and call _validate_input with a 2D array
        with only 1 row (signal too short for FFT).

        **Expected Result**: The call should raise a ValueError with a message containing
        "too short". This validates that input validation works correctly, preventing errors
        downstream when signals are too short for spectral analysis.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        arr = np.array([[1.0, 2.0]])  # Only 1 row

        with pytest.raises(ValueError, match="too short"):
            transform._validate_input(arr, "test")

    def test_validate_input_raises_on_too_short_signal_length(self):
        """Test _validate_input raises error on too short signal (alternative test).

        **Assumption**: Same as test_validate_input_short_signal_error - validates signal
        length checking.

        **Action**: Create a SpectralStatsTransform and call _validate_input with a 2D array
        with only 1 row.

        **Expected Result**: The call should raise a ValueError with a message containing "too short".
        This validates that input validation works correctly.
        """
        t = SpectralStatsTransform(stats_to_compute=["mean"])
        # Rows == 1 -> signal length along rows is too short for FFT
        short_arr = np.ones((1, 3))
        with pytest.raises(ValueError) as exc:
            t._validate_input(short_arr, "features")
        assert "too short" in str(exc.value)

    def test_validate_input_returns_array_for_valid_input(self):
        """Test _validate_input returns array for valid input.

        **Assumption**: SpectralStatsTransform._validate_input should return the input array
        unchanged when validation passes. This allows the method to be used as both a validator
        and a pass-through function.

        **Action**: Create a SpectralStatsTransform and call _validate_input with a valid
        2D array (multiple rows, no infinite values, no NaNs).

        **Expected Result**: The method should return the same array content (possibly a copy).
        This validates that validation works correctly for valid inputs.
        """
        t = SpectralStatsTransform(stats_to_compute=["mean", "maximum"])
        valid = np.array([[1.0, 2.0], [3.0, 4.0]])
        returned = t._validate_input(valid, "features")
        # Should return the same array content
        assert_array_equal(returned, valid)

    # ========================================================================
    # TRANSFORM DATA TESTS
    # ========================================================================

    def test_transform_data_mean(self):
        """Test transform_data with mean statistic.

        **Assumption**: SpectralStatsTransform should compute statistics on the FFT
        spectrum of each signal. The "mean" statistic computes the average magnitude
        across all frequency bins in the spectrum, providing a measure of overall
        spectral energy. The transform processes each column (signal) independently.

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["mean"].
        Provide input data with 2 signals (columns) and 128 time points (rows). Each
        signal is a sinusoid. Apply the transform to compute spectral statistics.

        **Expected Result**: The result should be a numpy array with at least 1 dimension
        and all values should be finite. The output represents spectral mean statistics
        for each input signal. This validates that spectral feature extraction works
        correctly, which is essential for frequency-domain analysis of time-series data,
        useful for detecting periodic patterns or characterizing signal properties in
        the frequency domain.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        data = NamedTransformInput(
            features=np.array(
                [
                    np.sin(np.linspace(0, 4 * np.pi, 128)),
                    np.cos(np.linspace(0, 4 * np.pi, 128)),
                ]
            ).T
        )  # Shape (128, 2)
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        # Output shape depends on implementation - check it's valid
        assert result.ndim >= 1
        assert result.shape[0] >= 1  # At least one row
        assert np.all(np.isfinite(result))

    def test_transform_data_multiple_stats(self):
        """Test transform_data with multiple statistics.

        **Assumption**: SpectralStatsTransform should compute multiple statistics on each
        signal's FFT spectrum, concatenating the results into a flattened array. The output
        shape should be (n_signals * n_stats,), where statistics are grouped by signal.

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["mean", "variance", "kurtosis"].
        Provide input data with 2 signals and apply the transform.

        **Expected Result**: The result should be a numpy array with shape (6,) representing
        2 signals × 3 stats. All values should be finite. This validates that multiple
        statistics can be computed simultaneously, which is essential for comprehensive
        spectral feature extraction.
        """
        transform = SpectralStatsTransform(
            stats_to_compute=["mean", "variance", "kurtosis"]
        )
        data = NamedTransformInput(
            features=np.array(
                [
                    np.sin(np.linspace(0, 4 * np.pi, 128)),
                    np.cos(np.linspace(0, 4 * np.pi, 128)),
                ]
            ).T
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        # Output is flattened: (n_signals * n_stats,) = (2 * 3,) = (6,)
        assert result.shape == (6,)
        assert np.all(np.isfinite(result))

    def test_transform_data_entropy_stats(self):
        """Test transform_data with entropy statistics.

        **Assumption**: SpectralStatsTransform should compute entropy-based statistics
        (spectral_entropy, shannon_entropy) on the FFT spectrum. These statistics measure
        the randomness or complexity of the frequency distribution and are typically
        normalized to the range [0, 1].

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["spectral_entropy", "shannon_entropy"].
        Provide input data with 2 signals and apply the transform.

        **Expected Result**: The result should be a numpy array with all values finite and
        in the range [0, 1] (normalized entropy). This validates that entropy statistics
        work correctly, which are useful for characterizing signal complexity and randomness.
        """
        transform = SpectralStatsTransform(
            stats_to_compute=["spectral_entropy", "shannon_entropy"]
        )
        data = NamedTransformInput(
            features=np.array(
                [
                    np.sin(np.linspace(0, 4 * np.pi, 128)),
                    np.cos(np.linspace(0, 4 * np.pi, 128)),
                ]
            ).T
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))
        # Entropy should be between 0 and 1 (normalized)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_transform_data_all_valid_stats(self):
        """Test transform_data with all valid statistics.

        **Assumption**: SpectralStatsTransform should support computing all available
        statistics simultaneously. This allows comprehensive spectral feature extraction
        in a single pass, which is more efficient than multiple transform calls.

        **Action**: Create a SpectralStatsTransform with stats_to_compute containing all
        14 valid statistics. Provide input data with 1 signal and apply the transform.

        **Expected Result**: The result should be a numpy array with shape (14,) representing
        all statistics for the single signal. All values should be finite. This validates
        that all statistics can be computed together, which is essential for comprehensive
        spectral analysis.
        """
        all_stats = [
            "mean",
            "maximum",
            "minimum",
            "root_mean_square",
            "peak_to_peak_value",
            "variance",
            "skewness",
            "kurtosis",
            "abs_energy",
            "peak_factor",
            "change_coefficient",
            "clearance_factor",
            "spectral_entropy",
            "shannon_entropy",
        ]
        transform = SpectralStatsTransform(stats_to_compute=all_stats)
        data = NamedTransformInput(
            features=np.array([np.sin(np.linspace(0, 4 * np.pi, 128))]).T
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        # Output is flattened: (n_signals * n_stats,) = (1 * 14,) = (14,)
        assert result.shape == (len(all_stats),)
        assert np.all(np.isfinite(result))

    def test_transform_data_single_signal(self):
        """Test transform_data with single signal.

        **Assumption**: SpectralStatsTransform should work correctly with single-signal
        input data (single column). This is a common use case when analyzing individual
        time-series signals.

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["mean", "variance"].
        Provide input data with 1 signal (single column) and apply the transform.

        **Expected Result**: The result should be a numpy array with at least 1 dimension
        and all values should be finite. This validates that single-signal processing
        works correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean", "variance"])
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert result.ndim >= 1
        assert np.all(np.isfinite(result))

    def test_transform_data_permutation_entropy(self):
        """Test transform_data with permutation_entropy.

        **Assumption**: SpectralStatsTransform should compute permutation entropy when
        it's included in stats_to_compute. This requires additional parameters pe_dim
        and pe_tau to be set during initialization.

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["permutation_entropy"],
        pe_dim=3, and pe_tau=1. Provide input data and apply the transform.

        **Expected Result**: The result should be a numpy array with all values finite
        and non-negative. This validates that permutation entropy computation works correctly
        in the transform pipeline.
        """
        transform = SpectralStatsTransform(
            stats_to_compute=["permutation_entropy"], pe_dim=3, pe_tau=1
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))

    def test_transform_data_with_spectral_entropy(self):
        """Test transform_data with spectral_entropy statistic.

        **Assumption**: SpectralStatsTransform should compute spectral entropy, which measures
        the randomness or complexity of the frequency distribution. Higher entropy indicates
        more uniform energy distribution across frequencies, while lower entropy indicates
        energy concentrated in specific frequencies. Entropy values are always non-negative.

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["spectral_entropy"].
        Provide a sinusoidal signal (which has energy concentrated at specific frequencies) and
        apply the transform.

        **Expected Result**: The result should have shape (1,), representing a single entropy
        value. The value should be finite and non-negative. For a sine wave, entropy should be
        relatively low since energy is concentrated at specific frequencies. This validates that
        spectral entropy computation works correctly, which is essential for characterizing signal
        complexity and randomness in frequency-domain analysis.
        """
        t = SpectralStatsTransform(stats_to_compute=["spectral_entropy"])

        # Simple sine wave signal
        n = 128
        signal = np.sin(np.linspace(0, 4 * np.pi, n))
        data = {"features": signal.reshape(-1, 1)}

        result = t.transform_data(data, metadata={})

        assert result.shape == (1,)
        assert np.isfinite(result[0])
        assert result[0] >= 0  # Entropy should be non-negative

    def test_transform_data_with_permutation_entropy(self):
        """Test transform_data with permutation_entropy statistic.

        **Assumption**: SpectralStatsTransform should compute permutation entropy, which measures
        complexity based on ordinal patterns in the signal. This requires a signal long enough
        for the embedding dimension.

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["permutation_entropy"],
        pe_dim=3, and pe_tau=1. Provide a signal long enough for permutation entropy computation.

        **Expected Result**: The result should have shape (1,), representing a single entropy
        value. The value should be finite and non-negative. This validates that permutation
        entropy computation works correctly in the transform pipeline.
        """
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

    def test_transform_data_multiple_signals_multiple_stats(self):
        """Test transform_data with multiple signals and multiple statistics.

        **Assumption**: SpectralStatsTransform should process multiple signals (columns)
        independently, computing all specified statistics for each signal. The output
        should be flattened: (n_signals * n_stats,).

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["mean", "maximum", "minimum"].
        Provide input data with 3 signals (columns) and 5 samples (rows). Apply the transform.

        **Expected Result**: The result should be a numpy array with shape (9,) representing
        3 signals × 3 stats. All values should be finite. This validates that multi-signal
        processing works correctly, which is essential for analyzing multiple time-series
        signals simultaneously.
        """
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

    def test_transform_data_with_named_transform_input(self):
        """Test transform_data with a proper NamedTransformInput dict.

        **Assumption**: SpectralStatsTransform.transform_data should accept NamedTransformInput
        (a dict with data keys like "features"). The transform should extract the data from
        the dict and process it correctly.

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["mean", "maximum", "variance"].
        Provide input data as a dict with "features" key containing a 2D array (4 rows, 3 columns).
        Apply the transform.

        **Expected Result**: The result should be a numpy array with shape (9,) representing
        3 signals × 3 stats. All values should be finite. This validates that NamedTransformInput
        format is handled correctly, which is essential for compatibility with the framework's
        data pipeline.
        """
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

    def test_transform_data_empty(self):
        """Test transform_data with empty data.

        **Assumption**: SpectralStatsTransform.transform_data should handle edge cases like
        empty input data gracefully, either returning an empty result or raising an appropriate
        error.

        **Action**: Create a SpectralStatsTransform and call transform_data with empty data
        (0 rows, 1 column).

        **Expected Result**: The method should either return an empty array or raise an appropriate
        error (ValueError or IndexError). This validates that edge cases are handled correctly.
        """
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

    def test_transform_data_raises_on_empty_data(self):
        """Test transform_data raises error on empty dict.

        **Assumption**: SpectralStatsTransform.transform_data should raise a ValueError
        when an empty dict is provided, as there's no data to process.

        **Action**: Create a SpectralStatsTransform and call transform_data with an empty
        dict {}.

        **Expected Result**: The call should raise a ValueError with a message containing
        "No data provided". This validates that error handling works correctly for invalid
        inputs.
        """
        t = SpectralStatsTransform(stats_to_compute=["mean"])

        with pytest.raises(ValueError) as exc:
            t.transform_data({}, metadata={})
        assert "No data provided" in str(exc.value)

    def test_transform_data_with_nans(self):
        """Test transform_data with NaN values in signal.

        **Assumption**: SpectralStatsTransform.transform_data should detect NaN values
        in input signals and raise a ValueError, as NaNs indicate invalid or missing data
        that cannot be processed meaningfully. The error message should indicate which
        signal and statistic contain NaNs.

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["mean"]. Provide
        input data with a signal containing NaN values. Apply the transform.

        **Expected Result**: The transform should raise a ValueError with a message containing
        "is NaN". This validates that NaN detection works correctly, which is essential for
        data quality assurance and preventing silent failures when invalid data is provided.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        signal[10:20] = np.nan  # Add NaNs
        data = NamedTransformInput(features=signal.reshape(-1, 1))
        metadata = {}

        # Should log warning and return NaN stats
        with pytest.raises(ValueError, match="is NaN"):
            transform.transform_data(data, metadata)

    def test_transform_data_raises_error_on_nans(self):
        """Test transform_data raises error on NaNs (alternative test).

        **Assumption**: Same as test_transform_data_with_nans - validates NaN detection
        and error handling.

        **Action**: Create a SpectralStatsTransform and provide input data with NaN values
        in one of the signals. Apply the transform.

        **Expected Result**: The transform should raise a ValueError with a message indicating
        which signal and statistic contain NaNs. This validates that NaN detection works
        correctly.
        """
        # Arrange
        stats = ["mean", "maximum"]
        t = SpectralStatsTransform(stats_to_compute=stats)
        data = {
            # Column 0: valid signal [1, 3, 5]
            # Column 1: contains NaN [NaN, 4, 6] -> This should trigger the error
            "features": np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        }

        # Act & Assert
        # We expect a ValueError because the signal in column 1 has NaNs,
        # which causes the computed stats to be NaN.
        with pytest.raises(ValueError, match="Stat 'mean' for signal 1 is NaN"):
            t.transform_data(data, metadata={})

    # ========================================================================
    # FEATURE NAMES TESTS
    # ========================================================================

    def test_get_feature_names(self):
        """Test get_feature_names method.

        **Assumption**: SpectralStatsTransform.get_feature_names should generate descriptive
        feature names for the computed spectral statistics. The names should indicate that
        they are spectral features ("spec_"), which column/signal they come from ("col"),
        and which statistic was computed. With 3 signals and 2 stats, there should be 6
        feature names total.

        **Action**: Create a SpectralStatsTransform with stats_to_compute=["mean", "variance"].
        Call get_feature_names with input_keys=["features"] and input_shapes indicating 3
        columns (signals).

        **Expected Result**: The result should be a list of 6 feature names (3 signals × 2
        stats), and all names should contain "spec_" and "col" to indicate they are spectral
        features from specific columns. This validates that feature naming works correctly,
        which is essential for interpretability and tracking which features correspond to
        which signals and statistics.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean", "variance"])
        input_keys = ["features"]
        input_shapes = {"features": (128, 3)}  # 128 rows, 3 columns

        feature_names = transform.get_feature_names(input_keys, input_shapes)

        assert len(feature_names) == 6  # 3 signals * 2 stats
        assert all("spec_" in name for name in feature_names)
        assert all("col" in name for name in feature_names)

    def test_get_feature_names_empty_keys(self):
        """Test get_feature_names with empty input_keys.

        **Assumption**: SpectralStatsTransform.get_feature_names should handle edge cases
        like empty input_keys gracefully, returning an empty list when there are no input
        keys to process.

        **Action**: Create a SpectralStatsTransform and call get_feature_names with empty
        input_keys [] and empty input_shapes {}.

        **Expected Result**: The result should be an empty list []. This validates that
        edge cases are handled correctly.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        feature_names = transform.get_feature_names([], {})

        assert feature_names == []

    def test_get_feature_names_missing_shape_error(self):
        """Test get_feature_names with missing shape raises error.

        **Assumption**: SpectralStatsTransform.get_feature_names should validate that
        input_shapes contains shapes for all input_keys. If a shape is missing, it should
        raise a KeyError with a descriptive error message.

        **Action**: Create a SpectralStatsTransform and call get_feature_names with
        input_keys=["features"] but input_shapes={} (missing shape for "features").

        **Expected Result**: The call should raise a KeyError with a message containing
        "not provided". This validates that error handling works correctly for missing
        configuration.
        """
        transform = SpectralStatsTransform(stats_to_compute=["mean"])
        input_keys = ["features"]
        input_shapes = {}  # Missing shape

        with pytest.raises(KeyError, match="not provided"):
            transform.get_feature_names(input_keys, input_shapes)
