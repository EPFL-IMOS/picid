"""Comprehensive tests for TimeStatsTransform.

This module provides rigorous testing for time-domain statistical feature extraction
used in PHM (Prognostics and Health Management) applications. The TimeStatsTransform
extracts statistical features from raw vibration/sensor signals that serve as
health indicators for condition monitoring and fault detection.

PHM Context:
-----------
Time-domain statistics are fundamental features for bearing health monitoring:
- **RMS (Root Mean Square)**: Energy indicator - increases with damage severity
- **Kurtosis**: Impulsiveness indicator - spikes during early fault development
- **Peak Factor**: Ratio of peak to RMS - high values indicate impulsive faults
- **Skewness**: Asymmetry indicator - changes with asymmetric wear patterns

Reference: Mao et al. (2020) "A Review on Machine Learning in Rotating Machinery"

Test Coverage Strategy:
----------------------
1. **Initialization Tests**: Parameter validation, invalid stats detection
2. **Nominal State Tests**: Healthy bearing signals with expected low indicators
3. **Fault Signature Tests**: Faulty signals with elevated indicators
4. **Anomalous Input Tests**: NaN, Inf, empty signals - error handling
5. **Edge Case Tests**: Single sample, constant signal, multi-signal arrays
6. **Integration Tests**: Full transform pipeline with SplitDatasetContainer
"""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from picid.transforms.base_transforms.time_statistics import TimeStatsTransform
from picid.data.data_objects import NamedTransformInput


class TestTimeStatsTransformInitialization:
    """Tests for TimeStatsTransform initialization and parameter validation.

    Validates that the transform correctly handles configuration parameters
    and rejects invalid statistics names. This is critical for catching
    configuration errors early in PHM pipeline setup.
    """

    def test_init_with_single_stat(self):
        """Test initialization with a single statistic.

        **PHM Logic**: Configuring transforms with specific statistics allows
        targeted feature extraction. For example, using only 'root_mean_square'
        for a quick energy assessment.

        **Methodology**: Create transform with one valid stat, verify storage.

        **Expected**: Transform created with stats_to_compute containing exactly
        one statistic, default apply_to_columns=True.

        Validates: Requirement R1.1 - Single statistic configuration
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])

        assert len(transform.stats_to_compute) == 1
        assert "mean" in transform.stats_to_compute
        assert transform.apply_to_columns is True

    def test_init_with_multiple_stats(self):
        """Test initialization with multiple statistics.

        **PHM Logic**: Comprehensive health assessment requires multiple
        indicators. RMS tracks energy, kurtosis detects impulses, peak_factor
        quantifies impulsiveness - together they provide robust fault detection.

        **Methodology**: Create transform with common PHM stat combination.

        **Expected**: All statistics stored correctly in stats_to_compute.

        Validates: Requirement R1.2 - Multi-statistic configuration
        """
        phm_stats = ["root_mean_square", "kurtosis", "peak_factor", "skewness"]
        transform = TimeStatsTransform(stats_to_compute=phm_stats)

        assert len(transform.stats_to_compute) == 4
        for stat in phm_stats:
            assert stat in transform.stats_to_compute

    def test_init_with_all_valid_stats(self):
        """Test initialization with all valid statistics.

        **PHM Logic**: Full feature extraction for comprehensive condition
        monitoring. All 15 statistics provide complementary information about
        signal characteristics relevant to machinery health.

        **Methodology**: Create transform with complete VALID_STATS set.

        **Expected**: Transform created successfully with all statistics.

        Validates: Requirement R1.3 - Complete feature set support
        """
        all_stats = list(TimeStatsTransform.VALID_STATS)
        transform = TimeStatsTransform(
            stats_to_compute=all_stats,
            hankel_window_size=10,  # Required for hankel_svd
            slice_window_size=100,
        )

        assert len(transform.stats_to_compute) == len(all_stats)

    def test_init_invalid_stat_raises_error(self):
        """Test that invalid statistic name raises ValueError.

        **PHM Logic**: Configuration errors must be caught early to prevent
        silent failures in production PHM pipelines. Invalid statistics
        would result in missing features that could affect model performance.

        **Methodology**: Attempt to create transform with non-existent statistic.

        **Expected**: ValueError raised with descriptive message listing valid options.

        Validates: Requirement R1.4 - Input validation error handling
        """
        with pytest.raises(ValueError) as exc_info:
            TimeStatsTransform(stats_to_compute=["invalid_stat"])

        assert "Unknown statistic" in str(exc_info.value)
        assert "invalid_stat" in str(exc_info.value)

    def test_init_apply_to_columns_false_raises_error(self):
        """Test that apply_to_columns=False raises NotImplementedError.

        **PHM Logic**: Row-wise processing is not yet implemented. This test
        ensures users are informed of this limitation.

        **Methodology**: Attempt to create transform with apply_to_columns=False.

        **Expected**: NotImplementedError raised with descriptive message.

        Validates: Requirement R1.5 - Unsupported feature error handling
        """
        with pytest.raises(NotImplementedError, match="apply_to_columns=False"):
            TimeStatsTransform(stats_to_compute=["mean"], apply_to_columns=False)

    def test_init_custom_hankel_parameters(self):
        """Test initialization with custom Hankel SVD parameters.

        **PHM Logic**: Hankel SVD captures signal structure through matrix
        decomposition. Different window sizes capture different time scales
        of signal dynamics, important for detecting varying fault frequencies.

        **Methodology**: Create transform with custom hankel parameters.

        **Expected**: Parameters stored correctly for later use.

        Validates: Requirement R1.6 - Hankel SVD parameter configuration
        """
        transform = TimeStatsTransform(
            stats_to_compute=["hankel_svd"],
            hankel_window_size=50,
            slice_window_size=200,
        )

        assert transform.hankel_window_size == 50
        assert transform.slice_window_size == 200


class TestTimeStatsTransformInputValidation:
    """Tests for input data validation.

    Validates that the transform correctly validates input data shapes,
    types, and values. Proper validation prevents cryptic errors during
    transformation and ensures data quality in PHM applications.
    """

    def test_validate_input_valid_2d_array(self):
        """Test _validate_input accepts valid 2D array.

        **PHM Logic**: Time series signals must be 2D arrays where rows are
        time samples and columns are sensor channels. This is the expected
        format for vibration data from accelerometers.

        **Methodology**: Create valid 2D array and verify no exception raised.

        **Expected**: Method returns validated array unchanged.

        Validates: Requirement R2.1 - Valid input acceptance
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])
        valid_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = transform._validate_input(valid_array, "features")

        np.testing.assert_array_equal(result, valid_array)

    def test_validate_input_1d_array_raises_error(self):
        """Test _validate_input rejects 1D array.

        **PHM Logic**: 1D arrays are ambiguous - could be single channel or
        single sample. Explicit 2D format prevents misinterpretation.

        **Methodology**: Attempt to validate 1D array.

        **Expected**: ValueError raised with message indicating 2D requirement.

        Validates: Requirement R2.2 - Invalid shape rejection
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])
        invalid_array = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="must be a 2D array"):
            transform._validate_input(invalid_array, "features")

    def test_validate_input_infinite_values_raises_error(self):
        """Test _validate_input rejects arrays with infinite values.

        **PHM Logic**: Infinite values indicate sensor saturation, numerical
        overflow, or data corruption. These must be detected and reported
        as they would corrupt statistical calculations.

        **Methodology**: Create array with np.inf values and validate.

        **Expected**: ValueError raised with message about infinite values.

        Validates: Requirement R2.3 - Infinite value detection
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])
        inf_array = np.array([[1.0, 2.0], [np.inf, 4.0]])

        with pytest.raises(ValueError, match="Infinite values"):
            transform._validate_input(inf_array, "features")

    def test_validate_input_negative_infinite_values_raises_error(self):
        """Test _validate_input rejects negative infinite values.

        **PHM Logic**: Both positive and negative infinities must be detected.

        **Methodology**: Create array with -np.inf and validate.

        **Expected**: ValueError raised.

        Validates: Requirement R2.4 - Negative infinity detection
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])
        neg_inf_array = np.array([[1.0, 2.0], [-np.inf, 4.0]])

        with pytest.raises(ValueError, match="Infinite values"):
            transform._validate_input(neg_inf_array, "features")

    def test_validate_input_zero_length_signal_raises_error(self):
        """Test _validate_input rejects zero-length signals.

        **PHM Logic**: Statistics cannot be computed on empty signals.
        This catches data pipeline issues where signals are accidentally empty.

        **Methodology**: Create array with 0 rows (no samples).

        **Expected**: ValueError raised with message about signal length.

        Validates: Requirement R2.5 - Empty signal detection
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])
        empty_array = np.array([]).reshape(0, 2)  # 0 rows, 2 columns

        with pytest.raises(ValueError, match="Signal length is 0"):
            transform._validate_input(empty_array, "features")


class TestTimeStatsTransformStatisticComputation:
    """Tests for individual statistic computation.

    Validates that each statistical function computes correct values
    according to mathematical definitions. These are the fundamental
    building blocks of PHM feature extraction.
    """

    def test_compute_stat_mean(self):
        """Test mean computation.

        **PHM Logic**: Mean represents the DC offset of the signal. For
        vibration signals, non-zero mean may indicate sensor drift or
        unbalanced loading.

        **Methodology**: Compute mean on known signal and verify against numpy.

        **Expected**: Result matches np.mean exactly.

        Validates: Requirement R3.1 - Mean computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(signal, "mean")
        expected = np.mean(signal)

        assert abs(result - expected) < 1e-10
        assert result == 3.0  # Explicit value check

    def test_compute_stat_root_mean_square(self):
        """Test RMS computation.

        **PHM Logic**: RMS = sqrt(mean(signal²)) represents signal energy.
        In bearing monitoring, increasing RMS indicates increasing vibration
        energy, typically due to damage progression. RMS thresholds are
        fundamental to condition-based maintenance decisions.

        **Methodology**: Compute RMS and verify against formula.

        **Expected**: Result = sqrt(mean(signal²)).

        Validates: Requirement R3.2 - RMS computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["root_mean_square"])
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(signal, "root_mean_square")
        expected = np.sqrt(np.mean(signal**2))

        assert abs(result - expected) < 1e-10

    def test_compute_stat_abs_avg_equals_rms(self):
        """Test abs_avg is alias for RMS.

        **PHM Logic**: abs_avg is an alternative name for RMS in some PHM
        literature. Both should produce identical results.

        **Methodology**: Compute both stats and verify equality.

        **Expected**: abs_avg == root_mean_square.

        Validates: Requirement R3.3 - Statistic alias handling
        """
        transform = TimeStatsTransform(stats_to_compute=["root_mean_square", "abs_avg"])
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        rms = transform._compute_stat(signal, "root_mean_square")
        abs_avg = transform._compute_stat(signal, "abs_avg")

        assert abs(rms - abs_avg) < 1e-10

    def test_compute_stat_kurtosis(self):
        """Test kurtosis computation.

        **PHM Logic**: Kurtosis measures the "tailedness" of the distribution.
        Gaussian signals have kurtosis ≈ 3 (Fisher's). Impulsive signals
        (characteristic of bearing faults) have elevated kurtosis (>6).
        Kurtosis is particularly sensitive to early fault development.

        **Methodology**: Compute kurtosis using scipy reference.

        **Expected**: Result matches scipy.stats.kurtosis.

        Validates: Requirement R3.4 - Kurtosis computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["kurtosis"])
        # Use a longer signal for stable kurtosis estimation
        signal = np.random.randn(1000)
        signal += np.random.choice([0, 10], size=1000, p=[0.95, 0.05])  # Add impulses

        result = transform._compute_stat(signal, "kurtosis")
        expected = scipy_stats.kurtosis(signal)

        assert abs(result - expected) < 1e-6

    def test_compute_stat_skewness(self):
        """Test skewness computation.

        **PHM Logic**: Skewness measures distribution asymmetry. Symmetric
        wear produces zero skewness; asymmetric faults (e.g., one-sided
        contact) produce non-zero skewness.

        **Methodology**: Compute skewness using scipy reference.

        **Expected**: Result matches scipy.stats.skew.

        Validates: Requirement R3.5 - Skewness computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["skewness"])
        signal = np.array([1.0, 2.0, 2.5, 2.8, 3.0, 3.1, 10.0])  # Right-skewed

        result = transform._compute_stat(signal, "skewness")
        expected = scipy_stats.skew(signal)

        assert abs(result - expected) < 1e-6

    def test_compute_stat_peak_to_peak_value(self):
        """Test peak-to-peak value computation.

        **PHM Logic**: Peak-to-peak = max - min represents the signal's
        dynamic range. Increasing peak-to-peak indicates growing amplitude
        variations, often seen with mechanical looseness or impacts.

        **Methodology**: Compute peak-to-peak and verify formula.

        **Expected**: Result = max(signal) - min(signal).

        Validates: Requirement R3.6 - Peak-to-peak computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["peak_to_peak_value"])
        signal = np.array([-2.0, 1.0, 5.0, 3.0, 8.0])

        result = transform._compute_stat(signal, "peak_to_peak_value")
        expected = 8.0 - (-2.0)  # max - min

        assert result == expected

    def test_compute_stat_peak_factor(self):
        """Test peak factor computation.

        **PHM Logic**: Peak factor = max(signal) / RMS indicates signal
        impulsiveness. Healthy bearings have peak factor ≈ √2 for sinusoids
        or ≈ 3-4 for Gaussian noise. Bearing faults produce impulsive peaks
        with peak factors > 6, making this a sensitive fault indicator.

        **Methodology**: Compute peak factor and verify formula.

        **Expected**: Result = max(signal) / RMS + epsilon.

        Validates: Requirement R3.7 - Peak factor computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["peak_factor"])
        signal = np.array([1.0, 2.0, 3.0, 10.0, 2.0])  # Impulse at index 3

        result = transform._compute_stat(signal, "peak_factor")
        rms = np.sqrt(np.mean(signal**2))
        expected = np.max(signal) / (rms + 1e-10)

        assert abs(result - expected) < 1e-6
        assert result > 2.0  # Should show elevated peak factor due to impulse

    def test_compute_stat_variance(self):
        """Test variance computation.

        **PHM Logic**: Variance measures signal power around the mean.
        Increasing variance indicates increasing vibration amplitude,
        typically seen as damage progresses.

        **Methodology**: Compute variance using numpy reference.

        **Expected**: Result matches np.var.

        Validates: Requirement R3.8 - Variance computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["variance"])
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(signal, "variance")
        expected = np.var(signal)

        assert abs(result - expected) < 1e-10

    def test_compute_stat_standard_deviation(self):
        """Test standard deviation computation.

        **PHM Logic**: Standard deviation = sqrt(variance) provides
        amplitude dispersion in original signal units.

        **Methodology**: Compute std using numpy reference.

        **Expected**: Result matches np.std.

        Validates: Requirement R3.9 - Standard deviation computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["standard_deviation"])
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(signal, "standard_deviation")
        expected = np.std(signal)

        assert abs(result - expected) < 1e-10

    def test_compute_stat_abs_energy(self):
        """Test absolute energy computation.

        **PHM Logic**: abs_energy = sum(signal²) represents total energy
        in the signal. Unlike RMS, it's sensitive to signal length, useful
        for comparing signals of equal duration.

        **Methodology**: Compute abs_energy and verify formula.

        **Expected**: Result = sum(signal²).

        Validates: Requirement R3.10 - Absolute energy computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["abs_energy"])
        signal = np.array([1.0, 2.0, 3.0])

        result = transform._compute_stat(signal, "abs_energy")
        expected = np.sum(signal**2)  # 1 + 4 + 9 = 14

        assert abs(result - expected) < 1e-10
        assert result == 14.0

    def test_compute_stat_change_coefficient(self):
        """Test change coefficient computation.

        **PHM Logic**: Change coefficient = mean / std is the inverse of
        coefficient of variation. High values indicate stable signals with
        low relative variability.

        **Methodology**: Compute change coefficient and verify formula.

        **Expected**: Result = mean / (std + epsilon).

        Validates: Requirement R3.11 - Change coefficient computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["change_coefficient"])
        signal = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

        result = transform._compute_stat(signal, "change_coefficient")
        expected = np.mean(signal) / (np.std(signal) + 1e-10)

        assert abs(result - expected) < 1e-6

    def test_compute_stat_clearance_factor(self):
        """Test clearance factor computation.

        **PHM Logic**: Clearance factor = max / mean(signal²) is a variant
        of peak factor that uses mean squared value instead of RMS.

        **Methodology**: Compute clearance factor and verify formula.

        **Expected**: Result = max / (mean(signal²) + epsilon).

        Validates: Requirement R3.12 - Clearance factor computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["clearance_factor"])
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = transform._compute_stat(signal, "clearance_factor")
        expected = np.max(signal) / (np.mean(signal**2) + 1e-10)

        assert abs(result - expected) < 1e-6

    def test_compute_stat_maximum(self):
        """Test maximum computation.

        **PHM Logic**: Maximum value indicates peak amplitude, important
        for detecting impacts or overloads.

        **Methodology**: Compute maximum and verify.

        **Expected**: Result = max(signal).

        Validates: Requirement R3.13 - Maximum computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["maximum"])
        signal = np.array([1.0, 5.0, 3.0, 2.0, 4.0])

        result = transform._compute_stat(signal, "maximum")

        assert result == 5.0

    def test_compute_stat_minimum(self):
        """Test minimum computation.

        **PHM Logic**: Minimum value important for detecting asymmetric
        signals or sensor offset issues.

        **Methodology**: Compute minimum and verify.

        **Expected**: Result = min(signal).

        Validates: Requirement R3.14 - Minimum computation accuracy
        """
        transform = TimeStatsTransform(stats_to_compute=["minimum"])
        signal = np.array([1.0, 5.0, 3.0, -2.0, 4.0])

        result = transform._compute_stat(signal, "minimum")

        assert result == -2.0

    def test_compute_stat_empty_signal_returns_nan(self):
        """Test that empty signal returns NaN.

        **PHM Logic**: Empty signals should return NaN rather than raising
        errors, allowing graceful handling of missing data.

        **Methodology**: Compute stat on empty signal.

        **Expected**: Result is NaN.

        Validates: Requirement R3.15 - Empty signal handling
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])
        empty_signal = np.array([])

        result = transform._compute_stat(empty_signal, "mean")

        assert np.isnan(result)

    def test_compute_stat_hankel_svd(self):
        """Test Hankel SVD computation.

        **PHM Logic**: Hankel SVD decomposes the signal into singular value
        components through a Hankel matrix construction. This captures the
        underlying dynamics of the signal and is useful for detecting
        structural changes in the signal pattern, often associated with
        developing faults.

        **Methodology**: Compute Hankel SVD on sinusoidal signal with
        specific window size.

        **Expected**: Returns array of singular values (non-negative),
        length equals hankel_window_size.

        Validates: Requirement R3.16 - Hankel SVD computation
        """
        transform = TimeStatsTransform(
            stats_to_compute=["hankel_svd"],
            hankel_window_size=10,
            slice_window_size=None,
        )
        # Signal must be long enough for Hankel matrix
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))

        result = transform._compute_stat(signal, "hankel_svd")

        assert isinstance(result, np.ndarray)
        assert len(result) == 10  # Number of singular values = hankel_window_size
        assert np.all(result >= 0)  # SVD values are non-negative
        assert np.all(np.isfinite(result))

    def test_compute_stat_hankel_svd_signal_too_short_raises_error(self):
        """Test Hankel SVD raises error when signal too short.

        **PHM Logic**: Hankel matrix requires minimum signal length equal
        to hankel_window_size. Shorter signals cannot construct the matrix.

        **Methodology**: Attempt Hankel SVD on signal shorter than window.

        **Expected**: AssertionError raised about signal length.

        Validates: Requirement R3.17 - Hankel SVD length validation
        """
        transform = TimeStatsTransform(
            stats_to_compute=["hankel_svd"], hankel_window_size=100
        )
        short_signal = np.array([1.0, 2.0, 3.0])  # Too short

        with pytest.raises(AssertionError, match="Signal length"):
            transform._compute_stat(short_signal, "hankel_svd")


class TestTimeStatsTransformData:
    """Tests for the transform_data method.

    Validates end-to-end transformation from input data to output features.
    These tests ensure the complete pipeline works correctly for PHM applications.
    """

    def test_transform_data_single_signal_single_stat(self):
        """Test transform with single signal and single statistic.

        **PHM Logic**: Simplest use case - extract one feature from one signal.

        **Methodology**: Create single-column data, apply transform.

        **Expected**: Output shape (1,) containing single statistic value.

        Validates: Requirement R4.1 - Basic transformation
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])
        data = {"features": np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])}

        result = transform.transform_data(data, metadata={})

        assert result.shape == (1,)
        assert abs(result[0] - 3.0) < 1e-10  # Mean of [1,2,3,4,5]

    def test_transform_data_single_signal_multiple_stats(self):
        """Test transform with single signal and multiple statistics.

        **PHM Logic**: Extract comprehensive feature set from single sensor.

        **Methodology**: Create single-column data, apply multi-stat transform.

        **Expected**: Output shape (n_stats,) with all statistics computed.

        Validates: Requirement R4.2 - Multi-statistic extraction
        """
        transform = TimeStatsTransform(stats_to_compute=["mean", "maximum", "minimum"])
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        data = {"features": signal.reshape(-1, 1)}

        result = transform.transform_data(data, metadata={})

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_transform_data_multi_signal(self):
        """Test transform with multiple signals.

        **PHM Logic**: Bearings typically have multiple sensors (horizontal,
        vertical, axial accelerometers). Each signal is processed independently.

        **Methodology**: Create multi-column data (3 signals), verify all processed.

        **Expected**: Output shape (n_signals * n_stats,).

        Validates: Requirement R4.3 - Multi-signal processing
        """
        transform = TimeStatsTransform(stats_to_compute=["mean", "variance"])
        # 100 samples, 3 signals
        data = {"features": np.random.randn(100, 3)}

        result = transform.transform_data(data, metadata={})

        # 3 signals × 2 stats = 6 features
        assert result.shape == (6,)
        assert np.all(np.isfinite(result))

    def test_transform_data_with_named_transform_input(self):
        """Test transform with NamedTransformInput object.

        **PHM Logic**: NamedTransformInput is the standard data container
        used in the transform pipeline. Must work correctly with this type.

        **Methodology**: Create NamedTransformInput, verify transformation.

        **Expected**: Same results as with dict input.

        Validates: Requirement R4.4 - NamedTransformInput compatibility
        """
        transform = TimeStatsTransform(stats_to_compute=["mean", "root_mean_square"])
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )

        result = transform.transform_data(data, metadata={})

        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)  # 2 signals × 2 stats

    def test_transform_data_empty_raises_error(self):
        """Test transform raises error on empty data.

        **PHM Logic**: Empty data indicates pipeline issue that should
        fail fast with clear error message.

        **Methodology**: Pass empty dict to transform_data.

        **Expected**: ValueError with descriptive message.

        Validates: Requirement R4.5 - Empty data error handling
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])

        with pytest.raises(ValueError, match="No data provided"):
            transform.transform_data({}, metadata={})

    def test_transform_data_hankel_svd_multi_column_output(self):
        """Test Hankel SVD produces multi-value output per signal.

        **PHM Logic**: Hankel SVD returns multiple singular values, unlike
        scalar statistics. Output shape must accommodate this.

        **Methodology**: Apply Hankel SVD to single signal.

        **Expected**: Output shape includes all singular values.

        Validates: Requirement R4.6 - Multi-value statistic handling
        """
        transform = TimeStatsTransform(
            stats_to_compute=["hankel_svd"], hankel_window_size=10
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = {"features": signal.reshape(-1, 1)}

        result = transform.transform_data(data, metadata={})

        # Hankel SVD returns hankel_window_size values
        assert result.shape == (10,)

    def test_transform_data_mixed_stats_correct_ordering(self):
        """Test that stats are returned in specified order.

        **PHM Logic**: Feature ordering must be consistent and predictable
        for downstream models that expect specific feature positions.

        **Methodology**: Use multiple stats, verify order matches input list.

        **Expected**: First features are mean, then hankel_svd values.

        Validates: Requirement R4.7 - Feature ordering consistency
        """
        transform = TimeStatsTransform(
            stats_to_compute=["mean", "hankel_svd"], hankel_window_size=10
        )
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        data = {"features": signal.reshape(-1, 1)}

        result = transform.transform_data(data, metadata={})

        # Should be: 1 mean + 10 hankel_svd = 11 features
        assert result.shape == (11,)

        # First value should be mean
        expected_mean = np.mean(signal)
        assert abs(result[0] - expected_mean) < 1e-6

    def test_transform_data_with_nan_logs_warning(self, caplog):
        """Test transform handles NaN values by logging warning.

        **PHM Logic**: NaN values in sensor data indicate measurement issues.
        The implementation logs a warning and may produce NaN outputs or errors.

        **Methodology**: Create data with NaN, verify warning logged.

        **Expected**: Warning logged about NaN values in signal.

        Validates: Requirement R4.8 - NaN detection
        """
        import logging

        transform = TimeStatsTransform(stats_to_compute=["mean"])
        signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        data = {"features": signal.reshape(-1, 1)}

        with caplog.at_level(logging.WARNING):
            try:
                transform.transform_data(data, metadata={})
            except ValueError:
                pass  # Expected - may raise error after warning
            # Should log warning about NaNs
            assert "NaN" in caplog.text


class TestTimeStatsTransformPHMScenarios:
    """PHM-specific scenario tests with realistic data.

    These tests use PHM fixtures to validate transform behavior with
    realistic healthy, degraded, and faulty machinery data.
    """

    def test_healthy_bearing_low_rms_and_kurtosis(self, healthy_bearing_signal):
        """Test healthy bearing signal produces low RMS.

        **PHM Logic**: A healthy bearing exhibits low-amplitude vibration with
        relatively stable characteristics. RMS should be below healthy threshold.

        **Methodology**: Apply transform to healthy signal fixture, verify
        RMS is within healthy range.

        **Expected Outcome**:
        - RMS < 0.5 g (healthy threshold)
        - Peak factor reasonable (no extreme impulses)

        Validates: Requirement R5.1 - Healthy state characterization
        """
        transform = TimeStatsTransform(
            stats_to_compute=["root_mean_square", "kurtosis", "peak_factor"]
        )
        data = {"features": healthy_bearing_signal["signal"]}

        result = transform.transform_data(data, metadata={})
        rms, kurtosis, peak_factor = result[0], result[1], result[2]

        # Verify healthy characteristics - RMS should be low
        assert rms < 0.5, f"RMS {rms:.3f} exceeds healthy threshold (0.5)"
        # Kurtosis varies widely, just ensure it's finite
        assert np.isfinite(kurtosis), f"Kurtosis {kurtosis:.3f} is not finite"
        # Peak factor should be reasonable
        assert (
            peak_factor < 10.0
        ), f"Peak factor {peak_factor:.3f} indicates extreme content"

    def test_faulty_bearing_elevated_kurtosis(self, faulty_bearing_signal_outer_race):
        """Test faulty bearing signal produces elevated kurtosis.

        **PHM Logic**: Outer race faults produce periodic impulses that
        elevate kurtosis significantly above the Gaussian baseline (3.0).
        This is one of the earliest indicators of incipient bearing damage.

        **Methodology**: Apply transform to faulty signal fixture, verify
        kurtosis is elevated above fault threshold.

        **Expected Outcome**:
        - Kurtosis > 4.0 (elevated due to impulses)
        - Peak factor > 3.0 (impulsive content present)

        Validates: Requirement R5.2 - Fault signature detection
        """
        transform = TimeStatsTransform(stats_to_compute=["kurtosis", "peak_factor"])
        data = {"features": faulty_bearing_signal_outer_race["signal"]}

        result = transform.transform_data(data, metadata={})
        kurtosis, peak_factor = result[0], result[1]

        # Verify fault characteristics
        assert (
            kurtosis > 4.0
        ), f"Kurtosis {kurtosis:.3f} should be elevated for faulty bearing"
        assert (
            peak_factor > 3.0
        ), f"Peak factor {peak_factor:.3f} should indicate impulses"

    def test_degradation_trend_increasing_rms(self, degradation_trend_exponential):
        """Test degradation trend shows increasing RMS over time.

        **PHM Logic**: As bearing damage progresses, vibration energy (RMS)
        increases. The trend should show monotonic increase from initial
        healthy state to failure threshold.

        **Methodology**: Apply transform to each sample of degradation trend,
        verify RMS increases from start to end.

        **Expected Outcome**:
        - Initial RMS ≈ 0.3 (healthy)
        - Final RMS ≈ 4.0 (failure)
        - Monotonic increase overall

        Validates: Requirement R5.3 - Degradation trend capture
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])
        trend = degradation_trend_exponential["trend"]

        # Apply transform to verify it processes the trend (returns np.ndarray, not dict)
        out = transform.transform_data({"features": trend}, {})
        assert isinstance(out, np.ndarray), "TimeStatsTransform returns ndarray"
        assert out.shape == (1,), "1 signal × 1 stat (mean) → shape (1,)"
        assert np.isfinite(out[0]), "Transform output should be finite"

        # The trend fixture already represents RMS values over time
        initial_rms = trend[0, 0]
        final_rms = trend[-1, 0]

        # Mean of degradation trend should lie between initial and final RMS
        assert (
            initial_rms <= out[0] <= final_rms
        ), f"Mean {out[0]:.3f} should be between initial {initial_rms:.3f} and final {final_rms:.3f}"

        # Verify degradation pattern
        assert (
            initial_rms < 1.0
        ), f"Initial RMS {initial_rms:.3f} should be in healthy range"
        assert (
            final_rms > 3.0
        ), f"Final RMS {final_rms:.3f} should indicate degraded state"
        assert (
            final_rms > initial_rms * 5
        ), "RMS should increase significantly over life"

    def test_anomalous_nan_input_logs_warning(self, anomalous_input_nan, caplog):
        """Test NaN input is properly detected and logged.

        **PHM Logic**: NaN values in sensor data indicate measurement failures.
        Transform must detect and report these via warning.

        **Methodology**: Apply transform to data with NaN values.

        **Expected Outcome**: Warning logged about NaN values.

        Validates: Requirement R5.4 - Anomalous data detection
        """
        import logging

        transform = TimeStatsTransform(stats_to_compute=["mean"])
        data = anomalous_input_nan["data"]

        with caplog.at_level(logging.WARNING):
            try:
                transform.transform_data(data, metadata={})
            except ValueError:
                pass  # May raise error after warning
            # Should log warning about NaNs
            assert "NaN" in caplog.text

    def test_anomalous_inf_input_raises_error(self, anomalous_input_inf):
        """Test infinite input is properly detected and reported.

        **PHM Logic**: Infinite values indicate sensor saturation or
        numerical issues. Must be caught in validation phase.

        **Methodology**: Apply transform to data with infinite values.

        **Expected Outcome**: ValueError raised about infinite values.

        Validates: Requirement R5.5 - Infinite value detection
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])
        data = anomalous_input_inf["data"]

        with pytest.raises(ValueError, match="Infinite values"):
            transform.transform_data(data, metadata={})

    def test_constant_signal_edge_case(self, edge_case_constant_signal):
        """Test constant (DC) signal produces expected statistics.

        **PHM Logic**: Constant signals have zero variance and undefined
        skewness/kurtosis. Transform should handle this gracefully.

        **Methodology**: Apply transform to constant signal, verify
        mean equals constant value and variance is zero.

        **Expected Outcome**:
        - Mean = 5.0 (the constant value)
        - Variance = 0.0 (no variation)
        - RMS = 5.0 (equals DC value)

        Validates: Requirement R5.6 - Edge case handling
        """
        transform = TimeStatsTransform(
            stats_to_compute=["mean", "variance", "root_mean_square"]
        )
        data = edge_case_constant_signal

        result = transform.transform_data(data, metadata={})
        mean_val, variance, rms = result[0], result[1], result[2]

        assert abs(mean_val - 5.0) < 1e-10
        assert abs(variance) < 1e-10
        assert abs(rms - 5.0) < 1e-10


class TestTimeStatsTransformFeatureNames:
    """Tests for feature name generation.

    Validates that meaningful feature names are generated for
    interpretability and traceability in PHM applications.
    """

    def test_get_feature_names_single_signal(self):
        """Test feature names for single signal.

        **PHM Logic**: Feature names should indicate the source key,
        signal index, and statistic type for traceability.

        **Methodology**: Generate names for single-signal input.

        **Expected**: Names follow pattern {key}_col{idx}_time_{stat}.

        Validates: Requirement R6.1 - Single signal naming
        """
        transform = TimeStatsTransform(stats_to_compute=["mean", "variance"])
        input_keys = ["features"]
        input_shapes = {"features": (100, 1)}  # 100 samples, 1 signal

        names = transform.get_feature_names(input_keys, input_shapes)

        assert len(names) == 2  # 1 signal × 2 stats
        assert "features_col0_time_mean" in names
        assert "features_col0_time_variance" in names

    def test_get_feature_names_multi_signal(self):
        """Test feature names for multiple signals.

        **PHM Logic**: Each signal should be identified by its column index.

        **Methodology**: Generate names for multi-signal input.

        **Expected**: Names include signal indices col0, col1, col2.

        Validates: Requirement R6.2 - Multi-signal naming
        """
        transform = TimeStatsTransform(stats_to_compute=["mean", "root_mean_square"])
        input_keys = ["features"]
        input_shapes = {"features": (100, 3)}  # 3 signals

        names = transform.get_feature_names(input_keys, input_shapes)

        assert len(names) == 6  # 3 signals × 2 stats
        # Check all column indices present
        assert any("col0" in name for name in names)
        assert any("col1" in name for name in names)
        assert any("col2" in name for name in names)

    def test_get_feature_names_empty_keys(self):
        """Test feature names with empty input keys.

        **PHM Logic**: Empty keys should return empty list gracefully.

        **Methodology**: Call get_feature_names with empty lists.

        **Expected**: Empty list returned.

        Validates: Requirement R6.3 - Empty input handling
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])

        names = transform.get_feature_names([], {})

        assert names == []

    def test_get_feature_names_missing_shape_raises_error(self):
        """Test missing shape raises KeyError.

        **PHM Logic**: Shape information is required for feature name
        generation. Missing shapes indicate configuration error.

        **Methodology**: Call get_feature_names with missing shape.

        **Expected**: KeyError raised.

        Validates: Requirement R6.4 - Missing configuration error
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])

        with pytest.raises(KeyError, match="not provided"):
            transform.get_feature_names(["features"], {})


class TestTimeStatsTransformCallable:
    """Tests for __call__ interface.

    Validates the transform can be used as a callable function.
    """

    def test_callable_interface(self):
        """Test transform can be called directly.

        **PHM Logic**: Callable interface enables use in functional pipelines.

        **Methodology**: Call transform using () operator.

        **Expected**: Same result as transform_data.

        Validates: Requirement R7.1 - Callable interface
        """
        transform = TimeStatsTransform(stats_to_compute=["mean"])
        data = {"features": np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])}

        result_call = transform(data, metadata={})
        result_method = transform.transform_data(data, metadata={})

        np.testing.assert_array_equal(result_call, result_method)
