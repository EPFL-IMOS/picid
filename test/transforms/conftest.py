"""Pytest configuration and fixtures for transform tests.

This module provides comprehensive PHM (Prognostics and Health Management) mock data
fixtures that mirror real-world telemetry patterns. These fixtures serve as "Gold Standards"
for testing transforms with:

1. **Nominal States**: Data representing healthy assets with expected sensor readings
2. **Fault Signatures**: Data simulating degradation and failure modes
3. **Anomalous Inputs**: Out-of-bounds sensor readings, missing data, edge cases

The mock data follows PHM domain conventions:
- Sampling rates: 25.6 kHz (XJTU-SY bearings), 20 kHz (PRONOSTIA), etc.
- Unit scales: g (acceleration), mm/s (velocity), Hz (frequency)
- Timestamps: Monotonically increasing cycle/runtime values

Reference: Based on PRONOSTIA and XJTU-SY bearing datasets, N-CMAPSS turbofan engine data,
and UniBo battery degradation datasets.
"""

import numpy as np
import pytest
import awkward as ak
from typing import Tuple, Optional
from dataclasses import dataclass
from picid.data.data_objects import NamedTransformInput, SplitDatasetContainer


# =============================================================================
# PHM CONSTANTS - Based on real-world datasets
# =============================================================================

# Sampling frequencies (Hz)
PRONOSTIA_FS = 25600  # PRONOSTIA bearing dataset sampling rate
XJTU_SY_FS = 25600  # XJTU-SY bearing dataset sampling rate
CMAPSS_CYCLES_PER_SAMPLE = 1  # N-CMAPSS is cycle-based

# Typical bearing fault frequencies (normalized to shaft frequency)
BALL_PASS_FREQUENCY_OUTER = 3.585  # BPFO for typical bearing
BALL_PASS_FREQUENCY_INNER = 5.415  # BPFI
BALL_SPIN_FREQUENCY = 2.357  # BSF
FUNDAMENTAL_TRAIN_FREQUENCY = 0.398  # FTF

# RMS thresholds for bearing health (mm/s or g depending on sensor)
RMS_THRESHOLD_HEALTHY = 0.5  # Below this: healthy
RMS_THRESHOLD_WARNING = 2.0  # Above this: warning
RMS_THRESHOLD_CRITICAL = 4.0  # Above this: critical/failure imminent

# Kurtosis thresholds for impulsive faults
KURTOSIS_HEALTHY = 3.0  # Gaussian baseline
KURTOSIS_FAULT_THRESHOLD = 6.0  # Above this: likely fault


# =============================================================================
# DATA CLASSES FOR STRUCTURED TEST DATA
# =============================================================================


@dataclass
class PHMSignalCharacteristics:
    """Characteristics of a PHM signal for validation.

    Attributes:
        rms: Root Mean Square value (energy indicator)
        peak_factor: Peak/RMS ratio (impulsiveness indicator)
        kurtosis: Distribution tail heaviness (fault indicator)
        health_state: 'healthy', 'degraded', 'faulty'
        dominant_freq: Dominant frequency if periodic signal
    """

    rms: float
    peak_factor: float
    kurtosis: float
    health_state: str
    dominant_freq: Optional[float] = None


@dataclass
class BearingTestUnit:
    """Test unit configuration for bearing datasets.

    Based on PRONOSTIA/XJTU-SY bearing dataset structure.
    """

    condition: int  # Operating condition (1, 2, or 3)
    bearing_id: int  # Bearing number within condition
    total_life: float  # Total life in seconds
    fault_type: str  # 'outer_race', 'inner_race', 'ball', 'cage', 'mixed'


# Default test units based on PRONOSTIA dataset
PRONOSTIA_TEST_UNITS = {
    (1, 1): BearingTestUnit(1, 1, 28020.0, "outer_race"),
    (1, 2): BearingTestUnit(1, 2, 8700.0, "inner_race"),
    (2, 1): BearingTestUnit(2, 1, 9100.0, "ball"),
    (2, 2): BearingTestUnit(2, 2, 7960.0, "cage"),
    (3, 1): BearingTestUnit(3, 1, 5140.0, "mixed"),
}


# =============================================================================
# PHM SIGNAL GENERATORS - Core signal generation utilities
# =============================================================================


def generate_healthy_vibration_signal(
    n_samples: int = 2560,
    fs: float = PRONOSTIA_FS,
    base_amplitude: float = 0.1,
    noise_level: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a healthy bearing vibration signal.

    Healthy bearings exhibit low-amplitude, broad-spectrum noise with Gaussian
    characteristics (kurtosis ≈ 3). This simulates normal operational vibration
    without impulsive fault signatures.

    PHM Logic: Healthy bearings have RMS < 0.5 g, kurtosis ≈ 3.0, and no
    dominant frequency peaks at bearing fault frequencies.

    Args:
        n_samples: Number of samples to generate
        fs: Sampling frequency in Hz
        base_amplitude: Base vibration amplitude (g)
        noise_level: Additive Gaussian noise level
        seed: Random seed for reproducibility

    Returns:
        1D numpy array representing healthy vibration signal

    Example:
        >>> signal = generate_healthy_vibration_signal(n_samples=2560)
        >>> assert signal.shape == (2560,)
        >>> assert np.sqrt(np.mean(signal**2)) < RMS_THRESHOLD_HEALTHY
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n_samples) / fs

    # Healthy signal: low-amplitude sinusoid + Gaussian noise
    # Represents shaft rotation with minor imbalance
    shaft_freq = 30.0  # 1800 RPM = 30 Hz
    signal = base_amplitude * np.sin(2 * np.pi * shaft_freq * t)
    signal += noise_level * np.random.randn(n_samples)

    return signal


def generate_fault_vibration_signal(
    n_samples: int = 2560,
    fs: float = PRONOSTIA_FS,
    fault_type: str = "outer_race",
    severity: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a bearing fault vibration signal.

    Fault signals exhibit characteristic periodic impulses at bearing fault
    frequencies (BPFO, BPFI, BSF, FTF). As damage progresses, impulse amplitude
    increases while becoming more periodic.

    PHM Logic: Faulty bearings show elevated kurtosis (>6), increased peak factor,
    and spectral peaks at fault characteristic frequencies. RMS typically > 2.0 g.

    Fault Types (per ISO 15243):
        - outer_race: Impulses at BPFO, typically first to fail
        - inner_race: Impulses at BPFI, modulated by shaft rotation
        - ball: Impulses at 2×BSF (double frequency due to ball geometry)
        - cage: Low-frequency modulation at FTF

    Args:
        n_samples: Number of samples to generate
        fs: Sampling frequency in Hz
        fault_type: Type of bearing fault
        severity: Fault severity multiplier (0.5 = incipient, 1.0 = developed)
        seed: Random seed for reproducibility

    Returns:
        1D numpy array representing faulty vibration signal
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n_samples) / fs
    shaft_freq = 30.0  # 30 Hz shaft rotation

    # Select fault frequency based on fault type
    fault_frequencies = {
        "outer_race": BALL_PASS_FREQUENCY_OUTER * shaft_freq,
        "inner_race": BALL_PASS_FREQUENCY_INNER * shaft_freq,
        "ball": 2 * BALL_SPIN_FREQUENCY * shaft_freq,
        "cage": FUNDAMENTAL_TRAIN_FREQUENCY * shaft_freq,
    }
    fault_freq = fault_frequencies.get(fault_type, fault_frequencies["outer_race"])

    # Generate impulse train at fault frequency
    impulse_period = 1.0 / fault_freq
    n_impulses = int(n_samples / fs / impulse_period) + 1

    signal = np.zeros(n_samples)

    # Add impulsive responses (exponentially decaying)
    for i in range(n_impulses):
        impulse_sample = int(i * impulse_period * fs)
        if impulse_sample < n_samples:
            decay_length = min(100, n_samples - impulse_sample)
            decay = np.exp(-np.arange(decay_length) / 20)
            amplitude = severity * (2.0 + 0.5 * np.random.randn())
            signal[impulse_sample : impulse_sample + decay_length] += amplitude * decay

    # Add background noise
    signal += 0.1 * np.random.randn(n_samples)

    # For inner race: add amplitude modulation at shaft frequency
    if fault_type == "inner_race":
        modulation = 0.5 * (1 + np.sin(2 * np.pi * shaft_freq * t))
        signal *= modulation

    return signal


def generate_degradation_trend(
    n_cycles: int = 100,
    degradation_type: str = "exponential",
    initial_rms: float = 0.3,
    final_rms: float = 4.0,
    noise_ratio: float = 0.05,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a degradation trend signal (RMS/Health Indicator over time).

    Models the typical degradation pattern observed in rotating machinery where
    condition indicators (RMS, kurtosis) remain stable during healthy operation
    then increase exponentially as damage accumulates.

    PHM Logic: Degradation follows bathtub curve - initial run-in period,
    stable operation, then accelerating degradation until failure.

    Args:
        n_cycles: Number of measurement cycles
        degradation_type: 'exponential', 'linear', or 'bathtub'
        initial_rms: Initial RMS value (healthy state)
        final_rms: Final RMS value (failure state)
        noise_ratio: Measurement noise as ratio of trend amplitude
        seed: Random seed for reproducibility

    Returns:
        1D numpy array representing degradation trend
    """
    if seed is not None:
        np.random.seed(seed)

    t_normalized = np.linspace(0, 1, n_cycles)

    if degradation_type == "exponential":
        # Exponential degradation: typical for accelerating damage
        trend = initial_rms + (final_rms - initial_rms) * (
            np.exp(3 * t_normalized) - 1
        ) / (np.exp(3) - 1)
    elif degradation_type == "linear":
        # Linear degradation: uniform wear
        trend = initial_rms + (final_rms - initial_rms) * t_normalized
    elif degradation_type == "bathtub":
        # Bathtub curve: stable middle with accelerating ends
        # Stable region: 20%-80% of life
        stable_start, stable_end = 0.2, 0.8
        trend = np.zeros(n_cycles)
        for i, t in enumerate(t_normalized):
            if t < stable_start:
                # Initial run-in (slight decrease)
                trend[i] = initial_rms * (1 + 0.2 * (stable_start - t) / stable_start)
            elif t > stable_end:
                # End-of-life acceleration
                progress = (t - stable_end) / (1 - stable_end)
                trend[i] = initial_rms + (final_rms - initial_rms) * progress**2
            else:
                # Stable operation
                trend[i] = initial_rms
    else:
        raise ValueError(f"Unknown degradation_type: {degradation_type}")

    # Add measurement noise
    noise = noise_ratio * (final_rms - initial_rms) * np.random.randn(n_cycles)
    trend += noise

    return trend


def generate_rul_sequence(
    n_samples: int, max_rul: float = 100.0, unit_id: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a RUL (Remaining Useful Life) sequence.

    Creates monotonically decreasing RUL values and corresponding runtime values.
    Used for testing Health Index and RUL prediction transforms.

    PHM Logic: RUL decreases linearly from max_rul to 0 as runtime increases
    from 0 to total_life. This is the ground truth for prognostics models.

    Args:
        n_samples: Number of samples in sequence
        max_rul: Maximum RUL value (at start of life)
        unit_id: Optional unit identifier tuple

    Returns:
        Tuple of (runtime_array, rul_array)
    """
    runtime = np.linspace(0, max_rul, n_samples)
    rul = max_rul - runtime
    return runtime, rul


# =============================================================================
# PYTEST FIXTURES - Reusable test data
# =============================================================================


@pytest.fixture
def sample_2d_array():
    """Create a sample 2D numpy array for testing.

    Basic 3x3 float array for testing transforms that require 2D input.
    """
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def sample_1d_array():
    """Create a sample 1D numpy array for testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_named_transform_input():
    """Create a sample NamedTransformInput for testing."""
    return NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


@pytest.fixture
def sample_multi_key_input():
    """Create a NamedTransformInput with multiple keys."""
    return NamedTransformInput(
        features=np.array([[1.0, 2.0], [3.0, 4.0]]), target=np.array([[0.5], [0.7]])
    )


@pytest.fixture
def sample_split_dataset_container():
    """Create a sample SplitDatasetContainer for integration tests."""
    return SplitDatasetContainer(
        features={
            "train": [np.array([[1.0, 2.0], [3.0, 4.0]])],
            "val": [np.array([[5.0, 6.0]])],
            "test": [np.array([[7.0, 8.0]])],
        },
        target={
            "train": [np.array([[0.1], [0.2]])],
            "val": [np.array([[0.3]])],
            "test": [np.array([[0.4]])],
        },
    )


@pytest.fixture
def sample_ragged_array():
    """Create a sample ragged (awkward) array for testing."""
    return ak.Array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    )


@pytest.fixture
def sample_time_series_signal():
    """Create a sample time series signal for spectral/time statistics tests."""
    n = 128
    t = np.linspace(0, 4 * np.pi, n)
    signal = np.sin(t)
    return signal.reshape(-1, 1)


@pytest.fixture
def sample_multi_signal_array():
    """Create a multi-signal array for testing."""
    n = 100
    signals = [
        np.sin(np.linspace(0, 4 * np.pi, n)),
        np.cos(np.linspace(0, 4 * np.pi, n)),
        np.sin(2 * np.linspace(0, 4 * np.pi, n)),
    ]
    return np.column_stack(signals)


# =============================================================================
# PHM-SPECIFIC FIXTURES - Realistic PHM telemetry data
# =============================================================================


@pytest.fixture
def healthy_bearing_signal():
    """Generate realistic healthy bearing vibration signal.

    **PHM Context**: This fixture provides a vibration signal characteristic
    of a healthy rolling element bearing under normal operating conditions.
    The signal exhibits:
    - Low RMS amplitude (< 0.5 g)
    - Gaussian noise characteristics (kurtosis ≈ 3.0)
    - No impulsive content (low peak factor)
    - Broad-spectrum noise without dominant fault frequencies

    **Use Case**: Testing that transforms correctly identify and preserve
    healthy state characteristics. The signal should result in low fault
    indicator values when processed through time/spectral statistics.

    Returns:
        Dict with signal, sampling frequency, and expected characteristics
    """
    signal = generate_healthy_vibration_signal(n_samples=2560, seed=42)
    return {
        "signal": signal.reshape(-1, 1),
        "fs": PRONOSTIA_FS,
        "n_samples": 2560,
        "expected": PHMSignalCharacteristics(
            rms=np.sqrt(np.mean(signal**2)),
            peak_factor=np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
            kurtosis=3.0,  # Approximately Gaussian
            health_state="healthy",
            dominant_freq=30.0,  # Shaft frequency
        ),
    }


@pytest.fixture
def faulty_bearing_signal_outer_race():
    """Generate realistic outer race fault vibration signal.

    **PHM Context**: Outer race defects are the most common bearing failure mode
    (~40% of all bearing failures). They produce periodic impulses at the Ball
    Pass Frequency Outer (BPFO). This fixture simulates a developed outer race
    defect with:
    - Elevated RMS (> 2.0 g)
    - High kurtosis (> 6.0) due to impulsive content
    - Dominant spectral peak at BPFO (≈107.55 Hz for typical bearing at 30 Hz shaft)

    **Use Case**: Testing that transforms correctly detect fault signatures,
    particularly the impulsive content and fault frequency characteristics.

    Returns:
        Dict with signal, sampling frequency, and expected characteristics
    """
    signal = generate_fault_vibration_signal(
        n_samples=2560, fault_type="outer_race", severity=1.0, seed=42
    )
    return {
        "signal": signal.reshape(-1, 1),
        "fs": PRONOSTIA_FS,
        "fault_type": "outer_race",
        "expected": PHMSignalCharacteristics(
            rms=np.sqrt(np.mean(signal**2)),
            peak_factor=np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
            kurtosis=6.0,  # Elevated due to impulses
            health_state="faulty",
            dominant_freq=BALL_PASS_FREQUENCY_OUTER * 30.0,  # BPFO
        ),
    }


@pytest.fixture
def faulty_bearing_signal_inner_race():
    """Generate realistic inner race fault vibration signal.

    **PHM Context**: Inner race defects produce amplitude-modulated impulses
    because the defect rotates with the shaft. The impulses occur at BPFI
    but are modulated at the shaft frequency, creating characteristic sidebands.

    Returns:
        Dict with signal, sampling frequency, and expected characteristics
    """
    signal = generate_fault_vibration_signal(
        n_samples=2560, fault_type="inner_race", severity=1.0, seed=43
    )
    return {
        "signal": signal.reshape(-1, 1),
        "fs": PRONOSTIA_FS,
        "fault_type": "inner_race",
        "expected": PHMSignalCharacteristics(
            rms=np.sqrt(np.mean(signal**2)),
            peak_factor=np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
            kurtosis=5.0,
            health_state="faulty",
            dominant_freq=BALL_PASS_FREQUENCY_INNER * 30.0,  # BPFI
        ),
    }


@pytest.fixture
def degradation_trend_exponential():
    """Generate exponential degradation trend (RMS over time).

    **PHM Context**: This represents the classic degradation pattern where
    a bearing operates normally, then experiences accelerating damage until
    failure. This is used for testing Health Index calculations and RUL
    estimation transforms.

    The trend follows: RMS(t) = RMS_0 + (RMS_f - RMS_0) * (exp(3t) - 1) / (exp(3) - 1)

    Returns:
        Dict with trend array, runtime array, and expected characteristics
    """
    n_cycles = 100
    trend = generate_degradation_trend(
        n_cycles=n_cycles,
        degradation_type="exponential",
        initial_rms=0.3,
        final_rms=4.0,
        seed=42,
    )
    runtime = np.linspace(0, 28020.0, n_cycles)  # Based on PRONOSTIA (1,1)

    return {
        "trend": trend.reshape(-1, 1),
        "runtime": runtime.reshape(-1, 1),
        "n_cycles": n_cycles,
        "total_life": 28020.0,
        "initial_rms": 0.3,
        "final_rms": 4.0,
        "degradation_type": "exponential",
    }


@pytest.fixture
def health_index_test_data():
    """Generate Health Index test data based on PRONOSTIA dataset.

    **PHM Context**: Health Index (HI) is calculated as HI = Runtime / Total_Life,
    ranging from 0 (start of life) to 1 (end of life). This fixture provides
    realistic runtime sequences for multiple bearings with known total life.

    Based on PRONOSTIA dataset Condition 1, Bearing 1 with Total_Life = 28020s.

    Returns:
        Dict with runtime, unit_id, expected HI values, and metadata
    """
    total_life = 28020.0
    n_samples = 100
    runtime = np.linspace(0, total_life, n_samples)
    expected_hi = runtime / total_life

    return {
        "runtime": runtime.reshape(-1, 1),
        "unit_id": np.array([[1, 1]] * n_samples),  # Condition 1, Bearing 1
        "expected_hi": expected_hi.reshape(-1, 1),
        "total_life": total_life,
        "dataset_name": "PRONOSTIA",
        "unit_key": (1, 1),
        "n_samples": n_samples,
    }


@pytest.fixture
def multi_unit_split_container():
    """Create a multi-unit SplitDatasetContainer for PHM testing.

    **PHM Context**: PHM datasets often contain multiple units (bearings,
    engines, batteries) with different operating conditions and failure times.
    This fixture creates a realistic multi-unit dataset structure.

    Structure:
    - 3 units per split
    - Each unit has features (2D) and target (1D)
    - Variable sequence lengths (simulating run-to-failure)

    Returns:
        SplitDatasetContainer with multi-unit data
    """

    def create_unit_data(n_samples: int, n_features: int = 14):
        """Create data for a single unit."""
        features = np.random.randn(n_samples, n_features)
        target = np.linspace(0, 1, n_samples).reshape(-1, 1)  # HI
        return features, target

    # Variable length units (simulating different failure times)
    train_lengths = [100, 80, 120]
    val_lengths = [50, 40, 60]
    test_lengths = [30, 25, 35]

    return SplitDatasetContainer(
        features={
            "train": [create_unit_data(n)[0] for n in train_lengths],
            "val": [create_unit_data(n)[0] for n in val_lengths],
            "test": [create_unit_data(n)[0] for n in test_lengths],
        },
        target={
            "train": [create_unit_data(n)[1] for n in train_lengths],
            "val": [create_unit_data(n)[1] for n in val_lengths],
            "test": [create_unit_data(n)[1] for n in test_lengths],
        },
    )


@pytest.fixture
def cmapss_style_features():
    """Generate N-CMAPSS style sensor features.

    **PHM Context**: N-CMAPSS turbofan engine dataset contains 14 virtual
    sensors and 4 operating conditions. This fixture generates realistic
    sensor readings following the expected ranges and correlations.

    Virtual Sensors:
    - T2, T24, T30, T50: Temperature sensors (°R)
    - P2, P15, P30: Pressure sensors (psia)
    - Nf, Nc: Fan/Core shaft speeds (rpm)
    - epr, phi: Engine performance ratios
    - NRf, NRc: Corrected shaft speeds
    - BPR, farB: Bypass ratio, fuel-air ratio

    Returns:
        Dict with features array, sensor names, and expected ranges
    """
    n_samples = 100
    n_sensors = 14

    # Realistic sensor data (simplified but representative)
    sensors = {
        "T2": np.random.uniform(440, 520, n_samples),  # Total temperature
        "T24": np.random.uniform(550, 650, n_samples),  # LPC outlet temp
        "T30": np.random.uniform(1550, 1650, n_samples),  # HPC outlet temp
        "T50": np.random.uniform(1300, 1400, n_samples),  # LPT outlet temp
        "P2": np.random.uniform(14.5, 15.5, n_samples),  # Fan inlet pressure
        "P15": np.random.uniform(21, 23, n_samples),  # Bypass pressure
        "P30": np.random.uniform(380, 420, n_samples),  # HPC outlet pressure
        "Nf": np.random.uniform(2380, 2400, n_samples),  # Fan speed (rpm)
        "Nc": np.random.uniform(9000, 9100, n_samples),  # Core speed (rpm)
        "epr": np.random.uniform(1.0, 1.2, n_samples),  # Engine pressure ratio
        "phi": np.random.uniform(520, 530, n_samples),  # Fuel flow
        "NRf": np.random.uniform(2380, 2400, n_samples),  # Corrected fan speed
        "NRc": np.random.uniform(9000, 9100, n_samples),  # Corrected core speed
        "BPR": np.random.uniform(8.0, 9.0, n_samples),  # Bypass ratio
    }

    features = np.column_stack(list(sensors.values()))

    return {
        "features": features,
        "sensor_names": list(sensors.keys()),
        "n_samples": n_samples,
        "n_sensors": n_sensors,
    }


@pytest.fixture
def anomalous_input_nan():
    """Generate input data with NaN values for testing error handling.

    **PHM Context**: Real-world sensor data often contains missing values
    due to sensor failures, communication dropouts, or data corruption.
    Transforms must handle these gracefully with appropriate errors or
    warnings.

    Returns:
        NamedTransformInput with NaN values at known positions
    """
    n_samples = 100
    signal = np.sin(np.linspace(0, 4 * np.pi, n_samples))
    # Introduce NaNs at specific positions
    signal[10:15] = np.nan
    signal[50] = np.nan

    return {
        "data": NamedTransformInput(features=signal.reshape(-1, 1)),
        "nan_indices": [10, 11, 12, 13, 14, 50],
        "n_nans": 6,
    }


@pytest.fixture
def anomalous_input_inf():
    """Generate input data with infinite values for testing error handling.

    **PHM Context**: Infinite values can occur from sensor saturation,
    division by zero in preprocessing, or numerical overflow. Transforms
    should detect and report these anomalies.

    Returns:
        NamedTransformInput with infinite values at known positions
    """
    n_samples = 100
    signal = np.sin(np.linspace(0, 4 * np.pi, n_samples))
    # Introduce infinities
    signal[25] = np.inf
    signal[75] = -np.inf

    return {
        "data": NamedTransformInput(features=signal.reshape(-1, 1)),
        "inf_indices": [25, 75],
        "n_infs": 2,
    }


@pytest.fixture
def edge_case_empty_signal():
    """Generate empty signal for edge case testing.

    Returns:
        Empty NamedTransformInput for testing error handling
    """
    return NamedTransformInput(features=np.array([]).reshape(0, 1))


@pytest.fixture
def edge_case_single_sample():
    """Generate single-sample signal for edge case testing.

    **PHM Context**: Some transforms require minimum sample lengths
    (e.g., FFT requires at least 2 samples). This tests boundary conditions.

    Returns:
        NamedTransformInput with single sample
    """
    return NamedTransformInput(features=np.array([[1.0]]))


@pytest.fixture
def edge_case_constant_signal():
    """Generate constant (DC) signal for edge case testing.

    **PHM Context**: A constant signal has zero variance and undefined
    kurtosis/skewness. Tests should handle this gracefully.

    Returns:
        NamedTransformInput with constant values
    """
    n_samples = 100
    return NamedTransformInput(features=np.ones((n_samples, 1)) * 5.0)


# =============================================================================
# VALIDATION HELPER FUNCTIONS
# =============================================================================


def assert_phm_signal_healthy(
    rms: float, kurtosis: float, peak_factor: float, tolerance: float = 0.5
) -> None:
    """Assert that signal characteristics indicate healthy state.

    Args:
        rms: Computed RMS value
        kurtosis: Computed kurtosis value
        peak_factor: Computed peak factor
        tolerance: Tolerance for threshold comparisons
    """
    assert (
        rms < RMS_THRESHOLD_HEALTHY + tolerance
    ), f"RMS {rms:.3f} exceeds healthy threshold {RMS_THRESHOLD_HEALTHY}"
    assert (
        abs(kurtosis - KURTOSIS_HEALTHY) < tolerance * 2
    ), f"Kurtosis {kurtosis:.3f} deviates significantly from Gaussian ({KURTOSIS_HEALTHY})"
    assert (
        peak_factor < 5.0
    ), f"Peak factor {peak_factor:.3f} indicates impulsive content (not healthy)"


def assert_phm_signal_faulty(
    rms: float, kurtosis: float, peak_factor: float, tolerance: float = 0.5
) -> None:
    """Assert that signal characteristics indicate faulty state.

    Args:
        rms: Computed RMS value
        kurtosis: Computed kurtosis value
        peak_factor: Computed peak factor
        tolerance: Tolerance for threshold comparisons
    """
    assert (
        rms > RMS_THRESHOLD_WARNING - tolerance
    ), f"RMS {rms:.3f} below warning threshold - may not detect fault"
    assert (
        kurtosis > KURTOSIS_FAULT_THRESHOLD - tolerance
    ), f"Kurtosis {kurtosis:.3f} below fault threshold - may not detect impulsive content"


def validate_health_index(
    hi: np.ndarray, runtime: np.ndarray, total_life: float
) -> None:
    """Validate Health Index calculations.

    Args:
        hi: Computed Health Index array
        runtime: Runtime array used for calculation
        total_life: Total life value used for calculation
    """
    expected_hi = runtime / total_life
    np.testing.assert_allclose(hi, expected_hi, rtol=1e-6)
    assert np.all(hi >= 0.0), "Health Index contains negative values"
    assert np.all(hi <= 1.0), "Health Index exceeds 1.0"
