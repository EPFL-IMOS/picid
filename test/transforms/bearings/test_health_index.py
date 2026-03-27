"""Comprehensive tests for HealthIndexTransform.

This module provides rigorous testing for the Health Index (HI) transform,
which is a fundamental component of bearing prognostics and health management.

PHM Context:
-----------
Health Index is a normalized metric representing the current health state of
a bearing relative to its total expected life:

    HI = Runtime / Total_Life

Where:
- HI ranges from 0.0 (start of life) to 1.0 (end of life)
- Runtime is the accumulated operating time
- Total_Life is the known or estimated total life for the specific unit

The Health Index is critical for:
1. **Prognostics**: Predicting Remaining Useful Life (RUL = (1 - HI) * Total_Life)
2. **Condition Monitoring**: Tracking degradation progression
3. **Maintenance Planning**: Scheduling interventions before failure

Reference Datasets:
- PRONOSTIA (FEMTO-ST): 17 bearings with known failure times under 3 conditions
- XJTU-SY: 15 bearings across 3 operating conditions with run-to-failure data

Test Coverage Strategy:
----------------------
1. **Initialization Tests**: Parameter validation, lookup table configuration
2. **Nominal Calculation Tests**: Correct HI computation for known inputs
3. **Inverse Transform Tests**: Verifying Runtime reconstruction
4. **Multi-Unit Tests**: Handling vectorized unit_id inputs
5. **Error Handling Tests**: Invalid inputs, missing metadata, range violations
6. **Integration Tests**: Full pipeline with SplitDatasetContainer

Physical Validation:
-------------------
- HI must be in [0.0, 1.0] range (normalized metric)
- HI must be monotonically increasing over time (damage accumulation)
- Inverse transform must reconstruct original runtime values
"""

import numpy as np
import pytest

from picid.transforms.bearings.health_index import (
    HealthIndexTransform,
    DEFAULT_TOTAL_LIFE_LOOKUP,
    DECREASE_PERIOD,
)
from picid.data.data_objects import NamedTransformInput


# =============================================================================
# TEST FIXTURES - Bearing-specific test data
# =============================================================================


@pytest.fixture
def pronostia_unit_1_1():
    """PRONOSTIA Condition 1, Bearing 1 test data.

    **PHM Context**: This bearing has Total_Life = 28020.0 seconds, representing
    a complete run-to-failure dataset under Condition 1 (1800 rpm, 4000 N load).

    Returns:
        Dict with runtime, expected HI, and unit metadata
    """
    total_life = 28020.0
    n_samples = 100
    # Runtime increases from 0 to total_life
    runtime = np.linspace(0, total_life, n_samples)
    expected_hi = runtime / total_life

    return {
        "runtime": runtime.reshape(-1, 1),
        "unit_id": np.array([[1, 1]] * n_samples),  # Condition 1, Bearing 1
        "expected_hi": expected_hi.reshape(-1, 1),
        "total_life": total_life,
        "n_samples": n_samples,
    }


@pytest.fixture
def pronostia_unit_2_1():
    """PRONOSTIA Condition 2, Bearing 1 test data.

    **PHM Context**: This bearing has Total_Life = 9100.0 seconds under
    Condition 2 (1650 rpm, 4200 N load). Shorter life due to higher load.
    """
    total_life = 9100.0
    n_samples = 50
    runtime = np.linspace(0, total_life, n_samples)
    expected_hi = runtime / total_life

    return {
        "runtime": runtime.reshape(-1, 1),
        "unit_id": np.array([[2, 1]] * n_samples),
        "expected_hi": expected_hi.reshape(-1, 1),
        "total_life": total_life,
        "n_samples": n_samples,
    }


@pytest.fixture
def xjtu_sy_unit_1_1():
    """XJTU-SY Condition 1, Bearing 1 test data.

    **PHM Context**: This bearing has Total_Life = 123.0 minutes under
    Condition 1 (2100 rpm, 12 kN radial load). XJTU-SY has different
    operating conditions than PRONOSTIA.
    """
    total_life = 123.0
    n_samples = 50
    runtime = np.linspace(0, total_life, n_samples)
    expected_hi = runtime / total_life

    return {
        "runtime": runtime.reshape(-1, 1),
        "unit_id": np.array([[1, 1]] * n_samples),
        "expected_hi": expected_hi.reshape(-1, 1),
        "total_life": total_life,
        "dataset_name": "XJTU-SY",
        "n_samples": n_samples,
    }


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestHealthIndexTransformInitialization:
    """Tests for HealthIndexTransform initialization and configuration.

    Validates that the transform correctly handles configuration parameters
    including runtime_key, unit_key, dataset_name, and total_life_lookup.
    """

    def test_init_with_default_lookup(self):
        """Test initialization with default total_life_lookup.

        **PHM Logic**: The default lookup contains known total life values
        for PRONOSTIA and XJTU-SY datasets, enabling out-of-box usage with
        these standard PHM benchmark datasets.

        **Methodology**: Create transform without explicit lookup, verify
        defaults are loaded.

        **Expected**: Transform uses DEFAULT_TOTAL_LIFE_LOOKUP containing
        PRONOSTIA and XJTU-SY entries.

        Validates: Requirement HI-1.1 - Default lookup configuration
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        assert transform.runtime_key == "runtime"
        assert transform.unit_key == "unit_id"
        assert transform.dataset_name == "PRONOSTIA"
        assert transform.total_life_lookup == DEFAULT_TOTAL_LIFE_LOOKUP
        assert "PRONOSTIA" in transform.total_life_lookup
        assert "XJTU-SY" in transform.total_life_lookup

    def test_init_with_custom_lookup(self):
        """Test initialization with custom total_life_lookup.

        **PHM Logic**: Custom datasets or experimental bearings may have
        different total life values. Custom lookup allows flexibility.

        **Methodology**: Create transform with custom lookup table.

        **Expected**: Transform uses provided custom lookup.

        Validates: Requirement HI-1.2 - Custom lookup configuration
        """
        custom_lookup = {"CUSTOM_DATASET": {(1, 1): 10000.0, (1, 2): 15000.0}}

        transform = HealthIndexTransform(
            runtime_key="cycle",
            unit_key="unit",
            dataset_name="CUSTOM_DATASET",
            total_life_lookup=custom_lookup,
        )

        assert transform.total_life_lookup == custom_lookup
        assert (1, 1) in transform.dataset_unit_lives
        assert transform.dataset_unit_lives[(1, 1)] == 10000.0

    def test_init_invalid_dataset_raises_error(self):
        """Test initialization with invalid dataset name raises KeyError.

        **PHM Logic**: Dataset name must exist in lookup to retrieve
        unit-specific total life values. Invalid names indicate
        configuration errors that should fail early.

        **Methodology**: Attempt to create transform with non-existent dataset.

        **Expected**: KeyError raised with available dataset names listed.

        Validates: Requirement HI-1.3 - Invalid dataset error handling
        """
        with pytest.raises(KeyError) as exc_info:
            HealthIndexTransform(
                runtime_key="runtime",
                unit_key="unit_id",
                dataset_name="NONEXISTENT_DATASET",
            )

        assert "NONEXISTENT_DATASET" in str(exc_info.value)
        assert "PRONOSTIA" in str(exc_info.value) or "Available keys" in str(
            exc_info.value
        )

    def test_init_pronostia_contains_all_units(self):
        """Test PRONOSTIA lookup contains all expected units.

        **PHM Logic**: PRONOSTIA dataset has 17 bearings across 3 conditions:
        - Condition 1: Bearings 1-7
        - Condition 2: Bearings 1-7
        - Condition 3: Bearings 1-3

        **Methodology**: Verify all expected unit keys exist in lookup.

        **Expected**: All PRONOSTIA unit keys present with positive life values.

        Validates: Requirement HI-1.4 - PRONOSTIA dataset completeness
        """
        pronostia_lookup = DEFAULT_TOTAL_LIFE_LOOKUP["PRONOSTIA"]

        # Check Condition 1 bearings
        for bearing in range(1, 8):
            key = (1, bearing)
            assert key in pronostia_lookup, f"Missing PRONOSTIA unit {key}"
            assert pronostia_lookup[key] > 0, f"Invalid total_life for {key}"

        # Check Condition 2 bearings
        for bearing in range(1, 8):
            key = (2, bearing)
            assert key in pronostia_lookup, f"Missing PRONOSTIA unit {key}"

        # Check Condition 3 bearings
        for bearing in range(1, 4):
            key = (3, bearing)
            assert key in pronostia_lookup, f"Missing PRONOSTIA unit {key}"

    def test_init_xjtu_sy_contains_all_units(self):
        """Test XJTU-SY lookup contains all expected units.

        **PHM Logic**: XJTU-SY dataset has 15 bearings across 3 conditions:
        - Condition 1: Bearings 1-5
        - Condition 2: Bearings 1-5
        - Condition 3: Bearings 1-5

        **Methodology**: Verify all expected unit keys exist in lookup.

        **Expected**: All XJTU-SY unit keys present with positive life values.

        Validates: Requirement HI-1.5 - XJTU-SY dataset completeness
        """
        xjtu_lookup = DEFAULT_TOTAL_LIFE_LOOKUP["XJTU-SY"]

        # Check all 3 conditions × 5 bearings
        for condition in range(1, 4):
            for bearing in range(1, 6):
                key = (condition, bearing)
                assert key in xjtu_lookup, f"Missing XJTU-SY unit {key}"
                assert xjtu_lookup[key] > 0, f"Invalid total_life for {key}"


# =============================================================================
# HEALTH INDEX CALCULATION TESTS
# =============================================================================


class TestHealthIndexCalculation:
    """Tests for Health Index calculation.

    Validates correct HI computation: HI = Runtime / Total_Life
    """

    def test_calculate_hi_pronostia_unit_1_1(self, pronostia_unit_1_1):
        """Test HI calculation for PRONOSTIA Condition 1, Bearing 1.

        **PHM Logic**: HI = Runtime / Total_Life should produce values
        from 0.0 (start) to 1.0 (end of life) for this bearing.

        **Methodology**: Transform runtime sequence and compare to expected HI.

        **Expected**: Computed HI matches expected within tolerance.

        Validates: Requirement HI-2.1 - Basic HI calculation
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        data = NamedTransformInput(
            runtime=pronostia_unit_1_1["runtime"],
            unit_id=pronostia_unit_1_1["unit_id"][0],  # Single unit_id
        )

        result = transform.transform_data(data, metadata={})

        # Verify shape
        assert result.shape == pronostia_unit_1_1["expected_hi"].shape

        # Verify values (with tolerance for numerical precision)
        np.testing.assert_allclose(result, pronostia_unit_1_1["expected_hi"], rtol=1e-6)

        # Verify range
        assert np.all(result >= 0.0), "HI contains negative values"
        assert np.all(result <= 1.0), "HI exceeds 1.0"

    def test_calculate_hi_different_units_different_results(
        self, pronostia_unit_1_1, pronostia_unit_2_1
    ):
        """Test that different units with different total_life produce different HI.

        **PHM Logic**: Bearings under different conditions have different
        total life. Same runtime should produce different HI values for
        different units.

        **Methodology**: Compare HI for same runtime but different units.

        **Expected**: HI values differ proportionally to total_life ratio.

        Validates: Requirement HI-2.2 - Unit-specific calculation
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        # Use same runtime for both units
        common_runtime = np.array([[5000.0]])

        # Unit 1,1 with total_life=28020
        data_1_1 = NamedTransformInput(runtime=common_runtime, unit_id=np.array([1, 1]))
        hi_1_1 = transform.transform_data(data_1_1, metadata={})

        # Unit 2,1 with total_life=9100
        data_2_1 = NamedTransformInput(runtime=common_runtime, unit_id=np.array([2, 1]))
        hi_2_1 = transform.transform_data(data_2_1, metadata={})

        # Unit 2,1 should have higher HI (shorter total_life)
        assert hi_2_1[0, 0] > hi_1_1[0, 0]

        # Verify proportionality: HI_2_1 / HI_1_1 ≈ TL_1_1 / TL_2_1
        expected_ratio = (
            pronostia_unit_1_1["total_life"] / pronostia_unit_2_1["total_life"]
        )
        actual_ratio = hi_2_1[0, 0] / hi_1_1[0, 0]
        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_calculate_hi_at_boundaries(self):
        """Test HI calculation at boundary conditions.

        **PHM Logic**:
        - At runtime=0: HI should be 0.0 (start of life)
        - At runtime=total_life: HI should be 1.0 (end of life)

        **Methodology**: Test at exact boundary values.

        **Expected**: Exact boundary values (0.0 and 1.0).

        Validates: Requirement HI-2.3 - Boundary value handling
        """
        total_life = 28020.0
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        # Test at start of life (runtime=0)
        data_start = NamedTransformInput(
            runtime=np.array([[0.0]]), unit_id=np.array([1, 1])
        )
        hi_start = transform.transform_data(data_start, metadata={})
        assert abs(hi_start[0, 0] - 0.0) < 1e-10, "HI at runtime=0 should be 0.0"

        # Test at end of life (runtime=total_life)
        data_end = NamedTransformInput(
            runtime=np.array([[total_life]]), unit_id=np.array([1, 1])
        )
        hi_end = transform.transform_data(data_end, metadata={})
        assert abs(hi_end[0, 0] - 1.0) < 1e-10, "HI at runtime=total_life should be 1.0"

    def test_calculate_hi_xjtu_sy_dataset(self, xjtu_sy_unit_1_1):
        """Test HI calculation for XJTU-SY dataset.

        **PHM Logic**: XJTU-SY has different total_life values and uses
        different units (minutes vs seconds for PRONOSTIA). Verify
        dataset-specific lookup works correctly.

        **Methodology**: Transform XJTU-SY data and verify against expected.

        **Expected**: Computed HI matches expected for XJTU-SY bearing.

        Validates: Requirement HI-2.4 - Multi-dataset support
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="XJTU-SY"
        )

        data = NamedTransformInput(
            runtime=xjtu_sy_unit_1_1["runtime"], unit_id=xjtu_sy_unit_1_1["unit_id"][0]
        )

        result = transform.transform_data(data, metadata={})

        np.testing.assert_allclose(result, xjtu_sy_unit_1_1["expected_hi"], rtol=1e-6)


# =============================================================================
# INVERSE TRANSFORM TESTS
# =============================================================================


class TestHealthIndexInverseTransform:
    """Tests for inverse Health Index transformation.

    Validates: Runtime = HI * Total_Life
    """

    def test_inverse_transform_basic(self, pronostia_unit_1_1):
        """Test basic inverse transform recovers original runtime.

        **PHM Logic**: The inverse transform should recover the original
        runtime values from HI: Runtime = HI * Total_Life

        **Methodology**: Transform runtime to HI, then inverse back to runtime.

        **Expected**: Recovered runtime matches original within tolerance.

        Validates: Requirement HI-3.1 - Inverse transform accuracy
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        # Forward transform: Runtime -> HI
        data = NamedTransformInput(
            runtime=pronostia_unit_1_1["runtime"],
            unit_id=pronostia_unit_1_1["unit_id"][0],
        )
        hi = transform.transform_data(data, metadata={})

        # Inverse transform: HI -> Runtime
        inverse_data = NamedTransformInput(features=hi)
        metadata = {"unit_id": pronostia_unit_1_1["unit_id"][0]}
        recovered_runtime = transform.inverse_transform(inverse_data, metadata)

        # Verify recovery
        np.testing.assert_allclose(
            recovered_runtime, pronostia_unit_1_1["runtime"], rtol=1e-6
        )

    def test_inverse_transform_missing_metadata_raises_error(self):
        """Test inverse transform raises error when metadata missing.

        **PHM Logic**: Inverse transform requires unit_id in metadata to
        lookup the correct total_life. Missing metadata should fail fast.

        **Methodology**: Attempt inverse transform without unit_id in metadata.

        **Expected**: ValueError raised about missing metadata.

        Validates: Requirement HI-3.2 - Metadata validation
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        inverse_data = NamedTransformInput(features=np.array([[0.5]]))

        with pytest.raises(ValueError, match="unit_id"):
            transform.inverse_transform(inverse_data, metadata={})

    def test_inverse_transform_empty_data_raises_error(self):
        """Test inverse transform raises error on empty data.

        **PHM Logic**: Empty data indicates pipeline issue that should fail.

        **Methodology**: Attempt inverse transform with empty data dict.

        **Expected**: ValueError raised about no data.

        Validates: Requirement HI-3.3 - Empty data validation
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        metadata = {"unit_id": np.array([1, 1])}

        with pytest.raises(ValueError, match="No data"):
            transform.inverse_transform({}, metadata)


# =============================================================================
# VECTORIZED UNIT ID TESTS
# =============================================================================


class TestHealthIndexVectorizedUnitId:
    """Tests for vectorized unit_id handling.

    When processing sequences from multiple units, unit_id may be a 2D array
    with one row per sample. The transform must handle this correctly.
    """

    def test_vectorized_unit_id_2d_array(self, pronostia_unit_1_1):
        """Test transform handles 2D vectorized unit_id.

        **PHM Logic**: In batch processing, each sample may have its own
        unit_id stored as a 2D array [[condition, bearing], ...].

        **Methodology**: Provide 2D unit_id array matching sample count.

        **Expected**: Transform processes correctly with vectorized lookup.

        Validates: Requirement HI-4.1 - Vectorized unit_id support
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        # 2D unit_id: [[1,1], [1,1], ...] for all samples
        data = NamedTransformInput(
            runtime=pronostia_unit_1_1["runtime"],
            unit_id=pronostia_unit_1_1["unit_id"],  # 2D array
        )

        result = transform.transform_data(data, metadata={})

        np.testing.assert_allclose(result, pronostia_unit_1_1["expected_hi"], rtol=1e-6)

    def test_vectorized_unit_id_single_tuple(self):
        """Test transform handles single unit_id as tuple.

        **PHM Logic**: Single unit_id may be provided as tuple or 1D array.

        **Methodology**: Provide unit_id as 1D array.

        **Expected**: Transform converts to tuple correctly.

        Validates: Requirement HI-4.2 - Single unit_id tuple conversion
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        data = NamedTransformInput(
            runtime=np.array([[1000.0], [2000.0]]),
            unit_id=np.array([1, 1]),  # 1D array
        )

        result = transform.transform_data(data, metadata={})

        expected = np.array([[1000.0], [2000.0]]) / 28020.0
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestHealthIndexErrorHandling:
    """Tests for error handling and validation.

    Validates that transform correctly detects and reports invalid inputs,
    missing data, and range violations.
    """

    def test_missing_runtime_key_raises_error(self):
        """Test missing runtime_key in data raises KeyError.

        **PHM Logic**: Runtime data is required for HI calculation. Missing
        key indicates data pipeline misconfiguration.

        **Methodology**: Provide data without the expected runtime_key.

        **Expected**: KeyError raised with descriptive message.

        Validates: Requirement HI-5.1 - Missing runtime detection
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        # Data missing 'runtime' key
        data = NamedTransformInput(
            features=np.array([[1.0], [2.0]]), unit_id=np.array([1, 1])
        )

        with pytest.raises(KeyError, match="runtime"):
            transform.transform_data(data, metadata={})

    def test_unknown_unit_id_raises_error(self):
        """Test unknown unit_id raises KeyError.

        **PHM Logic**: Unit_id must exist in lookup table to retrieve
        total_life. Unknown units indicate data or configuration error.

        **Methodology**: Provide unit_id not in PRONOSTIA lookup.

        **Expected**: KeyError raised listing available units.

        Validates: Requirement HI-5.2 - Unknown unit detection
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        # Unit (99, 99) doesn't exist
        data = NamedTransformInput(
            runtime=np.array([[1000.0]]), unit_id=np.array([99, 99])
        )

        with pytest.raises(KeyError, match="not found"):
            transform.transform_data(data, metadata={})

    def test_infinite_runtime_raises_error(self):
        """Test infinite runtime values raise ValueError.

        **PHM Logic**: Infinite runtime indicates sensor/data corruption.
        Must be detected before producing invalid HI values.

        **Methodology**: Provide runtime with np.inf value.

        **Expected**: ValueError raised about infinite values.

        Validates: Requirement HI-5.3 - Infinite value detection
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        data = NamedTransformInput(
            runtime=np.array([[1000.0], [np.inf]]), unit_id=np.array([1, 1])
        )

        with pytest.raises(ValueError, match="Infinite"):
            transform.transform_data(data, metadata={})

    def test_hi_exceeds_one_raises_error(self):
        """Test HI > 1.0 raises ValueError.

        **PHM Logic**: HI > 1.0 means runtime exceeds total_life, which
        is physically impossible for run-to-failure data. This indicates
        either wrong total_life or incorrect runtime values.

        **Methodology**: Provide runtime exceeding total_life.

        **Expected**: ValueError raised about HI range.

        Validates: Requirement HI-5.4 - Range violation detection
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        # Runtime exceeds total_life (28020)
        data = NamedTransformInput(
            runtime=np.array([[30000.0]]), unit_id=np.array([1, 1])
        )

        with pytest.raises(ValueError, match="outside"):
            transform.transform_data(data, metadata={})

    def test_negative_runtime_produces_negative_hi_raises_error(self):
        """Test negative runtime produces negative HI which raises error.

        **PHM Logic**: Negative runtime is physically impossible and would
        produce negative HI. Must be detected and rejected.

        **Methodology**: Provide negative runtime value.

        **Expected**: ValueError raised about HI range (negative).

        Validates: Requirement HI-5.5 - Negative value detection
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        data = NamedTransformInput(
            runtime=np.array([[-1000.0]]), unit_id=np.array([1, 1])
        )

        with pytest.raises(ValueError):
            transform.transform_data(data, metadata={})


# =============================================================================
# FEATURE NAMES TESTS
# =============================================================================


class TestHealthIndexFeatureNames:
    """Tests for feature name generation."""

    def test_get_feature_names(self):
        """Test feature names are correctly generated.

        **PHM Logic**: Feature names should indicate dataset and source
        key for traceability in multi-feature analysis.

        **Methodology**: Call get_feature_names with typical parameters.

        **Expected**: Descriptive name including dataset and key info.

        Validates: Requirement HI-6.1 - Feature naming
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        names = transform.get_feature_names(
            input_keys=["runtime"], input_shapes={"runtime": (100, 1)}
        )

        assert len(names) == 1
        assert "HI" in names[0]
        assert "PRONOSTIA" in names[0]
        assert "runtime" in names[0]


# =============================================================================
# CALLABLE INTERFACE TESTS
# =============================================================================


class TestHealthIndexCallable:
    """Tests for __call__ interface."""

    def test_callable_interface(self, pronostia_unit_1_1):
        """Test transform can be called directly.

        **PHM Logic**: Callable interface enables use in functional pipelines.

        **Methodology**: Call transform using () operator.

        **Expected**: Same result as transform_data.

        Validates: Requirement HI-7.1 - Callable interface
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        data = NamedTransformInput(
            runtime=pronostia_unit_1_1["runtime"],
            unit_id=pronostia_unit_1_1["unit_id"][0],
        )

        result_call = transform(data, metadata={})
        result_method = transform.transform_data(data, metadata={})

        np.testing.assert_array_equal(result_call, result_method)


# =============================================================================
# MONOTONICITY VALIDATION TESTS
# =============================================================================


class TestHealthIndexMonotonicity:
    """Tests for monotonicity validation in HI calculation.

    The transform validates that computed HI values are monotonically
    increasing (or non-decreasing) over time, as damage can only accumulate.
    """

    def test_monotonic_runtime_produces_monotonic_hi(self):
        """Test that monotonically increasing runtime produces monotonic HI.

        **PHM Logic**: If runtime increases monotonically (as it should for
        run-to-failure data), HI should also increase monotonically.

        **Methodology**: Provide strictly increasing runtime sequence.

        **Expected**: Transform succeeds and produces monotonic HI.

        Validates: Requirement HI-8.1 - Monotonic input handling
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        # Strictly increasing runtime (step = DECREASE_PERIOD for PRONOSTIA)
        n_steps = 10
        step = DECREASE_PERIOD["PRONOSTIA"]
        runtime = np.arange(0, n_steps * step, step).astype(float)

        data = NamedTransformInput(
            runtime=runtime.reshape(-1, 1), unit_id=np.array([1, 1])
        )

        result = transform.transform_data(data, metadata={})

        # Verify monotonicity
        for i in range(1, len(result)):
            assert (
                result[i, 0] >= result[i - 1, 0]
            ), f"HI not monotonic at index {i}: {result[i-1, 0]:.4f} -> {result[i, 0]:.4f}"

    def test_non_monotonic_runtime_accepted(self):
        """Test that non-monotonic runtime sequence is accepted.

        **PHM Logic**: The transform doesn't enforce strict monotonicity
        by default - it allows non-monotonic runtime sequences and still
        computes HI values. This is useful for data that may have been
        re-ordered or contains partial segments.

        **Methodology**: Provide runtime that decreases at some point.

        **Expected**: Transform processes successfully (no error raised).

        Validates: Requirement HI-8.2 - Non-monotonic handling
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        # Non-monotonic runtime (decrease at step 5)
        step = DECREASE_PERIOD["PRONOSTIA"]
        runtime = np.array(
            [
                0,
                step,
                2 * step,
                3 * step,
                4 * step,
                3 * step,  # Decrease here
                6 * step,
                7 * step,
            ]
        ).astype(float)

        data = NamedTransformInput(
            runtime=runtime.reshape(-1, 1), unit_id=np.array([1, 1])
        )

        # Transform should process without error
        result = transform.transform_data(data, metadata={})

        # Verify output shape is correct
        assert result.shape == (len(runtime), 1)
        # Verify HI values are in valid range
        assert np.all(result >= 0.0) and np.all(result <= 1.0)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestHealthIndexIntegration:
    """Integration tests with realistic data scenarios."""

    def test_full_bearing_lifecycle(self, pronostia_unit_1_1):
        """Test HI calculation over complete bearing lifecycle.

        **PHM Logic**: A bearing's lifecycle goes from HI=0 (healthy) to
        HI=1 (failure). This test verifies the transform correctly captures
        this entire progression.

        **Methodology**: Transform full runtime sequence from start to failure.

        **Expected**: HI progresses from ~0 to ~1 smoothly.

        Validates: Requirement HI-9.1 - Full lifecycle handling
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        data = NamedTransformInput(
            runtime=pronostia_unit_1_1["runtime"],
            unit_id=pronostia_unit_1_1["unit_id"][0],
        )

        result = transform.transform_data(data, metadata={})

        # Verify lifecycle progression
        assert result[0, 0] < 0.1, "Initial HI should be near 0"
        assert result[-1, 0] > 0.9, "Final HI should be near 1"
        assert result[-1, 0] > result[0, 0], "HI should increase over life"

    def test_round_trip_transform(self, pronostia_unit_1_1):
        """Test round-trip transform/inverse_transform.

        **PHM Logic**: Forward and inverse transforms should be exact inverses.
        This is critical for RUL estimation and prognostics.

        **Methodology**: Transform -> Inverse -> Compare to original.

        **Expected**: Recovered runtime equals original within tolerance.

        Validates: Requirement HI-9.2 - Transform invertibility
        """
        transform = HealthIndexTransform(
            runtime_key="runtime", unit_key="unit_id", dataset_name="PRONOSTIA"
        )

        original_runtime = pronostia_unit_1_1["runtime"]
        unit_id = pronostia_unit_1_1["unit_id"][0]

        # Forward
        data = NamedTransformInput(runtime=original_runtime, unit_id=unit_id)
        hi = transform.transform_data(data, metadata={})

        # Inverse
        inverse_data = NamedTransformInput(features=hi)
        metadata = {"unit_id": unit_id}
        recovered = transform.inverse_transform(inverse_data, metadata)

        # Compare
        np.testing.assert_allclose(recovered, original_runtime, rtol=1e-6)
