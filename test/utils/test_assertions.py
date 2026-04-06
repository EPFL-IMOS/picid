"""Comprehensive tests for assertion utilities.

This module tests the validation functions used to ensure data
integrity throughout the PHM data pipeline.

PHM Context:
-----------
Data validation is critical in PHM systems where incorrect data
shapes or types can lead to silent failures in model training.

Test Coverage Strategy:
----------------------
1. **List of ndarray Validation**: Variable-length array lists
2. **Single ndarray Validation**: Fixed-size array handling
3. **Dimension Checking**: 2D, 3D, N-D arrays
4. **Edge Cases**: Empty lists, single arrays, mixed types
"""

import pytest
import numpy as np
from omegaconf import ListConfig

from picid.utils.assertions import assert_list_of_ndarray_or_nd_array_for_dims


class TestAssertListOfNdarray:
    """Tests for assert_list_of_ndarray_or_nd_array_for_dims."""

    def test_valid_list_of_2d_arrays(self):
        """Test validation of list of 2D arrays.

        **PHM Logic**: Multi-unit data is often stored as list of arrays.

        **Methodology**: Pass valid list of 2D arrays; assert return is ``None``.

        **Expected**: Returns ``None`` (validation success).

        Validates: Requirement ASS-1.1 - Valid 2D list validation
        """
        arrays = [
            np.random.randn(100, 5),
            np.random.randn(80, 5),
            np.random.randn(120, 5),
        ]

        assert assert_list_of_ndarray_or_nd_array_for_dims(arrays, dims=2) is None

    def test_valid_single_2d_array(self):
        """Test validation of single 2D array.

        **PHM Logic**: Single-unit data is a single array.

        **Methodology**: Pass single 2D array; assert return is ``None``.

        **Expected**: Returns ``None`` (validation success).

        Validates: Requirement ASS-1.2 - Valid single array validation
        """
        array = np.random.randn(100, 5)

        assert assert_list_of_ndarray_or_nd_array_for_dims(array, dims=2) is None

    def test_valid_list_of_3d_arrays(self):
        """Test validation of list of 3D arrays.

        **PHM Logic**: Time series with channels may be 3D (samples, time, channels).

        **Methodology**: Pass valid list of 3D arrays; assert return is ``None``.

        **Expected**: Returns ``None`` (validation success).

        Validates: Requirement ASS-1.3 - Valid 3D list validation
        """
        arrays = [
            np.random.randn(100, 10, 3),
            np.random.randn(80, 10, 3),
        ]

        assert assert_list_of_ndarray_or_nd_array_for_dims(arrays, dims=3) is None

    def test_listconfig_not_supported_for_arrays(self):
        """Test that ListConfig cannot hold numpy arrays.

        **PHM Logic**: OmegaConf doesn't support numpy arrays directly.

        **Methodology**: Attempt to create ListConfig with arrays.

        **Expected**: UnsupportedValueType error from OmegaConf.

        Validates: Requirement ASS-1.4 - ListConfig limitation
        """
        from omegaconf.errors import UnsupportedValueType

        # OmegaConf cannot hold numpy arrays
        with pytest.raises(UnsupportedValueType):
            ListConfig(
                [
                    np.random.randn(100, 5),
                    np.random.randn(80, 5),
                ]
            )

    def test_invalid_non_array_in_list(self):
        """Test that non-array elements raise error.

        **PHM Logic**: List must contain only ndarrays.

        **Methodology**: Include non-array element in list.

        **Expected**: AssertionError raised.

        Validates: Requirement ASS-1.5 - Non-array detection
        """
        invalid_list = [
            np.random.randn(100, 5),
            "not an array",  # Invalid
            np.random.randn(80, 5),
        ]

        with pytest.raises(AssertionError, match="Element is not a numpy ndarray\\."):
            assert_list_of_ndarray_or_nd_array_for_dims(invalid_list, dims=2)

    def test_invalid_wrong_dimensions(self):
        """Test that wrong dimensions raise error.

        **PHM Logic**: All arrays must have expected dimensionality.

        **Methodology**: Pass 2D array, request 3D validation.

        **Expected**: AssertionError raised about dimensions.

        Validates: Requirement ASS-1.6 - Dimension mismatch detection
        """
        array_2d = np.random.randn(100, 5)  # 2D

        with pytest.raises(AssertionError, match=r"Array does not have 3 dimensions\."):
            assert_list_of_ndarray_or_nd_array_for_dims(array_2d, dims=3)

    def test_invalid_mixed_dimensions(self):
        """Test that mixed dimensions in list raise error.

        **PHM Logic**: All arrays in list must have same dimensionality.

        **Methodology**: Pass list with 2D and 3D arrays mixed.

        **Expected**: AssertionError raised.

        Validates: Requirement ASS-1.7 - Mixed dimension detection
        """
        mixed_list = [
            np.random.randn(100, 5),  # 2D
            np.random.randn(80, 5, 3),  # 3D
        ]

        with pytest.raises(AssertionError, match=r"Array does not have 2 dimensions\."):
            assert_list_of_ndarray_or_nd_array_for_dims(mixed_list, dims=2)

    def test_invalid_unsupported_type(self):
        """Test that unsupported types raise error.

        **PHM Logic**: Only list, ListConfig, or ndarray accepted.

        **Methodology**: Pass dict (unsupported type).

        **Expected**: AssertionError raised.

        Validates: Requirement ASS-1.8 - Unsupported type detection
        """
        invalid_input = {"key": np.random.randn(100, 5)}

        with pytest.raises(
            AssertionError,
            match="Input is neither a list/ListConfig nor a numpy ndarray\\.",
        ):
            assert_list_of_ndarray_or_nd_array_for_dims(invalid_input, dims=2)

    def test_tuple_of_ndarrays_rejected_not_list_like_contract(self):
        """Tuple of ndarrays must not be accepted as a list/ListConfig stand-in.

        **Bug sentinel**: ``tuple`` is iterable like ``list`` but must stay unsupported so
        callers do not silently pass a non-contract container type.
        """
        tuple_input = (np.random.randn(100, 5), np.random.randn(80, 5))

        with pytest.raises(
            AssertionError,
            match="Input is neither a list/ListConfig nor a numpy ndarray\\.",
        ):
            assert_list_of_ndarray_or_nd_array_for_dims(tuple_input, dims=2)

    def test_empty_list(self):
        """Test handling of empty list.

        **PHM Logic**: Empty list should pass (no arrays to validate).

        **Methodology**: Pass empty list.

        **Expected**: Empty list iterates zero elements and returns ``None``.

        Validates: Requirement ASS-1.9 - Empty list handling
        """
        assert assert_list_of_ndarray_or_nd_array_for_dims([], dims=2) is None

    def test_empty_listconfig_returns_none(self):
        """Empty ``ListConfig`` iterates zero elements; validation succeeds with ``None``.

        **Contract**: Same branch as list; no elements means no ndarray checks run.
        """
        empty_cfg = ListConfig([])
        assert assert_list_of_ndarray_or_nd_array_for_dims(empty_cfg, dims=2) is None

    def test_single_element_list(self):
        """Test validation of single-element list.

        **PHM Logic**: Single-element list is valid.

        **Methodology**: Pass list with one array.

        **Expected**: Returns ``None`` (validation success).

        Validates: Requirement ASS-1.10 - Single element handling
        """
        single_list = [np.random.randn(100, 5)]

        assert assert_list_of_ndarray_or_nd_array_for_dims(single_list, dims=2) is None


class TestAssertDimensions:
    """Additional dimension validation tests."""

    def test_1d_array_validation(self):
        """Test validation of 1D arrays.

        **PHM Logic**: Some transforms produce 1D output (e.g., statistics).

        **Methodology**: Pass 1D array, validate for dims=1.

        **Expected**: Returns ``None`` (validation success).

        Validates: Requirement ASS-2.1 - 1D validation
        """
        array_1d = np.random.randn(100)

        assert assert_list_of_ndarray_or_nd_array_for_dims(array_1d, dims=1) is None

    def test_4d_array_validation(self):
        """Test validation of 4D arrays (image-like data).

        **PHM Logic**: Some PHM data includes images (batch, height, width, channels).

        **Methodology**: Pass 4D array, validate for dims=4.

        **Expected**: Returns ``None`` (validation success).

        Validates: Requirement ASS-2.2 - 4D validation
        """
        array_4d = np.random.randn(10, 32, 32, 3)

        assert assert_list_of_ndarray_or_nd_array_for_dims(array_4d, dims=4) is None
