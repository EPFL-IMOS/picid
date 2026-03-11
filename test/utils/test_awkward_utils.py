import pytest
import numpy as np
import awkward as ak

# We assume your utility functions are in a file named `awkward_utils.py`
# in the same directory or on the python path.
from picid.utils.awkward_utils import (
    ak_flatten_variable_dims,
    ak_unflatten_discontinous_groups,
    ak_find_var_dims,
    find_singular_ragged_dim,
    ragged_index_tuples,
    iterate_ragged_sub_blocks,
    blocks_to_ragged_array,
    ak_regularize_regular_axes,
    get_ak_shape,
)


def test_ak_flatten_variable_dims():
    """Tests flattening specified or all variable dimensions across 5 scenarios."""

    # Scenario 1: (N, var, var) -> (2, var, var)
    arr_n_var_var = ak.Array([[[1, 2], [3]], [[4, 5, 6]]])
    # Case 1a: Automatic (axis=None) -> flattens axes 1, 2
    res_1a = ak_flatten_variable_dims(arr_n_var_var, axis=None)
    assert ak.all(res_1a == [1, 2, 3, 4, 5, 6])
    assert res_1a.ndim == 1
    # Case 1b: Specific axis (axis=1) -> flattens axis 1
    res_1b = ak_flatten_variable_dims(arr_n_var_var, axis=1)
    assert ak.all(res_1b == [[1, 2], [3], [4, 5, 6]])
    assert str(ak.type(res_1b)) == "3 * var * int64"
    # Case 1c: Specific axis (axis=2) -> flattens axis 2
    res_1c = ak_flatten_variable_dims(arr_n_var_var, axis=2)
    assert ak.all(res_1c == [[1, 2, 3], [4, 5, 6]])
    assert str(ak.type(res_1c)) == "2 * var * int64"
    # Case 1d: List of axes (axis=[1, 2])
    res_1d = ak_flatten_variable_dims(arr_n_var_var, axis=[1, 2])
    assert ak.all(res_1d == [1, 2, 3, 4, 5, 6])
    assert res_1d.ndim == 1

    # Scenario 2: (N, var, F) -> (2, var, 2)
    arr_n_var_f_base = ak.Array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]]
    )
    arr_n_var_f = ak.to_regular(arr_n_var_f_base, axis=2)
    assert str(ak.type(arr_n_var_f)) == "2 * var * 2 * float64"
    # Case 2a: Automatic (axis=None) -> flattens axis 1
    res_2a = ak_flatten_variable_dims(arr_n_var_f, axis=None)
    assert ak.all(res_2a == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    assert str(ak.type(res_2a)) == "5 * 2 * float64"
    # Case 2b: Specific axis (axis=1)
    res_2b = ak_flatten_variable_dims(arr_n_var_f, axis=1)
    assert ak.all(res_2a == res_2b)

    # Scenario 3: (N, C, var) -> (2, 2, var)
    arr_n_c_var_base = ak.Array([[[1, 2], [3]], [[4, 5, 6], [7]]])
    arr_n_c_var = ak.to_regular(arr_n_c_var_base, axis=1)
    assert str(ak.type(arr_n_c_var)) == "2 * 2 * var * int64"
    # Case 3a: Automatic (axis=None) -> flattens axis 2
    res_3a = ak_flatten_variable_dims(arr_n_c_var, axis=None)
    assert ak.all(res_3a == [[1, 2, 3], [4, 5, 6, 7]])

    # --- START FIX ---
    # The result is (2, var) because the inner lists have lengths 3 and 4.
    # The assertion was wrong.
    assert str(ak.type(res_3a)) == "2 * var * int64"
    # --- END FIX ---

    # Case 3b: Specific axis (axis=2)
    res_3b = ak_flatten_variable_dims(arr_n_c_var, axis=2)
    assert ak.all(res_3a == res_3b)

    # Scenario 4: (N, var) -> (3, var)
    arr_n_var = ak.Array([[1, 2], [3], [4, 5, 6]])
    # Case 4a: Automatic (axis=None) -> flattens axis 1
    res_4a = ak_flatten_variable_dims(arr_n_var, axis=None)
    assert ak.all(res_4a == [1, 2, 3, 4, 5, 6])
    assert res_4a.ndim == 1

    # Scenario 5: (N, C, F) -> (2, 3, 2)
    arr_n_c_f = ak.Array(np.arange(12).reshape(2, 3, 2))
    # Case 5a: Automatic (axis=None) -> does nothing
    res_5a = ak_flatten_variable_dims(arr_n_c_f, axis=None)
    assert ak.all(res_5a == arr_n_c_f)
    assert str(ak.type(res_5a)) == "2 * 3 * 2 * int64"
    # Case 5b: Specific axis (axis=1) -> flattens axis 1
    res_5b = ak_flatten_variable_dims(arr_n_c_f, axis=1)
    assert ak.all(res_5b == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
    assert str(ak.type(res_5b)) == "6 * 2 * int64"


def test_ak_unflatten_discontinous_groups():
    """Tests reshaping a flat array based on a discontinuous group index."""
    values = ak.Array([10, 20, 30, 40, 50])
    groups = ak.Array([1, 2, 1, 2, 1])  # Discontinuous groups

    # The function should sort by group, so group 1 = [10, 30, 50]
    # and group 2 = [20, 40]
    result = ak_unflatten_discontinous_groups(values, groups)
    expected = ak.Array([[10, 30, 50], [20, 40]])

    assert ak.all(result == expected)


def test_ak_unflatten_discontinous_groups_errors():
    """Tests assertion errors for mismatched lengths."""
    values = ak.Array([1, 2, 3])
    groups_short = ak.Array([1, 2])

    with pytest.raises(
        AssertionError, match="Values and groups must have the same length"
    ):
        ak_unflatten_discontinous_groups(values, groups_short)


def test_ak_find_var_dims():
    """Tests finding the axes of variable-length dimensions."""
    # (2, var, var)
    arr_ragged = ak.Array([[[1, 2], [3]], [[4, 5, 6]]])
    assert ak_find_var_dims(arr_ragged) == [1, 2]

    # (2, 3, 2) - Regular numpy array
    arr_regular = ak.Array(np.arange(12).reshape(2, 3, 2))
    assert ak_find_var_dims(arr_regular) == []

    # (2, var)
    arr_simple_ragged = ak.Array([[1], [2, 3]])
    assert ak_find_var_dims(arr_simple_ragged) == [1]

    # (3, var) with an optional type (None)
    arr_option = ak.Array([[1, 2], [3], None])
    assert ak_find_var_dims(arr_option) == [1]


def test_ragged_index_tuples():
    """Tests generation of index tuples for ragged dimensions."""
    arr = ak.Array([[[1], [2, 3]], [[4, 5]]])  # Shape: (2, var, var)

    # Test with 1 ragged dim
    result_1d = ragged_index_tuples(arr, ragged_dims=1)
    expected_1d = [(0,), (1,)]
    assert result_1d == expected_1d

    # Test with 2 ragged dims
    result_2d = ragged_index_tuples(arr, ragged_dims=2)
    # (0,0), (0,1), (1,0)
    expected_2d = [(0, 0), (0, 1), (1, 0)]
    assert result_2d == expected_2d


def test_iterate_ragged_sub_blocks():
    """Tests iteration over sub-blocks of a ragged array."""
    arr = ak.Array([[[10], [20, 30]], [[40, 50]]])

    # We test with ragged_dims=2, which means the sub-blocks are the
    # innermost variable lists: [10], [20, 30], and [40, 50]

    results = list(iterate_ragged_sub_blocks(arr, ragged_dims=2))

    indices = [idx for idx, block in results]
    blocks = [
        block.to_list() for idx, block in results
    ]  # Convert blocks to list for easy comparison

    expected_indices = [(0, 0), (0, 1), (1, 0)]
    expected_blocks = [[10], [20, 30], [40, 50]]

    assert indices == expected_indices
    assert blocks == expected_blocks


def test_blocks_to_ragged_array():
    """Tests constructing a ragged array from coordinates and blocks."""
    # Use the output from the previous test
    coords = [(0, 0), (0, 1), (1, 0)]
    # Use np.array as specified in the function's docstring
    blocks = [np.array([10]), np.array([20, 30]), np.array([40, 50])]

    result = blocks_to_ragged_array(coords, blocks)
    expected = ak.Array([[[10], [20, 30]], [[40, 50]]])

    assert ak.all(result == expected)

    # Test with a different ordering
    coords_shuffled = [(1, 0), (0, 1), (0, 0)]
    blocks_shuffled = [np.array([40, 50]), np.array([20, 30]), np.array([10])]

    result_shuffled = blocks_to_ragged_array(coords_shuffled, blocks_shuffled)
    # The function should sort the keys, so the result is identical
    assert ak.all(result_shuffled == expected)


def test_ak_regularize_regular_axes():
    """Tests converting regular axes (typed as var) to regular."""
    # This array is regular (2, 2, 2) but typed as (2, var, var)
    arr_irregular = ak.Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert "var" in str(ak.type(arr_irregular))  # Check it's irregular

    result = ak_regularize_regular_axes(arr_irregular)

    # The result should be regular
    assert "var" not in str(ak.type(result))
    assert str(ak.type(result)) == "2 * 2 * 2 * int64"

    # Test a mixed array
    # This array is (2, var, var) with a complex union type.
    arr_mixed = ak.Array([[[1, 2], [3, 4]], [[[5, 6]]]])
    assert "var" in str(ak.type(arr_mixed))

    result_mixed = ak_regularize_regular_axes(arr_mixed)

    # Your function correctly identifies that axis 1 is 'var' (lengths 2 and 1)
    # and axis 2 is 'var' (lengths 2 and 1).
    # The innermost type is a union.
    # The assertion is now corrected to match the actual, correct output.
    assert str(ak.type(result_mixed)) == "2 * var * var * union[int64, var * int64]"


def test_get_ak_shape():
    """Tests getting the shape of an awkward array."""
    # (2, var, var)
    arr_ragged = ak.Array([[[1, 2], [3]], [[4, 5, 6]]])
    assert get_ak_shape(arr_ragged) == [2, "var", "var"]

    # (2, 3, 2) - Regular numpy array
    arr_regular = ak.Array(np.arange(12).reshape(2, 3, 2))
    assert get_ak_shape(arr_regular) == [2, 3, 2]

    # (3, var) with an optional type (None)
    arr_option = ak.Array([[1, 2], [3], None])
    assert get_ak_shape(arr_option) == [3, "var"]

    # (2, 2, var)
    # This array is *constructed* from Python lists, so its type is
    # (2, var, var). Your get_ak_shape function correctly reports this type.
    # The test expectation was wrong.
    arr_inner_ragged = ak.Array([[[1, 2], [3]], [[4, 5, 6], [7]]])
    assert get_ak_shape(arr_inner_ragged) == [2, "var", "var"]


class TestFindSingularRaggedDim:
    """Tests for find_singular_ragged_dim (ragged-dimension detection)."""

    def test_find_singular_ragged_dim_none_regular_array(self):
        """No variable-length dim (regular array) returns None."""
        regular = ak.from_numpy(np.zeros((2, 3, 4)))
        result = find_singular_ragged_dim(regular)
        assert result is None

    def test_find_singular_ragged_dim_none_iterable_of_regular(self):
        """Iterable of regular arrays returns None."""
        regular = ak.from_numpy(np.zeros((2, 3)))
        result = find_singular_ragged_dim([regular])
        assert result is None

    def test_find_singular_ragged_dim_single_ragged(self):
        """Single ragged dim returns that axis index."""
        ragged = ak.Array([[1.0, 2.0], [3.0]])  # (n, var) -> axis 1 is ragged
        result = find_singular_ragged_dim(ragged)
        assert result == 1

    def test_find_singular_ragged_dim_single_ragged_iterable(self):
        """Single ragged dim from iterable (e.g. nti.values()) returns axis."""
        ragged = ak.Array([[1.0, 2.0], [3.0]])  # (n, var) -> axis 1 is ragged
        result = find_singular_ragged_dim([ragged])
        assert result == 1

    def test_find_singular_ragged_dim_multiple_ragged_dims_raises(self):
        """More than one variable-length dim raises ValueError."""
        arr_two_var = ak.Array([[[1.0], [2.0, 3.0]], [[4.0]]])
        with pytest.raises(
            ValueError, match="Expected at most one variable-length dimension"
        ):
            find_singular_ragged_dim(arr_two_var)
