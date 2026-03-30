"""
Characterization tests for picid.data.data_objects.utils only.

This module imports from picid.data.data_objects.utils; tests use real arrays
(numpy, pandas, awkward) and assert concrete expected outcomes.
Run: pytest test/data/data_objects/test_utils_characterization.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import awkward as ak

from picid.data.data_objects.utils import (
    get_length,
    check_length_consistency,
    check_for_nans,
    convert_to_numpy,
)


# ----- get_length -----


def test_get_length_ndarray():
    assert get_length(np.zeros((10, 2))) == 10
    assert get_length(np.zeros((0, 2))) == 0


def test_get_length_list_tuple():
    assert get_length([1, 2, 3]) == 3
    assert get_length((1, 2)) == 2


def test_get_length_0dim_returns_none():
    assert get_length(np.array(3)) is None


# ----- check_length_consistency -----


def test_check_length_consistency_ok():
    check_length_consistency(
        [np.zeros((5, 2)), np.zeros(5), [1, 2, 3, 4, 5]],
        ["a", "b", "c"],
    )


def test_check_length_consistency_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        check_length_consistency(
            [np.zeros((5, 2)), np.zeros(3)],
            ["a", "b"],
        )


def test_check_length_consistency_warn_only():
    with pytest.warns(UserWarning, match="Length mismatch"):
        check_length_consistency(
            [np.zeros((5, 2)), np.zeros(3)],
            ["a", "b"],
            raise_on_error=False,
        )


# ----- check_for_nans: ndarray -----


def test_check_for_nans_ndarray_raises():
    """Numeric ndarray with NaN raises ValueError with key in message."""
    arr = np.array([1.0, np.nan, 2.0])
    with pytest.raises(ValueError) as exc_info:
        check_for_nans([arr], ["x"])
    assert "NaN" in exc_info.value.args[0]
    assert "x" in exc_info.value.args[0]


def test_check_for_nans_ndarray_ok():
    """Numeric ndarray without NaN returns without error."""
    result = check_for_nans([np.array([1.0, 2.0])], ["x"])
    assert result is None


def test_check_for_nans_ndarray_non_numeric_no_raise():
    """Non-numeric ndarray is not checked for NaN; no error."""
    result = check_for_nans([np.array(["a", "b"])], ["x"])
    assert result is None


def test_check_for_nans_ndarray_warn_only():
    """With raise_on_error=False, NaN triggers warning only."""
    with pytest.warns(UserWarning) as warn_info:
        result = check_for_nans(
            [np.array([1.0, np.nan])],
            ["x"],
            raise_on_error=False,
        )
    assert result is None
    assert len(warn_info) == 1
    assert "NaN" in str(warn_info[0].message)
    assert "x" in str(warn_info[0].message)


# ----- check_for_nans: DataFrame -----


def test_check_for_nans_dataframe_with_nan_raises():
    """DataFrame with NaN in numeric column raises; message includes key."""
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [3.0, 4.0]})
    with pytest.raises(ValueError) as exc_info:
        check_for_nans([df], ["feat"])
    assert "NaN" in exc_info.value.args[0]
    assert "feat" in exc_info.value.args[0]


def test_check_for_nans_dataframe_no_nan_ok():
    """DataFrame with no NaN in numeric columns returns None."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = check_for_nans([df], ["feat"])
    assert result is None


def test_check_for_nans_dataframe_empty_numeric_ok():
    """Empty numeric DataFrame does not raise."""
    df = pd.DataFrame({"a": [], "b": []})
    result = check_for_nans([df], ["feat"])
    assert result is None


# ----- check_for_nans: Series -----


def test_check_for_nans_series_numeric_with_nan_raises():
    """Numeric Series with NaN raises; message includes key."""
    s = pd.Series([1.0, np.nan, 2.0])
    with pytest.raises(ValueError) as exc_info:
        check_for_nans([s], ["x"])
    assert "NaN" in exc_info.value.args[0]
    assert "x" in exc_info.value.args[0]


def test_check_for_nans_series_numeric_ok():
    """Numeric Series without NaN returns None."""
    result = check_for_nans([pd.Series([1.0, 2.0])], ["x"])
    assert result is None


def test_check_for_nans_series_non_numeric_skipped():
    """Non-numeric Series is not checked; no error."""
    result = check_for_nans([pd.Series(["a", "b"])], ["x"])
    assert result is None


# ----- check_for_nans: awkward.Array -----


def test_check_for_nans_awkward_numeric_with_nan_raises():
    """Numeric awkward array with NaN raises ValueError; message includes key."""
    arr = ak.Array([1.0, np.nan, 2.0])
    with pytest.raises(ValueError) as exc_info:
        check_for_nans([arr], ["ak_feat"])
    assert "NaN" in exc_info.value.args[0]
    assert "ak_feat" in exc_info.value.args[0]


def test_check_for_nans_awkward_numeric_no_nan_ok():
    """Numeric awkward array without NaN returns None."""
    arr = ak.Array([1.0, 2.0, 3.0])
    result = check_for_nans([arr], ["ak_feat"])
    assert result is None


def test_check_for_nans_awkward_nested_numeric_with_nan_raises():
    """Nested numeric awkward array with NaN raises; flatten is applied."""
    arr = ak.Array([[1.0, 2.0], [np.nan, 4.0]])
    with pytest.raises(ValueError) as exc_info:
        check_for_nans([arr], ["nested"])
    assert "NaN" in exc_info.value.args[0]
    assert "nested" in exc_info.value.args[0]


def test_check_for_nans_awkward_nested_numeric_no_nan_ok():
    """Nested numeric awkward array without NaN returns None."""
    arr = ak.Array([[1.0, 2.0], [3.0, 4.0]])
    result = check_for_nans([arr], ["nested"])
    assert result is None


def test_check_for_nans_awkward_non_numeric_skipped():
    """Non-numeric awkward array (e.g. strings) is not checked; no error."""
    arr = ak.Array(["a", "b", "c"])
    result = check_for_nans([arr], ["labels"])
    assert result is None


def test_check_for_nans_awkward_warn_only():
    """awkward array with NaN and raise_on_error=False issues warning only."""
    arr = ak.Array([1.0, np.nan])
    with pytest.warns(UserWarning) as warn_info:
        result = check_for_nans([arr], ["ak_x"], raise_on_error=False)
    assert result is None
    assert len(warn_info) == 1
    assert "NaN" in str(warn_info[0].message)
    assert "ak_x" in str(warn_info[0].message)


def test_check_for_nans_mixed_list_awkward_nan_raises():
    """Multiple values: one awkward with NaN raises for that key."""
    ok_arr = np.array([1.0, 2.0])
    bad_ak = ak.Array([1.0, np.nan])
    with pytest.raises(ValueError) as exc_info:
        check_for_nans([ok_arr, bad_ak], ["ok", "bad"])
    assert "bad" in exc_info.value.args[0]


# ----- convert_to_numpy -----


def test_convert_to_numpy_ndarray():
    a = np.array([[1, 2], [3, 4]])
    out = convert_to_numpy(a)
    np.testing.assert_array_equal(out, a)
    out1 = convert_to_numpy(np.array([1, 2, 3]), ensure_2d=True)
    assert out1.shape == (3, 1)


def test_convert_to_numpy_ndarray_ensure_2d_false():
    a = np.array([1, 2, 3])
    out = convert_to_numpy(a, ensure_2d=False)
    np.testing.assert_array_equal(out, a)


def test_convert_to_numpy_ndarray_bad_ndim_raises():
    with pytest.raises(ValueError, match="unsupported dimensions"):
        convert_to_numpy(np.zeros((2, 3, 4)), ensure_2d=True)


def test_convert_to_numpy_dataframe_numeric():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = convert_to_numpy(df)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
    assert out.dtype == np.float32


def test_convert_to_numpy_dataframe_non_numeric():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = convert_to_numpy(df)
    assert isinstance(out, np.ndarray)


def test_convert_to_numpy_series_numeric():
    s = pd.Series([1.0, 2.0, 3.0])
    out = convert_to_numpy(s)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 1)
    assert out.dtype == np.float32


def test_convert_to_numpy_series_non_numeric_returns_series():
    s = pd.Series(["a", "b", "c"])
    with pytest.warns(UserWarning, match="non-numeric"):
        out = convert_to_numpy(s)
    assert isinstance(out, pd.Series)
    pd.testing.assert_series_equal(out, s)


def test_convert_to_numpy_list():
    out = convert_to_numpy([1, 2, 3], ensure_2d=True)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 1)
