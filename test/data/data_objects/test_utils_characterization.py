"""
Characterization tests for lmetk.data.data_objects.utils only.

This module imports ONLY from lmetk.data.data_objects.utils (no data.py/base.py),
so it can be run in isolation without pulling in awkward/pyarrow via data layer.
Run: pytest test/data/data_objects/test_utils_characterization.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Intentionally only utils – no data.py/base.py
from lmetk.data.data_objects.utils import (
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
    with pytest.raises(ValueError, match="NaN"):
        check_for_nans([np.array([1.0, np.nan, 2.0])], ["x"])


def test_check_for_nans_ndarray_ok():
    check_for_nans([np.array([1.0, 2.0])], ["x"])


def test_check_for_nans_ndarray_non_numeric_no_raise():
    check_for_nans([np.array(["a", "b"])], ["x"])


def test_check_for_nans_ndarray_warn_only():
    with pytest.warns(UserWarning, match="NaN"):
        check_for_nans(
            [np.array([1.0, np.nan])],
            ["x"],
            raise_on_error=False,
        )


# ----- check_for_nans: DataFrame -----


def test_check_for_nans_dataframe_with_nan_raises():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="NaN"):
        check_for_nans([df], ["feat"])


def test_check_for_nans_dataframe_no_nan_ok():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    check_for_nans([df], ["feat"])


def test_check_for_nans_dataframe_empty_numeric_ok():
    df = pd.DataFrame({"a": [], "b": []})
    check_for_nans([df], ["feat"])


# ----- check_for_nans: Series -----


def test_check_for_nans_series_numeric_with_nan_raises():
    s = pd.Series([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="NaN"):
        check_for_nans([s], ["x"])


def test_check_for_nans_series_numeric_ok():
    check_for_nans([pd.Series([1.0, 2.0])], ["x"])


def test_check_for_nans_series_non_numeric_skipped():
    check_for_nans([pd.Series(["a", "b"])], ["x"])


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
