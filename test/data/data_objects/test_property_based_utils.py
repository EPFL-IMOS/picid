"""
Phase 4: Property-based testing for data parsing/validation.

Uses Hypothesis to generate edge-case inputs for length consistency
and NaN checks. Skip if Hypothesis is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st

from picid.data.data_objects.utils import (
    get_length,
    check_length_consistency,
    check_for_nans,
)


@given(n=st.integers(min_value=0, max_value=100))
def test_get_length_ndarray_equals_n(n):
    """get_length(ndarray of shape (n, m)) == n."""
    if n == 0:
        arr = np.zeros((0, 2))
    else:
        arr = np.zeros((n, 2))
    assert get_length(arr) == n


@given(
    lengths=st.lists(st.integers(min_value=0, max_value=50), min_size=2, max_size=5),
)
def test_check_length_consistency_same_lengths(lengths):
    """When all arrays have the same length, check_length_consistency does not raise."""
    n = lengths[0]
    values = [np.zeros((n, 2)) for _ in lengths]
    keys = [f"k{i}" for i in range(len(values))]
    check_length_consistency(values, keys)


@given(
    size=st.integers(min_value=1, max_value=20),
)
def test_check_for_nans_no_nan_does_not_raise(size):
    """Numeric array without NaN does not raise."""
    arr = np.arange(size, dtype=np.float64)
    check_for_nans([arr], ["x"])
