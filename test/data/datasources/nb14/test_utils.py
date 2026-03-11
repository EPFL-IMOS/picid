"""Tests for nb14.utils (flatten_cycles, table_to_ak_array) — no I/O."""

import numpy as np
import awkward as ak

from picid.data.datasources.nb14.utils import flatten_cycles, table_to_ak_array


def test_flatten_cycles_basic():
    """flatten_cycles with (B=2, L=3, C=2) and no valid_lengths uses full L."""
    x = np.arange(12).reshape(2, 3, 2).astype(np.float64)
    y = np.array([[1.0], [2.0]])
    fx, fy, indices = flatten_cycles(x, y)
    assert fx.shape == (6, 2)
    assert fy.size == 6
    assert indices == [(0, 3), (3, 6)]
    np.testing.assert_array_equal(fx[:3], x[0])
    np.testing.assert_array_equal(np.unique(fy[:3]), [1.0])
    np.testing.assert_array_equal(np.unique(fy[3:]), [2.0])


def test_flatten_cycles_with_valid_lengths():
    """flatten_cycles with valid_lengths truncates each cycle."""
    x = np.arange(12).reshape(2, 3, 2).astype(np.float64)
    y = np.array([[1.0], [2.0]])
    valid_lengths = [2, 1]
    fx, fy, indices = flatten_cycles(x, y, valid_lengths=valid_lengths)
    assert fx.shape == (3, 2)
    assert fy.size == 3
    assert indices == [(0, 2), (2, 3)]
    np.testing.assert_array_equal(fx[:2], x[0, :2])
    np.testing.assert_array_equal(fx[2:], x[1, :1])


def test_table_to_ak_array_basic():
    """table_to_ak_array returns awkward array with correct layout."""
    x = np.arange(12).reshape(2, 3, 2).astype(np.float64)
    out = table_to_ak_array(x)
    assert isinstance(out, ak.Array)
    assert len(out) == 2
    assert ak.num(out[0], axis=0) == 3
    assert ak.num(out[1], axis=0) == 3


def test_table_to_ak_array_with_valid_lengths():
    """table_to_ak_array with valid_lengths truncates rows per batch."""
    x = np.arange(12).reshape(2, 3, 2).astype(np.float64)
    valid_lengths = [2, 1]
    out = table_to_ak_array(x, valid_lengths=valid_lengths)
    assert isinstance(out, ak.Array)
    assert len(out) == 2
    assert ak.num(out[0], axis=0) == 2
    assert ak.num(out[1], axis=0) == 1
