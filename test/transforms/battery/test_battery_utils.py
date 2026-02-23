import numpy as np

from picid.transforms.battery.utils import get_mode_change_right_indices


def test_get_mode_change_right_indices_basic():
    """Docstring example: [0,0,0,1,1,1,1,0,0,1,1,1] -> [3, 7, 9, 12]."""
    x = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1])
    result = get_mode_change_right_indices(x)
    assert result == [3, 7, 9, 12]


def test_get_mode_change_right_indices_single_mode():
    """All same value: only final boundary."""
    x = np.array([1, 1, 1, 1])
    result = get_mode_change_right_indices(x)
    assert result == [4]


def test_get_mode_change_right_indices_alternating():
    """Alternating modes."""
    x = np.array([0, 1, 0, 1, 0])
    result = get_mode_change_right_indices(x)
    assert result == [1, 2, 3, 4, 5]


def test_get_mode_change_right_indices_single_element():
    """Single element."""
    x = np.array([5])
    result = get_mode_change_right_indices(x)
    assert result == [1]


def test_get_mode_change_right_indices_multiple_modes():
    """Three different modes."""
    x = np.array([1, 1, 2, 2, 2, 3, 3])
    result = get_mode_change_right_indices(x)
    assert result == [2, 5, 7]
