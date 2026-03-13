import numpy as np
from typing import List


def get_mode_change_right_indices(x: np.ndarray) -> List[int]:
    """
    Identifies the right-exclusive boundaries (i.e., start of new regimes)
    of contiguous mode segments in the input array `x`.

    Parameters
    ----------
    x : np.ndarray
        A 1D NumPy array of mode values (e.g., 0s and 1s).

    Returns
    -------
    List[int]
        A list of indices marking the start of new segments, to be used as exclusive slice boundaries.

    Example
    -------
    >>> get_mode_change_right_indices(np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1]))
    [3, 7, 9, 12]
    """
    change_indices = np.where(x[:-1] != x[1:])[0]
    # +1 gives the exclusive right index where new mode starts
    right_boundaries = (change_indices + 1).tolist()
    # Add final end index (exclusive)
    right_boundaries.append(len(x))
    return right_boundaries
