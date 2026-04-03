"""Small, explicit NumPy RNG construction for reproducible test payloads."""

from __future__ import annotations

import numpy as np


def numpy_rs(seed: int) -> np.random.RandomState:
    """Return a legacy ``RandomState`` (MT19937) for stable draws across NumPy versions."""
    return np.random.RandomState(seed)
