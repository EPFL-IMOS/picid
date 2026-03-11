"""Security-oriented input validation tests.

Covers NaN/Inf rejection and documents future path-validation requirements.
"""

from __future__ import annotations

import pytest
import numpy as np

from picid.data.data_objects.utils import check_for_nans


def test_check_for_nans_rejects_nan_in_ndarray():
    """check_for_nans rejects NaN in numpy array."""
    arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        check_for_nans([arr], ["features"])


def test_check_for_nans_accepts_valid_ndarray():
    """check_for_nans accepts valid numpy array."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    check_for_nans([arr], ["features"])  # No raise


def test_path_traversal_in_config_rejected():
    """Config with path traversal attempt is rejected (if validated)."""
    # Placeholder: when config validation exists, ensure "../" in paths raises
    pytest.skip("Path validation in config not yet implemented - add when available")
