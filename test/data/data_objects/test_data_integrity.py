"""
Phase 1: Data integrity & schema validation for data objects.

Uses real/synthetic data (numpy, pandas) to validate:
- Shape and type invariance after get_sanitized_data
- Value ranges (e.g. normalized [0,1])
- Length consistency and NaN checks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picid.data.data_objects import NamedTransformInput, SplitDatasetContainer
from picid.data.data_objects.utils import check_for_nans, check_length_consistency


# ---------------------------------------------------------------------------
# NamedTransformInput.get_sanitized_data
# ---------------------------------------------------------------------------


def test_get_sanitized_data_preserves_shape_and_keys(synthetic_normalized_float32):
    """After sanitization, keys and array shapes are preserved."""
    inp = NamedTransformInput(features=synthetic_normalized_float32.copy())
    out = inp.get_sanitized_data(check_nans=True, check_lengths=True)
    assert "features" in out
    assert out["features"].shape == synthetic_normalized_float32.shape
    assert np.allclose(out["features"], synthetic_normalized_float32)


def test_get_sanitized_data_nan_raises_by_default():
    """NaN in data raises when check_nans=True (default)."""
    data = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    inp = NamedTransformInput(x=data)
    with pytest.raises(ValueError, match="NaN"):
        inp.get_sanitized_data(check_nans=True)


def test_get_sanitized_data_length_mismatch_raises():
    """Inconsistent lengths across keys raise when check_lengths=True."""
    inp = NamedTransformInput(
        a=np.zeros((10, 2)),
        b=np.zeros((5, 2)),
    )
    with pytest.raises(ValueError, match="Length mismatch"):
        inp.get_sanitized_data(check_lengths=True)


# ---------------------------------------------------------------------------
# Value range assertions (normalized [0,1] vs [0,255])
# ---------------------------------------------------------------------------


def test_value_range_normalized_float32(synthetic_normalized_float32):
    """Synthetic normalized data stays in [0,1] and is float32."""
    assert synthetic_normalized_float32.dtype == np.float32
    assert (
        synthetic_normalized_float32.min() >= 0.0
        and synthetic_normalized_float32.max() <= 1.0
    )


def test_value_range_uint8_image_like(synthetic_image_like_uint8):
    """Synthetic image-like data is uint8 in [0,255]."""
    assert synthetic_image_like_uint8.dtype == np.uint8
    assert (
        synthetic_image_like_uint8.min() >= 0
        and synthetic_image_like_uint8.max() <= 255
    )


# ---------------------------------------------------------------------------
# SplitDatasetContainer validation
# ---------------------------------------------------------------------------


def test_split_container_validate_consistent_units():
    """validate() passes when all feature lists have same number of units per split."""
    c = SplitDatasetContainer(
        features={
            "train": [np.zeros((10, 2)), np.zeros((10, 2))],
            "val": [np.zeros((5, 2))],
        },
        target={
            "train": [np.zeros((10, 1)), np.zeros((10, 1))],
            "val": [np.zeros((5, 1))],
        },
    )
    report = c.validate()
    assert report["is_consistent"] is True


def test_split_container_validate_inconsistent_units_reports_heterogeneity():
    """validate() reports when unit counts differ within a split."""
    c = SplitDatasetContainer(
        features={"train": [np.zeros((10, 2)), np.zeros((10, 2)), np.zeros((10, 2))]},
        target={"train": [np.zeros((10, 1))]},
    )
    report = c.validate()
    assert report["is_consistent"] is False
    assert report["unit_counts_match"] is False


# ---------------------------------------------------------------------------
# Utils: check_for_nans / check_length_consistency (real assertions)
# ---------------------------------------------------------------------------


def test_check_for_nans_dataframe_numeric_only():
    """DataFrame with NaN in numeric column raises."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10, 20, 30]})
    with pytest.raises(ValueError, match="NaN"):
        check_for_nans([df], ["df"])


def test_check_length_consistency_ndarray_and_list():
    """Mix of ndarray and list with same length passes."""
    check_length_consistency(
        [np.zeros((7, 2)), [1, 2, 3, 4, 5, 6, 7]],
        ["arr", "list"],
    )
