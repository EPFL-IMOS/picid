"""
Tests for PronostiaLoader (PHMD) with mocked phmd dataset and _process_unit logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picid.data.datasources.base.phmd_loader import PHMDMultiSourceLoader
from picid.data.datasources.phmd_pronostia import (
    PronostiaLoader,
    SAMPLE_RANGE_SIZE,
)


@pytest.fixture
def pronostia_kwargs():
    return {
        "data_name": "pronostia",
        "task_mode": "rul",
        "fold": 1,
        "cache_dir": "/tmp/pronostia_cache",
        "use_ragged": False,
        "meta_data": {
            "features": ["f1", "f2"],
            "identifier": "unit",
            "rul": "RUL",
        },
    }


@pytest.fixture
def small_unit_df():
    """Single unit DataFrame as passed to _process_unit."""
    n = 20
    return pd.DataFrame(
        {
            "unit": ["1_3"] * n,
            "f1": np.random.randn(n).astype(np.float32),
            "f2": np.random.randn(n).astype(np.float32),
            "RUL": np.arange(n - 1, -1, -1, dtype=np.float32),
        }
    )


def test_pronostia_process_unit_standard_mode_returns_dict(
    pronostia_kwargs, small_unit_df
):
    """_process_unit with use_ragged=False returns features, target, unit_id, metadata."""
    loader = PronostiaLoader(**pronostia_kwargs)
    loader.task_mode = "rul"
    out = loader._process_unit(
        small_unit_df,
        unit_name="Bearing1_3",
        features_col=["f1", "f2"],
        target_col="RUL",
    )
    assert "features" in out
    assert "target" in out
    assert "unit_id" in out
    assert "metadata" in out
    assert out["metadata"]["unit_name"] == "Bearing 1_3"
    assert out["metadata"]["task"] == "rul"
    assert len(out["features"]) == 20
    assert len(out["target"]) == 20


def test_pronostia_loader_uses_refactored_phmd_base(pronostia_kwargs):
    """PronostiaLoader now inherits from the refactored PHMD base."""
    loader = PronostiaLoader(**pronostia_kwargs)
    assert isinstance(loader, PHMDMultiSourceLoader)


def test_pronostia_process_unit_raises_for_non_rul_task(
    pronostia_kwargs, small_unit_df
):
    """_process_unit raises ValueError when task_mode is not 'rul'."""
    loader = PronostiaLoader(**pronostia_kwargs)
    loader.task_mode = "classification"
    with pytest.raises(ValueError, match="configured for 'rul'"):
        loader._process_unit(
            small_unit_df,
            unit_name="Bearing1_3",
            features_col=["f1", "f2"],
            target_col="RUL",
        )


def test_pronostia_process_unit_ragged_mode_requires_divisible_length(
    pronostia_kwargs, small_unit_df
):
    """_process_unit with use_ragged=True raises when length not divisible by SAMPLE_RANGE_SIZE."""
    pronostia_kwargs["use_ragged"] = True
    loader = PronostiaLoader(**pronostia_kwargs)
    loader.task_mode = "rul"
    with pytest.raises(ValueError, match="not divisible"):
        loader._process_unit(
            small_unit_df,
            unit_name="Bearing1_3",
            features_col=["f1", "f2"],
            target_col="RUL",
        )


def test_pronostia_process_unit_ragged_mode_success(pronostia_kwargs):
    """_process_unit with use_ragged=True and divisible length returns awkward-style structure."""
    n = SAMPLE_RANGE_SIZE * 2
    df = pd.DataFrame(
        {
            "unit": ["1_3"] * n,
            "f1": np.random.randn(n).astype(np.float32),
            "f2": np.random.randn(n).astype(np.float32),
            "RUL": np.repeat(
                np.array([100.0, 50.0], dtype=np.float32), SAMPLE_RANGE_SIZE
            ),
        }
    )
    pronostia_kwargs["use_ragged"] = True
    loader = PronostiaLoader(**pronostia_kwargs)
    loader.task_mode = "rul"
    out = loader._process_unit(
        df,
        unit_name="Bearing1_3",
        features_col=["f1", "f2"],
        target_col="RUL",
    )
    assert "features" in out
    assert "target" in out
    assert out["metadata"]["unit_length"] == 2
