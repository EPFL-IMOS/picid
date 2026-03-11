"""
Archived tests for PHMDBatteryLoader (extracted from test_phmd_wrappers and test_phmd_methods_mocked).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from picid.data.datasources.phmd_battery import PHMDBatteryLoader


def _minimal_phmd_kwargs(cache_dir="/tmp/phmd_cache"):
    return {
        "fold": 0,
        "data_name": "NB14",
        "task_mode": "rul",
        "cache_dir": cache_dir,
    }


def _phmd_kwargs():
    return {
        "fold": 0,
        "data_name": "NB14",
        "task_mode": "rul",
        "cache_dir": "/tmp/phmd",
    }


def test_phmd_battery_loader_init():
    """PHMDBatteryLoader initializes with minimal kwargs."""
    loader = PHMDBatteryLoader(**_minimal_phmd_kwargs())
    assert loader.data_name == "NB14"
    assert loader.task_mode == "rul"


def test_phmd_battery_load_data_mocked():
    """PHMDBatteryLoader.load_data with mocked _load_data; get_data_name returns data_dict['metadata']."""
    loader = PHMDBatteryLoader(**_minimal_phmd_kwargs())
    # PHMD get_data_name() returns self.data_dict["metadata"]
    fake_data = {
        "features": {
            "train": [np.zeros((10, 2))],
            "val": [np.zeros((5, 2))],
            "test": [np.zeros((5, 2))],
        },
        "target": {
            "train": [np.zeros((10, 1))],
            "val": [np.zeros((5, 1))],
            "test": [np.zeros((5, 1))],
        },
        "metadata": ["NB14"],
    }
    with patch.object(loader, "_load_data", return_value=fake_data):
        loader.load_data()
    assert loader._is_loaded
    assert loader.get_data_name() == ["NB14"]


def test_phmd_battery_get_features_columns():
    """_get_features_columns returns meta_data['features'] or NB1 override."""
    loader = PHMDBatteryLoader(**_phmd_kwargs())
    loader.meta_data = {"features": ["v", "i", "t"]}
    assert loader._get_features_columns() == ["v", "i", "t"]


def test_phmd_battery_get_features_columns_nb1():
    """_get_features_columns returns fixed list when data_name == NB1."""
    loader = PHMDBatteryLoader(
        fold=0, data_name="NB1", task_mode="rul", cache_dir="/tmp"
    )
    loader.meta_data = {"features": ["other"]}
    assert loader._get_features_columns() == [
        "mode",
        "voltage",
        "current",
        "temperature",
    ]


def test_phmd_battery_get_target_column():
    """_get_target_column returns task_mode."""
    loader = PHMDBatteryLoader(**_phmd_kwargs())
    assert loader._get_target_column() == "rul"


def test_phmd_battery_get_unit_column_single():
    """_get_unit_column returns identifier[0], or 'unit' if identifier was 'test_name'."""
    loader = PHMDBatteryLoader(**_phmd_kwargs())
    loader.meta_data = {"identifier": ["unit_id"]}
    assert loader._get_unit_column() == "unit_id"


def test_phmd_battery_get_unit_column_test_name():
    """_get_unit_column maps test_name to unit."""
    loader = PHMDBatteryLoader(**_phmd_kwargs())
    loader.meta_data = {"identifier": ["test_name"]}
    assert loader._get_unit_column() == "unit"


def test_phmd_battery_get_unit_column_asserts_single():
    """_get_unit_column asserts len(identifier) == 1."""
    loader = PHMDBatteryLoader(**_phmd_kwargs())
    loader.meta_data = {"identifier": ["a", "b"]}
    with pytest.raises(AssertionError, match="only one"):
        loader._get_unit_column()


def test_phmd_battery_preprocess_fold_test_name_rename():
    """_preprocess_fold renames test_name to unit when identifier is test_name."""
    loader = PHMDBatteryLoader(**_phmd_kwargs())
    loader.meta_data = {"identifier": ["test_name"]}
    main_fold = {
        "train": pd.DataFrame({"test_name": [1, 2], "x": [0.1, 0.2]}),
        "val": pd.DataFrame({"test_name": [3], "x": [0.3]}),
        "test": pd.DataFrame({"test_name": [4], "x": [0.4]}),
    }
    out = loader._preprocess_fold(main_fold, {})
    assert "unit" in out["train"].columns
    assert "test_name" not in out["train"].columns


def test_phmd_battery_process_unit_basic():
    """_process_unit returns dict with features, target, metadata (no NaN/inf)."""
    loader = PHMDBatteryLoader(**_phmd_kwargs())
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "target": [0.5, 0.6]})
    out = loader._process_unit(df, "u1", ["f1", "f2"], "target")
    assert "features" in out
    assert out["features"].shape[0] == 2
    assert "target" in out
    assert "metadata" in out
    assert out["metadata"]["unit_id"] == "u1"


def test_phmd_battery_process_unit_with_nan():
    """_process_unit drops NaN rows and logs."""
    loader = PHMDBatteryLoader(**_phmd_kwargs())
    df = pd.DataFrame({"f1": [1.0, np.nan], "f2": [3.0, 4.0], "target": [0.5, 0.6]})
    out = loader._process_unit(df, "u1", ["f1", "f2"], "target")
    assert len(out["features"]) == 1


def test_phmd_battery_process_unit_with_inf():
    """_process_unit replaces inf in target and drops corresponding rows."""
    loader = PHMDBatteryLoader(**_phmd_kwargs())
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "target": [0.5, np.inf]})
    out = loader._process_unit(df, "u1", ["f1", "f2"], "target")
    assert len(out["features"]) == 1
