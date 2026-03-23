"""
Tests for phmd_* loader methods (_get_*_column, _preprocess_fold, _process_unit) with mocked data.
No real data loading; all inputs are in-memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picid.data.datasources.phmd_cbmv3 import CBMv3Loader
from picid.data.datasources.phmd_hsf15 import HSF15Loader
from picid.data.datasources.phmd_xjtu_sy import XJTU_SYLoader, merge_data_folds_on_key


def _phmd_kwargs():
    return {
        "fold": 0,
        "data_name": "NB14",
        "task_mode": "rul",
        "cache_dir": "/tmp/phmd",
    }


# ----- CBMv3Loader -----


def test_phmd_cbmv3_process_unit():
    """_process_unit returns features as values, unit_id from unit_name (array-like)."""
    loader = CBMv3Loader(**_phmd_kwargs())
    loader.meta_data = {}
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "rul": [10.0, 9.0]})
    out = loader._process_unit(df, np.array(1), ["a", "b"], "rul")
    assert out["features"].shape == (2, 2)
    assert out["rul"].shape == (2,)
    np.testing.assert_array_equal(np.asarray(out["unit_id"]).flatten(), [1])


# ----- HSF15Loader -----


def test_phmd_hsf15_get_unit_column():
    """_get_unit_column returns 'unit_name'."""
    loader = HSF15Loader(**_phmd_kwargs())
    assert loader._get_unit_column() == "unit_name"


def test_phmd_hsf15_preprocess_fold():
    """_preprocess_fold adds unit_name to each split."""
    loader = HSF15Loader(**_phmd_kwargs())
    main_fold = {
        "train": pd.DataFrame({"x": [1]}),
        "val": pd.DataFrame({"x": [2]}),
        "test": pd.DataFrame({"x": [3]}),
    }
    out = loader._preprocess_fold(main_fold, {})
    assert out["train"]["unit_name"].iloc[0] == "HSF15_Unit_1"


def test_phmd_hsf15_process_unit_not_monotonic_raises():
    """_process_unit raises when cycle column is not monotonic."""
    loader = HSF15Loader(**_phmd_kwargs())
    df = pd.DataFrame(
        {
            "cycle": [2, 1, 3],
            "f1": [1.0, 2.0, 3.0],
            "target": [0.1, 0.2, 0.3],
        }
    )
    with pytest.raises(ValueError, match="not monotonically increasing"):
        loader._process_unit(df, "u1", ["f1"], "target")


def test_phmd_hsf15_process_unit_monotonic_one_cycle():
    """_process_unit with one cycle of 6000 steps (minimal for reshape to (-1,6000,1))."""
    loader = HSF15Loader(**_phmd_kwargs())
    n = 6000
    df = pd.DataFrame(
        {
            "cycle": np.ones(n, dtype=int),
            "f1": np.arange(n, dtype=float),
            "target": np.zeros(n),
        }
    )
    out = loader._process_unit(df, "u1", ["f1"], "target")
    assert "features" in out
    assert out["unit_id"] == "no_id_available"


# ----- XJTU_SYLoader -----


def test_phmd_xjtu_get_unit_column():
    """_get_unit_column returns 'bearing'."""
    loader = XJTU_SYLoader(**_phmd_kwargs())
    assert loader._get_unit_column() == "bearing"


def test_phmd_xjtu_preprocess_fold_rul_returns_main():
    """_preprocess_fold returns main_fold when task_mode is not fault."""
    loader = XJTU_SYLoader(**_phmd_kwargs())
    main_fold = {
        "train": pd.DataFrame({"x": [1]}),
        "val": pd.DataFrame(),
        "test": pd.DataFrame(),
    }
    out = loader._preprocess_fold(main_fold, {})
    assert out is main_fold


def test_phmd_xjtu_process_unit_rul():
    """_process_unit in rul mode returns features and target."""
    loader = XJTU_SYLoader(**_phmd_kwargs())
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "rul": [10.0, 9.0]})
    out = loader._process_unit(df, "1_1", ["f1", "f2"], "rul")
    assert out["features"].shape == (2, 2)
    assert out["target"].shape == (2, 1)
    assert out["metadata"]["unit_id"] == (1, 1)


def test_phmd_xjtu_process_unit_fault():
    """_process_unit in fault mode uses FAULT_TARGET_COLS."""
    loader = XJTU_SYLoader(
        fold=0, data_name="XJTU", task_mode="fault", cache_dir="/tmp"
    )
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "Outer race": [0, 1],
            "Cage": [1, 0],
            "Inner race": [0, 0],
        }
    )
    out = loader._process_unit(df, "1_1", ["f1"], "rul")
    assert "target" in out
    assert out["target"].shape[0] == 2


def test_phmd_xjtu_process_unit_unknown_task_raises():
    """_process_unit raises for unknown task_mode."""
    loader = XJTU_SYLoader(
        fold=0, data_name="XJTU", task_mode="unknown", cache_dir="/tmp"
    )
    df = pd.DataFrame({"f1": [1.0], "rul": [10.0]})
    with pytest.raises(ValueError, match="Unknown task_mode"):
        loader._process_unit(df, "1_1", ["f1"], "rul")


# ----- merge_data_folds_on_key (used by XJTU _preprocess_fold for fault) -----


def test_merge_data_folds_on_key():
    """merge_data_folds_on_key merges main and aux folds on key (default bearing)."""
    main_fold = {
        "train": pd.DataFrame({"bearing": [1, 2], "rul": [10, 20]}),
        "val": pd.DataFrame(),
        "test": pd.DataFrame(),
    }
    aux_fold = {
        "train": pd.DataFrame({"bearing": [1, 2], "fault": [0, 1]}),
        "val": pd.DataFrame(),
        "test": pd.DataFrame(),
    }
    out = merge_data_folds_on_key(main_fold, aux_fold, key="bearing")
    assert "train" in out
    assert "rul" in out["train"].columns and "fault" in out["train"].columns


def test_merge_data_folds_on_key_split_missing_in_aux():
    """When a split is missing in aux_fold, main_df is copied."""
    main_fold = {
        "train": pd.DataFrame({"bearing": [1], "rul": [10]}),
        "val": pd.DataFrame({"bearing": [2], "rul": [20]}),
    }
    aux_fold = {"train": pd.DataFrame({"bearing": [1], "fault": [0]})}  # no "val"
    out = merge_data_folds_on_key(main_fold, aux_fold, key="bearing")
    assert "val" in out
    assert list(out["val"].columns) == ["bearing", "rul"]


def test_merge_data_folds_on_key_no_new_columns():
    """When aux has no new columns, merge still returns copy per split."""
    main_fold = {"train": pd.DataFrame({"bearing": [1, 2], "rul": [10, 20]})}
    aux_fold = {"train": pd.DataFrame({"bearing": [1, 2], "rul": [0, 0]})}  # same cols
    out = merge_data_folds_on_key(main_fold, aux_fold, key="bearing")
    assert "train" in out
    assert list(out["train"].columns) == ["bearing", "rul"]


def test_phmd_xjtu_process_unit_fault_ragged_raises():
    """_process_unit in fault mode with use_ragged=True raises NotImplementedError."""
    loader = XJTU_SYLoader(
        fold=0,
        data_name="XJTU",
        task_mode="fault",
        cache_dir="/tmp",
        use_ragged=True,
    )
    n = 32768 * 2  # divisible by SAMPLE_RANGE_SIZE
    df = pd.DataFrame(
        {
            "f1": np.zeros(n),
            "Outer race": np.zeros(n),
            "Cage": np.zeros(n),
            "Inner race": np.zeros(n),
        }
    )
    with pytest.raises(
        NotImplementedError,
        match="Ragged mode is not yet implemented for the 'fault' task",
    ):
        loader._process_unit(df, "1_1", ["f1"], "rul")
