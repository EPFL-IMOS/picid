"""
Tests for concepts_n_cmapss: module-level functions and N_CMAPSSDataSource (mocked I/O).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from lmetk.data.datasources.concepts_n_cmapss import (
    subsampling,
    binarize_concept,
    scale_concept,
    flatten_RUL,
    N_CMAPSSDataSource,
)


def test_subsampling():
    """subsampling returns every nth row."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    out = subsampling(df, 2)
    assert len(out) == 3
    assert out["a"].tolist() == [1, 3, 5]


def test_binarize_concept():
    """binarize_concept thresholds at -0.0015."""
    assert binarize_concept(-0.002) == True
    assert binarize_concept(0.0) == False


def test_scale_concept():
    """scale_concept clips to [0, 1]."""
    x = np.array([-0.035, 0.0])
    out = scale_concept(x)
    assert out[0] == 1.0 and out[1] == 0.0


def test_flatten_RUL_no_hs_change():
    """flatten_RUL returns df unchanged when no hs diff == -1."""
    df = pd.DataFrame({"cycle": [1, 2, 3], "hs": [1, 1, 1], "RUL": [10, 9, 8]})
    out = flatten_RUL(df)
    pd.testing.assert_frame_equal(out, df)


def test_flatten_RUL_with_hs_change():
    """flatten_RUL updates RUL for cycles before first hs change."""
    df = pd.DataFrame({
        "cycle": [1, 2, 3, 4],
        "hs": [1, 1, 0, 0],
        "RUL": [100, 100, 50, 25],
    })
    out = flatten_RUL(df)
    # First row where hs.diff() == -1 is cycle 3 (index 2); RUL at that row is 50.
    # Cycles 1,2 should get RUL 50.
    assert out["RUL"].iloc[0] == 50
    assert out["RUL"].iloc[1] == 50


def test_n_cmapss_init_requires_linear_or_flat():
    """N_CMAPSSDataSource raises if RUL not 'linear' or 'flat'."""
    with pytest.raises(AssertionError, match="RUL type must"):
        N_CMAPSSDataSource(
            path="/tmp",
            load_arguments=_make_load_args(),
            data_name="N_CMAPSS",
            task_mode="rul",
            RUL="invalid",
        )


def test_n_cmapss_init_requires_fc_or_units():
    """N_CMAPSSDataSource raises if both or neither of fc/units specified."""
    class LoadArgsBothNone:
        mode = "train"
        n_DS = 1
        concepts = ["LPT"]
        @staticmethod
        def get(k, d=None):
            return None
    with pytest.raises(ValueError, match="Specify exactly one"):
        N_CMAPSSDataSource(
            path="/tmp",
            load_arguments=LoadArgsBothNone(),
            data_name="N_CMAPSS",
            task_mode="rul",
        )


def _make_load_args():
    """LoadArgs-like object: mode, n_DS, concepts, get(units)=None, get(fc)=1."""
    class LoadArgs:
        mode = "train"
        n_DS = 1
        concepts = ["LPT"]
        @staticmethod
        def get(k, d=None):
            if k == "units":
                return None
            if k == "fc":
                return 1
            return d
    return LoadArgs()


def test_n_cmapss_init_success_with_fc():
    """N_CMAPSSDataSource initializes with load_arguments.fc set."""
    ds = N_CMAPSSDataSource(
        path="/tmp",
        load_arguments=_make_load_args(),
        data_name="N_CMAPSS",
        task_mode="rul",
    )
    assert ds.n_DS == 1
    assert ds.selected_fc == 1
    assert ds.selected_units is None


def test_n_cmapss_load_data_mocked(mocker):
    """N_CMAPSSDataSource.load_data with mocked _load_data."""
    mocker.patch.object(
        N_CMAPSSDataSource,
        "_load_data",
        return_value={"features": np.zeros((20, 5)), "target": np.zeros((20, 1))},
    )
    ds = N_CMAPSSDataSource(
        path="/tmp",
        load_arguments=_make_load_args(),
        data_name="N_CMAPSS",
        task_mode="rul",
    )
    ds.load_data()
    assert ds._is_loaded
    assert "features" in ds.data_dict


def test_n_cmapss_load_data_raises_when_file_missing():
    """_load_data raises FileNotFoundError when h5 file does not exist (no mock)."""
    ds = N_CMAPSSDataSource(
        path="/nonexistent",
        load_arguments=_make_load_args(),
        data_name="N_CMAPSS",
        task_mode="rul",
    )
    with pytest.raises(FileNotFoundError, match="File not found"):
        ds._load_data()
