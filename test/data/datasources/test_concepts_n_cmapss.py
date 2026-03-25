"""
Tests for concepts_n_cmapss: module-level functions and N_CMAPSSDataSource (mocked I/O).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from picid.data.datasources.concepts_n_cmapss import (
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
    assert binarize_concept(-0.002) is True
    assert binarize_concept(0.0) is False


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
    df = pd.DataFrame(
        {
            "cycle": [1, 2, 3, 4],
            "hs": [1, 1, 0, 0],
            "RUL": [100, 100, 50, 25],
        }
    )
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


def test_concepts_n_cmapss_base_config_is_neutral_for_subset_selection():
    """The base datasource config should leave fc/units unset for derived configs."""
    config = OmegaConf.load(
        Path(__file__).resolve().parents[3]
        / "configs"
        / "datasource"
        / "concepts_n_cmapss.yaml"
    )

    assert config.load_arguments.fc is None
    assert config.load_arguments.units is None


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


def _write_minimal_n_cmapss_h5(tmp_path: Path) -> None:
    """Write a minimal fixture-backed N-CMAPSS HDF5 file."""
    import h5py

    n_rows = 50
    A_var = ["unit", "cycle", "Fc", "hs"]
    A = np.column_stack(
        [
            np.ones(n_rows, dtype=int),  # unit
            np.arange(1, n_rows + 1, dtype=int),  # cycle
            np.ones(n_rows, dtype=int) * 2,  # Fc
            np.zeros(n_rows, dtype=int),  # hs (healthy)
        ]
    )
    W_var = ["alt", "Mach", "TRA", "T2"]
    W = np.ones((n_rows, 4)) * 0.5
    T_var = [
        "fan_eff_mod",
        "fan_flow_mod",
        "LPC_eff_mod",
        "LPC_flow_mod",
        "HPC_eff_mod",
        "HPC_flow_mod",
        "LPT_eff_mod",
        "LPT_flow_mod",
        "HPT_eff_mod",
        "HPT_flow_mod",
    ]
    T = np.ones((n_rows, 10)) * -0.02  # Slightly degraded
    X_s_var = [f"s{i}" for i in range(5)]
    X_s = np.random.randn(n_rows, 5).astype(np.float32)
    Y = np.arange(n_rows, 0, -1, dtype=np.float32).reshape(-1, 1)

    h5_path = tmp_path / "N-CMAPSS_DS1.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("W_dev", data=W)
        f.create_dataset("X_s_dev", data=X_s)
        f.create_dataset("X_v_dev", data=np.zeros((n_rows, 2)))
        f.create_dataset("T_dev", data=T)
        f.create_dataset("Y_dev", data=Y)
        f.create_dataset("A_dev", data=A)
        f.create_dataset("W_var", data=np.array(W_var, dtype="S20"))
        f.create_dataset("X_s_var", data=np.array(X_s_var, dtype="S20"))
        f.create_dataset("X_v_var", data=np.array(["v1", "v2"], dtype="S20"))
        f.create_dataset("T_var", data=np.array(T_var, dtype="S20"))
        f.create_dataset("A_var", data=np.array(A_var, dtype="S20"))

class _FixtureLoadArgs:
    """Load arguments for the minimal HDF5 fixture scenario."""

    mode = "train"
    n_DS = 1
    concepts = ["LPT", "HPT"]

    @staticmethod
    def get(k, d=None):
        """Return the configured selector for the fixture-backed test."""
        if k == "units":
            return None
        if k == "fc":
            return 2
        return d


def test_n_cmapss_load_with_minimal_h5_fixture(tmp_path):
    """
    Exercise full _load_data path with a minimal valid HDF5 fixture.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used to build the minimal HDF5 fixture.
    """
    _write_minimal_n_cmapss_h5(tmp_path)
    expected_rows = 50

    ds = N_CMAPSSDataSource(
        path=str(tmp_path),
        load_arguments=_FixtureLoadArgs(),
        data_name="N_CMAPSS",
        task_mode="rul",
        group_by_unit=False,
    )
    result = ds._load_data()
    assert "features" in result
    assert "rul" in result
    assert "concepts" in result
    assert result["features"].shape[0] == expected_rows
    assert result["concepts"].shape[0] == expected_rows


def test_n_cmapss_repeated_loads_on_same_instance_are_stable(tmp_path):
    """Repeated load_data() calls on the same datasource instance stay stable."""
    _write_minimal_n_cmapss_h5(tmp_path)
    ds = N_CMAPSSDataSource(
        path=str(tmp_path),
        load_arguments=_FixtureLoadArgs(),
        data_name="N_CMAPSS",
        task_mode="rul",
        group_by_unit=False,
    )

    ds.load_data()
    first = ds.data_dict["concepts"].copy()

    ds.load_data()
    second = ds.data_dict["concepts"]

    np.testing.assert_array_equal(first, second)
