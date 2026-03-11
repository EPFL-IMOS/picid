"""
Tests for agtf30k: module-level functions and AGTF30KDataSource (mocked I/O).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from picid.data.datasources.agtf30k import (
    subsampling,
    binarize_concept,
    scale_concept,
    AGTF30KDataSource,
)


def test_subsampling():
    """subsampling returns every nth row."""
    df = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [10, 11, 12, 13, 14, 15]})
    out = subsampling(df, 2)
    assert len(out) == 3
    assert out["a"].tolist() == [0, 2, 4]


def test_binarize_concept():
    """binarize_concept thresholds at -0.0015."""
    x = np.array([-0.002, -0.001, 0.0, 0.001])
    out = binarize_concept(x)
    assert out.tolist() == [True, False, False, False]


def test_scale_concept():
    """scale_concept clips x / -0.035 to [0, 1]."""
    x = np.array([-0.07, -0.035, 0.0])
    out = scale_concept(x)
    assert out[0] == 1.0
    assert out[1] == 1.0
    assert out[2] == 0.0


def test_agtf30k_data_source_init():
    """AGTF30KDataSource initializes with base kwargs (data_name, task_mode)."""
    ds = AGTF30KDataSource(
        path="/nonexistent",
        subsampling_rate=1,
        data_name="AGTF30K",
        task_mode="regression",
    )
    assert ds.data_name == "AGTF30K"
    assert ds.task_mode == "regression"


def test_agtf30k_load_data_calls_load_data(mocker):
    """load_data calls _load_data and sets data_dict (mocked _load_data)."""
    mocker.patch.object(
        AGTF30KDataSource,
        "_load_data",
        return_value={
            "features": np.zeros((10, 5)),
            "timestamps": np.arange(10),
            "descriptors": np.zeros((10, 3)),
            "fault-type": np.zeros((10, 1)),
        },
    )
    ds = AGTF30KDataSource(
        path="/fake",
        subsampling_rate=1,
        data_name="AGTF30K",
        task_mode="regression",
    )
    ds.load_data()
    assert ds._is_loaded
    assert ds.data_dict is not None
    assert "features" in ds.data_dict


def test_agtf30k_load_data_full_path_mocked_io(mocker):
    """_load_data path with mocked os.path.exists and pd.read_csv (no real files)."""
    required_cols = [
        "signaldate",
        "alt",
        "MN",
        "PLA",
        "Wf",
        "Pa",
        "S2_Pt",
        "S25_Pt",
        "S36_Pt",
        "S45_Tt",
        "S5_Pt",
        "VAFN",
        "N_LPC",
        "N_HPC",
        "V",
    ]
    n = 6
    fake_df = pd.DataFrame({c: np.arange(n, dtype=float) for c in required_cols})
    mocker.patch("picid.data.datasources.agtf30k.os.path.exists", return_value=True)
    mocker.patch("picid.data.datasources.agtf30k.pd.read_csv", return_value=fake_df)
    ds = AGTF30KDataSource(
        path="/fake",
        subsampling_rate=1,
        data_name="AGTF30K",
        task_mode="regression",
    )
    # AGTF30KDataSource does not set these from kwargs; set for _load_data path
    ds.path = "/fake"
    ds.mode = "train"
    ds.window_size = 10
    ds.stride = 1
    ds.subsampling_rate = 1
    ds.concepts = pd.DataFrame({"c1": [0, 1, 0, 1, 0, 1]})
    out = ds._load_data()
    assert "features" in out
    # _load_data concats train + test + val (each mocked as n rows) -> 3*n rows
    assert out["features"].shape[0] == 3 * n
    assert "timestamps" in out
    assert "descriptors" in out and out["descriptors"].shape[1] == 3
    assert "fault-type" in out
