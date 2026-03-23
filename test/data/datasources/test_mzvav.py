"""
Tests for MZVAVLoader with mocked I/O (read_data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from picid.data.datasources.base.exceptions import DatasourceConfigurationError
from picid.data.datasources.MZVAV import MZVAVLoader


@pytest.fixture
def mzvav_kwargs():
    return {
        "data_name": "MZVAV",
        "task_mode": "regression",
        "data_path": "/fake/mzvav.csv",
        "timestamp_name": "date",
        "target_name": "fault",
        "random_state": 42,
        "group_by_days": False,
    }


@pytest.fixture
def mock_mzvav_read_data():
    """Minimal (df, df_target, df_stamp, days_to_fault) for _load_data to split. Need >=2 days per fault for stratify."""
    dates = pd.date_range("2007-08-28", periods=12, freq="D")
    idx = pd.DatetimeIndex(np.repeat(dates.values, 10))
    n = len(idx)
    df = pd.DataFrame({"feat": np.random.randn(n).astype(np.float32)}, index=idx)
    df.index.name = "date"
    fault_per_day = [1, 1, 2, 2, 3, 3, 1, 2, 3, 1, 2, 3]
    fault_vals = np.repeat(fault_per_day, 10)
    df_target = pd.DataFrame({"fault": fault_vals}, index=idx)
    df_stamp = df.index.to_series()
    days_to_fault = pd.Series(fault_per_day, index=dates)
    return df, df_target, df_stamp, days_to_fault


def test_mzvav_load_data_returns_split_structure(mzvav_kwargs, mock_mzvav_read_data):
    """load_data (with read_data mocked) populates data_dict with train/val/test splits."""
    with patch.object(MZVAVLoader, "read_data", return_value=mock_mzvav_read_data):
        loader = MZVAVLoader(**mzvav_kwargs)
        loader.load_data()
    assert loader._is_loaded
    assert getattr(loader, "_is_splitted", loader._is_loaded)  # MZVAV uses _is_splitted
    data = loader.get_data()
    assert "features" in data
    assert "train" in data["features"]
    assert "val" in data["features"]
    assert "test" in data["features"]


def test_mzvav_get_data_name(mzvav_kwargs):
    """get_data_name returns data_name."""
    loader = MZVAVLoader(**mzvav_kwargs)
    assert loader.get_data_name() == "MZVAV"


def test_mzvav_get_meta_data_after_load(mzvav_kwargs, mock_mzvav_read_data):
    """get_meta_data returns dict after load_data (coverage)."""
    with patch.object(MZVAVLoader, "read_data", return_value=mock_mzvav_read_data):
        loader = MZVAVLoader(**mzvav_kwargs)
        loader.load_data()
    meta = loader.get_meta_data()
    assert isinstance(meta, dict)


def test_mzvav_split_data_no_op(mzvav_kwargs):
    """split_data is a no-op (logs that MZVAV does not support splitting) — coverage."""
    loader = MZVAVLoader(**mzvav_kwargs)
    loader.split_data()  # no raise


def test_mzvav_rejects_multisource_splitter_for_split_mode(mzvav_kwargs):
    """Predefined-split loaders should reject multisource splitters explicitly."""
    with pytest.raises(
        DatasourceConfigurationError,
        match="does not accept multisource_data_splitter",
    ):
        MZVAVLoader(**mzvav_kwargs, multisource_data_splitter=object())


def test_mzvav_read_data_with_real_csv(tmp_path, mzvav_kwargs):
    """read_data runs with a minimal CSV that has required dates and no NaN after ffill (coverage)."""
    dates = [
        "2/12/2008 00:00",
        "2/12/2008 01:00",
        "5/7/2008 00:00",
        "5/7/2008 01:00",
        "8/28/2007 00:00",
        "8/29/2007 00:00",
        "8/30/2007 00:00",
        "5/6/2008 00:00",
        "8/31/2007 00:00",
        "5/15/2008 00:00",
        "9/1/2007 00:00",
        "9/2/2007 00:00",
        "9/5/2007 00:00",
        "9/6/2007 00:00",
        "5/8/2008 00:00",
    ]
    csv_path = tmp_path / "mzvav.csv"
    df = pd.DataFrame(
        {
            "date": dates,
            "feat": np.ones(len(dates)),
            "fault": np.zeros(len(dates)),
        }
    )
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M")
    df = df.set_index("date")
    df.to_csv(csv_path, date_format="%m/%d/%Y %H:%M")
    kwargs = {**mzvav_kwargs, "data_path": str(csv_path)}
    loader = MZVAVLoader(**kwargs)
    df_out, df_target, df_stamp, days_to_fault = loader.read_data()
    assert df_out is not None
    assert df_target is not None
    assert len(days_to_fault) > 0
