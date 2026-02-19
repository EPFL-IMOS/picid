"""
Tests for RailwayLoader with mocked I/O (pd.read_csv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from lmetk.data.datasources.railway import RailwayLoader


@pytest.fixture
def railway_kwargs():
    return {
        "data_name": "Railway",
        "task_mode": "regression",
        "data_path": "/fake/railway.csv",
        "timestamp_name": "timestamp",
        "target_name": "target_col",
    }


def test_railway_extract_mask(railway_kwargs, mock_csv_df_battery_railway):
    """extract_mask returns boolean array True only for target column."""
    loader = RailwayLoader(**railway_kwargs)
    mask = loader.extract_mask(mock_csv_df_battery_railway)
    assert mask.dtype == bool
    assert np.sum(mask) == 1
    assert mask[mock_csv_df_battery_railway.columns.get_loc("target_col")] == True


def test_railway_read_data_returns_three_frames(railway_kwargs, mock_csv_df_battery_railway):
    """read_data returns (df, df_target, df_stamp)."""
    with patch("pandas.read_csv", return_value=mock_csv_df_battery_railway.copy()):
        loader = RailwayLoader(**railway_kwargs)
        df, df_target, df_stamp = loader.read_data()
    assert "timestamp" not in df.columns
    assert "target_col" not in df.columns
    assert list(df_target.columns) == ["target_col"]
    assert len(df_stamp) == len(mock_csv_df_battery_railway)


def test_railway_load_data_structure(railway_kwargs, mock_csv_df_battery_railway):
    """_load_data returns features, target, timestamps, initial_dates; get_meta_data includes mask."""
    with patch("pandas.read_csv", return_value=mock_csv_df_battery_railway.copy()):
        loader = RailwayLoader(**railway_kwargs)
        loader.load_data()
    assert "features" in loader.data_dict
    assert "target" in loader.data_dict
    assert "timestamps" in loader.data_dict
    assert "initial_dates" in loader.data_dict
    meta = loader.get_meta_data()
    assert "mask" in meta


def test_railway_get_data_name(railway_kwargs):
    """get_data_name returns data_name."""
    loader = RailwayLoader(**railway_kwargs)
    assert loader.get_data_name() == "Railway"
