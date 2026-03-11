"""
Tests for BatteryLoader with mocked I/O (pd.read_csv).
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

from picid.data.datasources.battery import BatteryLoader


@pytest.fixture
def battery_kwargs():
    return {
        "data_name": "Battery",
        "task_mode": "regression",
        "data_path": "/fake/battery.csv",
        "timestamp_name": "timestamp",
        "target_name": "target_col",
    }


@pytest.fixture
def mock_battery_df(mock_csv_df_battery_railway):
    return mock_csv_df_battery_railway.copy()


def test_battery_extract_mask(battery_kwargs, mock_battery_df):
    """extract_mask returns boolean array True only for target column."""
    loader = BatteryLoader(**battery_kwargs)
    loader.target_name = "target_col"
    mask = loader.extract_mask(mock_battery_df)
    assert mask.dtype == bool
    assert np.sum(mask) == 1
    assert mask[mock_battery_df.columns.get_loc("target_col")] == True


def test_battery_read_data_returns_three_frames(battery_kwargs, mock_battery_df):
    """read_data returns (df_features, df_target, df_stamp)."""
    with patch("pandas.read_csv", return_value=mock_battery_df.copy()):
        loader = BatteryLoader(**battery_kwargs)
        # BatteryLoader does not set self.data_path; ensure it exists for read_csv
        loader.data_path = battery_kwargs["data_path"]
        loader.timestamp_name = battery_kwargs["timestamp_name"]
        loader.target_name = battery_kwargs["target_name"]
        df, df_target, df_stamp = loader.read_data()
    assert "timestamp" not in df.columns
    assert "target_col" not in df.columns
    assert list(df_target.columns) == ["target_col"]
    assert len(df_stamp) == len(mock_battery_df)


def test_battery_load_data_structure(battery_kwargs, mock_battery_df):
    """_load_data returns features, target, timestamps, mask, initial_dates."""
    with patch("pandas.read_csv", return_value=mock_battery_df.copy()):
        loader = BatteryLoader(**battery_kwargs)
        loader.data_path = battery_kwargs["data_path"]
        loader.timestamp_name = battery_kwargs["timestamp_name"]
        loader.target_name = battery_kwargs["target_name"]
        loader.load_data()
    assert "features" in loader.data_dict
    assert "target" in loader.data_dict
    assert "timestamps" in loader.data_dict
    assert "mask" in loader.data_dict
    assert "initial_dates" in loader.data_dict
    assert loader.data_dict["features"].shape[1] >= 1
    assert loader.data_dict["target"].shape[1] == 1


def test_battery_get_data_name(battery_kwargs):
    """get_data_name returns data_name."""
    loader = BatteryLoader(**battery_kwargs)
    assert loader.get_data_name() == "Battery"
