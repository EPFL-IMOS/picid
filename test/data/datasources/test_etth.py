"""
Tests for ETThLoader with mocked I/O (pd.read_csv).
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from picid.data.datasources.ETTh import ETThLoader


@pytest.fixture
def etth_kwargs():
    return {
        "data_name": "ETTh",
        "data_path": "/fake/etth.csv",
        "timestamp_name": "date",
        "task_mode": "forecasting",
        "target_name": "OT",
    }


def test_etth_load_data_returns_features_and_timestamps(etth_kwargs, mock_csv_df_etth):
    """load_data populates data_dict with 'features' and 'timestamps' (read_data mocked)."""
    with patch.object(
        ETThLoader,
        "read_data",
        return_value=(
            mock_csv_df_etth.drop(columns=["date"]),
            mock_csv_df_etth["date"].copy(),
        ),
    ):
        loader = ETThLoader(**etth_kwargs)
        loader.load_data()
    assert "features" in loader.data_dict
    assert "timestamps" in loader.data_dict
    assert loader.data_dict["features"].shape[1] == 3
    assert len(loader.data_dict["timestamps"]) == len(mock_csv_df_etth)


def test_etth_read_data_univariate_selects_target_column(etth_kwargs, mock_csv_df_etth):
    """read_data in univariate mode keeps only target_name column."""
    with patch("pandas.read_csv", return_value=mock_csv_df_etth.copy()):
        loader = ETThLoader(**etth_kwargs)
        loader.task_mode = "univariate"
        loader.target_name = "OT"
        df, df_stamp = loader.read_data()
    assert list(df.columns) == ["OT"]
    assert df_stamp.name == "date"


def test_etth_read_data_multivariate_keeps_all_non_timestamp(
    etth_kwargs, mock_csv_df_etth
):
    """read_data in non-univariate mode drops only timestamp column."""
    with patch("pandas.read_csv", return_value=mock_csv_df_etth.copy()):
        loader = ETThLoader(**etth_kwargs)
        df, df_stamp = loader.read_data()
    assert "date" not in df.columns
    assert set(df.columns) == {"HUFL", "HULL", "OT"}
    assert len(df_stamp) == len(mock_csv_df_etth)


def test_etth_get_meta_data_includes_target_name(etth_kwargs, mock_csv_df_etth):
    """get_meta_data includes target_name."""
    with patch.object(
        ETThLoader,
        "read_data",
        return_value=(
            mock_csv_df_etth.drop(columns=["date"]),
            mock_csv_df_etth["date"].copy(),
        ),
    ):
        loader = ETThLoader(**etth_kwargs)
        loader.load_data()
        meta = loader.get_meta_data()
    assert meta["target_name"] == "OT"


def test_etth_get_data_name(etth_kwargs):
    """get_data_name returns data_name."""
    loader = ETThLoader(**etth_kwargs)
    assert loader.get_data_name() == "ETTh"


def test_etth_init_default_transforms(etth_kwargs):
    """_init_default_transforms returns loader_default_transforms (coverage)."""
    loader = ETThLoader(**etth_kwargs)
    result = loader._init_default_transforms()
    assert result == getattr(loader, "loader_default_transforms", {})
