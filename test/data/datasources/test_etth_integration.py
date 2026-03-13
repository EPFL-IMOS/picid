"""Integration test for ETThLoader with real CSV (no mocks)."""

from __future__ import annotations

import pytest
from pathlib import Path

from picid.data.datasources.ETTh import ETThLoader


@pytest.fixture
def etth_csv_path():
    """Path to minimal ETTh CSV fixture."""
    return Path(__file__).resolve().parent.parent.parent / "fixtures" / "etth_mini.csv"


def test_etth_load_data_from_real_csv(etth_csv_path):
    """ETThLoader reads real CSV and populates data_dict correctly."""
    loader = ETThLoader(
        data_path=str(etth_csv_path),
        data_name="ETTh",
        timestamp_name="date",
        task_mode="forecasting",
        target_name="OT",
    )
    loader.load_data()
    assert "features" in loader.data_dict
    assert "timestamps" in loader.data_dict
    assert loader.data_dict["features"].shape[1] == 3
    assert len(loader.data_dict["timestamps"]) == 5
