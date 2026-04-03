"""
Shared fixtures and mocks for datasource tests (I/O mocked).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from test.fixtures.datasource_layouts import (
    build_mock_umar_multivariate_pickles_dataframes,
)


@pytest.fixture
def mock_umar_data():
    """Pickle-shaped UMAR inputs/targets DataFrames for loader unit tests."""
    return build_mock_umar_multivariate_pickles_dataframes()


@pytest.fixture
def mock_csv_df_etth():
    """Minimal DataFrame as returned by pd.read_csv for ETTh: timestamp + feature columns."""
    n = 50
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="h"),
            "HUFL": np.random.randn(n).astype(np.float32),
            "HULL": np.random.randn(n).astype(np.float32),
            "OT": np.random.randn(n).astype(np.float32),
        }
    )


@pytest.fixture
def mock_csv_df_battery_railway():
    """DataFrame for Battery/Railway: index, timestamp, target column, and other features."""
    n = 40
    df = pd.DataFrame(
        {
            "timestamp": np.arange(n, dtype=np.float64),
            "target_col": np.random.randn(n).astype(np.float32),
            "feat_a": np.random.randn(n).astype(np.float32),
            "feat_b": np.random.randn(n).astype(np.float32),
        }
    )
    df.index.name = "index"
    return df
