"""
Shared fixtures and mocks for datasource tests (I/O mocked).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Mock phmd so modules that import phmd.download (nb14, unibo) or phmd.datasets (phmd_loader)
# can be collected and run without the real phmd package.
if "phmd" not in sys.modules:
    _phmd = MagicMock()
    _phmd.download = MagicMock()
    _phmd.download.download = MagicMock(return_value=None)
    _phmd.datasets = MagicMock()
    sys.modules["phmd"] = _phmd
    sys.modules["phmd.download"] = _phmd.download
    sys.modules["phmd.datasets"] = _phmd.datasets


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
