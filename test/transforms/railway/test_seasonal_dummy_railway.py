"""Tests for picid.transforms.railway.seasonal_dummy_railway."""

import pytest
import numpy as np

from picid.transforms.railway.seasonal_dummy_railway import SeasonalDummyTransform


class TestSeasonalDummyTransform:
    def test_apply_basic(self):
        """Apply transform with minimal valid data."""
        # Use enough samples so each seasonal pos has multiple values (avoids scalar .extend)
        t = SeasonalDummyTransform(shift=2, seq_len=4)
        n = 10
        features = np.arange(n, dtype=float).reshape(-1, 1)
        target = np.arange(n, dtype=float).reshape(-1, 1)
        data = {"features": features, "target": target}
        metadata = {"mode": "train"}
        result = t.transform_data(data, metadata)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == n
        assert result.shape[1] >= 4

    def test_apply_missing_target_raises(self):
        t = SeasonalDummyTransform(shift=1, seq_len=4)
        data = {"features": np.array([[1.0], [2.0]])}
        with pytest.raises(ValueError, match="target"):
            t.transform_data(data, metadata={"mode": "train"})

    def test_apply_missing_features_raises(self):
        t = SeasonalDummyTransform(shift=1, seq_len=4)
        data = {"target": np.array([[1.0], [2.0]])}
        with pytest.raises(ValueError, match="features"):
            t.transform_data(data, metadata={"mode": "train"})

    def test_apply_target_feature_length_mismatch_raises(self):
        t = SeasonalDummyTransform(shift=1, seq_len=4)
        data = {
            "features": np.array([[1.0], [2.0], [3.0]]),
            "target": np.array([[1.0], [2.0]]),
        }
        with pytest.raises(ValueError, match="doesn't match"):
            t.transform_data(data, metadata={"mode": "train"})

    def test_reset_history(self):
        t = SeasonalDummyTransform(shift=2, seq_len=4)
        data = {
            "features": np.arange(8).reshape(-1, 1).astype(float),
            "target": np.arange(8).reshape(-1, 1).astype(float),
        }
        t.transform_data(data, metadata={"mode": "train"})
        t.reset_history()
        summary = t.get_seasonal_history_summary()
        for pos in range(4):
            assert summary[pos]["count"] == 0

    def test_get_seasonal_history_summary(self):
        t = SeasonalDummyTransform(shift=2, seq_len=4)
        data = {
            "features": np.arange(10).reshape(-1, 1).astype(float),
            "target": np.arange(10).reshape(-1, 1).astype(float),
        }
        t.transform_data(data, metadata={"mode": "train"})
        summary = t.get_seasonal_history_summary()
        assert isinstance(summary, dict)
        assert len(summary) == 4
