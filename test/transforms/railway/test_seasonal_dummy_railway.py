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

    def test_repr_contains_class_name_and_params(self):
        """__repr__ includes class name, shift, and seq_len."""
        t = SeasonalDummyTransform(shift=3, seq_len=7)
        r = repr(t)
        assert "SeasonalDummyTransform" in r
        assert "3" in r
        assert "7" in r

    def test_list_target_is_converted(self):
        """Non-ndarray target (list) is converted to numpy internally."""
        t = SeasonalDummyTransform(shift=1, seq_len=3)
        n = 6
        data = {
            "features": np.ones((n, 2)),
            "target": list(range(n)),  # plain Python list, not ndarray
        }
        result = t.transform_data(data, metadata={"mode": "train"})
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == n

    def test_1d_target_is_reshaped(self):
        """1-D target array is silently promoted to 2-D."""
        t = SeasonalDummyTransform(shift=1, seq_len=3)
        n = 6
        data = {
            "features": np.ones((n, 2)),
            "target": np.ones(n),  # 1-D
        }
        result = t.transform_data(data, metadata={"mode": "train"})
        assert result.shape[0] == n

    def test_list_features_is_converted(self):
        """Non-ndarray features are converted to numpy internally."""
        t = SeasonalDummyTransform(shift=1, seq_len=3)
        n = 6
        data = {
            "features": [[float(i)] for i in range(n)],  # list of lists
            "target": np.ones((n, 1)),
        }
        result = t.transform_data(data, metadata={"mode": "train"})
        assert result.shape[0] == n

    def test_1d_features_is_reshaped(self):
        """1-D features array is silently promoted to 2-D."""
        t = SeasonalDummyTransform(shift=1, seq_len=3)
        n = 6
        data = {
            "features": np.ones(n),  # 1-D
            "target": np.ones((n, 1)),
        }
        result = t.transform_data(data, metadata={"mode": "train"})
        assert result.shape[0] == n

    def test_polynomial_trend_computed_with_sufficient_history(self):
        """Trend fitting is triggered when enough observations accumulate (n>=16)."""
        t = SeasonalDummyTransform(shift=2, seq_len=4)
        n = 16  # enough for seasonal pos to see >=3 observations → triggers polyfit
        data = {
            "features": np.ones((n, 1)),
            "target": np.arange(n, dtype=float).reshape(-1, 1),
        }
        result = t.transform_data(data, metadata={"mode": "train"})
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == n

    def test_call_delegates_to_transform_data(self):
        """__call__ is a pass-through to transform_data."""
        t = SeasonalDummyTransform(shift=1, seq_len=4)
        n = 8
        data = {
            "features": np.ones((n, 2)),
            "target": np.ones((n, 1)),
        }
        result_direct = t.transform_data(data, metadata={"mode": "train"})
        t.reset_history()
        result_call = t(data, metadata={"mode": "train"})
        np.testing.assert_array_equal(result_direct, result_call)
