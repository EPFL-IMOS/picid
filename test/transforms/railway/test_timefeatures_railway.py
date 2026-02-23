"""Tests for picid.transforms.railway.timefeatures_railway."""

import pytest
import numpy as np
import pandas as pd

from picid.transforms.railway.timefeatures_railway import (
    RailwayTimeFeaturesTransform,
    time_features,
    reconstruct_datetime_from_features,
)


class TestRailwayTimeFeaturesTransform:

    def test_transform_data_basic(self):
        """Extract time features from timestamps."""
        t = RailwayTimeFeaturesTransform(
            use_features=["month", "day", "hour", "minute"],
            normalize_time_features=True,
        )
        # Mix of dates so no column is all-zero
        timestamps = pd.Series(pd.to_datetime(["2023-01-01 10:00", "2023-01-15 14:30", "2023-02-01 09:00", "2023-02-20 16:45"]))
        data = {"timestamps": timestamps}
        metadata = {}
        result = t.transform_data(data, metadata)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 4
        assert result.shape[1] == 4  # month, day, hour, minute

    def test_transform_data_missing_timestamps_raises(self):
        t = RailwayTimeFeaturesTransform(
            use_features=["year", "month"],
            normalize_time_features=True,
        )
        data = {"features": np.array([[1, 2]])}
        with pytest.raises(AssertionError, match="timestamps"):
            t.transform_data(data, {})

    def test_call_delegates_to_transform_data(self):
        t = RailwayTimeFeaturesTransform(
            use_features=["hour", "minute"],
            normalize_time_features=True,
        )
        timestamps = pd.Series(pd.date_range("2023-01-01 10:00", periods=3, freq="15min"))
        data = {"timestamps": timestamps}
        result = t(data, metadata={})
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)

    def test_repr(self):
        t = RailwayTimeFeaturesTransform(
            use_features=["year"],
            normalize_time_features=False,
        )
        r = repr(t)
        assert "RailwayTimeFeaturesTransform" in r
        assert "year" in r


class TestTimeFeaturesFunction:

    def test_time_features_series(self):
        # Multiple months so month/day vary (no all-zero column)
        dates = pd.Series(pd.to_datetime(["2023-01-15", "2023-02-15", "2023-03-15", "2023-04-15"]))
        df = time_features(dates, use_features=["month", "day"], normalize=True)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["month", "day"]
        assert len(df) == 4
        assert not (df == 0).all().any()

    def test_time_features_non_series_raises(self):
        dates = np.array(["2023-01-01", "2023-01-02"], dtype="datetime64")
        with pytest.raises(ValueError, match="dates should be a pandas Series"):
            time_features(dates)

    def test_time_features_normalize_false(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=2, freq="D"))
        df = time_features(dates, use_features=["day"], normalize=False)
        assert df["day"].tolist() == [1, 2]


class TestReconstructDatetimeFromFeatures:

    def test_reconstruct_from_dataframe(self):
        df = pd.DataFrame({
            "year": [2023, 2023],
            "month": [1, 1],
            "day": [1, 2],
            "hour": [0, 12],
            "minute": [0, 30],
            "second": [0, 0],
        })
        result = reconstruct_datetime_from_features(df, list(df.columns))
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_reconstruct_from_numpy(self):
        arr = np.array([[2023, 1, 1, 10, 0, 0], [2023, 1, 2, 12, 30, 0]])
        cols = ["year", "month", "day", "hour", "minute", "second"]
        result = reconstruct_datetime_from_features(arr, cols)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_reconstruct_numpy_non_2d_raises(self):
        arr = np.array([2023, 1, 1])
        with pytest.raises(ValueError, match="NumPy array must be 2D"):
            reconstruct_datetime_from_features(arr, ["year", "month", "day"])

    def test_reconstruct_wrong_columns_raises(self):
        # Array has 2 columns but we request 3 -> shape mismatch
        arr = np.array([[2023, 1]])
        with pytest.raises(ValueError, match="Number of columns"):
            reconstruct_datetime_from_features(arr, ["year", "month", "day"])
