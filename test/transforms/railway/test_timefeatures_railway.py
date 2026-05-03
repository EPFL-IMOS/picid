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
        timestamps = pd.Series(
            pd.to_datetime(
                [
                    "2023-01-01 10:00",
                    "2023-01-15 14:30",
                    "2023-02-01 09:00",
                    "2023-02-20 16:45",
                ]
            )
        )
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
        timestamps = pd.Series(
            pd.date_range("2023-01-01 10:00", periods=3, freq="15min")
        )
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
        dates = pd.Series(
            pd.to_datetime(["2023-01-15", "2023-02-15", "2023-03-15", "2023-04-15"])
        )
        df = time_features(dates, use_features=["month", "day"], normalize=True)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["month", "day"]
        assert len(df) == 4
        assert not (df == 0).all().any()

    def test_time_features_numpy_array_accepted(self):
        """Numpy datetime array is converted to Series; use varied dates to avoid zero-column."""
        dates = np.array(
            pd.to_datetime(["2023-01-15", "2023-02-15", "2023-03-15"]),
            dtype="datetime64[ns]",
        )
        df = time_features(dates, use_features=["month", "day"], normalize=True)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["month", "day"]
        assert len(df) == 3

    def test_time_features_normalize_false(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=2, freq="D"))
        df = time_features(dates, use_features=["day"], normalize=False)
        assert df["day"].tolist() == [1, 2]

    def test_time_features_year_normalize_false(self):
        """Year column with normalize=False returns raw integer year values."""
        dates = pd.Series(pd.to_datetime(["2022-06-01 08:00", "2023-07-15 12:30"]))
        df = time_features(dates, use_features=["year", "month"], normalize=False)
        assert df["year"].tolist() == [2022, 2023]


class TestReconstructDatetimeFromFeatures:
    def test_reconstruct_from_dataframe(self):
        df = pd.DataFrame(
            {
                "year": [2023, 2023],
                "month": [1, 1],
                "day": [1, 2],
                "hour": [0, 12],
                "minute": [0, 30],
                "second": [0, 0],
            }
        )
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

    def test_reconstruct_float_numpy_rounds_to_int(self):
        """Float array is rounded and cast to int before reconstruction."""
        arr = np.array([[2023.0, 1.0, 15.0, 10.0, 30.0, 0.0]])
        cols = ["year", "month", "day", "hour", "minute", "second"]
        result = reconstruct_datetime_from_features(arr, cols)
        assert isinstance(result, pd.Series)
        assert len(result) == 1

    def test_reconstruct_float_dataframe_rounds_to_int(self):
        """DataFrame with float columns is rounded before reconstruction."""
        df = pd.DataFrame({"year": [2023.0], "month": [3.0], "day": [5.0]})
        result = reconstruct_datetime_from_features(df, ["year", "month", "day"])
        assert isinstance(result, pd.Series)

    def test_reconstruct_dataframe_missing_columns_raises(self):
        """DataFrame missing requested columns raises ValueError."""
        df = pd.DataFrame({"year": [2023], "month": [1]})
        with pytest.raises(ValueError, match="Missing columns"):
            reconstruct_datetime_from_features(df, ["year", "month", "day"])

    def test_reconstruct_invalid_type_raises(self):
        """Non-array, non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="numpy.ndarray or pandas.DataFrame"):
            reconstruct_datetime_from_features([[2023, 1, 1]], ["year", "month", "day"])


class TestTimeFeaturesEdgeCases:
    """Cover uncovered branches in time_features and transform_data."""

    def test_use_features_none_uses_default_set(self):
        """When use_features=None, all 6 default features are extracted."""
        # Dates span multiple years/months/weekdays/hours/minutes to avoid zero-columns.
        # Specifically: varied weekdays (Mon=0, Tue=1, Fri=4) so no all-zero column.
        dates = pd.Series(
            [
                pd.Timestamp("2023-01-09 10:15"),  # Monday
                pd.Timestamp("2023-04-11 14:30"),  # Tuesday
                pd.Timestamp("2024-07-05 18:45"),  # Friday
            ]
        )
        df = time_features(dates, use_features=None, normalize=True)
        assert set(df.columns) == {"year", "month", "day", "weekday", "hour", "minute"}

    def test_numeric_unix_seconds_converted(self):
        """Integer Unix-epoch timestamps (seconds) are parsed correctly."""
        # 2023-06-15 12:00:00 UTC in seconds
        ts = pd.Series([1686830400])
        df = time_features(ts, use_features=["month", "day"], normalize=False)
        assert df["month"].iloc[0] == 6

    def test_numeric_large_int_converted_via_to_datetime(self):
        """Numeric values >= 1e12 (nanoseconds) are parsed via pd.to_datetime."""
        # Same datetime expressed in nanoseconds
        ts_ns = pd.Series([1686830400 * 1_000_000_000])
        df = time_features(ts_ns, use_features=["month"], normalize=False)
        assert df["month"].iloc[0] == 6

    def test_year_feature_normalized(self):
        """Year feature is normalised to [0, 1] when normalize=True."""
        dates = pd.Series(pd.to_datetime(["2020-06-15", "2021-06-15", "2022-06-15"]))
        df = time_features(dates, use_features=["year", "month"], normalize=True)
        assert df["year"].min() == pytest.approx(0.0, abs=1e-6)
        assert df["year"].max() == pytest.approx(1.0, abs=1e-6)

    def test_weekday_feature_included(self):
        """Weekday feature is extracted and in [0, 1] when normalised."""
        # Span multiple months to avoid a zero month column
        dates = pd.Series([pd.Timestamp(f"2023-{m:02d}-15") for m in range(1, 8)])
        df = time_features(dates, use_features=["weekday", "month"], normalize=True)
        assert "weekday" in df.columns
        assert df["weekday"].max() <= 1.0 + 1e-6

    def test_zero_column_raises(self):
        """All-zero feature column raises ValueError."""
        # All timestamps at midnight → hour column is all zero
        dates = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]))
        with pytest.raises(ValueError, match="zero column"):
            time_features(dates, use_features=["hour"], normalize=False)

    def test_transform_data_calls_transform_method(self):
        """transform() delegates to transform_data without error."""
        t = RailwayTimeFeaturesTransform(
            use_features=["month", "hour", "minute"],
            normalize_time_features=True,
        )
        # Span 3 months with varied hours/minutes to avoid zero columns
        timestamps = pd.Series(
            [
                pd.Timestamp("2023-01-15 10:15"),
                pd.Timestamp("2023-03-20 14:30"),
                pd.Timestamp("2023-06-25 18:45"),
            ]
        )
        data = {"timestamps": timestamps}
        result = t.transform(data, metadata={})
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
