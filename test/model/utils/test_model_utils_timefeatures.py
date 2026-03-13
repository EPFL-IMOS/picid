import numpy as np
import pandas as pd
import pytest

from picid.model.utils.timefeatures import (
    TimeFeature,
    SecondOfMinute,
    MinuteOfHour,
    HourOfDay,
    DayOfWeek,
    DayOfMonth,
    MonthOfYear,
    time_features_from_frequency_str,
    time_features,
)


def test_second_of_minute():
    idx = pd.DatetimeIndex(["2024-01-15 12:30:45"])
    feat = SecondOfMinute()
    out = feat(idx)
    assert out.shape == (1,)
    assert -0.5 <= out[0] <= 0.5
    assert np.isclose(out[0], 45 / 59.0 - 0.5)


def test_minute_of_hour():
    idx = pd.DatetimeIndex(["2024-01-15 12:30:00"])
    feat = MinuteOfHour()
    out = feat(idx)
    assert out.shape == (1,)
    assert np.isclose(out[0], 30 / 59.0 - 0.5)


def test_hour_of_day():
    idx = pd.DatetimeIndex(["2024-01-15 14:00:00"])
    feat = HourOfDay()
    out = feat(idx)
    assert out.shape == (1,)
    assert np.isclose(out[0], 14 / 23.0 - 0.5)


def test_day_of_week():
    idx = pd.DatetimeIndex(["2024-01-15"])  # Monday
    feat = DayOfWeek()
    out = feat(idx)
    assert out.shape == (1,)
    assert -0.5 <= out[0] <= 0.5


def test_day_of_month():
    idx = pd.DatetimeIndex(["2024-01-15"])
    feat = DayOfMonth()
    out = feat(idx)
    assert out.shape == (1,)
    assert np.isclose(out[0], (15 - 1) / 30.0 - 0.5)


def test_month_of_year():
    idx = pd.DatetimeIndex(["2024-01-15"])
    feat = MonthOfYear()
    out = feat(idx)
    assert out.shape == (1,)
    assert np.isclose(out[0], 0 / 11.0 - 0.5)


def test_time_features_from_frequency_str_hourly():
    feats = time_features_from_frequency_str("1H")
    assert len(feats) > 0
    assert all(isinstance(f, TimeFeature) for f in feats)


def test_time_features_from_frequency_str_daily():
    feats = time_features_from_frequency_str("1D")
    assert len(feats) > 0


def test_time_features_from_frequency_str_unsupported_raises():
    with pytest.raises(RuntimeError, match="Unsupported frequency"):
        time_features_from_frequency_str("1ms")


def test_time_features():
    dates = pd.DatetimeIndex(["2024-01-15 12:00:00", "2024-01-16 14:30:00"])
    out = time_features(dates, freq="h")

    assert out.ndim == 2
    # vstack stacks features: shape is (n_features, n_timestamps)
    assert out.shape[1] == 2  # two timestamps
    assert out.shape[0] >= 1  # at least one feature
