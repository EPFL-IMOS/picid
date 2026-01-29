"""Extended tests for timefeatures.py to reach 90% coverage."""
import numpy as np
import pytest
import pandas as pd
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.timefeatures import (
    TimePFNTimeFeaturesTransform,
    TimeFeature,
    SecondOfMinute,
    MinuteOfHour,
    HourOfDay,
    DayOfWeek,
    DayOfMonth,
    DayOfYear,
    MonthOfYear,
    WeekOfYear,
)


class TestTimePFNTimeFeaturesTransformExtended:
    """Extended tests for TimePFNTimeFeaturesTransform."""
    
    def test_repr_with_kwargs(self):
        """Test __repr__ with additional kwargs."""
        transform = TimePFNTimeFeaturesTransform(freq="h", custom_param=123)
        repr_str = repr(transform)
        assert "TimePFNTimeFeaturesTransform" in repr_str
        assert "freq=" in repr_str


class TestTimeFeatureClasses:
    """Tests for TimeFeature base class and subclasses."""
    
    def test_time_feature_base(self):
        """Test TimeFeature base class."""
        feature = TimeFeature()
        assert feature is not None
        
        # Test __repr__
        repr_str = repr(feature)
        assert "TimeFeature" in repr_str
    
    def test_second_of_minute(self):
        """Test SecondOfMinute feature."""
        feature = SecondOfMinute()
        index = pd.date_range("2023-01-01 12:00:00", periods=5, freq="5s")
        
        result = feature(index)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
        # Values should be between -0.5 and 0.5
        assert np.all(result >= -0.5)
        assert np.all(result <= 0.5)
    
    def test_minute_of_hour(self):
        """Test MinuteOfHour feature."""
        feature = MinuteOfHour()
        index = pd.date_range("2023-01-01 12:00:00", periods=5, freq="10min")
        
        result = feature(index)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
        assert np.all(result >= -0.5)
        assert np.all(result <= 0.5)
    
    def test_hour_of_day(self):
        """Test HourOfDay feature."""
        feature = HourOfDay()
        index = pd.date_range("2023-01-01", periods=24, freq="h")
        
        result = feature(index)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 24
        assert np.all(result >= -0.5)
        assert np.all(result <= 0.5)
    
    def test_day_of_week(self):
        """Test DayOfWeek feature."""
        feature = DayOfWeek()
        index = pd.date_range("2023-01-01", periods=7, freq="d")
        
        result = feature(index)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 7
        assert np.all(result >= -0.5)
        assert np.all(result <= 0.5)
    
    def test_day_of_month(self):
        """Test DayOfMonth feature."""
        feature = DayOfMonth()
        index = pd.date_range("2023-01-01", periods=10, freq="d")
        
        result = feature(index)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert np.all(result >= -0.5)
        assert np.all(result <= 0.5)
    
    def test_day_of_year(self):
        """Test DayOfYear feature."""
        feature = DayOfYear()
        index = pd.date_range("2023-01-01", periods=10, freq="d")
        
        result = feature(index)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert np.all(result >= -0.5)
        assert np.all(result <= 0.5)
    
    def test_month_of_year(self):
        """Test MonthOfYear feature."""
        feature = MonthOfYear()
        index = pd.date_range("2023-01-01", periods=12, freq="M")
        
        result = feature(index)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 12
        assert np.all(result >= -0.5)
        assert np.all(result <= 0.5)
    
    def test_week_of_year(self):
        """Test WeekOfYear feature."""
        feature = WeekOfYear()
        index = pd.date_range("2023-01-01", periods=10, freq="W")
        
        result = feature(index)
        
        # WeekOfYear returns pandas Series, not numpy array directly
        # Convert to numpy if needed
        if hasattr(result, 'values'):
            result = result.values
        assert isinstance(result, (np.ndarray, pd.Series))
        assert len(result) == 10
        # Values might be Series, check values
        if isinstance(result, pd.Series):
            assert np.all(result.values >= -0.5)
            assert np.all(result.values <= 0.5)
        else:
            assert np.all(result >= -0.5)
            assert np.all(result <= 0.5)
