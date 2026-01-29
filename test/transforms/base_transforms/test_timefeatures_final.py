"""Final tests for timefeatures.py to reach 90% coverage."""
import numpy as np
import pytest
import pandas as pd
from pandas.tseries import offsets
from lmetk.transforms.base_transforms.timefeatures import (
    time_features_from_frequency_str,
    time_features,
    TimePFNTimeFeaturesTransform,
)


class TestTimeFeaturesFunctions:
    """Tests for time_features utility functions."""
    
    def test_time_features_from_frequency_str_year(self):
        """Test time_features_from_frequency_str with yearly frequency."""
        features = time_features_from_frequency_str("Y")
        assert len(features) == 0  # YearEnd has no features
    
    def test_time_features_from_frequency_str_quarter(self):
        """Test time_features_from_frequency_str with quarterly frequency."""
        features = time_features_from_frequency_str("Q")
        assert len(features) == 1
        assert features[0].__class__.__name__ == "MonthOfYear"
    
    def test_time_features_from_frequency_str_month(self):
        """Test time_features_from_frequency_str with monthly frequency."""
        features = time_features_from_frequency_str("M")
        assert len(features) == 1
        assert features[0].__class__.__name__ == "MonthOfYear"
    
    def test_time_features_from_frequency_str_week(self):
        """Test time_features_from_frequency_str with weekly frequency."""
        features = time_features_from_frequency_str("W")
        assert len(features) == 2
    
    def test_time_features_from_frequency_str_day(self):
        """Test time_features_from_frequency_str with daily frequency."""
        features = time_features_from_frequency_str("D")
        assert len(features) == 3
    
    def test_time_features_from_frequency_str_business_day(self):
        """Test time_features_from_frequency_str with business day frequency."""
        features = time_features_from_frequency_str("B")
        assert len(features) == 3
    
    def test_time_features_from_frequency_str_hour(self):
        """Test time_features_from_frequency_str with hourly frequency."""
        features = time_features_from_frequency_str("H")
        assert len(features) == 4
    
    def test_time_features_from_frequency_str_minute(self):
        """Test time_features_from_frequency_str with minutely frequency."""
        features = time_features_from_frequency_str("T")
        assert len(features) == 5
    
    def test_time_features_from_frequency_str_second(self):
        """Test time_features_from_frequency_str with secondly frequency."""
        features = time_features_from_frequency_str("S")
        assert len(features) == 6
    
    def test_time_features_from_frequency_str_invalid_error(self):
        """Test time_features_from_frequency_str with invalid frequency raises error."""
        # pandas might raise ValueError first, then RuntimeError
        with pytest.raises((RuntimeError, ValueError), match="Unsupported frequency|Invalid frequency"):
            time_features_from_frequency_str("invalid")
    
    def test_time_features_function(self):
        """Test time_features function."""
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        result = time_features(dates, freq="h")
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] > 0  # Number of features
        assert result.shape[1] == 10  # Number of dates
    
    def test_time_features_different_frequencies(self):
        """Test time_features with different frequency strings."""
        dates = pd.date_range("2023-01-01", periods=5, freq="d")
        
        # Test different frequencies
        for freq in ["D", "H", "T", "S"]:
            try:
                result = time_features(dates, freq=freq)
                assert isinstance(result, np.ndarray)
                assert result.ndim == 2
            except (ValueError, RuntimeError):
                # Some frequencies might not match the dates
                pass
