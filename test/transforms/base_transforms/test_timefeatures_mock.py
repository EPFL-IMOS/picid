"""Tests for timefeatures.py using mocking to reach 99% coverage."""
import numpy as np
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.timefeatures import TimePFNTimeFeaturesTransform


class TestTimePFNTimeFeaturesTransformMock:
    """Tests using mocking to cover exception paths."""
    
    def test_transform_data_date_column_path(self):
        """Test transform_data with 'date' column (successful path)."""
        transform = TimePFNTimeFeaturesTransform(freq="h")
        timestamps = pd.date_range("2023-01-01", periods=10, freq="h")
        timestamps_array = timestamps.values
        data = NamedTransformInput(timestamps=timestamps_array)
        metadata = {}
        
        # Patch the DataFrame's __getitem__ to return 'date' column successfully
        # The transform creates: initial_dates = pd.DataFrame(pd.to_datetime(data_segment))
        # Then tries: initial_dates["date"]
        original_dataframe = pd.DataFrame
        
        def mock_dataframe_with_date(*args, **kwargs):
            df = original_dataframe(*args, **kwargs)
            # Replace the integer column with 'date' column
            if len(df.columns) == 1 and df.columns[0] == 0:
                df.columns = ['date']
            return df
        
        with patch('lmetk.transforms.base_transforms.timefeatures.pd.DataFrame', side_effect=mock_dataframe_with_date):
            with patch('lmetk.transforms.base_transforms.timefeatures.time_features') as mock_time_features:
                mock_time_features.return_value = np.random.randn(4, 10)
                
                result = transform.transform_data(data, metadata)
                assert isinstance(result, pd.DataFrame)
    
    def test_transform_data_datetime_column_path(self):
        """Test transform_data with 'datetime' column (exception path)."""
        transform = TimePFNTimeFeaturesTransform(freq="h")
        timestamps = pd.date_range("2023-01-01", periods=10, freq="h")
        timestamps_array = timestamps.values
        data = NamedTransformInput(timestamps=timestamps_array)
        metadata = {}
        
        # Patch to trigger the except branch - DataFrame without 'date' but with 'datetime'
        original_dataframe = pd.DataFrame
        
        def mock_dataframe_with_datetime(*args, **kwargs):
            df = original_dataframe(*args, **kwargs)
            # Replace column with 'datetime' (no 'date')
            if len(df.columns) == 1:
                df.columns = ['datetime']
            return df
        
        with patch('lmetk.transforms.base_transforms.timefeatures.pd.DataFrame', side_effect=mock_dataframe_with_datetime):
            with patch('lmetk.transforms.base_transforms.timefeatures.time_features') as mock_time_features:
                mock_time_features.return_value = np.random.randn(4, 10)
                
                result = transform.transform_data(data, metadata)
                assert isinstance(result, pd.DataFrame)
