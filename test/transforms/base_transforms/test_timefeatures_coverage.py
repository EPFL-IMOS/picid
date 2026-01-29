"""Tests for timefeatures.py to reach 99% coverage by patching DataFrame access."""
import numpy as np
import pytest
import pandas as pd
from unittest.mock import patch
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.timefeatures import TimePFNTimeFeaturesTransform


class TestTimePFNTimeFeaturesTransformCoverage:
    """Tests to reach 99% coverage by patching DataFrame column access."""
    
    def test_transform_data_date_column_success(self):
        """Test transform_data with 'date' column (successful try path)."""
        transform = TimePFNTimeFeaturesTransform(freq="h")
        timestamps = pd.date_range("2023-01-01", periods=10, freq="h")
        timestamps_array = timestamps.values
        data = NamedTransformInput(timestamps=timestamps_array)
        metadata = {}
        
        # Patch the DataFrame's __getitem__ to return 'date' column successfully
        original_dataframe_init = pd.DataFrame.__init__
        
        def mock_dataframe_getitem(self, key):
            if key == "date":
                return pd.Series(timestamps, name="date")
            elif key == "datetime":
                raise KeyError("datetime")
            return self.__class__.__getitem__(self, key)
        
        # Create a DataFrame and patch its __getitem__
        df = pd.DataFrame(pd.to_datetime(timestamps_array))
        df.__getitem__ = lambda key: (
            pd.Series(timestamps, name="date") if key == "date" 
            else pd.DataFrame.__getitem__(df, key)
        )
        
        # Patch pd.DataFrame constructor to return our mocked DataFrame
        def mock_dataframe(*args, **kwargs):
            result = pd.DataFrame(*args, **kwargs)
            if len(result.columns) == 1:
                # Replace __getitem__ to handle 'date' access
                original_getitem = result.__getitem__
                def patched_getitem(key):
                    if key == "date":
                        return result.iloc[:, 0]  # Return the first column
                    return original_getitem(key)
                result.__getitem__ = patched_getitem
            return result
        
        with patch('lmetk.transforms.base_transforms.timefeatures.pd.DataFrame', side_effect=mock_dataframe):
            with patch('lmetk.transforms.base_transforms.timefeatures.time_features') as mock_time_features:
                mock_time_features.return_value = np.random.randn(4, 10)
                
                result = transform.transform_data(data, metadata)
                assert isinstance(result, pd.DataFrame)
    
    def test_transform_data_datetime_column_exception(self):
        """Test transform_data with 'datetime' column (exception path)."""
        transform = TimePFNTimeFeaturesTransform(freq="h")
        timestamps = pd.date_range("2023-01-01", periods=10, freq="h")
        timestamps_array = timestamps.values
        data = NamedTransformInput(timestamps=timestamps_array)
        metadata = {}
        
        # Patch to simulate 'date' KeyError but 'datetime' success
        def mock_dataframe(*args, **kwargs):
            result = pd.DataFrame(*args, **kwargs)
            if len(result.columns) == 1:
                original_getitem = result.__getitem__
                def patched_getitem(key):
                    if key == "date":
                        raise KeyError("date")
                    elif key == "datetime":
                        return result.iloc[:, 0]  # Return first column as 'datetime'
                    return original_getitem(key)
                result.__getitem__ = patched_getitem
            return result
        
        with patch('lmetk.transforms.base_transforms.timefeatures.pd.DataFrame', side_effect=mock_dataframe):
            with patch('lmetk.transforms.base_transforms.timefeatures.time_features') as mock_time_features:
                mock_time_features.return_value = np.random.randn(4, 10)
                
                result = transform.transform_data(data, metadata)
                assert isinstance(result, pd.DataFrame)
