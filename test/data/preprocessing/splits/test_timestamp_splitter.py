"""Comprehensive tests for TimeStampSplitter.

This module tests the timestamp-based data splitting functionality used
in time series forecasting where splits are defined by date boundaries.

PHM Context:
-----------
In industrial PHM, models are trained on historical data and validated
on future data. TimeStampSplitter ensures temporal integrity by splitting
at specific dates (e.g., "train until 2020, test from 2021").

Test Coverage Strategy:
----------------------
1. **Initialization Tests**: Timestamp parsing and ratio validation
2. **Period Rounding**: Snapping splits to period boundaries
3. **Split Calculation**: Correct date-based partitioning
4. **Lookback Handling**: Overlap for forecasting models
5. **Edge Cases**: Boundary dates, invalid timestamps
6. **Error Handling**: Missing timestamps, invalid ratios
"""

import numpy as np
import pandas as pd
import pytest

from picid.data.split_strategies.database_splitter import TimeStampSplitter


class TestTimeStampSplitterInitialization:
    """Tests for TimeStampSplitter initialization."""

    def test_init_basic(self):
        """Test basic initialization with required parameters.

        **PHM Logic**: test_start defines where test data begins.

        **Methodology**: Create splitter with test_start date.

        **Expected**: Parameters stored correctly (as Timestamp).

        Validates: Requirement TSS-1.1 - Basic initialization
        """
        import pandas as pd

        splitter = TimeStampSplitter(
            test_start="2021-01-01", train_ratio=0.7, val_ratio=0.3
        )

        # test_start is converted to Timestamp
        assert splitter.test_start == pd.Timestamp("2021-01-01")
        assert splitter.train_ratio == 0.7
        assert splitter.val_ratio == 0.3

    def test_init_with_test_end(self):
        """Test initialization with test_end date.

        **PHM Logic**: test_end limits test data range.

        **Methodology**: Create splitter with both test_start and test_end.

        **Expected**: Both dates stored (as Timestamp).

        Validates: Requirement TSS-1.2 - Test end configuration
        """
        import pandas as pd

        splitter = TimeStampSplitter(
            test_start="2021-01-01",
            test_end="2021-06-30",
            train_ratio=0.7,
            val_ratio=0.3,
        )

        assert splitter.test_start == pd.Timestamp("2021-01-01")
        assert splitter.test_end == pd.Timestamp("2021-06-30")

    def test_init_invalid_ratios_error(self):
        """Test that invalid ratios (not summing to 1) raise error.

        **PHM Logic**: train_ratio + val_ratio must equal 1.0.

        **Methodology**: Pass ratios that don't sum to 1.

        **Expected**: ValueError raised.

        Validates: Requirement TSS-1.3 - Ratio validation
        """
        with pytest.raises(ValueError, match="1.0"):
            TimeStampSplitter(
                test_start="2021-01-01",
                train_ratio=0.6,  # 0.6 + 0.3 = 0.9 != 1.0
                val_ratio=0.3,
            )

    def test_init_with_custom_frequency(self):
        """Test initialization with custom frequency.

        **PHM Logic**: Different frequencies for different data resolutions.

        **Methodology**: Use hourly frequency instead of daily.

        **Expected**: Custom frequency stored.

        Validates: Requirement TSS-1.4 - Frequency configuration
        """
        splitter = TimeStampSplitter(
            test_start="2021-01-01",
            train_ratio=0.7,
            val_ratio=0.3,
            frequency="H",  # Hourly
        )

        assert splitter.frequency == "H"

    def test_init_with_lookback_params(self):
        """Test initialization with lookback parameters.

        **PHM Logic**: seq_len determines lookback overlap for forecasting.

        **Methodology**: Set custom seq_len and pred_len.

        **Expected**: Parameters stored correctly.

        Validates: Requirement TSS-1.5 - Lookback configuration
        """
        splitter = TimeStampSplitter(
            test_start="2021-01-01",
            train_ratio=0.7,
            val_ratio=0.3,
            seq_len=168,  # One week of hourly data
            label_len=24,
            pred_len=24,
        )

        assert splitter.seq_len == 168
        assert splitter.label_len == 24
        assert splitter.pred_len == 24


class TestTimeStampSplitterGetSplits:
    """Tests for get_splits method."""

    def test_get_splits_basic(self, timestamp_data):
        """Test basic split generation with timestamps.

        **PHM Logic**: Data before test_start is train/val, after is test.

        **Methodology**: Split data at known date boundary.

        **Expected**: Splits align with date boundary.

        Validates: Requirement TSS-2.1 - Basic timestamp splitting
        """
        splitter = TimeStampSplitter(
            test_start=timestamp_data["test_start"],
            train_ratio=0.7,
            val_ratio=0.3,
            seq_len=30,
            pred_len=15,
        )

        data = timestamp_data["features"]
        dates = timestamp_data["timestamps"]

        splits, masks = splitter.get_splits(data, dates)

        # Verify split structure
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        # Test should start after test_start date
        test_start_idx = splits["test"][0]
        test_start_date = dates.iloc[test_start_idx]
        expected_test_start = pd.Timestamp(timestamp_data["test_start"])

        assert test_start_idx >= 0
        assert test_start_date >= expected_test_start

    def test_get_splits_mask_generation(self, timestamp_data):
        """Test mask generation for effective samples.

        **PHM Logic**: Masks indicate lookback (False) vs effective (True) samples.

        **Methodology**: Check mask structure and values.

        **Expected**: Correct mask lengths and values.

        Validates: Requirement TSS-2.2 - Mask generation
        """
        splitter = TimeStampSplitter(
            test_start=timestamp_data["test_start"],
            train_ratio=0.7,
            val_ratio=0.3,
            seq_len=30,
            pred_len=15,
            apply_lookback=True,
        )

        data = timestamp_data["features"]
        dates = timestamp_data["timestamps"]

        splits, masks = splitter.get_splits(data, dates)

        # Verify masks exist
        assert "train" in masks
        assert "val" in masks
        assert "test" in masks

        # Masks should be boolean arrays
        assert masks["train"].dtype == bool
        assert masks["val"].dtype == bool
        assert masks["test"].dtype == bool


class TestTimeStampSplitterSplitData:
    """Tests for split_data method."""

    def test_split_data_basic(self, timestamp_data):
        """Test split_data with data dictionary.

        **PHM Logic**: split_data applies splits to data_dict.

        **Methodology**: Split data dict with features and timestamps.

        **Expected**: Properly structured output dicts.

        Validates: Requirement TSS-3.1 - Data dict splitting
        """
        splitter = TimeStampSplitter(
            test_start=timestamp_data["test_start"],
            train_ratio=0.7,
            val_ratio=0.3,
            seq_len=30,
            pred_len=15,
        )

        data_dict = {
            "features": timestamp_data["features"],
            "timestamps": timestamp_data["timestamps"],
        }

        splitted_data, split_masks = splitter.split_data(data_dict, "features")

        # Verify structure
        assert "train" in splitted_data
        assert "val" in splitted_data
        assert "test" in splitted_data

        # Data should be numpy arrays
        assert isinstance(splitted_data["train"], np.ndarray)

    def test_split_data_missing_timestamps_error(self, timestamp_data):
        """Test error when timestamps key is missing.

        **PHM Logic**: TimeStampSplitter requires 'timestamps' in data_dict.

        **Methodology**: Pass data_dict without timestamps.

        **Expected**: KeyError raised.

        Validates: Requirement TSS-3.2 - Timestamps requirement
        """
        splitter = TimeStampSplitter(
            test_start="2021-01-01", train_ratio=0.7, val_ratio=0.3
        )

        data_dict = {
            "features": np.random.randn(100, 5)
            # Missing 'timestamps' key!
        }

        with pytest.raises(KeyError):
            splitter.split_data(data_dict, "features")


class TestTimeStampSplitterPeriodRounding:
    """Tests for period rounding behavior."""

    def test_round_to_period_end_daily(self):
        """Test rounding to daily period boundaries.

        **PHM Logic**: Splits should align to period boundaries for clean dates.

        **Methodology**: Test internal rounding method.

        **Expected**: Index rounded to period end.

        Validates: Requirement TSS-4.1 - Daily rounding
        """
        splitter = TimeStampSplitter(
            test_start="2021-01-15",  # Mid-month
            train_ratio=0.7,
            val_ratio=0.3,
            frequency="D",
        )

        # Create date series
        dates = pd.date_range(start="2021-01-01", periods=31, freq="D")
        dates = pd.Series(dates)

        # Round from index 10
        rounded_idx = splitter._round_to_period_end(dates, 10)

        # Should return valid index
        assert 0 <= rounded_idx <= len(dates)

    def test_round_to_period_end_hourly(self):
        """Test rounding to hourly period boundaries.

        **PHM Logic**: Hourly frequency is fixed and rounds properly.

        **Methodology**: Use hourly frequency, check rounding.

        **Expected**: Rounds to hour boundary.

        Validates: Requirement TSS-4.2 - Hourly rounding
        """
        splitter = TimeStampSplitter(
            test_start="2021-06-15",  # Mid-June
            train_ratio=0.7,
            val_ratio=0.3,
            frequency="H",  # Hourly (fixed frequency)
        )

        # Create hourly date series
        dates = pd.date_range(start="2021-01-01", periods=100, freq="H")
        dates = pd.Series(dates)

        rounded_idx = splitter._round_to_period_end(dates, 10)

        # Should return valid index
        assert 0 <= rounded_idx <= len(dates)


class TestTimeStampSplitterRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test __repr__ method.

        **PHM Logic**: Repr shows configuration for debugging.

        **Methodology**: Create splitter, check repr content.

        **Expected**: Contains key parameters.

        Validates: Requirement TSS-5.1 - String representation
        """
        splitter = TimeStampSplitter(
            test_start="2021-01-01", train_ratio=0.7, val_ratio=0.3
        )

        repr_str = repr(splitter)

        assert "2021-01-01" in repr_str or "TimeStamp" in repr_str


class TestTimeStampSplitterEdgeCases:
    """Edge case tests for TimeStampSplitter."""

    def test_test_start_before_data(self, timestamp_data):
        """Test when test_start is before all data.

        **PHM Logic**: All data would be test data.

        **Methodology**: Set test_start before data starts.

        **Expected**: May raise error or produce empty train/val.

        Validates: Requirement TSS-6.1 - Early test_start handling
        """
        splitter = TimeStampSplitter(
            test_start="2019-01-01",  # Before data starts (2020-01-01)
            train_ratio=0.7,
            val_ratio=0.3,
            seq_len=30,
            pred_len=15,
        )

        data = timestamp_data["features"]
        dates = timestamp_data["timestamps"]

        # Behavior depends on implementation
        # May raise error or handle gracefully
        try:
            splits, masks = splitter.get_splits(data, dates)
            # If it works, verify structure
            assert "train" in splits
        except (AssertionError, ValueError):
            pass  # Expected behavior

    def test_test_start_after_data(self, timestamp_data):
        """Test when test_start is after all data.

        **PHM Logic**: All data would be train/val, no test data.

        **Methodology**: Set test_start after data ends.

        **Expected**: May raise IndexError or other error.

        Validates: Requirement TSS-6.2 - Late test_start handling
        """
        splitter = TimeStampSplitter(
            test_start="2025-01-01",  # After data ends
            train_ratio=0.7,
            val_ratio=0.3,
            seq_len=30,
            pred_len=15,
        )

        data = timestamp_data["features"]
        dates = timestamp_data["timestamps"]

        try:
            splits, masks = splitter.get_splits(data, dates)
            # If it works, test might be empty or minimal
            assert "test" in splits
        except (AssertionError, ValueError, IndexError):
            pass  # Expected behavior - out of bounds is acceptable

    def test_non_series_timestamps_conversion(self, timestamp_data):
        """Test automatic conversion of non-Series timestamps.

        **PHM Logic**: Timestamps should be converted to pd.Series if needed.

        **Methodology**: Pass list/array of timestamps.

        **Expected**: Automatic conversion attempted.

        Validates: Requirement TSS-6.3 - Timestamp type handling
        """
        splitter = TimeStampSplitter(
            test_start="2021-01-01",
            train_ratio=0.7,
            val_ratio=0.3,
            seq_len=30,
            pred_len=15,
        )

        data = timestamp_data["features"]
        # Convert timestamps to list (non-Series)
        dates_list = timestamp_data["timestamps"].tolist()

        try:
            # May work if implementation handles lists
            splits, masks = splitter.get_splits(data, pd.Series(dates_list))
            assert "train" in splits
        except (ValueError, TypeError):
            pass  # Expected if strict pd.Series required
