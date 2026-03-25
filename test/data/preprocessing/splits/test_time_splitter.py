"""Comprehensive tests for TimeSplitter.

This module tests the time-based data splitting functionality used in
PHM forecasting and prognostics tasks. The TimeSplitter creates train/val/test
splits from sequential time series data with proper handling of lookback
overlaps for forecasting models.

PHM Context:
-----------
Time series forecasting for RUL prediction requires careful splitting:
- Training data must not leak future information
- Lookback overlap ensures models can make predictions at split boundaries
- Masks indicate which samples are "effective" vs "lookback padding"

Test Coverage Strategy:
----------------------
1. **Initialization Tests**: Parameter validation
2. **Float Ratio Splitting**: Proportional splits (0.7/0.15/0.15)
3. **Integer Size Splitting**: Exact sample counts
4. **Lookback Overlap**: Proper overlap calculation for seq_len
5. **Mask Generation**: Correct effective sample masks
6. **Edge Cases**: Very short data, boundary conditions
7. **Error Handling**: Invalid ratios, type mixing
"""

import numpy as np
import pytest

from picid.data.split_strategies.time_splitter import TimeSplitter, ValueWarning


class TestTimeSplitterInitialization:
    """Tests for TimeSplitter initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters.

        **PHM Logic**: Default seq_len=384 and pred_len=96 are typical for
        long-horizon forecasting in time series analysis.

        **Methodology**: Create splitter without arguments, verify defaults.

        **Expected**: seq_len=384, label_len=96, pred_len=96, no splits defined.

        Validates: Requirement TS-1.1 - Default parameter handling
        """
        splitter = TimeSplitter()

        assert splitter.train is None
        assert splitter.val is None
        assert splitter.test is None
        assert splitter.seq_len == 384
        assert splitter.label_len == 96
        assert splitter.pred_len == 96
        assert splitter.splits_dict is None

    def test_init_with_float_ratios(self):
        """Test initialization with float ratio parameters.

        **PHM Logic**: Float ratios (0.7, 0.15, 0.15) are common for
        splitting datasets proportionally.

        **Methodology**: Create splitter with float train/val/test.

        **Expected**: Parameters stored correctly as floats.

        Validates: Requirement TS-1.2 - Float ratio configuration
        """
        splitter = TimeSplitter(train=0.7, val=0.15, test=0.15)

        assert splitter.train == 0.7
        assert splitter.val == 0.15
        assert splitter.test == 0.15

    def test_init_with_integer_sizes(self):
        """Test initialization with integer size parameters.

        **PHM Logic**: Integer sizes allow exact control over split sizes,
        useful when specific sample counts are required.

        **Methodology**: Create splitter with integer train/val/test.

        **Expected**: Parameters stored correctly as integers.

        Validates: Requirement TS-1.3 - Integer size configuration
        """
        splitter = TimeSplitter(train=700, val=150, test=150)

        assert splitter.train == 700
        assert splitter.val == 150
        assert splitter.test == 150

    def test_init_with_custom_seq_len(self):
        """Test initialization with custom sequence length.

        **PHM Logic**: Different applications require different lookback
        windows. seq_len controls how much history the model sees.

        **Methodology**: Create splitter with custom seq_len and pred_len.

        **Expected**: Custom parameters stored correctly.

        Validates: Requirement TS-1.4 - Custom sequence parameters
        """
        splitter = TimeSplitter(
            train=0.7, val=0.15, test=0.15, seq_len=128, label_len=32, pred_len=32
        )

        assert splitter.seq_len == 128
        assert splitter.label_len == 32
        assert splitter.pred_len == 32

    def test_init_create_splits_for_kwarg(self):
        """Test initialization with create_splits_for kwarg.

        **PHM Logic**: Not all data keys need splitting (e.g., metadata).
        create_splits_for specifies which keys to split.

        **Methodology**: Create splitter with custom create_splits_for.

        **Expected**: Parameter extracted from kwargs correctly.

        Validates: Requirement TS-1.5 - Selective key splitting
        """
        splitter = TimeSplitter(
            train=0.7, val=0.3, create_splits_for=["features", "target", "aux"]
        )

        assert splitter.create_splits_for == ["features", "target", "aux"]


class TestTimeSplitterGetSplits:
    """Tests for get_splits method."""

    def test_get_splits_float_ratios(self, sample_time_series_data):
        """Test get_splits with float ratio splitting (test=None).

        **PHM Logic**: 70/15 split with test derived from remainder.
        The splitter should divide data proportionally.

        **Methodology**: Create 1000-sample data, split with 0.7/0.15/None.

        **Expected**:
        - Train: ~700 samples (0-699)
        - Val: starts with overlap, includes ~150 effective samples
        - Test: remaining data with overlap

        Validates: Requirement TS-2.1 - Float ratio splitting
        """
        # Note: When using float ratios, test should be None (derived from remainder)
        splitter = TimeSplitter(
            train=0.7, val=0.15, test=None, seq_len=100, pred_len=50
        )
        data = sample_time_series_data["features"]

        splits, masks = splitter.get_splits(data)

        # Verify split structure
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        # Verify ranges are tuples
        assert len(splits["train"]) == 2
        assert len(splits["val"]) == 2
        assert len(splits["test"]) == 2

        # Train should start at 0
        assert splits["train"][0] == 0

        # Verify no gaps between actual data (considering overlap)
        # Train ends, val starts with lookback overlap
        train_start, train_end = splits["train"]
        val_start, val_end = splits["val"]
        test_start, test_end = splits["test"]

        # Val should start before train ends (lookback overlap)
        assert val_start < train_end
        # Test should start before val ends (lookback overlap)
        assert test_start < val_end

    def test_get_splits_integer_sizes(self, sample_time_series_data):
        """Test get_splits with integer size splitting.

        **PHM Logic**: Exact sample counts useful for reproducibility
        and when comparing models on identical splits.

        **Methodology**: Create splitter with exact sample counts.

        **Expected**: Splits have exactly specified sizes (before overlap).

        Validates: Requirement TS-2.2 - Integer size splitting
        """
        splitter = TimeSplitter(train=600, val=200, test=200, seq_len=50, pred_len=25)
        data = sample_time_series_data["features"]

        splits, masks = splitter.get_splits(data)

        # Train should have exactly train_len samples
        train_start, train_end = splits["train"]
        assert train_end - train_start == 600

    def test_get_splits_float_ratios_not_exceeding_one(self):
        """Test that float ratios cannot exceed 1.0.

        **PHM Logic**: Ratios > 1.0 are invalid as they would require
        more data than exists.

        **Methodology**: Attempt to split with ratios summing > 1.0.

        **Expected**: ValueError raised about proportions.

        Validates: Requirement TS-2.3 - Ratio validation
        """
        splitter = TimeSplitter(
            train=0.7,
            val=0.3,
            test=0.2,  # Sum = 1.2 > 1.0
            seq_len=50,
            pred_len=25,
        )
        data = np.random.randn(1000, 5)

        with pytest.raises(ValueError, match="cannot exceed 1.0"):
            splitter.get_splits(data)

    def test_get_splits_mixed_types_error(self):
        """Test that mixing int/float types raises error.

        **PHM Logic**: Mixing types is ambiguous - is 0.7 a ratio or 0 samples?

        **Methodology**: Create splitter with mixed int/float types.

        **Expected**: ValueError about type mixing.

        Validates: Requirement TS-2.4 - Type consistency validation
        """
        splitter = TimeSplitter(
            train=0.7,
            val=150,
            test=150,  # Mixed types
            seq_len=50,
            pred_len=25,
        )
        data = np.random.randn(1000, 5)

        with pytest.raises(ValueError, match="must all be float or int"):
            splitter.get_splits(data)

    def test_get_splits_lookback_overlap(self, sample_time_series_data):
        """Test lookback overlap between splits.

        **PHM Logic**: For forecasting, validation/test sets need lookback
        samples to form the first prediction. seq_len determines this overlap.

        **Methodology**: Split data and verify overlap equals seq_len.

        **Expected**: Val/test start seq_len samples before previous split ends.

        Validates: Requirement TS-2.5 - Lookback overlap
        """
        seq_len = 100
        # Use test=None so float ratios work correctly
        splitter = TimeSplitter(
            train=0.7, val=0.15, test=None, seq_len=seq_len, pred_len=50
        )
        data = sample_time_series_data["features"]

        splits, masks = splitter.get_splits(data)

        train_start, train_end = splits["train"]
        val_start, val_end = splits["val"]
        test_start, test_end = splits["test"]

        # Val should start seq_len before train ends (if train > seq_len)
        expected_val_start = max(0, train_end - seq_len)
        assert val_start == expected_val_start

        # Test should start seq_len before val ends
        expected_test_start = max(0, val_end - seq_len)
        assert test_start == expected_test_start

    def test_get_splits_mask_generation(self, sample_time_series_data):
        """Test mask generation for effective samples.

        **PHM Logic**: Masks indicate which samples are "effective" (ones)
        vs "lookback padding" (zeros). Training loss should only be computed
        on effective samples.

        **Methodology**: Verify mask lengths and values.

        **Expected**:
        - Train mask: all ones (no lookback needed)
        - Val/test masks: zeros for lookback, ones for effective

        Validates: Requirement TS-2.6 - Mask correctness
        """
        seq_len = 100
        # Use test=None so float ratios work correctly
        splitter = TimeSplitter(
            train=0.7, val=0.15, test=None, seq_len=seq_len, pred_len=50
        )
        data = sample_time_series_data["features"]
        n = len(data)

        splits, masks = splitter.get_splits(data)

        # Verify mask structure
        assert "train" in masks
        assert "val" in masks
        assert "test" in masks

        # Train mask should be all ones
        train_start, train_end = splits["train"]
        assert train_end <= n
        assert len(masks["train"]) == train_end - train_start
        assert np.all(masks["train"])

        # Val mask should have lookback zeros then ones
        val_start, val_end = splits["val"]
        assert len(masks["val"]) == val_end - val_start
        assert np.sum(~masks["val"][:seq_len]) == seq_len

        # Test mask should have lookback zeros then ones
        test_start, test_end = splits["test"]
        assert len(masks["test"]) == test_end - test_start
        assert np.sum(~masks["test"][:seq_len]) == seq_len

    def test_get_splits_minimum_size_validation(self):
        """Test minimum split size validation.

        **PHM Logic**: Each split must be at least seq_len + pred_len to
        form a single valid sample for training/evaluation.

        **Methodology**: Create data too small for splits with integer sizes.

        **Expected**: AssertionError about split being too short.

        Validates: Requirement TS-2.7 - Minimum size validation
        """
        # Use integer sizes for explicit control
        splitter = TimeSplitter(
            train=100,
            val=50,
            test=50,  # Total 200 samples
            seq_len=400,
            pred_len=100,  # Requires 500 samples per split
        )
        # Only 200 total samples - not enough for any split
        data = np.random.randn(200, 5)

        with pytest.raises(AssertionError, match="split too short"):
            splitter.get_splits(data)

    def test_get_splits_test_size_specified(self):
        """Test splitting with explicit test size.

        **PHM Logic**: Sometimes test size is fixed (e.g., 100 samples for
        standardized evaluation) while train/val are proportional.

        **Methodology**: Specify test as integer with train/val as floats.
        Note: This actually requires all to be integers per implementation.

        **Expected**: Warning about specified test size.

        Validates: Requirement TS-2.8 - Explicit test size handling
        """
        splitter = TimeSplitter(train=500, val=200, test=200, seq_len=50, pred_len=25)
        data = np.random.randn(1000, 5)  # 100 extra samples

        with pytest.warns(ValueWarning, match="Test size is specified"):
            splits, masks = splitter.get_splits(data)


class TestTimeSplitterSplitData:
    """Tests for split_data method."""

    def test_split_data_basic(self, sample_time_series_data):
        """Test basic data splitting functionality.

        **PHM Logic**: split_data applies the computed splits to a data
        dictionary, extracting the appropriate ranges for each split.

        **Methodology**: Split data dict and verify structure.

        **Expected**: Dict with train/val/test keys, each containing data slice.

        Validates: Requirement TS-3.1 - Basic split_data functionality
        """
        # Use test=None so float ratios work correctly
        splitter = TimeSplitter(train=0.7, val=0.15, test=None, seq_len=50, pred_len=25)
        data_dict = {
            "features": sample_time_series_data["features"],
            "timestamps": sample_time_series_data["timestamps"],
        }

        splitted_data, split_masks = splitter.split_data(data_dict, "features")

        assert "train" in splitted_data
        assert "val" in splitted_data
        assert "test" in splitted_data

        # Verify data types preserved
        assert isinstance(splitted_data["train"], np.ndarray)
        assert isinstance(splitted_data["val"], np.ndarray)
        assert isinstance(splitted_data["test"], np.ndarray)

    def test_split_data_caches_splits(self, sample_time_series_data):
        """Test that split computation is cached.

        **PHM Logic**: splits_dict is computed once and reused for
        subsequent split_data calls, ensuring consistency.

        **Methodology**: Call split_data twice, verify same splits used.

        **Expected**: splits_dict computed on first call, reused on second.

        Validates: Requirement TS-3.2 - Split caching
        """
        # Use test=None so float ratios work correctly
        splitter = TimeSplitter(train=0.7, val=0.15, test=None, seq_len=50, pred_len=25)
        data_dict = {
            "features": sample_time_series_data["features"],
            "timestamps": sample_time_series_data["timestamps"],
        }

        # First call computes splits
        assert splitter.splits_dict is None
        splitter.split_data(data_dict, "features")
        first_splits = splitter.splits_dict.copy()

        # Second call should use cached splits
        splitter.split_data(data_dict, "features")
        assert splitter.splits_dict == first_splits

    def test_split_data_missing_key_error(self, sample_time_series_data):
        """Test error handling for missing split variable.

        **PHM Logic**: split_variable must exist in data_dict.

        **Methodology**: Request split on non-existent key.

        **Expected**: KeyError raised.

        Validates: Requirement TS-3.3 - Missing key validation
        """
        splitter = TimeSplitter(train=0.7, val=0.15, test=0.15)
        data_dict = {"features": sample_time_series_data["features"]}

        with pytest.raises(KeyError):
            splitter.split_data(data_dict, "nonexistent_key")


class TestTimeSplitterRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test __repr__ method.

        **PHM Logic**: Repr should show configuration for debugging.

        **Methodology**: Create splitter and check repr.

        **Expected**: Repr contains train/val/test values.

        Validates: Requirement TS-4.1 - String representation
        """
        splitter = TimeSplitter(train=0.7, val=0.15, test=0.15)
        repr_str = repr(splitter)

        assert "0.7" in repr_str
        assert "0.15" in repr_str
        assert "SimpleSplitter" in repr_str or "TimeSplitter" in repr_str


class TestTimeSplitterEdgeCases:
    """Edge case tests for TimeSplitter."""

    def test_test_none_uses_remaining(self):
        """Test that test=None uses remaining data.

        **PHM Logic**: If test is None, test split gets whatever is left
        after train and val.

        **Methodology**: Set only train and val, verify test uses remainder.

        **Expected**: All data used, no gaps.

        Validates: Requirement TS-5.1 - Automatic test size
        """
        splitter = TimeSplitter(train=0.7, val=0.15, test=None, seq_len=50, pred_len=25)
        data = np.random.randn(1000, 5)

        splits, masks = splitter.get_splits(data)

        # Test should get remaining 15%
        # Verify all 1000 samples are covered (accounting for overlaps)
        train_len = splits["train"][1] - splits["train"][0]
        val_effective = np.sum(masks["val"])
        test_effective = np.sum(masks["test"])

        total_effective = train_len + val_effective + test_effective
        assert total_effective == 1000

    def test_pred_len_greater_than_seq_len(self):
        """Test behavior when pred_len > seq_len.

        **PHM Logic**: When pred_len > seq_len, lookback = seq_len which
        is still positive, so no error is raised by default.

        **Methodology**: Create splitter with pred_len > seq_len.

        **Expected**: No error raised (lookback still positive).

        Validates: Requirement TS-5.2 - pred_len behavior
        """
        # Use test=None so float ratios work correctly
        splitter = TimeSplitter(
            train=0.7,
            val=0.15,
            test=None,
            seq_len=50,
            pred_len=100,  # pred_len > seq_len
        )
        data = np.random.randn(1000, 5)

        # Implementation uses lookback = seq_len (not seq_len - pred_len)
        # so this doesn't raise ValueError
        splits, masks = splitter.get_splits(data)

        # Should still produce valid splits
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
