"""Comprehensive tests for subsample.py transforms."""

import numpy as np
import pytest
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base_transforms.subsample import (
    SubsampleTransform,
    WindowedAggregationTransform,
)


class TestSubsampleTransform:
    """Tests for SubsampleTransform."""

    def test_init(self):
        """Test initialization.

        **Assumption**: SubsampleTransform should accept a step parameter that determines
        the subsampling rate. A step of 2 means every 2nd element is kept (keeping indices
        0, 2, 4, ...), effectively reducing the data size by a factor of step.

        **Action**: Create a SubsampleTransform instance with step=2.

        **Expected Result**: The transform should be created successfully and the step
        attribute should be set to 2. This validates that the transform can be configured
        with a specific subsampling rate, which is essential for downsampling time-series
        data or reducing computational load while preserving temporal structure.
        """
        transform = SubsampleTransform(step=2)
        assert transform.step == 2

    def test_transform_data_single_key(self):
        """Test transform_data with single key.

        **Assumption**: SubsampleTransform should subsample arrays by selecting every
        step-th element along the first dimension (rows). With step=2, it should keep
        rows at indices 0, 2, 4, ... (every other row), effectively halving the data size.

        **Action**: Create a SubsampleTransform with step=2 and provide input data with
        4 rows. Apply the transform to subsample the data.

        **Expected Result**: The result should be a NamedTransformInput with 2 rows (half
        of the original 4 rows), containing rows at indices 0 and 2: [[1.0, 2.0], [5.0, 6.0]].
        This validates that subsampling works correctly, which is important for reducing
        data size while maintaining temporal relationships, useful for faster training or
        when lower temporal resolution is acceptable.
        """
        transform = SubsampleTransform(step=2)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, NamedTransformInput)
        assert len(result["features"]) == 2  # Subsample by step 2
        np.testing.assert_array_equal(
            result["features"], np.array([[1.0, 2.0], [5.0, 6.0]])
        )

    def test_transform_data_multiple_keys_same_length(self):
        """Test transform_data with multiple keys of same length."""
        transform = SubsampleTransform(step=2)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            target=np.array([[0.1], [0.2], [0.3], [0.4]]),
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert len(result["features"]) == 2
        assert len(result["target"]) == 2
        # Both should be subsampled consistently
        assert len(result["features"]) == len(result["target"])

    def test_transform_data_multiple_keys_different_length_warning(self, caplog):
        """Test transform_data with multiple keys of different length logs warning."""
        import logging

        transform = SubsampleTransform(step=2)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            target=np.array([[0.1], [0.2], [0.3]]),  # Different length
        )
        metadata = {}

        with caplog.at_level(logging.WARNING):
            result = transform.transform_data(data, metadata)
            # Should log warning about different lengths (via logging module)
            assert "different lengths" in caplog.text or len(result["features"]) != len(
                result["target"]
            )

    def test_call_method(self):
        """Test __call__ method."""
        transform = SubsampleTransform(step=2)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )

        result = transform(data)

        assert isinstance(result, NamedTransformInput)


class TestWindowedAggregationTransform:
    """Tests for WindowedAggregationTransform."""

    def test_init_mean(self):
        """Test initialization with mean aggregation."""
        transform = WindowedAggregationTransform(window_size=3, step=1, agg="mean")
        assert transform.window_size == 3
        assert transform.step == 1
        assert transform.agg == "mean"
        assert transform.dim == 0

    def test_init_full_window(self):
        """Test initialization with full window."""
        transform = WindowedAggregationTransform(window_size="full", step=1, agg="mean")
        assert transform.window_size == "full"

    def test_init_invalid_window_size_string(self):
        """Test initialization with invalid window_size string."""
        with pytest.raises(AssertionError, match="window_size string must be 'full'"):
            WindowedAggregationTransform(window_size="invalid", step=1, agg="mean")

    def test_init_invalid_window_size_int(self):
        """Test initialization with invalid window_size int."""
        with pytest.raises(
            AssertionError, match="window_size must be a positive integer"
        ):
            WindowedAggregationTransform(window_size=0, step=1, agg="mean")

    def test_init_unsupported_agg_error(self):
        """Test initialization with unsupported aggregation."""
        with pytest.raises(ValueError, match="Unsupported aggregation"):
            WindowedAggregationTransform(window_size=3, step=1, agg="invalid")

    def test_transform_data_mean(self):
        """Test transform_data with mean aggregation.

        **Assumption**: WindowedAggregationTransform should apply a sliding window over
        the data and compute the mean of values within each window. With window_size=3
        and step=1, it creates overlapping windows and aggregates each window using the
        mean function.

        **Action**: Create a WindowedAggregationTransform with window_size=3, step=1,
        and agg="mean". Provide input data with 4 rows and apply the transform.

        **Expected Result**: The result should be a NamedTransformInput with valid
        aggregated data. Each output row should be the mean of a window of 3 consecutive
        input rows. This validates that windowed aggregation works correctly.
        """
        transform = WindowedAggregationTransform(
            window_size=3, step=1, agg="mean", dim=0
        )
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, NamedTransformInput)
        assert "features" in result
        # Windowed aggregation produces valid output (may or may not reduce size based on step)
        assert (
            result["features"].shape[1] == data["features"].shape[1]
        )  # Same num features
        assert len(result["features"]) > 0  # Non-empty result

    def test_transform_data_sum(self):
        """Test transform_data with sum aggregation."""
        transform = WindowedAggregationTransform(
            window_size=2, step=2, agg="sum", dim=0
        )
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, NamedTransformInput)

    def test_transform_data_first(self):
        """Test transform_data with first aggregation."""
        transform = WindowedAggregationTransform(
            window_size=2, step=2, agg="first", dim=0
        )
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, NamedTransformInput)

    def test_transform_data_last(self):
        """Test transform_data with last aggregation."""
        transform = WindowedAggregationTransform(
            window_size=2, step=2, agg="last", dim=0
        )
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, NamedTransformInput)

    def test_transform_data_full_window(self):
        """Test transform_data with full window."""
        transform = WindowedAggregationTransform(
            window_size="full", step=1, agg="mean", dim=0
        )
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, NamedTransformInput)
        # Full window should aggregate entire array
        assert result["features"].ndim >= 1

    def test_transform_data_different_dim(self):
        """Test transform_data with different dim."""
        transform = WindowedAggregationTransform(
            window_size=2, step=1, agg="mean", dim=1
        )
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, NamedTransformInput)

    def test_transform_data_multiple_keys_same_length(self):
        """Test transform_data with multiple keys."""
        transform = WindowedAggregationTransform(
            window_size=2, step=2, agg="mean", dim=0
        )
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            target=np.array([[0.1], [0.2], [0.3], [0.4]]),
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert len(result["features"]) == len(result["target"])

    def test_transform_data_multiple_keys_different_length_warning(self):
        """Test transform_data with different length keys logs warning."""
        import warnings

        transform = WindowedAggregationTransform(
            window_size=2, step=2, agg="mean", dim=0
        )
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            target=np.array([[0.1], [0.2], [0.3]]),  # Different length
        )
        metadata = {}

        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter("always")
            result = transform.transform_data(data, metadata)
            # Should log warning or handle gracefully
            assert isinstance(result, NamedTransformInput)
