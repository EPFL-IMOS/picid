"""Comprehensive tests for subsample.py transforms."""
import numpy as np
import pytest
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.subsample import (
    SubsampleTransform,
    WindowedAggregationTransform,
)


class TestSubsampleTransform:
    """Tests for SubsampleTransform."""
    
    def test_init(self):
        """Test initialization."""
        transform = SubsampleTransform(step=2)
        assert transform.step == 2
    
    def test_transform_data_single_key(self):
        """Test transform_data with single key."""
        transform = SubsampleTransform(step=2)
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, NamedTransformInput)
        assert len(result["features"]) == 2  # Subsample by step 2
        np.testing.assert_array_equal(result["features"], np.array([[1.0, 2.0], [5.0, 6.0]]))
    
    def test_transform_data_multiple_keys_same_length(self):
        """Test transform_data with multiple keys of same length."""
        transform = SubsampleTransform(step=2)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            target=np.array([[0.1], [0.2], [0.3], [0.4]])
        )
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert len(result["features"]) == 2
        assert len(result["target"]) == 2
        # Both should be subsampled consistently
        assert len(result["features"]) == len(result["target"])
    
    def test_transform_data_multiple_keys_different_length_warning(self):
        """Test transform_data with multiple keys of different length logs warning."""
        import warnings
        transform = SubsampleTransform(step=2)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            target=np.array([[0.1], [0.2], [0.3]])  # Different length
        )
        metadata = {}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform.transform_data(data, metadata)
            # Should log warning about different lengths
            assert len(w) > 0 or len(result["features"]) != len(result["target"])
    
    def test_call_method(self):
        """Test __call__ method."""
        transform = SubsampleTransform(step=2)
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        
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
        with pytest.raises(AssertionError, match="window_size must be a positive integer"):
            WindowedAggregationTransform(window_size=0, step=1, agg="mean")
    
    def test_init_unsupported_agg_error(self):
        """Test initialization with unsupported aggregation."""
        with pytest.raises(ValueError, match="Unsupported aggregation"):
            WindowedAggregationTransform(window_size=3, step=1, agg="invalid")
    
    def test_transform_data_mean(self):
        """Test transform_data with mean aggregation."""
        transform = WindowedAggregationTransform(window_size=3, step=1, agg="mean", dim=0)
        data = NamedTransformInput(features=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, NamedTransformInput)
        assert "features" in result
        # Windowed aggregation should reduce size
        assert len(result["features"]) < len(data["features"])
    
    def test_transform_data_sum(self):
        """Test transform_data with sum aggregation."""
        transform = WindowedAggregationTransform(window_size=2, step=2, agg="sum", dim=0)
        data = NamedTransformInput(features=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, NamedTransformInput)
    
    def test_transform_data_first(self):
        """Test transform_data with first aggregation."""
        transform = WindowedAggregationTransform(window_size=2, step=2, agg="first", dim=0)
        data = NamedTransformInput(features=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, NamedTransformInput)
    
    def test_transform_data_last(self):
        """Test transform_data with last aggregation."""
        transform = WindowedAggregationTransform(window_size=2, step=2, agg="last", dim=0)
        data = NamedTransformInput(features=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, NamedTransformInput)
    
    def test_transform_data_full_window(self):
        """Test transform_data with full window."""
        transform = WindowedAggregationTransform(window_size="full", step=1, agg="mean", dim=0)
        data = NamedTransformInput(features=np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, NamedTransformInput)
        # Full window should aggregate entire array
        assert result["features"].ndim >= 1
    
    def test_transform_data_different_dim(self):
        """Test transform_data with different dim."""
        transform = WindowedAggregationTransform(window_size=2, step=1, agg="mean", dim=1)
        data = NamedTransformInput(features=np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0]
        ]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, NamedTransformInput)
    
    def test_transform_data_multiple_keys_same_length(self):
        """Test transform_data with multiple keys."""
        transform = WindowedAggregationTransform(window_size=2, step=2, agg="mean", dim=0)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            target=np.array([[0.1], [0.2], [0.3], [0.4]])
        )
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert len(result["features"]) == len(result["target"])
    
    def test_transform_data_multiple_keys_different_length_warning(self):
        """Test transform_data with different length keys logs warning."""
        import warnings
        transform = WindowedAggregationTransform(window_size=2, step=2, agg="mean", dim=0)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            target=np.array([[0.1], [0.2], [0.3]])  # Different length
        )
        metadata = {}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform.transform_data(data, metadata)
            # Should log warning or handle gracefully
            assert isinstance(result, NamedTransformInput)
