"""Comprehensive tests for tabularizers.py transform."""
import numpy as np
import pytest
from omegaconf import OmegaConf
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.tabularizers import TimeseriesTabularizer


class TestTimeseriesTabularizer:
    """Tests for TimeseriesTabularizer."""
    
    def test_init(self):
        """Test initialization."""
        select_features = {"features": "t", "target": "history"}
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        assert transform.timestep_dimension == 1
        assert transform.seq_len == 10
        assert transform.label_len == 2
        assert transform.pred_len == 1
        assert transform.stride == 1
    
    def test_init_with_horizon(self):
        """Test initialization with horizon in select_features."""
        select_features = {"target": "horizon"}
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        assert len(transform.select_features) == 1
    
    def test_init_horizon_multiple_keys_error(self):
        """Test initialization with horizon and multiple keys raises error."""
        select_features = {"features": "t", "target": "horizon"}
        
        with pytest.raises(AssertionError, match="If 'horizon' is present"):
            TimeseriesTabularizer(
                select_features=select_features,
                timestep_dimension=1,
                seq_len=10,
                label_len=2,
                pred_len=1,
                stride=1
            )
    
    def test_init_with_subset_ratio(self):
        """Test initialization with subset_ratio."""
        select_features = {"features": "t"}
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1,
            subset_ratio=0.5,
            subset_seed=42
        )
        assert transform.subset_ratio == 0.5
        assert transform.subset_seed == 42
    
    def test_repr(self):
        """Test __repr__ method."""
        select_features = {"features": "t"}
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        repr_str = repr(transform)
        assert "TimeseriesTabularizer" in repr_str
    
    def test_fit_data(self):
        """Test fit_data."""
        select_features = {"features": "t"}
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        # Create 3D data: (batch, time, features)
        data = NamedTransformInput(features=np.random.randn(5, 20, 3))
        metadata = {}
        
        # Should not raise
        transform.fit_data(data, metadata)
    
    def test_transform_data_time_features(self):
        """Test transform_data with time features selection."""
        select_features = {"features": "t"}
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        # Create 3D data: (batch, time, features)
        data = NamedTransformInput(features=np.random.randn(5, 20, 3))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # Should be tabularized (2D)
        assert result.ndim == 2
    
    def test_transform_data_history(self):
        """Test transform_data with history selection."""
        select_features = {"target": "history"}
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        data = NamedTransformInput(target=np.random.randn(5, 20, 1))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
    
    def test_transform_data_multiple_keys(self):
        """Test transform_data with multiple keys."""
        select_features = {"features": "t", "target": "history"}
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        data = NamedTransformInput(
            features=np.random.randn(5, 20, 3),
            target=np.random.randn(5, 20, 1)
        )
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # Should concatenate features and target
        assert result.ndim == 2
    
    def test_fit_multi_source(self):
        """Test fit_multi_source."""
        select_features = {"features": "t"}
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        data_segments = [
            NamedTransformInput(features=np.random.randn(5, 20, 3)),
            NamedTransformInput(features=np.random.randn(5, 20, 3)),
        ]
        metadata = {"apply_to_keys": ["features"]}
        
        # Should not raise
        transform.fit_multi_source(data_segments, metadata)
