"""Comprehensive tests for tabularizers.py to reach 90% coverage."""
import numpy as np
import pytest
from omegaconf import OmegaConf
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.tabularizers import TimeseriesTabularizer


class TestTimeseriesTabularizerComprehensive:
    """Comprehensive tests for TimeseriesTabularizer to reach 90% coverage."""
    
    def test_init_with_list_select_features(self):
        """Test initialization with list of dicts for select_features."""
        select_features = [{"features": "t"}, {"target": "history"}]
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        assert "features" in transform.select_features
        assert "target" in transform.select_features
    
    def test_init_with_dict_select_features(self):
        """Test initialization with dict for select_features."""
        select_features = {"features": "t", "target": "history"}
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        assert "features" in transform.select_features
        assert "target" in transform.select_features
    
    def test_transform_data_collect_sequences_subset(self):
        """Test transform_data with subset_ratio for collect_sequences."""
        select_features = OmegaConf.create([{"features": "t"}])
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=2,
            subset_ratio=0.6,
            subset_seed=42
        )
        # Create enough data for subsetting
        data = NamedTransformInput(features=np.random.randn(5, 100, 3))
        metadata = {"mode": "train"}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
    
    def test_transform_data_collect_sequences_no_subset(self):
        """Test transform_data without subset_ratio."""
        select_features = OmegaConf.create([{"features": "t"}])
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1,
            subset_ratio=None
        )
        data = NamedTransformInput(features=np.random.randn(5, 30, 3))
        metadata = {"mode": "train"}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
    
    def test_transform_data_rearrange_t_selection(self):
        """Test transform_data with 't' selection rearrangement."""
        select_features = OmegaConf.create([{"features": "t"}])
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=3,  # Creates 3 tasks
            stride=1
        )
        data = NamedTransformInput(features=np.random.randn(5, 30, 3))
        metadata = {"mode": "train"}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
    
    def test_transform_data_rearrange_present_selection(self):
        """Test transform_data with 'present' selection rearrangement."""
        select_features = OmegaConf.create([{"features": "present"}])
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        data = NamedTransformInput(features=np.random.randn(5, 30, 3))
        metadata = {"mode": "train"}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
    
    def test_transform_data_rearrange_history_selection(self):
        """Test transform_data with 'history' selection rearrangement."""
        select_features = OmegaConf.create([{"features": "history"}])
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        data = NamedTransformInput(features=np.random.randn(5, 30, 3))
        metadata = {"mode": "train"}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
    
    def test_transform_data_rearrange_horizon_selection(self):
        """Test transform_data with 'horizon' selection rearrangement."""
        select_features = OmegaConf.create([{"target": "horizon"}])
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=3,  # 3-step horizon
            stride=1
        )
        data = NamedTransformInput(target=np.random.randn(5, 30, 1))
        metadata = {"mode": "train"}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
    
    def test_transform_data_padding_history_present(self):
        """Test transform_data with history/present that needs padding."""
        select_features = OmegaConf.create([{"features": "t"}, {"target": "history"}])
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=3,  # Creates 3 tasks, history needs padding
            stride=1
        )
        data = NamedTransformInput(
            features=np.random.randn(5, 30, 3),
            target=np.random.randn(5, 30, 1)
        )
        metadata = {"mode": "train"}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
    
    def test_transform_data_concatenation_multiple_keys(self):
        """Test transform_data with multiple keys that get concatenated."""
        select_features = OmegaConf.create([{"features": "t"}, {"target": "history"}])
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1
        )
        data = NamedTransformInput(
            features=np.random.randn(5, 30, 3),
            target=np.random.randn(5, 30, 1)
        )
        metadata = {"mode": "train"}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # Should concatenate along feature dimension
        assert result.ndim >= 2
