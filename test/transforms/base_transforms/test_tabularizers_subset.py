"""Tests for tabularizers.py subset_ratio paths to reach 99% coverage."""
import numpy as np
import pytest
from omegaconf import OmegaConf
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.tabularizers import TimeseriesTabularizer


class TestTimeseriesTabularizerSubset:
    """Tests for subset_ratio code paths."""
    
    def test_transform_data_subset_ratio_with_logging(self):
        """Test transform_data with subset_ratio that triggers logging."""
        select_features = OmegaConf.create([{"features": "t"}])
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1,
            subset_ratio=0.5,  # 50% subset
            subset_seed=42
        )
        # Use 2D array (Time, Features) with enough data for subsetting
        # Need enough sequences: with 50 time steps, seq_len=10, label_len=2, pred_len=1
        # We get about 40 sequences, so subset_ratio=0.5 gives ~20 sequences
        data = NamedTransformInput(features=np.random.randn(100, 3))  # More data for subsetting
        metadata = {"mode": "train"}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
    
    def test_transform_data_subset_ratio_batch_idx_logging(self):
        """Test transform_data with subset_ratio that logs batch_idx[:100]."""
        select_features = OmegaConf.create([{"features": "t"}])
        transform = TimeseriesTabularizer(
            select_features=select_features,
            timestep_dimension=1,
            seq_len=10,
            label_len=2,
            pred_len=1,
            stride=1,
            subset_ratio=0.3,  # 30% subset
            subset_seed=42
        )
        # Create enough data to get >100 sequences so batch_idx[:100] is logged
        # With 200 time steps, we get ~190 sequences, so 30% = ~57 sequences
        # Actually need >100 sequences total to trigger batch_idx[:100] logging
        data = NamedTransformInput(features=np.random.randn(200, 3))
        metadata = {"mode": "train"}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2
