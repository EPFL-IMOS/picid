"""Comprehensive tests for identity.py transform."""
import numpy as np
import pytest
import awkward as ak
from lmetk.data.data_objects.data import NamedTransformInput, SplitDatasetContainer
from lmetk.transforms.base_transforms.identity import IdentityPassThrough
from test.transforms.base.test_base_comprehensive import create_dummy_split_container


class TestIdentityPassThrough:
    """Tests for IdentityPassThrough transform."""
    
    def test_init(self):
        """Test initialization."""
        transform = IdentityPassThrough()
        assert transform is not None
        
        # Test with args and kwargs (should be ignored)
        transform2 = IdentityPassThrough(1, 2, 3, a=4, b=5)
        assert transform2 is not None
    
    def test_fit_data_does_nothing(self):
        """Test that fit_data does nothing."""
        transform = IdentityPassThrough()
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        # Should not raise
        result = transform.fit_data(data, metadata)
        assert result is None
    
    def test_transform_data_returns_input(self):
        """Test that transform_data returns input unchanged."""
        transform = IdentityPassThrough()
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        # Should return the same object
        assert result is data
        np.testing.assert_array_equal(result["features"], data["features"])
    
    def test_transform_data_with_multiple_keys(self):
        """Test transform_data with multiple keys."""
        transform = IdentityPassThrough()
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0]]),
            target=np.array([[0.5]])
        )
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert result is data
        assert "features" in result
        assert "target" in result
    
    def test_transform_data_with_ragged_array(self):
        """Test transform_data with ragged arrays."""
        transform = IdentityPassThrough()
        ragged = ak.Array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0]
        ])
        data = NamedTransformInput(features=ragged)
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert result is data
        assert isinstance(result["features"], ak.Array)
    
    def test_fit_multi_source(self):
        """Test fit_multi_source method."""
        transform = IdentityPassThrough()
        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0]])),
            NamedTransformInput(features=np.array([[3.0, 4.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}
        
        # Should not raise
        transform.fit_multi_source(data_segments, metadata)
    
    def test_transform_multi_source(self):
        """Test transform_multi_source method."""
        transform = IdentityPassThrough()
        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0]])),
            NamedTransformInput(features=np.array([[3.0, 4.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}
        
        result, log = transform.transform_multi_source(data_segments, metadata)
        
        assert isinstance(result, list)
        # transform_function_to_data_segments processes each segment individually
        assert len(result) == 2  # One result per input segment
        assert isinstance(log, dict)
        # Each result should be the same as input (identity transform)
        assert "features" in result[0]
        np.testing.assert_array_equal(result[0]["features"], data_segments[0]["features"])
        np.testing.assert_array_equal(result[1]["features"], data_segments[1]["features"])
    
    def test_with_split_dataset_container(self):
        """Test with SplitDatasetContainer."""
        transform = IdentityPassThrough()
        container = create_dummy_split_container(n_units=2)
        
        # Test on train split
        train_data = container.features.train[0]
        data = NamedTransformInput(features=train_data)
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        assert result is data
        np.testing.assert_array_equal(result["features"], train_data)
    
    def test_empty_data(self):
        """Test with empty data."""
        transform = IdentityPassThrough()
        data = NamedTransformInput()
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        assert result is data
        assert len(result) == 0
