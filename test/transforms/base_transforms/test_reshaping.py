"""Comprehensive tests for reshaping.py transform."""
import numpy as np
import pytest
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.reshaping import ReshapeTransform


class TestReshapeTransform:
    """Tests for ReshapeTransform."""
    
    def test_init(self):
        """Test initialization."""
        transform = ReshapeTransform(pattern="b h w -> (b h) w")
        assert transform.pattern == "b h w -> (b h) w"
    
    def test_transform_data_simple_reshape(self):
        """Test transform_data with simple reshape pattern."""
        transform = ReshapeTransform(pattern="h w -> (h w)")
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)  # Flattened
        
    def test_transform_data_3d_reshape(self):
        """Test transform_data with 3D reshape."""
        transform = ReshapeTransform(pattern="b h w -> b (h w)")
        arr_3d = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        data = NamedTransformInput(features=arr_3d)
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 4)  # (b, h*w)
    
    def test_transform_data_multiple_keys_error(self):
        """Test transform_data with multiple keys raises error."""
        transform = ReshapeTransform(pattern="h w -> (h w)")
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0]]),
            target=np.array([[0.5]])
        )
        metadata = {}
        
        with pytest.raises(AssertionError, match="exactly one entry"):
            transform.transform_data(data, metadata)
    
    def test_call_method(self):
        """Test __call__ method."""
        transform = ReshapeTransform(pattern="h w -> (h w)")
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        result = transform(data, metadata)
        
        assert isinstance(result, np.ndarray)
