"""Comprehensive tests for padding2length.py transform."""
import numpy as np
import pytest
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.padding2length import PadToLength


class TestPadToLength:
    """Tests for PadToLength transform."""
    
    def test_init(self):
        """Test initialization."""
        transform = PadToLength(target_length=10, axis=0, pad_value=0.0)
        assert transform.target_length == 10
        assert transform.axis == 0
        assert transform.pad_value == 0.0
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        transform = PadToLength(target_length=5)
        assert transform.target_length == 5
        assert transform.axis == 0
        assert transform.pad_value == 0.0
    
    def test_fit_data(self):
        """Test fit_data does nothing."""
        transform = PadToLength(target_length=10)
        data = NamedTransformInput(features=np.array([[1.0, 2.0]]))
        metadata = {}
        
        # Should not raise
        transform.fit_data(data, metadata)
    
    def test_transform_data_no_padding_needed(self):
        """Test transform_data when padding is not needed."""
        transform = PadToLength(target_length=2, axis=0)
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        # Should return original array unchanged
        np.testing.assert_array_equal(result, data["features"])
    
    def test_transform_data_padding_axis_0(self):
        """Test transform_data with padding along axis 0."""
        transform = PadToLength(target_length=5, axis=0, pad_value=0.0)
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert result.shape == (5, 2)
        # Original data should be preserved
        np.testing.assert_array_equal(result[:2, :], data["features"])
        # Padding should be zeros
        np.testing.assert_array_equal(result[2:, :], np.zeros((3, 2)))
    
    def test_transform_data_padding_axis_1(self):
        """Test transform_data with padding along axis 1."""
        transform = PadToLength(target_length=5, axis=1, pad_value=0.0)
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert result.shape == (2, 5)
        # Original data should be preserved
        np.testing.assert_array_equal(result[:, :2], data["features"])
        # Padding should be zeros
        np.testing.assert_array_equal(result[:, 2:], np.zeros((2, 3)))
    
    def test_transform_data_custom_pad_value(self):
        """Test transform_data with custom pad value."""
        transform = PadToLength(target_length=5, axis=0, pad_value=-1.0)
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert result.shape == (5, 2)
        # Padding should be -1.0
        np.testing.assert_array_equal(result[2:, :], np.full((3, 2), -1.0))
    
    def test_transform_data_negative_axis(self):
        """Test transform_data with negative axis indexing."""
        transform = PadToLength(target_length=5, axis=-1, pad_value=0.0)
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        # axis=-1 should be the last axis (axis=1 for 2D array)
        assert result.shape == (2, 5)
    
    def test_transform_data_3d_array(self):
        """Test transform_data with 3D array."""
        transform = PadToLength(target_length=5, axis=1, pad_value=0.0)
        data = NamedTransformInput(features=np.array([[[1.0], [2.0]], [[3.0], [4.0]]]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert result.shape == (2, 5, 1)
    
    def test_transform_data_invalid_axis_error(self):
        """Test transform_data with invalid axis raises error."""
        transform = PadToLength(target_length=5, axis=10, pad_value=0.0)
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        with pytest.raises(ValueError, match="Invalid axis"):
            transform.transform_data(data, metadata)
    
    def test_transform_data_not_numpy_error(self):
        """Test transform_data with non-numpy array raises error."""
        transform = PadToLength(target_length=5)
        data = NamedTransformInput(features=[[1.0, 2.0], [3.0, 4.0]])  # List, not array
        metadata = {}
        
        with pytest.raises(TypeError, match="expects numpy array"):
            transform.transform_data(data, metadata)
