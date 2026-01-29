"""Tests for base transform functionality."""
import numpy as np
import pytest
from lmetk.transforms.base.base_transform import BaseTransform, DenseTransform, RaggedTransform
from lmetk.data.data_objects.data import NamedTransformInput


class DummyTransform(BaseTransform):
    """A dummy transform for testing BaseTransform interface."""
    
    def transform_data(self, data: NamedTransformInput, metadata: dict):
        """Simple transform that doubles the values."""
        key = list(data.keys())[0]
        return data[key] * 2


class DummyDenseTransform(DenseTransform):
    """A dummy dense transform for testing."""
    
    def transform_data(self, data: NamedTransformInput, metadata: dict):
        """Simple transform that adds 1."""
        key = list(data.keys())[0]
        return data[key] + 1


class TestBaseTransform:
    """Test suite for BaseTransform interface."""
    
    def test_base_transform_initialization(self):
        """Test that BaseTransform can be initialized."""
        transform = DummyTransform()
        assert transform is not None
        assert transform.exclude_keys == []
    
    def test_base_transform_with_exclude_keys(self):
        """Test BaseTransform with exclude_keys."""
        transform = DummyTransform(exclude_keys=["target"])
        assert "target" in transform.exclude_keys
    
    def test_base_transform_call(self):
        """Test that __call__ method works."""
        transform = DummyTransform()
        data = NamedTransformInput(features=np.array([1.0, 2.0, 3.0]))
        result = transform(data)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_base_transform_repr(self):
        """Test string representation of transform."""
        transform = DummyTransform(exclude_keys=["target"], some_param=42)
        repr_str = repr(transform)
        assert "DummyTransform" in repr_str
        assert "exclude_keys" in repr_str


class TestDenseTransform:
    """Test suite for DenseTransform marker class."""
    
    def test_dense_transform_inheritance(self):
        """Test that DenseTransform is a BaseTransform."""
        transform = DummyDenseTransform()
        assert isinstance(transform, BaseTransform)
        assert isinstance(transform, DenseTransform)
