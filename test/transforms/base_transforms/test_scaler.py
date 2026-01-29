"""Comprehensive tests for scaler.py transforms."""
import numpy as np
import pytest
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.scaler import (
    ConstantScaler,
    MinMaxScalerSklearn,
    StandardScalerSklearn,
)


class TestConstantScaler:
    """Tests for ConstantScaler."""
    
    def test_init_default(self):
        """Test initialization with default factor."""
        transform = ConstantScaler()
        assert transform.factor == 1.0
    
    def test_init_custom_factor(self):
        """Test initialization with custom factor."""
        transform = ConstantScaler(factor=2.5)
        assert transform.factor == 2.5
    
    def test_fit_data(self):
        """Test fit_data does nothing."""
        transform = ConstantScaler(factor=2.0)
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        result = transform.fit_data(data, metadata)
        assert result is transform
    
    def test_transform_data(self):
        """Test transform_data multiplies by factor."""
        transform = ConstantScaler(factor=2.0)
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        expected = np.array([[2.0, 4.0], [6.0, 8.0]])
        np.testing.assert_array_equal(result, expected)
    
    def test_transform_data_multiple_keys_error(self):
        """Test transform_data with multiple keys raises error."""
        transform = ConstantScaler(factor=2.0)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0]]),
            target=np.array([[0.5]])
        )
        metadata = {}
        
        with pytest.raises(AssertionError, match="only supports single key"):
            transform.transform_data(data, metadata)
    
    def test_inverse_transform(self):
        """Test inverse_transform divides by factor."""
        transform = ConstantScaler(factor=2.0)
        data = NamedTransformInput(features=np.array([[2.0, 4.0], [6.0, 8.0]]))
        metadata = {}
        
        result = transform.inverse_transform(data, metadata)
        
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(result, expected)
    
    def test_inverse_transform_multiple_keys_error(self):
        """Test inverse_transform with multiple keys raises error."""
        transform = ConstantScaler(factor=2.0)
        data = NamedTransformInput(
            features=np.array([[1.0]]),
            target=np.array([[0.5]])
        )
        metadata = {}
        
        with pytest.raises(AssertionError, match="only supports single key"):
            transform.inverse_transform(data, metadata)
    
    def test_fit_multi_source(self):
        """Test fit_multi_source."""
        transform = ConstantScaler(factor=2.0)
        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0]])),
            NamedTransformInput(features=np.array([[3.0, 4.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}
        
        # Should not raise
        transform.fit_multi_source(data_segments, metadata)


class TestMinMaxScalerSklearn:
    """Tests for MinMaxScalerSklearn."""
    
    def test_init(self):
        """Test initialization."""
        transform = MinMaxScalerSklearn()
        assert transform.scaler is not None
    
    def test_fit_data_2d(self):
        """Test fit_data with 2D array."""
        transform = MinMaxScalerSklearn()
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        metadata = {}
        
        transform.fit_data(data, metadata)
        
        # Verify scaler was fitted
        assert hasattr(transform.scaler, "data_min_")
        assert hasattr(transform.scaler, "data_max_")
    
    def test_fit_data_1d(self):
        """Test fit_data with 1D array (reshaped to 2D)."""
        transform = MinMaxScalerSklearn()
        data = NamedTransformInput(features=np.array([1.0, 2.0, 3.0, 4.0]))
        metadata = {}
        
        transform.fit_data(data, metadata)
        
        # Verify scaler was fitted
        assert hasattr(transform.scaler, "data_min_")
    
    def test_fit_data_multiple_keys_error(self):
        """Test fit_data with multiple keys raises error."""
        transform = MinMaxScalerSklearn()
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0]]),
            target=np.array([[0.5]])
        )
        metadata = {}
        
        with pytest.raises(AssertionError, match="only supports single key"):
            transform.fit_data(data, metadata)
    
    def test_transform_data_2d(self):
        """Test transform_data with 2D array."""
        transform = MinMaxScalerSklearn()
        # Fit first
        fit_data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        transform.fit_data(fit_data, {})
        
        # Transform
        data = NamedTransformInput(features=np.array([[2.0, 3.0], [4.0, 5.0]]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        # MinMaxScaler scales to [0, 1]
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_transform_data_1d(self):
        """Test transform_data with 1D array."""
        transform = MinMaxScalerSklearn()
        # Fit first
        fit_data = NamedTransformInput(features=np.array([1.0, 2.0, 3.0, 4.0]))
        transform.fit_data(fit_data, {})
        
        # Transform
        data = NamedTransformInput(features=np.array([2.0, 3.0]))
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # Should be reshaped to 2D
        assert result.shape[1] == 1
    
    def test_transform_data_not_fitted_error(self):
        """Test transform_data without fitting raises error."""
        transform = MinMaxScalerSklearn()
        data = NamedTransformInput(features=np.array([[1.0, 2.0]]))
        metadata = {}
        
        # Should raise error if not fitted
        with pytest.raises((ValueError, AttributeError)):
            transform.transform_data(data, metadata)
    
    def test_inverse_transform(self):
        """Test inverse_transform."""
        transform = MinMaxScalerSklearn()
        # Fit first
        fit_data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        transform.fit_data(fit_data, {})
        
        # Transform
        original = np.array([[2.0, 3.0], [4.0, 5.0]])
        data = NamedTransformInput(features=original)
        transformed = transform.transform_data(data, {})
        
        # Inverse transform
        inverse_data = NamedTransformInput(features=transformed)
        result = transform.inverse_transform(inverse_data, {})
        
        # Should recover original (approximately)
        np.testing.assert_allclose(result, original, rtol=1e-10)
    
    def test_fit_multi_source(self):
        """Test fit_multi_source."""
        transform = MinMaxScalerSklearn()
        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0]])),
            NamedTransformInput(features=np.array([[3.0, 4.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}
        
        transform.fit_multi_source(data_segments, metadata)
        
        # Verify scaler was fitted
        assert hasattr(transform.scaler, "data_min_")


class TestStandardScalerSklearn:
    """Tests for StandardScalerSklearn."""
    
    def test_init(self):
        """Test initialization."""
        transform = StandardScalerSklearn()
        assert transform.scaler is not None
    
    def test_fit_data_2d(self):
        """Test fit_data with 2D array."""
        transform = StandardScalerSklearn()
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        metadata = {}
        
        transform.fit_data(data, metadata)
        
        # Verify scaler was fitted
        assert hasattr(transform.scaler, "mean_")
        assert hasattr(transform.scaler, "scale_")
    
    def test_fit_data_1d(self):
        """Test fit_data with 1D array."""
        transform = StandardScalerSklearn()
        data = NamedTransformInput(features=np.array([1.0, 2.0, 3.0, 4.0]))
        metadata = {}
        
        transform.fit_data(data, metadata)
        
        # Verify scaler was fitted
        assert hasattr(transform.scaler, "mean_")
    
    def test_transform_data_2d(self):
        """Test transform_data with 2D array."""
        transform = StandardScalerSklearn()
        # Fit first
        fit_data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        transform.fit_data(fit_data, {})
        
        # Transform
        data = NamedTransformInput(features=np.array([[2.0, 3.0], [4.0, 5.0]]))
        metadata = {"assign_to_map": ["features"]}  # Required by StandardScaler
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        # StandardScaler should have mean ~0 and std ~1
        assert np.abs(np.mean(result)) < 1.0  # Approximate check
    
    def test_inverse_transform(self):
        """Test inverse_transform."""
        transform = StandardScalerSklearn()
        # Fit first
        fit_data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        transform.fit_data(fit_data, {})
        
        # Transform
        original = np.array([[2.0, 3.0], [4.0, 5.0]])
        data = NamedTransformInput(features=original)
        transformed = transform.transform_data(data, {"assign_to_map": ["features"]})
        
        # Inverse transform
        inverse_data = NamedTransformInput(features=transformed)
        result = transform.inverse_transform(inverse_data, {})
        
        # Should recover original (approximately)
        np.testing.assert_allclose(result, original, rtol=1e-10)
    
    def test_fit_multi_source(self):
        """Test fit_multi_source."""
        transform = StandardScalerSklearn()
        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0]])),
            NamedTransformInput(features=np.array([[3.0, 4.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}
        
        transform.fit_multi_source(data_segments, metadata)
        
        # Verify scaler was fitted
        assert hasattr(transform.scaler, "mean_")
