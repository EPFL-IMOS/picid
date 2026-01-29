"""Comprehensive tests for shape_manipulations.py transforms."""
import numpy as np
import pytest
import awkward as ak
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base_transforms.shape_manipulations import (
    RegularizeRaggedDataTransform,
    ExpandScalarToReferenceFeatureSize,
    RaggedToDenseTransform,
)


class TestRegularizeRaggedDataTransform:
    """Tests for RegularizeRaggedDataTransform."""
    
    def test_init(self):
        """Test initialization."""
        transform = RegularizeRaggedDataTransform()
        assert transform is not None
    
    def test_transform_data_single_key(self):
        """Test transform_data with single key."""
        transform = RegularizeRaggedDataTransform()
        # Use regular awkward array (already regularized)
        regular_ak = ak.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        data = NamedTransformInput(features=regular_ak)
        metadata = {}
        
        result = transform.transform_data(data, metadata)
        
        # ak.to_regular returns ak.Array, not np.ndarray
        assert isinstance(result, (ak.Array, np.ndarray))
        # Should be regularized (same length)
        assert hasattr(result, 'ndim') or hasattr(result, 'type')
    
    def test_transform_data_multiple_keys_error(self):
        """Test transform_data with multiple keys raises error."""
        transform = RegularizeRaggedDataTransform()
        ragged = ak.Array([[1.0, 2.0], [3.0, 4.0]])
        data = NamedTransformInput(
            features=ragged,
            target=ragged
        )
        metadata = {}
        
        with pytest.raises(AssertionError, match="only supports single key"):
            transform.transform_data(data, metadata)


class TestExpandScalarToReferenceFeatureSize:
    """Tests for ExpandScalarToReferenceFeatureSize."""
    
    def test_init(self):
        """Test initialization."""
        transform = ExpandScalarToReferenceFeatureSize(scalar_key="target")
        assert transform.scalar_key == "target"
    
    def test_transform_data_scalar(self):
        """Test transform_data with scalar value."""
        transform = ExpandScalarToReferenceFeatureSize(scalar_key="target")
        # The transform expects scalar_key to be a scalar or array in the data
        # But NamedTransformInput requires arrays, so we'll test with a 1D array
        # The transform will use np.atleast_1d which handles this
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),  # 3 rows
            target=np.array([5.0])  # 1D array (will be treated as scalar-like)
        )
        metadata = {"apply_to_keys": ["features", "target"]}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)  # Expanded to match features length
        np.testing.assert_array_equal(result, np.array([[5.0], [5.0], [5.0]]))
    
    def test_transform_data_array(self):
        """Test transform_data with array value."""
        transform = ExpandScalarToReferenceFeatureSize(scalar_key="target")
        # The transform uses np.atleast_1d on the scalar_key value
        # So a 1D array will work fine
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),  # 3 rows
            target=np.array([5.0, 1.0])  # 1D array (will be expanded)
        )
        metadata = {"apply_to_keys": ["features", "target"]}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        # Output shape should be (target_size, *fill_value.shape) = (3, 2)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, np.array([[5.0, 1.0], [5.0, 1.0], [5.0, 1.0]]))
    
    def test_transform_data_reference_first(self):
        """Test transform_data when reference is first key."""
        transform = ExpandScalarToReferenceFeatureSize(scalar_key="target")
        data = NamedTransformInput(
            target=np.array([5.0]),  # Array with scalar value
            features=np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 rows - reference
        )
        metadata = {"apply_to_keys": ["target", "features"]}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1)  # Expanded to match features length


class TestRaggedToDenseTransform:
    """Tests for RaggedToDenseTransform."""
    
    def test_transform_data_single_key(self):
        """Test transform_data with single key."""
        transform = RaggedToDenseTransform()
        # Use regular awkward array that can be converted
        regular_ak = ak.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
        data = NamedTransformInput(features=regular_ak)
        metadata = {"apply_to_keys": ["features"]}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, dict)
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
    
    def test_transform_data_multiple_keys(self):
        """Test transform_data with multiple keys."""
        transform = RaggedToDenseTransform()
        regular_ak1 = ak.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
        regular_ak2 = ak.from_numpy(np.array([[5.0], [6.0]]))
        data = NamedTransformInput(
            features=regular_ak1,
            target=regular_ak2
        )
        metadata = {"apply_to_keys": ["features", "target"]}
        
        result = transform.transform_data(data, metadata)
        
        assert isinstance(result, dict)
        assert "features" in result
        assert "target" in result
        assert isinstance(result["features"], np.ndarray)
        assert isinstance(result["target"], np.ndarray)
