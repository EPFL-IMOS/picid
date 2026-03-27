"""Comprehensive tests for concatenate.py transforms."""

import numpy as np
import pytest
import awkward as ak
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base_transforms.concatenate import (
    ConcatenateTransform,
    RuggedToDenseTransform,
    MultiDatasetRuggedToDenseTransform,
)


class TestConcatenateTransform:
    """Tests for ConcatenateTransform."""

    def test_init_dim_1(self):
        """Test initialization with dim=1.

        **Assumption**: ConcatenateTransform should accept dim=1 to concatenate arrays
        along the second dimension (columns/features). This is the most common use case
        for concatenating feature arrays horizontally.

        **Action**: Create a ConcatenateTransform instance with dim=1.

        **Expected Result**: The transform should be created successfully and the dim
        attribute should be set to 1. This validates that the transform can be configured
        for horizontal concatenation, which is essential for combining multiple feature
        arrays into a single feature matrix.
        """
        transform = ConcatenateTransform(dim=1)
        assert transform.dim == 1

    def test_init_dim_2(self):
        """Test initialization with dim=2.

        **Assumption**: ConcatenateTransform should accept dim=2 to concatenate arrays
        along the third dimension. This is useful for 3D arrays where we want to combine
        features along the depth dimension (e.g., combining multiple time-series features).

        **Action**: Create a ConcatenateTransform instance with dim=2.

        **Expected Result**: The transform should be created successfully and the dim
        attribute should be set to 2. This validates that the transform supports
        3D concatenation, which is important for time-series or multi-dimensional
        feature combination scenarios.
        """
        transform = ConcatenateTransform(dim=2)
        assert transform.dim == 2

    def test_init_invalid_dim(self):
        """Test initialization with invalid dim.

        **Assumption**: ConcatenateTransform should only support concatenation along
        dimensions 1 or 2 (not 0 or 3+). Dimension 0 would concatenate along samples
        (rows), which doesn't make sense for this transform's use case. Dimensions
        3+ are not supported to keep the implementation manageable.

        **Action**: Attempt to create ConcatenateTransform instances with invalid dim
        values: dim=0 (concatenate along samples) and dim=3 (unsupported dimension).

        **Expected Result**: Both should raise AssertionError with appropriate error
        messages. This validates that the transform enforces its constraints and
        prevents misuse, which is important for catching configuration errors early
        rather than producing incorrect results.
        """
        with pytest.raises(
            AssertionError, match="supports concatination along dim 1 or 2"
        ):
            ConcatenateTransform(dim=0)

        with pytest.raises(AssertionError):
            ConcatenateTransform(dim=3)

    def test_convert_to_numpy(self):
        """Test _convert_to_numpy helper."""
        transform = ConcatenateTransform(dim=1)

        # Already numpy array
        arr = np.array([[1.0, 2.0]])
        result = transform._convert_to_numpy(arr, "test")
        np.testing.assert_array_equal(result, arr)

        # List conversion
        lst = [[1.0, 2.0], [3.0, 4.0]]
        result = transform._convert_to_numpy(lst, "test")
        assert isinstance(result, np.ndarray)

    def test_convert_to_numpy_error(self):
        """Test _convert_to_numpy with invalid input."""
        transform = ConcatenateTransform(dim=1)

        # Create something that will fail conversion
        # Use a nested structure that numpy can't handle properly
        # Actually, numpy can convert most things, so let's test with None which might fail
        # Or use a mock that raises an exception
        class Unconvertible:
            def __array__(self):
                raise TypeError("Cannot convert to array")

        unconvertible = Unconvertible()
        with pytest.raises((ValueError, TypeError), match="Could not convert"):
            transform._convert_to_numpy(unconvertible, "test")

    def test_ensure_2d_1d_input(self):
        """Test _ensure_2d with 1D input."""
        transform = ConcatenateTransform(dim=1)
        arr_1d = np.array([1.0, 2.0, 3.0])
        result = transform._ensure_2d(arr_1d, "test")
        assert result.ndim == 2
        assert result.shape == (3, 1)

    def test_ensure_2d_2d_input(self):
        """Test _ensure_2d with 2D input."""
        transform = ConcatenateTransform(dim=1)
        arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = transform._ensure_2d(arr_2d, "test")
        np.testing.assert_array_equal(result, arr_2d)

    def test_ensure_2d_3d_error(self):
        """Test _ensure_2d with 3D input raises error."""
        transform = ConcatenateTransform(dim=1)
        arr_3d = np.array([[[1.0]]])

        with pytest.raises(ValueError, match="unsupported dimensions"):
            transform._ensure_2d(arr_3d, "test")

    def test_ensure_3d_1d_input(self):
        """Test _ensure_3d with 1D input."""
        transform = ConcatenateTransform(dim=2)
        arr_1d = np.array([1.0, 2.0])
        result = transform._ensure_3d(arr_1d, "test")
        assert result.ndim == 3
        assert result.shape == (2, 1, 1)

    def test_ensure_3d_2d_input(self):
        """Test _ensure_3d with 2D input."""
        transform = ConcatenateTransform(dim=2)
        arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = transform._ensure_3d(arr_2d, "test")
        assert result.ndim == 3
        assert result.shape == (2, 2, 1)

    def test_ensure_3d_3d_input(self):
        """Test _ensure_3d with 3D input."""
        transform = ConcatenateTransform(dim=2)
        arr_3d = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        result = transform._ensure_3d(arr_3d, "test")
        np.testing.assert_array_equal(result, arr_3d)

    def test_ensure_3d_4d_error(self):
        """Test _ensure_3d with 4D input raises error."""
        transform = ConcatenateTransform(dim=2)
        arr_4d = np.array([[[[1.0]]]])

        with pytest.raises(ValueError, match="unsupported dimensions"):
            transform._ensure_3d(arr_4d, "test")

    def test_check_shape_consistency_valid(self):
        """Test _check_shape_consistency with valid shapes."""
        transform = ConcatenateTransform(dim=1)
        arrays = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ]
        keys = ["a", "b"]

        n_rows, ref_shape = transform._check_shape_consistency(
            arrays, keys, concat_dim=1
        )
        assert n_rows == 2
        assert ref_shape == (2, 2)

    def test_check_shape_consistency_empty(self):
        """Test _check_shape_consistency with empty list."""
        transform = ConcatenateTransform(dim=1)

        with pytest.raises(ValueError, match="No arrays"):
            transform._check_shape_consistency([], [], concat_dim=1)

    def test_check_shape_consistency_mismatch(self):
        """Test _check_shape_consistency with shape mismatch."""
        transform = ConcatenateTransform(dim=1)
        arrays = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),  # Shape (2, 2)
            np.array([[5.0], [6.0], [7.0]]),  # Shape (3, 1) - different number of rows
        ]
        keys = ["a", "b"]

        with pytest.raises(ValueError, match="Shape mismatch"):
            transform._check_shape_consistency(arrays, keys, concat_dim=1)

    def test_check_for_nans_no_nans(self):
        """Test _check_for_nans with no NaN values."""
        transform = ConcatenateTransform(dim=1)
        arrays = [
            np.array([[1.0, 2.0]]),
            np.array([[3.0, 4.0]]),
        ]
        keys = ["a", "b"]

        # Should not raise
        transform._check_for_nans(arrays, keys)

    def test_check_for_nans_with_nans(self):
        """Test _check_for_nans with NaN values."""
        transform = ConcatenateTransform(dim=1)
        arrays = [
            np.array([[1.0, 2.0]]),
            np.array([[np.nan, 4.0]]),
        ]
        keys = ["a", "b"]

        with pytest.raises(ValueError, match="NaN values"):
            transform._check_for_nans(arrays, keys)

    def test_transform_data_dim_1(self):
        """Test transform_data with dim=1.

        **Assumption**: ConcatenateTransform with dim=1 should concatenate multiple
        feature arrays horizontally (along columns), combining them into a single
        feature matrix. The number of rows (samples) must match across all arrays.

        **Action**: Create a ConcatenateTransform with dim=1 and provide input data
        with two feature arrays (feat1: shape (2,2) and feat2: shape (2,2)). Both
        have the same number of rows, so they can be concatenated horizontally.

        **Expected Result**: The result should be a numpy array with shape (2, 4),
        where the first 2 columns come from feat1 and the last 2 columns come from
        feat2. This validates the core concatenation functionality, which is essential
        for combining multiple feature sets (e.g., combining different sensor readings
        or feature types) into a unified feature matrix for machine learning models.
        """
        transform = ConcatenateTransform(dim=1)
        data = NamedTransformInput(
            feat1=np.array([[1.0, 2.0], [3.0, 4.0]]),
            feat2=np.array([[5.0, 6.0], [7.0, 8.0]]),
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 4)  # Concatenated along axis 1
        np.testing.assert_array_equal(result[:, :2], data["feat1"])
        np.testing.assert_array_equal(result[:, 2:], data["feat2"])

    def test_transform_data_dim_2(self):
        """Test transform_data with dim=2."""
        transform = ConcatenateTransform(dim=2)
        data = NamedTransformInput(
            feat1=np.array([[[1.0], [2.0]], [[3.0], [4.0]]]),
            feat2=np.array([[[5.0], [6.0]], [[7.0], [8.0]]]),
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2, 2)  # Concatenated along axis 2

    def test_transform_data_empty(self):
        """Test transform_data with empty data."""
        transform = ConcatenateTransform(dim=1)
        data = NamedTransformInput()
        metadata = {}

        with pytest.raises(ValueError, match="No data provided"):
            transform.transform_data(data, metadata)

    def test_transform_data_max_dim_3(self):
        """Test transform_data with max_dim > 3 raises error."""
        transform = ConcatenateTransform(dim=1)
        # Create data that would result in max_dim > 3
        # Actually, the check is on max_dim <= 3, so 4D arrays would fail earlier
        data = NamedTransformInput(
            feat1=np.array([[[[1.0]]]])  # 4D array
        )
        metadata = {}

        with pytest.raises(AssertionError, match="supports up to 3D arrays"):
            transform.transform_data(data, metadata)

    def test_fit_multi_source(self):
        """ConcatenateTransform is stateless; fit_multi_source raises NotImplementedError."""
        transform = ConcatenateTransform(dim=1)
        data_segments = [
            NamedTransformInput(feat1=np.array([[1.0]]), feat2=np.array([[2.0]])),
            NamedTransformInput(feat1=np.array([[3.0]]), feat2=np.array([[4.0]])),
        ]
        metadata = {"apply_to_keys": ["feat1", "feat2"]}
        with pytest.raises(
            NotImplementedError, match="stateless|does not support fitting"
        ):
            transform.fit_multi_source(data_segments, metadata)

    def test_call_method(self):
        """Test __call__ method."""
        transform = ConcatenateTransform(dim=1)
        data = NamedTransformInput(
            feat1=np.array([[1.0, 2.0]]),
            feat2=np.array([[3.0, 4.0]]),
        )
        metadata = {}

        result = transform(data, metadata)
        assert isinstance(result, np.ndarray)


class TestRuggedToDenseTransform:
    """Tests for RuggedToDenseTransform."""

    def test_init(self):
        """Test initialization."""
        transform = RuggedToDenseTransform()
        assert transform is not None

    def test_transform_data_single_ragged_dim(self):
        """Test transform_data with single ragged dimension."""
        transform = RuggedToDenseTransform()
        ragged = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0]])
        data = NamedTransformInput(features=ragged)
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, NamedTransformInput)
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)

    def test_transform_data_multiple_ragged_dims_error(self):
        """Test transform_data with multiple ragged dims raises error."""
        transform = RuggedToDenseTransform()
        # Create ragged array with multiple variable dims
        ragged = ak.Array([[[1.0], [2.0, 3.0]], [[4.0]]])
        data = NamedTransformInput(features=ragged)
        metadata = {}

        with pytest.raises(ValueError, match="ragged dimensions"):
            transform.transform_data(data, metadata)

    def test_transform_data_not_awkward_error(self):
        """Test transform_data with non-awkward array raises error."""
        transform = RuggedToDenseTransform()
        data = NamedTransformInput(features=np.array([[1.0, 2.0]]))
        metadata = {}

        with pytest.raises(AssertionError, match="not an awkward array"):
            transform.transform_data(data, metadata)

    def test_fit_multi_source(self):
        """RuggedToDenseTransform is stateless; fit_multi_source raises NotImplementedError."""
        transform = RuggedToDenseTransform()
        ragged = ak.Array([[1.0, 2.0], [3.0, 4.0, 5.0]])
        data_segments = [
            NamedTransformInput(features=ragged),
            NamedTransformInput(features=ragged),
        ]
        metadata = {"apply_to_keys": ["features"]}
        with pytest.raises(
            NotImplementedError, match="stateless|does not support fitting"
        ):
            transform.fit_multi_source(data_segments, metadata)


class TestMultiDatasetRuggedToDenseTransform:
    """Tests for MultiDatasetRuggedToDenseTransform."""

    def test_init(self):
        """Test initialization."""
        transform = MultiDatasetRuggedToDenseTransform(axis=0)
        assert transform.axis == 0

    def test_transform_data(self):
        """Test transform_data."""
        transform = MultiDatasetRuggedToDenseTransform(axis=0)
        # Use regular (non-ragged) awkward array that can be converted
        regular_ak = ak.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
        data = NamedTransformInput(features=regular_ak)
        metadata = {"apply_to_keys": ["features"]}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, dict)
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)

    def test_transform_multi_source(self):
        """Test transform_multi_source."""
        transform = MultiDatasetRuggedToDenseTransform(axis=0)
        ragged1 = ak.Array([[1.0, 2.0], [3.0, 4.0]])
        ragged2 = ak.Array([[5.0, 6.0], [7.0, 8.0]])
        data_segments = [
            NamedTransformInput(features=ragged1),
            NamedTransformInput(features=ragged2),
        ]
        metadata = {"apply_to_keys": ["features"], "assign_to_map": ["features"]}

        result, log = transform.transform_multi_source(data_segments, metadata)

        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "features" in result[0]
