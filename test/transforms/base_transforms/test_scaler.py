"""Comprehensive tests for scaler.py transforms."""

import numpy as np
import pytest
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base_transforms.scaler import (
    ConstantScaler,
    MinMaxScalerSklearn,
    StandardScalerSklearn,
)


class TestConstantScaler:
    """Tests for ConstantScaler."""

    def test_init_default(self):
        """Test initialization with default factor.

        **Assumption**: ConstantScaler should initialize with a default factor of 1.0
        when no factor is specified. A factor of 1.0 means no scaling (multiply by 1),
        which is a sensible default for a scaling transform.

        **Action**: Create a ConstantScaler instance without providing a factor parameter.

        **Expected Result**: The transform should be created successfully and the factor
        attribute should be 1.0. This validates that the default initialization works
        correctly and provides a no-op scaling behavior by default, which is useful for
        optional scaling in pipelines.
        """
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
        """Test transform_data multiplies by factor.

        **Assumption**: ConstantScaler should multiply all values in the input array by
        the scaling factor. This is a simple linear scaling operation that doesn't depend
        on the data distribution (unlike MinMaxScaler or StandardScaler).

        **Action**: Create a ConstantScaler with factor=2.0 and provide input data with
        values [[1.0, 2.0], [3.0, 4.0]]. Call transform_data to apply the scaling.

        **Expected Result**: The result should be [[2.0, 4.0], [6.0, 8.0]] (each value
        multiplied by 2.0). This validates the core scaling functionality, which is
        essential for unit conversion, normalization by known constants, or simple
        feature scaling when the scaling factor is known a priori.
        """
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
            features=np.array([[1.0, 2.0]]), target=np.array([[0.5]])
        )
        metadata = {}

        with pytest.raises(AssertionError, match="only supports single key"):
            transform.transform_data(data, metadata)

    def test_inverse_transform(self):
        """Test inverse_transform divides by factor.

        **Assumption**: ConstantScaler should support inverse transformation by dividing
        by the factor (the inverse of multiplication). This allows recovering the original
        data from scaled data, which is important for interpreting model predictions or
        converting scaled features back to their original units.

        **Action**: Create a ConstantScaler with factor=2.0 and provide scaled data
        [[2.0, 4.0], [6.0, 8.0]] (which was originally [[1.0, 2.0], [3.0, 4.0]] scaled
        by 2.0). Call inverse_transform to reverse the scaling.

        **Expected Result**: The result should be [[1.0, 2.0], [3.0, 4.0]] (each value
        divided by 2.0). This validates that the inverse transformation correctly undoes
        the scaling, which is crucial for maintaining data integrity when scaling needs
        to be reversed (e.g., converting predictions back to original scale).
        """
        transform = ConstantScaler(factor=2.0)
        data = NamedTransformInput(features=np.array([[2.0, 4.0], [6.0, 8.0]]))
        metadata = {}

        result = transform.inverse_transform(data, metadata)

        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_inverse_transform_multiple_keys_error(self):
        """Test inverse_transform with multiple keys raises error."""
        transform = ConstantScaler(factor=2.0)
        data = NamedTransformInput(features=np.array([[1.0]]), target=np.array([[0.5]]))
        metadata = {}

        with pytest.raises(AssertionError, match="only supports single key"):
            transform.inverse_transform(data, metadata)

    @pytest.mark.skip(
        reason="ConstantScaler does not support fitting (NoFitPerSegmentMixin)"
    )
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
        """Test fit_data with 2D array.

        **Assumption**: MinMaxScalerSklearn should fit on training data to compute the
        minimum and maximum values for each feature column. These statistics are stored
        in the underlying sklearn scaler and used during transformation to scale data
        to the [0, 1] range.

        **Action**: Create a MinMaxScalerSklearn and call fit_data with a 2D array
        containing 3 samples and 2 features. The scaler should compute data_min_ and
        data_max_ for each feature column.

        **Expected Result**: After fitting, the scaler should have data_min_ and data_max_
        attributes, indicating that it has learned the data range. This validates that
        the fit process works correctly and stores the necessary statistics for scaling,
        which is essential for proper normalization of features to a consistent range.
        """
        transform = MinMaxScalerSklearn()
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
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
            features=np.array([[1.0, 2.0]]), target=np.array([[0.5]])
        )
        metadata = {}

        with pytest.raises(AssertionError, match="only supports single key"):
            transform.fit_data(data, metadata)

    def test_transform_data_2d(self):
        """Test transform_data with 2D array.

        **Assumption**: MinMaxScalerSklearn should scale data to the [0, 1] range based
        on the min/max values learned during fitting. The scaling formula is:
        (x - min) / (max - min), which maps the original range to [0, 1].

        **Action**: Create a MinMaxScalerSklearn, fit it on data with range [1.0-5.0]
        for first feature and [2.0-6.0] for second feature, then transform new data
        [[2.0, 3.0], [4.0, 5.0]] using the learned statistics.

        **Expected Result**: The result should be a numpy array with shape (2, 2), and
        all values should be in the [0, 1] range. This validates that MinMax scaling
        works correctly and produces normalized features, which is essential for machine
        learning algorithms that are sensitive to feature scales (e.g., neural networks,
        distance-based methods).
        """
        transform = MinMaxScalerSklearn()
        # Fit first
        fit_data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
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

    def test_transform_data_tolerates_pipeline_metadata_keys(self):
        """MinMax must accept the same metadata keys pipelines pass for other scalers."""
        transform = MinMaxScalerSklearn()
        fit_data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
        transform.fit_data(fit_data, {})
        data = NamedTransformInput(features=np.array([[2.0, 3.0], [4.0, 5.0]]))
        metadata = {"assign_to_map": ["features"], "apply_to_keys": ["features"]}
        result = transform.transform_data(data, metadata)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

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
        """Test inverse_transform.

        **Assumption**: MinMaxScalerSklearn should support inverse transformation that
        recovers the original data from scaled data. The inverse formula is:
        x = scaled * (max - min) + min, which reverses the scaling operation.

        **Action**: Create a MinMaxScalerSklearn, fit it on training data, transform
        some data to the [0, 1] range, then apply inverse_transform to the scaled data.

        **Expected Result**: The inverse transformed result should match the original
        data (within numerical precision). This validates that the scaling is reversible,
        which is crucial for converting model predictions back to their original scale
        or interpreting scaled features in their original units.
        """
        transform = MinMaxScalerSklearn()
        # Fit first
        fit_data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
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
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
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
        fit_data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
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
        fit_data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
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
