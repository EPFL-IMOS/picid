"""Comprehensive tests for padding2length.py transform."""

import numpy as np
import pytest
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base_transforms.padding2length import PadToLength


class TestPadToLength:
    """Tests for PadToLength transform."""

    def test_init(self):
        """Test initialization.

        **Assumption**: PadToLength should accept target_length (desired length after padding),
        axis (dimension to pad along), and pad_value (value to use for padding). All parameters
        should be stored correctly for use during transformation.

        **Action**: Create a PadToLength transform with explicit parameters: target_length=10,
        axis=0 (pad along rows), and pad_value=0.0 (zero padding).

        **Expected Result**: All three attributes should be set correctly. This validates that
        the transform can be configured with specific padding requirements, which is important
        for ensuring consistent array lengths in batch processing or when interfacing with
        models that require fixed-length inputs.
        """
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
        """Test transform_data when padding is not needed.

        **Assumption**: PadToLength should handle the case where the input array already
        has the target length (or is longer). In this case, no padding should be applied
        and the original array should be returned unchanged.

        **Action**: Create a PadToLength transform with target_length=2 and provide input
        data with shape (2, 2), which already has 2 rows (the target length along axis=0).

        **Expected Result**: The result should be identical to the input array (element-wise
        equality). This validates that the transform correctly handles the edge case where
        padding is unnecessary, preventing unnecessary copying or modification of data that
        already meets the length requirement.
        """
        transform = PadToLength(target_length=2, axis=0)
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}

        result = transform.transform_data(data, metadata)

        # Should return original array unchanged
        np.testing.assert_array_equal(result, data["features"])

    def test_transform_data_padding_axis_0(self):
        """Test transform_data with padding along axis 0.

        **Assumption**: PadToLength should pad arrays along the specified axis to reach
        the target length. When padding along axis=0 (rows), new rows should be added
        at the end, filled with the pad_value. The original data should be preserved
        at the beginning.

        **Action**: Create a PadToLength transform with target_length=5, axis=0, pad_value=0.0.
        Provide input data with shape (2, 2), which needs 3 additional rows to reach length 5.

        **Expected Result**: The result should have shape (5, 2). The first 2 rows should
        match the original data, and the last 3 rows should be zeros (the pad_value). This
        validates the core padding functionality along the row dimension, which is essential
        for standardizing sequence lengths in time-series data or ensuring consistent batch
        sizes in neural network inputs.
        """
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
        """Test transform_data with padding along axis 1.

        **Assumption**: PadToLength should support padding along different axes. When
        padding along axis=1 (columns), new columns should be added at the end, filled
        with the pad_value. This is useful for standardizing feature vector lengths.

        **Action**: Create a PadToLength transform with target_length=5, axis=1, pad_value=0.0.
        Provide input data with shape (2, 2), which needs 3 additional columns to reach length 5.

        **Expected Result**: The result should have shape (2, 5). The first 2 columns should
        match the original data, and the last 3 columns should be zeros. This validates that
        padding works correctly along the column dimension, which is important for scenarios
        where feature vectors need to be standardized to a fixed length (e.g., when combining
        features from different sources with varying dimensions).
        """
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
        """Test transform_data with custom pad value.

        **Assumption**: PadToLength should allow custom pad values (not just zeros).
        This is important for scenarios where a specific sentinel value is needed (e.g.,
        -1.0 to distinguish padding from actual zero values in the data, or NaN for
        certain processing pipelines).

        **Action**: Create a PadToLength transform with target_length=5, axis=0, and
        pad_value=-1.0 (negative one instead of zero). Provide input data with shape (2, 2).

        **Expected Result**: The result should have shape (5, 2), with the last 3 rows
        filled with -1.0 (not zeros). This validates that custom padding values work
        correctly, which is essential for data preprocessing scenarios where the padding
        value needs to be distinguishable from actual data values.
        """
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
