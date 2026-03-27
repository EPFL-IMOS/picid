"""Comprehensive tests for reshaping.py transform."""

import numpy as np
import pytest
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base_transforms.reshaping import ReshapeTransform


class TestReshapeTransform:
    """Tests for ReshapeTransform."""

    def test_init(self):
        """Test initialization.

        **Assumption**: ReshapeTransform should accept an einops-style pattern string
        that describes how to rearrange array dimensions. The pattern uses named dimensions
        (like "b" for batch, "h" for height, "w" for width) and allows grouping dimensions
        with parentheses, making complex reshapes more readable than raw numpy reshape.

        **Action**: Create a ReshapeTransform instance with pattern "b h w -> (b h) w",
        which reshapes a 3D array by flattening the first two dimensions.

        **Expected Result**: The transform should be created successfully and the pattern
        should be stored correctly. This validates that the transform can be initialized
        with einops patterns, which is important for flexible array manipulation in
        deep learning workflows where tensor shapes need to be rearranged for different
        model architectures.
        """
        transform = ReshapeTransform(pattern="b h w -> (b h) w")
        assert transform.pattern == "b h w -> (b h) w"

    def test_transform_data_simple_reshape(self):
        """Test transform_data with simple reshape pattern.

        **Assumption**: ReshapeTransform should use einops.rearrange to reshape arrays
        according to the specified pattern. A pattern like "h w -> (h w)" should flatten
        a 2D array into a 1D array by combining the height and width dimensions.

        **Action**: Create a ReshapeTransform with pattern "h w -> (h w)" and provide
        a 2D input array with shape (2, 2). The transform should flatten it to 1D.

        **Expected Result**: The result should be a numpy array with shape (4,), which
        is the flattened version of the 2x2 input. This validates that einops-based
        reshaping works correctly, which is essential for tensor manipulation in neural
        networks where arrays need to be reshaped between different layer requirements
        (e.g., flattening before fully connected layers).
        """
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
            features=np.array([[1.0, 2.0]]), target=np.array([[0.5]])
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
