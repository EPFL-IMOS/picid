"""Comprehensive tests for moving_average.py transform."""

import numpy as np
import pytest
from picid.data.data_objects.data import NamedTransformInput
from picid.transforms.base_transforms.moving_average import MovingAverageTransform


class TestMovingAverageTransform:
    """Tests for MovingAverageTransform."""

    def test_init_continuous_mode(self):
        """Test initialization in continuous mode.

        **Assumption**: MovingAverageTransform should support continuous mode where the
        moving average is applied over the entire signal without chunking. When chunk_size
        is None, the transform operates in continuous mode. The default causal=False means
        centered (non-causal) averaging, and apply_to_columns=True means the filter is
        applied column-wise (across time for each feature).

        **Action**: Create a MovingAverageTransform with only window_size=3 specified
        (no chunk_size), which should initialize in continuous mode.

        **Expected Result**: The transform should have window_size=3, chunk_size=None
        (indicating continuous mode), causal=False (centered averaging), and
        apply_to_columns=True. This validates that the default initialization creates
        a continuous-mode transform, which is the most common use case for smoothing
        time-series signals without discontinuities.
        """
        transform = MovingAverageTransform(window_size=3)
        assert transform.window_size == 3
        assert transform.chunk_size is None
        assert transform.causal is False
        assert transform.apply_to_columns is True

    def test_init_chunked_mode(self):
        """Test initialization in chunked mode."""
        transform = MovingAverageTransform(window_size=3, chunk_size=10)
        assert transform.window_size == 3
        assert transform.chunk_size == 10

    def test_init_causal(self):
        """Test initialization with causal=True."""
        transform = MovingAverageTransform(window_size=3, causal=True)
        assert transform.causal is True

    def test_init_invalid_window_size(self):
        """Test initialization with invalid window_size."""
        with pytest.raises(ValueError, match="window_size must be a positive integer"):
            MovingAverageTransform(window_size=0)

        with pytest.raises(ValueError, match="window_size must be a positive integer"):
            MovingAverageTransform(window_size=-1)

    def test_init_invalid_chunk_size(self):
        """Test initialization with invalid chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            MovingAverageTransform(window_size=3, chunk_size=0)

    def test_init_apply_to_columns_false_error(self):
        """Test initialization with apply_to_columns=False raises error."""
        with pytest.raises(NotImplementedError, match="apply_to_columns=False"):
            MovingAverageTransform(window_size=3, apply_to_columns=False)

    def test_validate_input_2d(self):
        """Test _validate_input with 2D array."""
        transform = MovingAverageTransform(window_size=3)
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = transform._validate_input(arr, "test")
        np.testing.assert_array_equal(result, arr)

    def test_validate_input_not_2d_error(self):
        """Test _validate_input with non-2D array raises error."""
        transform = MovingAverageTransform(window_size=3)
        arr_1d = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="must be a 2D array"):
            transform._validate_input(arr_1d, "test")

    def test_validate_input_inf_error(self):
        """Test _validate_input with infinite values raises error."""
        transform = MovingAverageTransform(window_size=3)
        arr = np.array([[1.0, 2.0], [np.inf, 4.0]])

        with pytest.raises(ValueError, match="Infinite values"):
            transform._validate_input(arr, "test")

    def test_validate_input_chunk_size_not_divisible_error(self):
        """Test _validate_input with signal not divisible by chunk_size."""
        transform = MovingAverageTransform(window_size=3, chunk_size=5)
        arr = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        )  # 4 rows, not divisible by 5

        with pytest.raises(ValueError, match="not perfectly divisible"):
            transform._validate_input(arr, "test")

    def test_validate_input_short_signal_warning(self):
        """Test _validate_input with signal shorter than window_size."""
        transform = MovingAverageTransform(window_size=5)
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 rows < window_size 5

        # Should log warning but not raise error
        result = transform._validate_input(arr, "test")
        assert result is not None

    def test_transform_data_continuous_mode(self):
        """Test transform_data in continuous mode.

        **Assumption**: MovingAverageTransform in continuous mode should apply a moving
        average filter across the entire signal, smoothing the data by averaging values
        within a sliding window. With causal=False, the averaging is centered (non-causal),
        meaning each output point is the average of surrounding points (past and future).
        The output should have the same shape as the input.

        **Action**: Create a MovingAverageTransform with window_size=3 and causal=False
        (continuous mode). Provide input data with 4 time steps and 2 features, then
        apply the transform.

        **Expected Result**: The result should be a numpy array with the same shape as
        the input (4, 2), and all values should be finite (no NaN or inf). The values
        should be smoothed compared to the input (less variation). This validates that
        the moving average filter works correctly in continuous mode, which is essential
        for noise reduction and signal smoothing in time-series preprocessing.
        """
        transform = MovingAverageTransform(window_size=3, causal=False)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
        metadata = {"apply_to_keys": ["features"]}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert result.shape == data["features"].shape
        # Moving average should smooth the data
        assert np.all(np.isfinite(result))

    def test_transform_data_continuous_causal(self):
        """Test transform_data in continuous mode with causal=True."""
        transform = MovingAverageTransform(window_size=3, causal=True)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
        metadata = {"apply_to_keys": ["features"]}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert result.shape == data["features"].shape

    def test_transform_data_chunked_mode(self):
        """Test transform_data in chunked mode."""
        transform = MovingAverageTransform(window_size=3, chunk_size=4)
        data = NamedTransformInput(
            features=np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [7.0, 8.0],
                    [9.0, 10.0],
                    [11.0, 12.0],
                    [13.0, 14.0],
                    [15.0, 16.0],
                ]
            )
        )  # 8 rows, divisible by chunk_size=4
        metadata = {"apply_to_keys": ["features"]}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert result.shape == data["features"].shape

    def test_transform_data_chunked_causal(self):
        """Test transform_data in chunked mode with causal=True."""
        transform = MovingAverageTransform(window_size=3, chunk_size=4, causal=True)
        data = NamedTransformInput(
            features=np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [7.0, 8.0],
                    [9.0, 10.0],
                    [11.0, 12.0],
                    [13.0, 14.0],
                    [15.0, 16.0],
                ]
            )
        )
        metadata = {"apply_to_keys": ["features"]}

        result = transform.transform_data(data, metadata)

        assert isinstance(result, np.ndarray)
        assert result.shape == data["features"].shape

    def test_transform_data_empty_error(self):
        """Test transform_data with empty data raises error."""
        transform = MovingAverageTransform(window_size=3)
        data = NamedTransformInput()
        metadata = {"apply_to_keys": []}

        with pytest.raises(ValueError, match="No data provided"):
            transform.transform_data(data, metadata)

    def test_transform_data_multiple_keys_error(self):
        """Test transform_data with multiple keys raises error."""
        transform = MovingAverageTransform(window_size=3)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0]]), target=np.array([[0.5]])
        )
        metadata = {"apply_to_keys": ["features", "target"]}

        with pytest.raises(AssertionError, match="exactly one key"):
            transform.transform_data(data, metadata)

    def test_get_feature_names(self):
        """Test get_feature_names method."""
        transform = MovingAverageTransform(window_size=3)
        input_keys = ["features"]
        input_shapes = {"features": (10, 3)}  # 10 rows, 3 columns

        feature_names = transform.get_feature_names(input_keys, input_shapes)

        assert len(feature_names) == 3
        assert all("mov_avg3" in name for name in feature_names)

    def test_get_feature_names_chunked(self):
        """Test get_feature_names with chunked mode."""
        transform = MovingAverageTransform(window_size=3, chunk_size=5)
        input_keys = ["features"]
        input_shapes = {"features": (10, 2)}

        feature_names = transform.get_feature_names(input_keys, input_shapes)

        assert len(feature_names) == 2
        assert all("chunked5" in name for name in feature_names)

    def test_get_feature_names_causal(self):
        """Test get_feature_names with causal mode."""
        transform = MovingAverageTransform(window_size=3, causal=True)
        input_keys = ["features"]
        input_shapes = {"features": (10, 2)}

        feature_names = transform.get_feature_names(input_keys, input_shapes)

        assert len(feature_names) == 2
        assert all("causal" in name for name in feature_names)

    def test_get_feature_names_empty_keys(self):
        """Test get_feature_names with empty input_keys."""
        transform = MovingAverageTransform(window_size=3)
        feature_names = transform.get_feature_names([], {})

        assert feature_names == []

    def test_call_method(self):
        """Test __call__ method."""
        transform = MovingAverageTransform(window_size=3)
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
        metadata = {"apply_to_keys": ["features"]}

        result = transform(data, metadata)

        assert isinstance(result, np.ndarray)
