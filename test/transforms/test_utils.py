"""Tests for picid.transforms.utils module.

Coverage target: >=95% of picid/transforms/utils.py

These utility functions convert data between numpy and other container
types (DataFrame, Series, Tensor) while preserving metadata such as
column names, index, and device placement.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from picid.transforms.utils import _convert_to_numpy, _convert_from_numpy


@pytest.mark.unit
class TestConvertToNumpy:
    """Tests for _convert_to_numpy — converts any container to numpy."""

    def test_numpy_passthrough_returns_same_object(self):
        """Numpy arrays pass through without copying.

        **Methodology**: Supply a numpy array and check identity.

        **Expected**: Returned array is the same object; type is np.ndarray.
        """
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result, orig_type = _convert_to_numpy(arr)
        assert result is arr
        assert orig_type is np.ndarray

    def test_numpy_preserves_dtype(self):
        """Dtype is preserved through passthrough.

        **Methodology**: Supply float32 array.

        **Expected**: Returned dtype is float32.
        """
        arr = np.array([1, 2, 3], dtype=np.float32)
        result, _ = _convert_to_numpy(arr)
        assert result.dtype == np.float32

    def test_dataframe_to_numpy(self):
        """DataFrame is converted via .values.

        **Methodology**: Supply a DataFrame with named columns.

        **Expected**: Shape matches, values equal, type recorded as DataFrame.
        """
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result, orig_type = _convert_to_numpy(df)
        assert orig_type is pd.DataFrame
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, df.values)

    def test_dataframe_with_many_columns(self):
        """DataFrame with multiple columns preserves shape.

        **Methodology**: 3-row, 5-column DataFrame.

        **Expected**: Returned array has shape (3, 5).
        """
        df = pd.DataFrame(np.arange(15).reshape(3, 5))
        result, _ = _convert_to_numpy(df)
        assert result.shape == (3, 5)

    def test_series_to_numpy_reshaped(self):
        """Series is reshaped to (-1, 1) column vector.

        **Methodology**: Supply a 3-element Series.

        **Expected**: Shape is (3, 1), type is pd.Series.
        """
        s = pd.Series([1.0, 2.0, 3.0], name="sensor")
        result, orig_type = _convert_to_numpy(s)
        assert orig_type is pd.Series
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.ravel(), [1.0, 2.0, 3.0])

    def test_tensor_to_numpy(self):
        """Torch Tensor is moved to CPU and converted to numpy.

        **Methodology**: Supply a CPU tensor.

        **Expected**: Values match, type is torch.Tensor.
        """
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result, orig_type = _convert_to_numpy(t)
        assert orig_type is torch.Tensor
        np.testing.assert_array_equal(result, t.numpy())

    def test_list_fallback(self):
        """Plain list is converted via np.asarray.

        **Methodology**: Supply [1, 2, 3].

        **Expected**: Result is np.ndarray, type recorded as list.
        """
        data = [1, 2, 3]
        result, orig_type = _convert_to_numpy(data)
        assert orig_type is list
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_unsupported_type_raises_type_error(self):
        """Non-convertible object raises TypeError.

        **Methodology**: Supply object whose __array__ raises.

        **Expected**: TypeError with descriptive message.
        """

        class Unconvertible:
            def __array__(self, *args, **kwargs):
                raise ValueError("cannot convert")

        with pytest.raises(TypeError, match="Unsupported data type"):
            _convert_to_numpy(Unconvertible())

    def test_empty_numpy_array(self):
        """Empty numpy array passes through.

        **Methodology**: Supply np.array([]).

        **Expected**: Empty array returned, type np.ndarray.
        """
        arr = np.array([])
        result, orig_type = _convert_to_numpy(arr)
        assert result is arr
        assert orig_type is np.ndarray
        assert result.size == 0

    def test_empty_dataframe(self):
        """Empty DataFrame converts to empty array.

        **Methodology**: Supply pd.DataFrame().

        **Expected**: Result has 0 rows, type pd.DataFrame.
        """
        df = pd.DataFrame()
        result, orig_type = _convert_to_numpy(df)
        assert orig_type is pd.DataFrame
        assert result.shape[0] == 0

    def test_empty_series(self):
        """Empty Series converts to shape (0, 1).

        **Methodology**: Supply pd.Series(dtype=float).

        **Expected**: Shape (0, 1), type pd.Series.
        """
        s = pd.Series(dtype=float)
        result, orig_type = _convert_to_numpy(s)
        assert orig_type is pd.Series
        assert result.shape == (0, 1)


@pytest.mark.unit
class TestConvertFromNumpy:
    """Tests for _convert_from_numpy — reconstructs original container."""

    def test_numpy_passthrough(self):
        """Numpy type returns the numpy array unchanged.

        **Methodology**: Round-trip with original_type=np.ndarray.

        **Expected**: Same array object returned.
        """
        arr = np.array([1.0, 2.0])
        result = _convert_from_numpy(arr, np.ndarray, arr)
        assert result is arr

    def test_dataframe_preserves_index_and_columns(self):
        """DataFrame reconstruction preserves index and column names.

        **Methodology**: Create DF with custom index/columns, convert, reconstruct.

        **Expected**: Reconstructed DF has identical index and columns.
        """
        original = pd.DataFrame(
            {"temp": [20.0, 25.0], "pressure": [1.0, 1.1]},
            index=["t0", "t1"],
        )
        numpy_data = original.values
        result = _convert_from_numpy(numpy_data, pd.DataFrame, original)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["temp", "pressure"]
        assert list(result.index) == ["t0", "t1"]
        np.testing.assert_array_equal(result.values, numpy_data)

    def test_dataframe_fallback_without_original(self):
        """DataFrame fallback when original_data is not a DataFrame.

        **Methodology**: Pass original_type=DataFrame but original_data="not_df".

        **Expected**: Returns a DataFrame (without metadata).
        """
        numpy_data = np.array([[1.0, 2.0]])
        result = _convert_from_numpy(numpy_data, pd.DataFrame, "not_a_df")
        assert isinstance(result, pd.DataFrame)
        np.testing.assert_array_equal(result.values, numpy_data)

    def test_series_preserves_index_and_name(self):
        """Series reconstruction preserves index and name.

        **Methodology**: Create named Series with custom index, reconstruct.

        **Expected**: Name and index preserved.
        """
        original = pd.Series([10.0, 20.0, 30.0], index=["a", "b", "c"], name="voltage")
        numpy_data = original.values.reshape(-1, 1)
        result = _convert_from_numpy(numpy_data, pd.Series, original)
        assert isinstance(result, pd.Series)
        assert result.name == "voltage"
        assert list(result.index) == ["a", "b", "c"]
        np.testing.assert_array_equal(result.values, original.values)

    def test_series_fallback_without_original(self):
        """Series fallback when original_data is not a Series.

        **Methodology**: Pass original_type=Series but original_data=None.

        **Expected**: Returns a flattened Series.
        """
        numpy_data = np.array([[1.0], [2.0]])
        result = _convert_from_numpy(numpy_data, pd.Series, None)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_tensor_preserves_device(self):
        """Tensor reconstruction preserves original device.

        **Methodology**: CPU tensor round-trip.

        **Expected**: Reconstructed tensor is on CPU, values match.
        """
        original = torch.tensor([1.0, 2.0, 3.0])
        numpy_data = original.numpy()
        result = _convert_from_numpy(numpy_data, torch.Tensor, original)
        assert isinstance(result, torch.Tensor)
        assert result.device == original.device
        torch.testing.assert_close(result, original)

    def test_tensor_fallback_without_original(self):
        """Tensor fallback when original_data is not a Tensor.

        **Methodology**: Pass original_type=Tensor but original_data="string".

        **Expected**: Returns CPU tensor.
        """
        numpy_data = np.array([1.0, 2.0])
        result = _convert_from_numpy(numpy_data, torch.Tensor, "not_tensor")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    def test_unknown_type_returns_numpy(self):
        """Unknown original_type returns numpy array as-is.

        **Methodology**: Pass original_type=list.

        **Expected**: Returns the numpy array unchanged.
        """
        numpy_data = np.array([1.0, 2.0, 3.0])
        result = _convert_from_numpy(numpy_data, list, [1.0, 2.0, 3.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, numpy_data)


@pytest.mark.unit
class TestRoundTrip:
    """End-to-end round-trip tests for type-preserving conversion."""

    @pytest.mark.parametrize(
        "make_input",
        [
            pytest.param(
                lambda: np.array([[1.0, 2.0], [3.0, 4.0]]),
                id="numpy",
            ),
            pytest.param(
                lambda: pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}),
                id="dataframe",
            ),
            pytest.param(
                lambda: pd.Series([1.0, 2.0, 3.0], name="s"),
                id="series",
            ),
            pytest.param(
                lambda: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                id="tensor",
            ),
        ],
    )
    def test_roundtrip_preserves_values(self, make_input):
        """Converting to numpy and back preserves numerical values.

        **Methodology**: Create input → _convert_to_numpy → _convert_from_numpy.

        **Expected**: Reconstructed values match original within float tolerance.
        """
        original = make_input()
        numpy_data, orig_type = _convert_to_numpy(original)
        reconstructed = _convert_from_numpy(numpy_data, orig_type, original)

        assert type(reconstructed) is orig_type

        if isinstance(reconstructed, np.ndarray):
            np.testing.assert_array_equal(reconstructed, original)
        elif isinstance(reconstructed, pd.DataFrame):
            np.testing.assert_array_equal(reconstructed.values, original.values)
        elif isinstance(reconstructed, pd.Series):
            np.testing.assert_array_equal(reconstructed.values, original.values)
        elif isinstance(reconstructed, torch.Tensor):
            torch.testing.assert_close(reconstructed, original)
