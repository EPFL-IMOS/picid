import pytest
import numpy as np
import awkward as ak
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin
from lmetk.transforms.base.base_transform import DenseTransform, RaggedTransform


class DummyDenseTransform(ConcatinatedFitMultiSourceTransformMixin, DenseTransform):
    def __init__(self):
        self.fit_called = False
        self.transform_called = False
        super().__init__()

    def fit_data(self, data, metadata=None):
        self.fit_called = True
        self.last_fit_data = data
        self.last_fit_metadata = metadata

    def transform_data(self, data, metadata=None):
        self.transform_called = True
        self.last_transform_data = data
        self.last_transform_metadata = metadata
        return {k: v + 1 for k, v in data.items()}


class DummyRaggedTransform(ConcatinatedFitMultiSourceTransformMixin, RaggedTransform):
    def __init__(self):
        self.fit_called = False
        self.transform_called = False
        super().__init__()

    def fit_data(self, data, metadata=None):
        self.fit_called = True
        self.last_fit_data = data
        self.last_fit_metadata = metadata

    def transform_data(self, data, metadata=None):
        self.transform_called = True
        self.last_transform_data = data
        self.last_transform_metadata = metadata
        return {k: ak.to_list(v) for k, v in data.items()}


class DummyChunk(dict):
    def __init__(self, data, instance_cls):
        super().__init__(data)
        self._instance_cls = instance_cls

    def get_instance_cls(self):
        return self._instance_cls


def test_fit_multi_source_dense():
    # Prepare dummy dense data
    data_segments = [
        DummyChunk({"a": np.array([1, 2]), "b": np.array([3, 4])}, np.ndarray),
        DummyChunk({"a": np.array([5, 6]), "b": np.array([7, 8])}, np.ndarray),
    ]
    metadata = {"apply_to_keys": ["a", "b"]}
    transform = DummyDenseTransform()
    transform.fit_multi_source(data_segments, metadata)
    assert transform.fit_called
    # Should concatenate along axis 0
    np.testing.assert_array_equal(transform.last_fit_data["a"], np.array([1, 2, 5, 6]))
    np.testing.assert_array_equal(transform.last_fit_data["b"], np.array([3, 4, 7, 8]))


def test_transform_multi_source_dense():
    data_segments = [
        DummyChunk({"a": np.array([1, 2]), "b": np.array([3, 4])}, np.ndarray),
        DummyChunk({"a": np.array([5, 6]), "b": np.array([7, 8])}, np.ndarray),
    ]
    metadata = {"apply_to_keys": ["a", "b"]}
    transform = DummyDenseTransform()
    out = transform.transform_multi_source(data_segments, metadata)
    # Should return transformed dict with +1 for each value
    assert isinstance(out, dict)
    np.testing.assert_array_equal(out["a"], np.array([2, 3, 6, 7]))
    np.testing.assert_array_equal(out["b"], np.array([4, 5, 8, 9]))


def test_fit_multi_source_ragged(monkeypatch):
    # Prepare dummy ragged data
    arr1 = ak.Array([[1, 2], [3]])
    arr2 = ak.Array([[4], [5, 6]])
    data_segments = [
        DummyChunk({"a": arr1}, ak.Array),
        DummyChunk({"a": arr2}, ak.Array),
    ]
    metadata = {"apply_to_keys": ["a"]}

    # Patch ak_find_var_dim to return [1]
    monkeypatch.setattr(
        "lmetk.transforms.base.multisource.ak_find_var_dim", lambda x: [1]
    )

    transform = DummyRaggedTransform()
    transform.fit_multi_source(data_segments, metadata)
    assert transform.fit_called
    # Should concatenate ragged arrays
    assert ak.all(transform.last_fit_data["a"] == ak.concatenate([arr1, arr2], axis=1))


def test_fit_multi_source_ragged_to_dense(monkeypatch):
    # Prepare dummy ragged data
    arr1 = ak.Array([[1, 2], [3]])
    arr2 = ak.Array([[4], [5, 6]])
    data_segments = [
        DummyChunk({"a": arr1}, ak.Array),
        DummyChunk({"a": arr2}, ak.Array),
    ]
    metadata = {"apply_to_keys": ["a"]}

    # Patch ak_find_var_dim to return [1]
    monkeypatch.setattr(
        "lmetk.transforms.base.multisource.ak_find_var_dim", lambda x: [1]
    )

    transform = DummyDenseTransform()
    transform.fit_multi_source(data_segments, metadata)
    assert transform.fit_called

    arr_cc = ak.concatenate([arr1, arr2], axis=1).to_numpy()

    # Should concatenate ragged arrays
    np.testing.assert_array_equal(arr_cc, np.array([[1, 2, 4], [3, 5, 6]]))


def test_transform_multi_source_ragged(monkeypatch):
    arr1 = ak.Array([[1, 2], [3]])
    arr2 = ak.Array([[4], [5, 6]])
    data_segments = [
        DummyChunk({"a": arr1}, ak.Array),
        DummyChunk({"a": arr2}, ak.Array),
    ]
    metadata = {"apply_to_keys": ["a"]}
    monkeypatch.setattr(
        "lmetk.transforms.base.multisource.ak_find_var_dim", lambda x: [0]
    )
    transform = DummyRaggedTransform()
    out = transform.transform_multi_source(data_segments, metadata)
    assert isinstance(out, dict)
    assert isinstance(out["a"], list)
    # Should be a list of lists after transform_data


def test_fit_multi_source_type_error():
    # Mismatched types
    data_segments = [
        DummyChunk({"a": np.array([1, 2])}, np.ndarray),
        DummyChunk({"a": ak.Array([[1, 2]])}, ak.Array),
    ]
    metadata = {"apply_to_keys": ["a"]}
    transform = DummyDenseTransform()
    with pytest.raises(ValueError):
        transform.fit_multi_source(data_segments, metadata)


def test_transform_multi_source_length_error():
    data_segments = [
        DummyChunk({"a": np.array([1, 2])}, np.ndarray),
        DummyChunk({"a": np.array([3, 4]), "b": np.array([5, 6])}, np.ndarray),
    ]
    metadata = {"apply_to_keys": ["a"]}
    transform = DummyDenseTransform()
    with pytest.raises(ValueError):
        transform.transform_multi_source(data_segments, metadata)
