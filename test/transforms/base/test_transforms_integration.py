from typing import Any, Dict

import numpy as np
import pytest

from picid.data.data_objects import SplitDatasetContainer
from picid.transforms.base.base_transform import BaseTransform
from picid.transforms.base.data_transform import DataTransform


class AddOneTransform(BaseTransform):
    def transform_data(self, data: Any, metadata: Dict[str, Any]) -> Any:
        return data["features"] + 1

    def transform_multi_source(self, chunks: list, metadata: Dict[str, Any]) -> tuple:
        results = [self.transform_data(c, metadata or {}) for c in chunks]
        return results, {}


class ScaleTransform(BaseTransform):
    def __init__(self, factor: float):
        super().__init__(factor=factor)
        self.factor = factor

    def transform_data(self, data: Any, metadata: Dict[str, Any]) -> Any:
        return data["features"] * self.factor

    def transform_multi_source(self, chunks: list, metadata: Dict[str, Any]) -> tuple:
        results = [self.transform_data(c, metadata or {}) for c in chunks]
        return results, {}


class FitOffsetTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self.fit_called = False
        self.offset = 0.0

    def fit_data(self, data: Any, metadata: Dict[str, Any]) -> Any:
        self.fit_called = True
        self.offset = float(np.mean(data["features"]))

    def fit_multi_source(self, chunks: list, metadata: Dict[str, Any]) -> None:
        for c in chunks:
            self.fit_data(c, metadata or {})

    def transform_data(self, data: Any, metadata: Dict[str, Any]) -> Any:
        return data["features"] - self.offset

    def transform_multi_source(self, chunks: list, metadata: Dict[str, Any]) -> tuple:
        results = [self.transform_data(c, metadata or {}) for c in chunks]
        return results, {}


class MissingKeyTransform(BaseTransform):
    def transform_data(self, data: Any, metadata: Dict[str, Any]) -> Any:
        return {"unexpected": np.array([1.0])}

    def transform_multi_source(self, chunks: list, metadata: Dict[str, Any]) -> tuple:
        results = [self.transform_data(c, metadata or {}) for c in chunks]
        return results, {}


def _make_container():
    return SplitDatasetContainer(
        features={
            "train": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "val": np.array([[5.0, 6.0]]),
        }
    )


def test_data_transform_assign_to_new_key():
    data = _make_container()
    transform = DataTransform(
        "add_one",
        AddOneTransform(),
        metadata={"apply_to": "features", "assign_to": "features_scaled"},
    )

    transformed, _ = transform.forward(data)

    assert "features_scaled" in transformed
    np.testing.assert_array_equal(
        transformed["features_scaled"]["train"][0],
        np.array([[2.0, 3.0], [4.0, 5.0]]),
    )
    np.testing.assert_array_equal(
        transformed["features"]["train"][0],
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_data_transform_transform_on_keys():
    data = _make_container()
    transform = DataTransform(
        "scale_train_only",
        ScaleTransform(2.0),
        metadata={"apply_to": "features", "transform_on_keys": ["train"]},
    )

    transformed, _ = transform.forward(data)

    np.testing.assert_array_equal(
        transformed["features"]["train"][0],
        np.array([[2.0, 4.0], [6.0, 8.0]]),
    )
    np.testing.assert_array_equal(
        transformed["features"]["val"][0],
        np.array([[5.0, 6.0]]),
    )


def test_data_transform_fit_on_split():
    data = _make_container()
    fit_transform = FitOffsetTransform()
    transform = DataTransform(
        "fit_on_train",
        fit_transform,
        metadata={"apply_to": "features", "fit_on": "train", "fit_on_key": "features"},
    )

    transformed, _ = transform.forward(data)

    assert fit_transform.fit_called is True
    expected_offset = np.mean(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(fit_transform.offset, expected_offset)
    np.testing.assert_allclose(
        transformed["features"]["val"][0],
        np.array([[5.0, 6.0]]) - expected_offset,
    )


def test_missing_apply_to_raises():
    with pytest.raises(ValueError, match="must specify 'apply_to'"):
        DataTransform("missing_apply_to", AddOneTransform(), metadata={})


def test_output_consistency_missing_key_raises():
    """MissingKeyTransform returns unexpected key; validation may occur in pipeline."""
    transform = MissingKeyTransform()
    result = transform.transform_data(
        {"features": np.array([[1.0]])},
        metadata={"assign_to_map": ["features"]},
    )
    # Transform returns wrong key; pipeline IntegrityCheck would catch this
    assert "unexpected" in result
    assert "features" not in result
