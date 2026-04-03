"""Shared fixtures and dummy transforms for base module tests.

This conftest.py provides all dummy transforms, helper functions, and fixtures
used across the base module test files.
"""

import numpy as np
import pytest
import awkward as ak
from typing import Any, Dict, List

from picid.transforms.base.base_transform import (
    DenseTransform,
    RaggedTransform,
    RaggedOrDenseTransform,
)
from picid.transforms.base.multisource import (
    ConcatFitAndPerSegmentTransformMixin,
    NoFitConcatAlongAxisMixin,
    NoFitPerSegmentMixin,
    InverseTransformMixin,
)
from picid.data.data_objects import (
    NamedTransformInput,
    SplitDatasetContainer,
)

from test.fixtures.builders import (
    make_standard_normal_2d,
    split_container_unit_seed,
)


# ============================================================================
# DUMMY TRANSFORMS FOR TESTING
# ============================================================================


class DummyStatelessTransform(NoFitPerSegmentMixin, DenseTransform):
    """A stateless transform that multiplies by 2."""

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> np.ndarray:
        key = list(data.keys())[0]
        return data[key] * 2


class DummyFittableTransform(ConcatFitAndPerSegmentTransformMixin, DenseTransform):
    """A fittable transform that learns a scaling factor."""

    def __init__(self, initial_factor=1.0):
        super().__init__()
        self.factor = initial_factor
        self.fitted = False

    def fit_data(self, data: NamedTransformInput, metadata: Dict[str, Any]) -> None:
        """Fit by computing mean and storing as factor."""
        key = list(data.keys())[0]
        self.factor = np.mean(data[key])
        self.fitted = True

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Transform must be fitted before transformation")
        key = list(data.keys())[0]
        return data[key] * self.factor


class DummyMultiKeyTransform(NoFitPerSegmentMixin, DenseTransform):
    """A transform that takes multiple keys and returns a dict."""

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        result = {}
        for key in data.keys():
            result[key] = data[key] * 2
        return result


class DummyRaggedTransform(NoFitPerSegmentMixin, RaggedTransform):
    """A transform that works with ragged arrays."""

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> ak.Array:
        key = list(data.keys())[0]
        # Simple transformation: multiply by 2
        return data[key] * 2


class DummyRaggedOrDenseTransform(NoFitPerSegmentMixin, RaggedOrDenseTransform):
    """A transform that works with both ragged and dense arrays."""

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> Any:
        key = list(data.keys())[0]
        result = data[key] * 2
        # Convert back to same type
        if isinstance(data[key], ak.Array):
            return ak.Array(result) if not isinstance(result, ak.Array) else result
        return result


class DummyInverseTransform(
    NoFitPerSegmentMixin, InverseTransformMixin, DenseTransform
):
    """Minimal transform implementing InverseTransformMixin for get_inverter_for_key tests."""

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> np.ndarray:
        key = list(data.keys())[0]
        return data[key] * 2

    def inverse_transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any] = None
    ) -> np.ndarray:
        key = list(data.keys())[0]
        return data[key] / 2

    def inverse_transform(
        self, data: NamedTransformInput, metadata: dict = None
    ) -> np.ndarray:
        return self.inverse_transform_data(data, metadata)


class DummyTransformWithMetadata(DenseTransform, NoFitPerSegmentMixin):
    """A transform that uses metadata."""

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> np.ndarray:
        key = list(data.keys())[0]
        multiplier = metadata.get("multiplier", 1.0)
        _mode = metadata.get("mode", "unknown")
        return data[key] * multiplier


class NamedParameterTransform(NoFitPerSegmentMixin, DenseTransform):
    """Transform that uses named parameters matching apply_to keys."""

    def transform_data(
        self, features: np.ndarray, metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Transform with named parameter matching apply_to key."""
        return features * 2


class MultiNamedParameterTransform(NoFitPerSegmentMixin, DenseTransform):
    """Transform with multiple named parameters."""

    def transform_data(
        self, features: np.ndarray, target: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Transform with multiple named parameters."""
        return {"features": features * 2, "target": target * 3}


class BadSignatureTransform(NoFitPerSegmentMixin, DenseTransform):
    """Transform with invalid signature for testing."""

    def transform_data(self, data):  # Missing metadata parameter
        return data


class TransformReturningDict(NoFitPerSegmentMixin, DenseTransform):
    """Transform that returns a dictionary."""

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        result = {}
        for key, value in data.items():
            result[key] = value * 2
        return result


class TransformReturningList(NoFitPerSegmentMixin, DenseTransform):
    """Transform that returns a list of arrays."""

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> List[np.ndarray]:
        return [value * 2 for value in data.values()]


class RaggedSupportingTransform(NoFitPerSegmentMixin, RaggedTransform):
    """Transform that supports ragged arrays."""

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> ak.Array:
        key = list(data.keys())[0]
        return data[key] * 2


class ConcatenateUnitsTransform(NoFitConcatAlongAxisMixin, DenseTransform):
    """Transform that concatenates units."""

    def __init__(self, axis: int = 0):
        DenseTransform.__init__(self)
        NoFitConcatAlongAxisMixin.__init__(self, axis=axis)

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> np.ndarray:
        key = list(data.keys())[0]
        return data[key] * 2


# ============================================================================
# DUMMY DATA GENERATORS
# ============================================================================


def create_dummy_dense_data(n_samples=10, n_features=3, seed=42):
    """Create dummy dense numpy array (deterministic unless ``seed`` is overridden)."""
    return make_standard_normal_2d(seed=seed, n_rows=n_samples, n_cols=n_features)


def create_dummy_ragged_data():
    """Create dummy ragged array."""
    return ak.Array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    )


def create_dummy_split_container(
    n_units=3, n_samples_per_unit=10, n_features=3, base_seed=42
):
    """Create a dummy SplitDatasetContainer."""
    return SplitDatasetContainer(
        features={
            "train": [
                make_standard_normal_2d(
                    seed=split_container_unit_seed(base_seed, "train", i, 0),
                    n_rows=n_samples_per_unit,
                    n_cols=n_features,
                )
                for i in range(n_units)
            ],
            "val": [
                make_standard_normal_2d(
                    seed=split_container_unit_seed(base_seed, "val", i, 0),
                    n_rows=n_samples_per_unit,
                    n_cols=n_features,
                )
                for i in range(n_units)
            ],
            "test": [
                make_standard_normal_2d(
                    seed=split_container_unit_seed(base_seed, "test", i, 0),
                    n_rows=n_samples_per_unit,
                    n_cols=n_features,
                )
                for i in range(n_units)
            ],
        },
        target={
            "train": [
                make_standard_normal_2d(
                    seed=split_container_unit_seed(base_seed, "train", i, 1),
                    n_rows=n_samples_per_unit,
                    n_cols=1,
                )
                for i in range(n_units)
            ],
            "val": [
                make_standard_normal_2d(
                    seed=split_container_unit_seed(base_seed, "val", i, 1),
                    n_rows=n_samples_per_unit,
                    n_cols=1,
                )
                for i in range(n_units)
            ],
            "test": [
                make_standard_normal_2d(
                    seed=split_container_unit_seed(base_seed, "test", i, 1),
                    n_rows=n_samples_per_unit,
                    n_cols=1,
                )
                for i in range(n_units)
            ],
        },
    )


def create_dummy_single_unit_container(n_samples=10, n_features=3, base_seed=42):
    """Create a single-unit SplitDatasetContainer."""
    return SplitDatasetContainer(
        features={
            "train": [create_dummy_dense_data(n_samples, n_features, seed=base_seed)],
            "val": [create_dummy_dense_data(n_samples, n_features, seed=base_seed + 1)],
        },
    )


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture
def dummy_stateless_transform():
    """Fixture providing a DummyStatelessTransform instance."""
    return DummyStatelessTransform()


@pytest.fixture
def dummy_fittable_transform():
    """Fixture providing a DummyFittableTransform instance."""
    return DummyFittableTransform()


@pytest.fixture
def sample_dense_data():
    """Fixture providing sample dense data."""
    return create_dummy_dense_data()


@pytest.fixture
def sample_ragged_data():
    """Fixture providing sample ragged data."""
    return create_dummy_ragged_data()


@pytest.fixture
def sample_split_container():
    """Fixture providing a sample SplitDatasetContainer."""
    return create_dummy_split_container()


@pytest.fixture
def sample_single_unit_container():
    """Fixture providing a single-unit SplitDatasetContainer."""
    return create_dummy_single_unit_container()
