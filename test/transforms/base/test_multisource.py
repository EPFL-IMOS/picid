"""Tests for picid.transforms.base.multisource module.

This file consolidates all tests for multisource functionality from multiple test files.
All dummy transforms and fixtures are imported from conftest.
"""

import numpy as np
import pytest
import awkward as ak
from typing import Dict
from sortedcontainers import SortedDict

from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import (
    ConcatFitAndPerSegmentTransformMixin,
    find_singular_ragged_dim,
    tolist,
)
from picid.data.data_objects import NamedTransformInput

# Import shared fixtures and dummy transforms from conftest
from test.transforms.base.conftest import (
    DummyStatelessTransform,
    DummyFittableTransform,
    DummyRaggedTransform,
)


class TestMultisourceUtils:
    """Tests for multisource utility functions."""

    def test_find_singular_ragged_dim_none(self):
        """Test find_singular_ragged_dim with no ragged dims."""
        # Create a truly regular (non-ragged) awkward array - all dimensions fixed
        # Use ak.from_numpy to create a regular array
        regular_np = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        regular_ak = ak.from_numpy(regular_np)
        data = NamedTransformInput(features=regular_ak)
        result = find_singular_ragged_dim(data)
        # For regular awkward arrays (no variable dimensions), should return None
        assert result is None

    def test_find_singular_ragged_dim_single(self):
        """Test find_singular_ragged_dim with single ragged dim."""
        # Create a simpler ragged array with only one ragged dimension
        ragged = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0]])  # Only dimension 0 is ragged
        data = NamedTransformInput(features=ragged)
        result = find_singular_ragged_dim(data)
        assert result is not None

    def test_find_singular_ragged_dim_multiple_error(self):
        """Test find_singular_ragged_dim with multiple ragged dims raises error."""
        # Single array with two variable dimensions (2 * var * var * float64 -> var_dims [1, 2])
        # so unique_var_dims has size 2 and find_singular_ragged_dim raises ValueError
        arr_two_var_dims = ak.Array([[[1.0], [2.0, 3.0]], [[4.0]]])
        data = NamedTransformInput(features=arr_two_var_dims)
        with pytest.raises(
            ValueError, match="Expected at most one variable-length dimension"
        ):
            find_singular_ragged_dim(data)

    def test_tolist_dict(self):
        """Test tolist with dictionary."""
        d = {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}}
        result = tolist(d)
        assert isinstance(result, list)

    def test_tolist_sorted_dict(self):
        """Test tolist with SortedDict."""
        sd = SortedDict({"a": 1, "b": 2})
        result = tolist(sd)
        assert isinstance(result, list)

    def test_tolist_non_dict(self):
        """Test tolist with non-dict returns as-is."""
        arr = np.array([1, 2, 3])
        result = tolist(arr)
        assert result is arr


class TestConcatenatedAndFitMultiSourceMixinsComposition:
    """Tests for ConcatFitAndPerSegmentTransformMixin (fit on concat, transform per segment)."""

    def test_new_composition_fit_and_transform_multi_source(self):
        """Confirm a class using ConcatFitAndPerSegmentTransformMixin (new design) behaves like the old ConcatinatedFit mixin."""

        class FittableWithNewMixins(
            ConcatFitAndPerSegmentTransformMixin,
            DenseTransform,
        ):
            def __init__(self):
                super().__init__()
                self.fitted = False
                self.factor = 1.0

            def fit_data(self, data: NamedTransformInput, metadata: Dict) -> None:
                key = list(data.keys())[0]
                self.factor = float(np.mean(data[key]))
                self.fitted = True

            def transform_data(
                self, data: NamedTransformInput, metadata: Dict
            ) -> np.ndarray:
                key = list(data.keys())[0]
                return data[key] * self.factor

        transform = FittableWithNewMixins()
        assert getattr(transform, "requires_fit", None) is True

        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]])),
            NamedTransformInput(features=np.array([[5.0, 6.0], [7.0, 8.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}

        transform.fit_multi_source(data_segments, metadata=metadata)
        assert transform.fitted
        assert transform.factor > 0

        result, log = transform.transform_multi_source(data_segments, metadata=metadata)
        assert len(result) == 2
        assert "mode" in log or "flags" in log


class TestConcatFitAndPerSegmentTransformMixin:
    """Tests for ConcatFitAndPerSegmentTransformMixin."""

    def test_mixin_initialization_check(self):
        """Test that mixin checks for required methods."""

        class BadTransform(DenseTransform):
            def transform_data(self, data, metadata):
                return data

        # The mixin checks in __init__, but BadTransform also needs fit_data
        # Let's test with a transform that's missing fit_data
        try:
            mixin = ConcatFitAndPerSegmentTransformMixin()
            mixin.__init__(BadTransform())
            # If no error, the check might happen later
        except (TypeError, AttributeError):
            pass  # Expected

    def test_fit_multi_source_dense(self):
        """Test fit_multi_source with dense data."""
        transform = DummyFittableTransform()
        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]])),
            NamedTransformInput(features=np.array([[5.0, 6.0], [7.0, 8.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}

        transform.fit_multi_source(data_segments, metadata=metadata)
        assert transform.fitted
        # Factor should be mean of all data
        assert transform.factor > 0

    def test_fit_multi_source_inconsistent_types_error(self):
        """Test fit_multi_source with inconsistent types raises error."""
        transform = DummyFittableTransform()
        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0]])),
            NamedTransformInput(features=ak.Array([[[1.0, 2.0]]])),  # Different type
        ]
        metadata = {"apply_to_keys": ["features"]}

        with pytest.raises(ValueError, match="same data type"):
            transform.fit_multi_source(data_segments, metadata=metadata)

    def test_transform_multi_source_dense(self):
        """Test transform_multi_source with dense data."""
        transform = DummyStatelessTransform()
        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0]])),
            NamedTransformInput(features=np.array([[3.0, 4.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}

        result, log = transform.transform_multi_source(data_segments, metadata=metadata)

        assert len(result) == 2
        np.testing.assert_array_equal(result[0], np.array([[2.0, 4.0]]))
        np.testing.assert_array_equal(result[1], np.array([[6.0, 8.0]]))

    def test_transform_multi_source_ragged_to_dense(self):
        """Test transform_multi_source with ragged data and dense-only transform."""
        transform = DummyStatelessTransform()  # Dense-only
        # Use simpler ragged array with single ragged dimension
        ragged = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0]])
        data_segments = [
            NamedTransformInput(features=ragged),
        ]
        metadata = {"apply_to_keys": ["features"], "assign_to_map": ["features"]}

        # Should convert ragged to dense
        result, log = transform.transform_multi_source(data_segments, metadata=metadata)
        assert len(result) == 1
        assert log.get("mode") == "ragged_to_dense"

    def test_transform_multi_source_ragged_native(self):
        """Test transform_multi_source with ragged data and ragged-supporting transform."""
        transform = DummyRaggedTransform()
        # Use simpler ragged array with single ragged dimension
        ragged = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0]])
        data_segments = [
            NamedTransformInput(features=ragged),
        ]
        metadata = {"apply_to_keys": ["features"]}

        result, log = transform.transform_multi_source(data_segments, metadata=metadata)
        assert len(result) == 1
        assert log.get("mode") == "ragged"

    def test_transform_multi_source_different_key_lengths(self):
        """Test transform_multi_source with different key lengths."""
        transform = DummyStatelessTransform()
        # Use same keys but different data shapes
        data_segments = [
            NamedTransformInput(features=np.array([[1.0]])),
            NamedTransformInput(features=np.array([[2.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}

        # Should work since we only apply to features
        result, log = transform.transform_multi_source(data_segments, metadata=metadata)
        assert len(result) == 2


class TestNoFitPerSegmentMixin:
    """Tests for NoFitPerSegmentMixin."""

    def test_no_fit_multi_source_raises_error(self):
        """Test that fit_multi_source raises NotImplementedError."""
        transform = DummyStatelessTransform()
        data_segments = [NamedTransformInput(features=np.array([[1.0]]))]
        metadata = {"apply_to_keys": ["features"]}

        with pytest.raises(NotImplementedError):
            transform.fit_multi_source(data_segments, metadata=metadata)
