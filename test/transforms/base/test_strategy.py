"""Tests for picid.transforms.base.strategy module.

This file consolidates all tests for TransformStrategy and postprocess_transformed_data
from multiple test files. All dummy transforms and fixtures are imported from conftest.
"""

import numpy as np
import pytest
import awkward as ak
from typing import Dict

from picid.exceptions import TransformError
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.strategy import (
    TransformStrategy,
    postprocess_transformed_data,
)
from picid.transforms.base.multisource import NoFitPerSegmentMixin
from picid.data.data_objects import (
    NamedTransformInput,
    SplitDatasetContainer,
    SimpleReturnObject,
    NamedDictReturnObject,
)

# Import shared fixtures and dummy transforms from conftest
from test.transforms.base.conftest import (
    DummyStatelessTransform,
    DummyFittableTransform,
    DummyRaggedOrDenseTransform,
    create_dummy_split_container,
    create_dummy_single_unit_container,
)


class TestTransformStrategy:
    """Comprehensive tests for TransformStrategy."""

    def test_strategy_apply_basic(self):
        """Test basic strategy apply."""
        strategy = TransformStrategy()
        transform = DummyStatelessTransform()
        container = create_dummy_single_unit_container()

        result, log = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
        )

        assert isinstance(result, SplitDatasetContainer)
        assert "features" in result

    def test_strategy_apply_multi_unit(self):
        """Test strategy apply with multi-unit data."""
        strategy = TransformStrategy()
        transform = DummyStatelessTransform()
        container = create_dummy_split_container(n_units=3)

        result, log = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
        )

        assert len(result.features.train) == 3

    def test_strategy_apply_with_fitting(self):
        """Test strategy apply with fitting."""
        strategy = TransformStrategy()
        transform = DummyFittableTransform()
        container = create_dummy_single_unit_container()

        result, log = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
            fit_on_split="train",
            fit_on_key="features",
        )

        assert transform.fitted

    def test_strategy_apply_ragged_or_dense_on_dense_handler_compat(self):
        """CAPABILITY_BOTH transform on numpy data must dispatch through dense handler."""
        strategy = TransformStrategy()
        transform = DummyRaggedOrDenseTransform()
        container = create_dummy_single_unit_container(n_samples=4, n_features=2)

        result, log = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
        )

        orig = container.features.train[0]
        np.testing.assert_array_equal(result.features.train[0], orig * 2)

    def test_strategy_apply_missing_apply_to_key_error(self):
        """Test that missing apply_to key raises error (wrapped in TransformError)."""
        strategy = TransformStrategy()
        transform = DummyStatelessTransform()
        container = create_dummy_single_unit_container()

        with pytest.raises(TransformError) as exc_info:
            strategy.apply(
                transform_instance=transform,
                data=container,
                apply_to_keys="nonexistent",
                assign_to_keys="nonexistent",
                assign_to_keys_map=["nonexistent"],
            )
        assert isinstance(exc_info.value.cause, KeyError)
        assert "not found" in str(exc_info.value.cause)

    def test_strategy_apply_no_apply_to_keys_error(self):
        """Test that empty apply_to_keys raises error (wrapped in TransformError)."""
        strategy = TransformStrategy()
        transform = DummyStatelessTransform()
        container = create_dummy_single_unit_container()

        with pytest.raises(TransformError) as exc_info:
            strategy.apply(
                transform_instance=transform,
                data=container,
                apply_to_keys=[],
                assign_to_keys="features",
                assign_to_keys_map=["features"],
            )
        assert isinstance(exc_info.value.cause, ValueError)
        assert "apply_to_keys" in str(exc_info.value.cause)

    def test_strategy_apply_invalid_structure_error(self):
        """Test that invalid data structure raises error (wrapped in TransformError)."""
        strategy = TransformStrategy()
        transform = DummyStatelessTransform()
        # Create container with missing features key
        container = SplitDatasetContainer(target={"train": [np.array([[1.0]])]})

        with pytest.raises(TransformError) as exc_info:
            strategy.apply(
                transform_instance=transform,
                data=container,
                apply_to_keys="features",
                assign_to_keys="features",
                assign_to_keys_map=["features"],
            )
        assert isinstance(exc_info.value.cause, KeyError)
        assert "not found" in str(exc_info.value.cause)

    def test_strategy_prepare_chunks_single_unit(self):
        """Test _prepare_chunks_for_split with single unit."""
        strategy = TransformStrategy()
        container = create_dummy_single_unit_container()

        chunks = strategy._prepare_chunks_for_split(
            data_to_prepare={"features": container.features},
            split="train",
        )

        assert len(chunks) == 1
        assert isinstance(chunks[0], NamedTransformInput)

    def test_strategy_prepare_chunks_multi_unit(self):
        """Test _prepare_chunks_for_split with multi-unit."""
        strategy = TransformStrategy()
        container = create_dummy_split_container(n_units=3)

        chunks = strategy._prepare_chunks_for_split(
            data_to_prepare={"features": container.features},
            split="train",
        )

        assert len(chunks) == 3
        for chunk in chunks:
            assert isinstance(chunk, NamedTransformInput)

    def test_strategy_merge_transformed_data_new_key(self):
        """Test _merge_transformed_data with new key."""
        strategy = TransformStrategy()
        base_data = create_dummy_single_unit_container()
        transformed_results = {
            "new_key": {
                "train": [np.array([[1.0, 2.0]])],
                "val": [np.array([[3.0, 4.0]])],
            }
        }

        strategy._merge_transformed_data(
            base_data=base_data,
            transformed_results=transformed_results,
            available_splits=["train", "val"],
        )

        assert "new_key" in base_data

    def test_strategy_merge_transformed_data_existing_key(self):
        """Test _merge_transformed_data with existing key."""
        strategy = TransformStrategy()
        base_data = create_dummy_single_unit_container()
        transformed_results = {"features": {"train": [np.array([[1.0, 2.0]])]}}

        strategy._merge_transformed_data(
            base_data=base_data,
            transformed_results=transformed_results,
            available_splits=["train"],
        )

        assert len(base_data.features.train) == 1


class TestPostprocessTransformedData:
    """Tests for postprocess_transformed_data function."""

    def test_postprocess_single_array(self):
        """Test postprocessing single array."""
        data = [np.array([1.0, 2.0, 3.0])]
        metadata = {"assign_to_map": ["features"]}
        result = postprocess_transformed_data(data, metadata)

        assert len(result) == 1
        assert isinstance(result[0], SimpleReturnObject)
        assert "features" in result[0]
        np.testing.assert_array_equal(result[0]["features"], np.array([1.0, 2.0, 3.0]))

    def test_postprocess_multiple_arrays(self):
        """Test postprocessing multiple arrays."""
        data = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        metadata = {"assign_to_map": ["features"]}
        result = postprocess_transformed_data(data, metadata)

        assert len(result) == 3
        for r in result:
            assert isinstance(r, SimpleReturnObject)
        np.testing.assert_array_equal(result[0]["features"], np.array([1.0]))
        np.testing.assert_array_equal(result[1]["features"], np.array([2.0]))
        np.testing.assert_array_equal(result[2]["features"], np.array([3.0]))

    def test_postprocess_named_dicts(self):
        """Test postprocessing named dictionaries."""
        data = [
            {"features": np.array([1.0]), "target": np.array([0.5])},
            {"features": np.array([2.0]), "target": np.array([0.6])},
        ]
        metadata = {"assign_to_map": ["features", "target"]}
        result = postprocess_transformed_data(data, metadata)

        assert len(result) == 2
        for r in result:
            assert isinstance(r, NamedDictReturnObject)
        np.testing.assert_array_equal(result[0]["features"], np.array([1.0]))
        np.testing.assert_array_equal(result[0]["target"], np.array([0.5]))

    def test_postprocess_missing_assign_to_map_error(self):
        """Test that missing assign_to_map raises error."""
        data = [np.array([1.0])]
        metadata = {}
        with pytest.raises(AssertionError, match="assign_to_map"):
            postprocess_transformed_data(data, metadata)

    def test_postprocess_unrecognized_format_error(self):
        """Mixed Mapping / non-Mapping chunks are rejected (same contract as strategy helper)."""
        data = [{"features": np.array([1.0])}, 42]
        metadata = {"assign_to_map": ["features"]}
        with pytest.raises(
            ValueError, match="not recognised|not recognized|not handled"
        ):
            postprocess_transformed_data(data, metadata)


class TestTransformStrategyAdvanced:
    """Advanced tests for TransformStrategy covering missing lines."""

    def test_strategy_assign_to_keys_map_string_conversion(self):
        """Test assign_to_keys_map string conversion (line 88)."""
        strategy = TransformStrategy()
        transform = DummyStatelessTransform()
        container = create_dummy_split_container()

        result, log = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map="features",  # String instead of list
        )

        assert isinstance(result, SplitDatasetContainer)
        np.testing.assert_array_equal(
            result.features.train[0], container.features.train[0] * 2
        )

    def test_strategy_ragged_array_copy_behavior(self):
        """Test strategy copy behavior with ragged arrays (line 99)."""
        strategy = TransformStrategy()
        transform = DummyStatelessTransform()

        # Create container with simple ragged arrays (single ragged dimension)
        ragged_data = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0]])
        container = SplitDatasetContainer(
            features={"train": [ragged_data], "val": [ragged_data]}
        )

        result, log = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
        )

        assert isinstance(result.features.train[0], ak.Array)
        assert ak.sum(result.features.train[0]) == ak.sum(ragged_data) * 2

    def test_strategy_invalid_mapping_structure_error(self):
        """Test strategy with invalid mapping structure (line 117)."""
        strategy = TransformStrategy()
        transform = DummyStatelessTransform()

        # Create container where features key doesn't exist or is wrong type
        container = SplitDatasetContainer(target={"train": [np.array([[1.0]])]})
        # features key is missing

        with pytest.raises(TransformError) as exc_info:
            strategy.apply(
                transform_instance=transform,
                data=container,
                apply_to_keys="features",
                assign_to_keys="features",
                assign_to_keys_map=["features"],
            )
        assert isinstance(exc_info.value.cause, KeyError)
        assert "not found" in str(exc_info.value.cause)

    def test_strategy_fit_on_split_with_fit_on_key(self):
        """Test strategy fit_on_split with fit_on_key (line 138)."""
        strategy = TransformStrategy()
        transform = DummyFittableTransform()
        container = create_dummy_split_container()

        result, log = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
            fit_on_split="train",
            fit_on_key="features",
        )

        assert transform.fitted
        assert isinstance(result, SplitDatasetContainer)

    def test_strategy_assign_to_map_missing_error(self):
        """Test strategy with missing assign_to_map in transformed chunk (line 192)."""
        strategy = TransformStrategy()

        # Transform that returns dict but with wrong key name
        class BadReturnTransform(NoFitPerSegmentMixin, DenseTransform):
            def transform_data(self, data: NamedTransformInput, metadata: Dict) -> Dict:
                # Returns dict but with wrong key
                return {"wrong_key": list(data.values())[0] * 2}

        transform = BadReturnTransform()
        container = create_dummy_split_container()

        # This should fail when trying to access assign_to_map in transformed chunk
        with pytest.raises(TransformError) as exc_info:
            strategy.apply(
                transform_instance=transform,
                data=container,
                apply_to_keys="features",
                assign_to_keys="scaled_features",
                assign_to_keys_map=[
                    "scaled_features"
                ],  # Transform returns "wrong_key" instead
            )
        assert isinstance(exc_info.value.cause, KeyError)
        assert "missing expected key" in str(
            exc_info.value.cause
        ) or "expected key" in str(exc_info.value.cause)


class TestStrategyCoverage:
    """Tests to cover remaining missing lines in strategy.py."""

    def test_strategy_ragged_array_shallow_copy(self):
        """Test strategy shallow copy for ragged arrays (line 99)."""
        strategy = TransformStrategy()
        transform = DummyStatelessTransform()

        # Create container with ragged arrays
        ragged = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0]])
        container = SplitDatasetContainer(features={"train": [ragged], "val": [ragged]})

        result, log = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
        )

        assert isinstance(result.features.train[0], ak.Array)
        assert ak.sum(result.features.train[0]) == ak.sum(ragged) * 2

    def test_strategy_transform_on_keys_filtering(self):
        """Test strategy with transform_on_keys filtering (line 150)."""
        strategy = TransformStrategy()
        transform = DummyStatelessTransform()
        container = create_dummy_split_container()

        result, log = strategy.apply(
            transform_instance=transform,
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
            transform_on_keys=["train", "val"],
        )

        orig_test0 = container.features.test[0].copy()
        np.testing.assert_array_equal(
            result.features.train[0], container.features.train[0] * 2
        )
        np.testing.assert_array_equal(
            result.features.val[0], container.features.val[0] * 2
        )
        np.testing.assert_array_equal(result.features.test[0], orig_test0)
