"""Tests for picid.transforms.base.data_transform module.

This file consolidates all tests for DataTransform from multiple test files.
All dummy transforms and fixtures are imported from conftest.
"""

import numpy as np
import pytest
from typing import Dict
from omegaconf import OmegaConf

from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin
from picid.transforms.base.data_transform import DataTransform, _type_matches
from picid.data.data_objects import (
    SplitDatasetContainer,
)
from picid.exceptions import TransformError

# Import shared fixtures and dummy transforms from conftest
from test.transforms.base.conftest import (
    DummyStatelessTransform,
    DummyFittableTransform,
    DummyMultiKeyTransform,
    DummyRaggedOrDenseTransform,
    NamedParameterTransform,
    MultiNamedParameterTransform,
    BadSignatureTransform,
    create_dummy_split_container,
    create_dummy_single_unit_container,
)


class TestDataTransform:
    """Comprehensive tests for DataTransform."""

    def test_data_transform_initialization_basic(self):
        """Test basic DataTransform initialization."""
        transform = DummyStatelessTransform()
        metadata = {"apply_to": "features", "assign_to": "features"}
        dt = DataTransform("test_transform", transform, metadata)

        assert dt.transform_name == "test_transform"
        assert dt.apply_to == "features"
        assert dt.assign_to == ["features"]
        assert dt.fit_on_split is None

    def test_data_transform_apply_to_string(self):
        """Test apply_to as string."""
        transform = DummyStatelessTransform()
        metadata = {"apply_to": "features"}
        dt = DataTransform("test", transform, metadata)
        assert dt.apply_to == "features"

    def test_data_transform_apply_to_list(self):
        """Test apply_to as list."""
        transform = DummyMultiKeyTransform()
        metadata = {"apply_to": ["features", "target"]}
        dt = DataTransform("test", transform, metadata)
        assert dt.apply_to == ["features", "target"]

    def test_data_transform_apply_to_missing_error(self):
        """Test that missing apply_to raises error."""
        transform = DummyStatelessTransform()
        metadata = {}
        with pytest.raises(ValueError, match="must specify 'apply_to'"):
            DataTransform("test", transform, metadata)

    def test_data_transform_assign_to_implicit(self):
        """Test implicit assign_to (same as apply_to)."""
        transform = DummyStatelessTransform()
        metadata = {"apply_to": "features"}
        dt = DataTransform("test", transform, metadata)
        assert dt.assign_to == "features"

    def test_data_transform_assign_to_explicit(self):
        """Test explicit assign_to."""
        transform = DummyStatelessTransform()
        metadata = {"apply_to": "features", "assign_to": "scaled_features"}
        dt = DataTransform("test", transform, metadata)
        assert dt.assign_to == ["scaled_features"]

    def test_data_transform_assign_to_list(self):
        """Test assign_to as list."""
        transform = DummyMultiKeyTransform()
        metadata = {
            "apply_to": ["features", "target"],
            "assign_to": ["feat_scaled", "targ_scaled"],
        }
        dt = DataTransform("test", transform, metadata)
        assert dt.assign_to == ["feat_scaled", "targ_scaled"]

    def test_data_transform_fit_on_split(self):
        """Test fit_on_split parameter."""
        transform = DummyFittableTransform()
        metadata = {"apply_to": "features", "fit_on": "train"}
        dt = DataTransform("test", transform, metadata)
        assert dt.fit_on_split == "train"
        assert dt.fit_on_key == "features"

    def test_data_transform_fit_on_split_with_fit_on_key(self):
        """Test fit_on_split with explicit fit_on_key."""
        transform = DummyFittableTransform()
        metadata = {"apply_to": "features", "fit_on": "train", "fit_on_key": "features"}
        dt = DataTransform("test", transform, metadata)
        assert dt.fit_on_split == "train"
        assert dt.fit_on_key == "features"

    def test_data_transform_fit_on_split_list_apply_to_error(self):
        """Test that fit_on with list apply_to requires fit_on_key."""
        transform = DummyFittableTransform()
        metadata = {"apply_to": ["features", "target"], "fit_on": "train"}
        with pytest.raises(ValueError, match="fit_on_key"):
            DataTransform("test", transform, metadata)

    def test_data_transform_fit_on_invalid_split_error(self):
        """Test that invalid fit_on split raises error."""
        transform = DummyFittableTransform()
        metadata = {"apply_to": "features", "fit_on": "invalid"}
        with pytest.raises(AssertionError):
            DataTransform("test", transform, metadata)

    def test_data_transform_non_base_transform_error(self):
        """Test that non-BaseTransform raises error."""

        class NotATransform:
            pass

        not_transform = NotATransform()
        metadata = {"apply_to": "features"}
        with pytest.raises(TypeError, match="BaseTransform"):
            DataTransform("test", not_transform, metadata)

    def test_data_transform_forward_basic(self):
        """Test basic forward pass."""
        transform = DummyStatelessTransform()
        metadata = {"apply_to": "features", "assign_to": "features"}
        dt = DataTransform("test", transform, metadata)

        container = create_dummy_single_unit_container(n_samples=5, n_features=2)
        result, log = dt.forward(container)

        assert "features" in result
        assert "train" in result.features
        assert len(result.features.train) == 1
        # Check transformation was applied (multiplied by 2)
        original = container.features.train[0]
        transformed = result.features.train[0]
        np.testing.assert_array_equal(transformed, original * 2)

    def test_data_transform_forward_ragged_or_dense_on_dense_compat(self):
        """RaggedOrDense (CAPABILITY_BOTH) on dense SplitDatasetContainer uses dense path."""
        transform = DummyRaggedOrDenseTransform()
        metadata = {"apply_to": "features", "assign_to": "features"}
        dt = DataTransform("rod_dense", transform, metadata)
        container = create_dummy_single_unit_container(n_samples=5, n_features=2)
        result, _ = dt.forward(container)
        original = container.features.train[0]
        np.testing.assert_array_equal(result.features.train[0], original * 2)

    def test_data_transform_forward_with_fitting(self):
        """Test forward pass with fitting."""
        transform = DummyFittableTransform()
        metadata = {"apply_to": "features", "assign_to": "features", "fit_on": "train"}
        dt = DataTransform("test", transform, metadata)

        container = create_dummy_single_unit_container(n_samples=5, n_features=2)
        result, log = dt.forward(container)

        assert "features" in result
        assert transform.fitted

    def test_fittable_transform_uses_train_fit_state_for_non_train_splits(self):
        """Non-train splits are transformed with train-fitted parameters, not refit per split.

        **Bug sentinel**: ``DummyFittableTransform`` scales by the mean learned at fit time.
        Train mean is 10; val mean is 2 and test mean is 3. Correct behavior uses factor
        10 for both non-train splits. Refitting per split would produce 4 and 9 instead.
        """
        transform = DummyFittableTransform()
        metadata = {"apply_to": "features", "assign_to": "features", "fit_on": "train"}
        dt = DataTransform("fit_sentinel", transform, metadata)

        container = SplitDatasetContainer(
            features={
                "train": [np.full((4, 2), 10.0, dtype=np.float64)],
                "val": [np.full((4, 2), 2.0, dtype=np.float64)],
                "test": [np.full((4, 2), 3.0, dtype=np.float64)],
            },
        )
        result, _ = dt.forward(container)

        expected_val = np.full((4, 2), 20.0, dtype=np.float64)
        expected_test = np.full((4, 2), 30.0, dtype=np.float64)
        np.testing.assert_array_equal(result.features.val[0], expected_val)
        np.testing.assert_array_equal(result.features.test[0], expected_test)
        assert not np.allclose(
            result.features.val[0], np.full((4, 2), 4.0, dtype=np.float64)
        )
        assert not np.allclose(
            result.features.test[0], np.full((4, 2), 9.0, dtype=np.float64)
        )

    def test_data_transform_forward_new_key(self):
        """Test forward pass assigning to new key."""
        transform = DummyStatelessTransform()
        metadata = {"apply_to": "features", "assign_to": "scaled_features"}
        dt = DataTransform("test", transform, metadata)

        container = create_dummy_single_unit_container(n_samples=5, n_features=2)
        result, log = dt.forward(container)

        assert "scaled_features" in result
        assert "features" in result  # Original should still be there

    def test_data_transform_forward_multi_unit(self):
        """Test forward pass with multi-unit data."""
        transform = DummyStatelessTransform()
        metadata = {"apply_to": "features", "assign_to": "features"}
        dt = DataTransform("test", transform, metadata)

        container = create_dummy_split_container(
            n_units=3, n_samples_per_unit=5, n_features=2
        )
        result, log = dt.forward(container)

        assert len(result.features.train) == 3
        for i in range(3):
            original = container.features.train[i]
            transformed = result.features.train[i]
            np.testing.assert_array_equal(transformed, original * 2)

    def test_data_transform_forward_missing_key_error(self):
        """Test that missing apply_to key raises error (wrapped in TransformError)."""
        transform = DummyStatelessTransform()
        metadata = {"apply_to": "nonexistent", "assign_to": "nonexistent"}
        dt = DataTransform("test", transform, metadata)

        container = create_dummy_single_unit_container()
        with pytest.raises(TransformError) as exc_info:
            dt.forward(container)
        assert isinstance(exc_info.value.cause, KeyError)
        assert "not found" in str(exc_info.value.cause)

    def test_data_transform_forward_transform_on_keys(self):
        """transform_on_keys=['train']: train scaled x2; val left unchanged."""
        transform = DummyStatelessTransform()
        metadata = {
            "apply_to": "features",
            "assign_to": "features",
            "transform_on_keys": ["train"],
        }
        dt = DataTransform("test", transform, metadata)

        container = create_dummy_single_unit_container()
        orig_val = container.features.val[0].copy()
        result, log = dt.forward(container)

        np.testing.assert_array_equal(
            result.features.train[0], container.features.train[0] * 2
        )
        np.testing.assert_array_equal(result.features.val[0], orig_val)

    def test_data_transform_process_apply_to_listconfig(self):
        """Test _process_apply_to with ListConfig."""
        transform = DummyStatelessTransform()
        metadata = {"apply_to": OmegaConf.create(["features", "target"])}
        dt = DataTransform("test", transform, metadata)
        assert isinstance(dt.apply_to, list)
        assert "features" in dt.apply_to
        assert "target" in dt.apply_to

    def test_data_transform_process_assign_to_mapping_error(self):
        """Test apply_to as mapping - init succeeds and processes mapping."""
        transform = DummyMultiKeyTransform()
        metadata = {"apply_to": {"feat": "features", "targ": "target"}}
        dt = DataTransform("test", transform, metadata)
        assert dt.apply_to_map == ["feat", "targ"]
        assert "features" in dt.apply_to and "target" in dt.apply_to

    def test_data_transform_process_assign_to_unsupported_type_error(self):
        """Test _process_assign_to with unsupported type."""
        transform = DummyStatelessTransform()
        metadata = {"apply_to": "features", "assign_to": 123}  # Invalid type
        with pytest.raises(ValueError, match="unsupported assign_to type"):
            DataTransform("test", transform, metadata)

    def test_data_transform_signature_validation(self):
        """Test transform signature validation."""
        bad_transform = BadSignatureTransform()
        metadata = {"apply_to": "features"}
        with pytest.raises(TypeError, match="signature"):
            DataTransform("test", bad_transform, metadata)

    def test_data_transform_forward_multi_key_unit_count_mismatch_raises(self):
        """Misaligned unit counts across apply_to keys for the same split raise TransformError."""
        transform = DummyMultiKeyTransform()
        metadata = {
            "apply_to": ["features", "target"],
            "assign_to": ["features", "target"],
        }
        dt = DataTransform("test", transform, metadata)
        container = SplitDatasetContainer(
            features={"train": [np.array([[1.0]]), np.array([[2.0]])]},
            target={"train": [np.array([[0.1]])]},
        )
        with pytest.raises(TransformError) as exc_info:
            dt.forward(container)
        assert isinstance(exc_info.value.cause, ValueError)
        assert "Misaligned" in str(exc_info.value.cause)


class TestDataTransformAdvanced:
    """Advanced tests for DataTransform covering missing lines."""

    def test_apply_to_as_mapping_raises_not_implemented(self):
        """Test that apply_to as mapping raises NotImplementedError (line 281)."""
        transform = DummyStatelessTransform()
        metadata = {
            "apply_to": {"feat": "features", "targ": "target"},
            "assign_to": {"feat": "scaled_features", "targ": "scaled_target"},
        }
        dt = DataTransform("test", transform, metadata)

        container = create_dummy_split_container(n_units=2)
        container.target = {
            "train": [np.array([[0.1], [0.2]]) for _ in range(2)],
            "val": [np.array([[0.3]]) for _ in range(2)],
        }

        with pytest.raises(NotImplementedError, match="apply_to as a mapping"):
            dt.forward(container)

    def test_named_parameter_transform_signature_validation(self):
        """Test signature validation with named parameters (lines 221-234)."""
        transform = NamedParameterTransform()
        metadata = {"apply_to": {"features": "features"}, "assign_to": "features"}

        # Should pass validation
        dt = DataTransform("test", transform, metadata)
        assert dt.transform_name == "test"
        assert dt.apply_to_map == ["features"]

    def test_multi_named_parameter_transform_signature_validation(self):
        """Test signature validation with multiple named parameters."""
        transform = MultiNamedParameterTransform()
        metadata = {
            "apply_to": {"features": "features", "target": "target"},
            "assign_to": {"features": "scaled_features", "target": "scaled_target"},
        }

        dt = DataTransform("test", transform, metadata)
        assert len(dt.apply_to_map) == 2
        assert "features" in dt.apply_to_map
        assert "target" in dt.apply_to_map

    def test_named_parameter_missing_metadata_error(self):
        """Test that named parameter transform without metadata raises error."""

        class TransformWithoutMetadata(NoFitPerSegmentMixin, DenseTransform):
            def transform_data(self, features: np.ndarray):  # No metadata
                return features * 2

        transform = TransformWithoutMetadata()
        metadata = {"apply_to": {"features": "features"}}

        with pytest.raises(AssertionError, match="metadata"):
            DataTransform("test", transform, metadata)

    def test_named_parameter_wrong_count_error(self):
        """Test that wrong number of named parameters raises error."""

        class WrongCountTransform(NoFitPerSegmentMixin, DenseTransform):
            def transform_data(
                self, features: np.ndarray, extra: np.ndarray, metadata: Dict
            ):
                return features * 2

        transform = WrongCountTransform()
        metadata = {"apply_to": {"features": "features"}}

        with pytest.raises(
            AssertionError, match="transform_data must have .* data parameter"
        ):
            DataTransform("test", transform, metadata)

    def test_named_parameter_missing_key_error(self):
        """Test that missing key in named parameters raises error."""

        class MissingKeyTransform(NoFitPerSegmentMixin, DenseTransform):
            def transform_data(self, wrong_name: np.ndarray, metadata: Dict):
                return wrong_name * 2

        transform = MissingKeyTransform()
        metadata = {"apply_to": {"features": "features"}}

        with pytest.raises(AssertionError, match="must include"):
            DataTransform("test", transform, metadata)

    def test_process_apply_to_with_mapping(self):
        """Test _process_apply_to with mapping (lines 75-77).

        **Methodology**: Testing that _process_apply_to correctly handles Mapping
        inputs during initialization. After __init__, apply_to should be a list
        of values and apply_to_map should be a list of keys.
        """
        transform = DummyStatelessTransform()
        metadata = {"apply_to": {"feat": "features", "targ": "target"}}

        # Create DataTransform - _process_apply_to is called in __init__
        dt = DataTransform("test", transform, metadata)

        # Verify the mapping was processed correctly during initialization
        # apply_to should be converted to list of values
        assert isinstance(dt.apply_to, list)
        assert "features" in dt.apply_to
        assert "target" in dt.apply_to
        # apply_to_map should contain the original keys
        assert dt.apply_to_map == ["feat", "targ"]

    def test_process_assign_to_with_mapping(self):
        """Test _process_assign_to with mapping (lines 103-105).

        **Methodology**: Testing _process_assign_to functionality through proper
        DataTransform initialization. The assign_to mapping is processed during
        __init__, so we verify the results through public attributes.
        """
        transform = DummyStatelessTransform()
        metadata = {"apply_to": "features", "assign_to": {"scaled": "scaled_features"}}

        # Create DataTransform properly - _process_assign_to is called in __init__
        dt = DataTransform("test", transform, metadata)

        # Verify assign_to was processed correctly through public attributes
        assert isinstance(dt.assign_to, list)
        assert "scaled_features" in dt.assign_to

        # Also test internal method directly for coverage
        assign_to, assign_to_map = dt._process_assign_to(
            metadata["assign_to"], "features"
        )
        assert isinstance(assign_to, list)
        assert "scaled_features" in assign_to
        assert assign_to_map == ["scaled"]


class TestDataTransformCoverage:
    """Tests to cover remaining missing lines in data_transform.py."""

    def test_type_matches_typeerror_handling(self):
        """Test _type_matches TypeError handling (line 42)."""
        # Create annotation that will cause TypeError in issubclass
        invalid_annotation = "not_a_class"

        # This should catch TypeError and return False
        result = _type_matches(invalid_annotation, type)
        assert result is False

    def test_process_apply_to_listconfig(self):
        """Test _process_apply_to with ListConfig (line 71).

        **Methodology**: Testing ListConfig handling through proper DataTransform
        initialization. OmegaConf.create is used to simulate Hydra config.
        """
        transform = DummyMultiKeyTransform()  # Use transform that accepts multiple keys
        metadata = {"apply_to": OmegaConf.create(["features", "target"])}

        # Create DataTransform properly - _process_apply_to is called in __init__
        dt = DataTransform("test", transform, metadata)

        # Verify through public attributes
        assert isinstance(dt.apply_to, list)
        assert "features" in dt.apply_to

        # Also test internal method directly for coverage
        apply_to, apply_to_map = dt._process_apply_to(metadata["apply_to"])
        assert isinstance(apply_to, list)
        assert "features" in apply_to

    def test_process_assign_to_sequence(self):
        """Test _process_assign_to with sequence (lines 111-113).

        **Methodology**: Testing sequence handling through proper DataTransform
        initialization.
        """
        transform = DummyStatelessTransform()
        metadata = {"apply_to": "features", "assign_to": ["feat1", "feat2"]}

        # Create DataTransform properly - _process_assign_to is called in __init__
        dt = DataTransform("test", transform, metadata)

        # Verify through public attributes
        assert isinstance(dt.assign_to, list)
        assert len(dt.assign_to) == 2

        # Also test internal method directly for coverage
        assign_to, assign_to_map = dt._process_assign_to(
            metadata["assign_to"], "features"
        )
        assert isinstance(assign_to, list)
        assert len(assign_to) == 2

    def test_forward_validation_checks(self):
        """Test forward validation checks (lines 318, 326, 337, 348, 360)."""
        transform = DummyStatelessTransform()
        metadata = {"apply_to": "features", "assign_to": "features"}
        dt = DataTransform("test", transform, metadata)

        container = create_dummy_split_container()

        # Test that normal forward works and validates correctly
        result, log = dt.forward(container)

        # Validate output structure
        assert isinstance(result, SplitDatasetContainer)
        assert "features" in result
        assert "train" in result.features
        assert "val" in result.features
        assert "test" in result.features

        # Validate transformation was applied
        original_train = container.features.train[0]
        transformed_train = result.features.train[0]
        np.testing.assert_array_equal(transformed_train, original_train * 2)

    def test_forward_normal_path_keeps_warning_branches_inactive(self):
        """Normal forward does not hit the mocked type-change warning scenarios."""
        import warnings

        transform = DummyStatelessTransform()
        metadata = {"apply_to": "features", "assign_to": "features"}
        dt = DataTransform("test", transform, metadata)

        container = create_dummy_split_container(n_units=3)

        # Normal forward should work without warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, log = dt.forward(container)
            # Should not generate warnings for normal case
            assert len(w) == 0 or all(
                "changed type" not in str(warning.message) for warning in w
            )
