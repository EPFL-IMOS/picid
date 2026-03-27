"""Comprehensive tests for identity.py transform."""

import numpy as np
import pytest
import awkward as ak
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base_transforms.identity import IdentityPassThrough
from test.transforms.base.conftest import create_dummy_split_container


class TestIdentityPassThrough:
    """Tests for IdentityPassThrough transform."""

    def test_init(self):
        """Test initialization.

        **Assumption**: IdentityPassThrough should be initializable without any required
        parameters, and should accept (but ignore) arbitrary positional and keyword arguments.
        This makes it flexible and compatible with various initialization patterns in the
        framework, even if parameters are passed that it doesn't use.

        **Action**: Create IdentityPassThrough instances: first with no arguments, then with
        arbitrary positional args (1, 2, 3) and keyword args (a=4, b=5) that the transform
        doesn't actually use.

        **Expected Result**: Both instances should be created successfully (not None). This
        validates that the transform can be instantiated in different ways without errors,
        which is important for configuration-driven initialization where parameters might
        be passed that aren't relevant to this particular transform.
        """
        transform = IdentityPassThrough()
        assert transform is not None

        # Test with args and kwargs (should be ignored)
        transform2 = IdentityPassThrough(1, 2, 3, a=4, b=5)
        assert transform2 is not None

    def test_fit_data_does_nothing(self):
        """Test that fit_data does nothing.

        **Assumption**: IdentityPassThrough is a stateless transform that doesn't need to
        learn any parameters from the data. Therefore, fit_data should be a no-op that
        doesn't modify the transform's state and returns None.

        **Action**: Create an IdentityPassThrough transform and call fit_data with sample
        data and metadata. The transform should not raise any errors and should return None.

        **Expected Result**: fit_data should return None without raising exceptions. This
        validates that the identity transform correctly implements the fit interface as a
        no-op, which is important because some transforms require fitting while others don't,
        and the framework needs to handle both cases uniformly.
        """
        transform = IdentityPassThrough()
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}

        # Should not raise
        result = transform.fit_data(data, metadata)
        assert result is None

    def test_transform_data_returns_input(self):
        """Test that transform_data returns input unchanged.

        **Assumption**: IdentityPassThrough should return the exact same input object without
        any modifications. This is the core behavior of an identity transform - it passes
        data through unchanged, which is useful for testing, debugging, or as a placeholder
        in transformation pipelines.

        **Action**: Create an IdentityPassThrough transform, provide input data with features
        array [[1.0, 2.0], [3.0, 4.0]], and call transform_data.

        **Expected Result**: The result should be the exact same object (identity check with
        `is`) and the features array should be unchanged (element-wise equality). This is a
        critical test because if the identity transform modifies data, it breaks the contract
        and could cause subtle bugs in pipelines where it's used as a no-op placeholder.
        """
        transform = IdentityPassThrough()
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        metadata = {}

        result = transform.transform_data(data, metadata)

        # Should return the same object
        assert result is data
        np.testing.assert_array_equal(result["features"], data["features"])

    def test_transform_data_with_multiple_keys(self):
        """Test transform_data with multiple keys.

        **Assumption**: IdentityPassThrough should handle NamedTransformInput objects with
        multiple keys (e.g., both "features" and "target") and preserve all keys unchanged.
        This is important because real-world data often contains multiple data arrays that
        need to be passed through together.

        **Action**: Create an IdentityPassThrough transform and provide input data with both
        "features" and "target" keys, then call transform_data.

        **Expected Result**: The result should be the same object, and both "features" and
        "target" keys should be present in the result. This validates that the identity
        transform correctly handles multi-key data structures, which is essential for
        compatibility with the framework's data model where transforms often work with
        multiple related arrays simultaneously.
        """
        transform = IdentityPassThrough()
        data = NamedTransformInput(
            features=np.array([[1.0, 2.0]]), target=np.array([[0.5]])
        )
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert result is data
        assert "features" in result
        assert "target" in result

    def test_transform_data_with_ragged_array(self):
        """Test transform_data with ragged arrays.

        **Assumption**: IdentityPassThrough should work with both dense (numpy) arrays and
        ragged (awkward) arrays, returning them unchanged. This is important because the
        framework supports both data types, and an identity transform should be agnostic
        to the specific array type.

        **Action**: Create an IdentityPassThrough transform and provide input data containing
        a ragged awkward Array with variable-length rows ([1.0, 2.0, 3.0] and [4.0, 5.0]),
        then call transform_data.

        **Expected Result**: The result should be the same object, and the features should
        remain an awkward Array (not converted to numpy). This validates that the identity
        transform preserves the data type and structure, which is crucial for maintaining
        type consistency in transformation pipelines that mix dense and ragged data.
        """
        transform = IdentityPassThrough()
        ragged = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0]])
        data = NamedTransformInput(features=ragged)
        metadata = {}

        result = transform.transform_data(data, metadata)

        assert result is data
        assert isinstance(result["features"], ak.Array)

    def test_fit_multi_source(self):
        """Test fit_multi_source method.

        **Assumption**: IdentityPassThrough uses NoFitPerSegmentMixin (stateless); it does
        not support fitting. fit_multi_source must raise NotImplementedError so the pipeline
        does not attempt to fit this transform.

        **Action**: Create an IdentityPassThrough transform and call fit_multi_source with
        a list of two data segments and metadata. Expect NotImplementedError.

        **Expected Result**: fit_multi_source raises NotImplementedError with a message
        indicating the transform is stateless and does not support fitting.
        """
        transform = IdentityPassThrough()
        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0]])),
            NamedTransformInput(features=np.array([[3.0, 4.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}
        with pytest.raises(
            NotImplementedError, match="stateless transform and does not"
        ):
            transform.fit_multi_source(data_segments, metadata)

    def test_transform_multi_source(self):
        """Test transform_multi_source method.

        **Assumption**: IdentityPassThrough should support multi-source transformation,
        processing each data segment independently and returning them unchanged. The method
        should return a list of results (one per input segment) and a log dictionary.

        **Action**: Create an IdentityPassThrough transform and call transform_multi_source
        with a list of two data segments, each with different feature arrays. The transform
        should process each segment through the identity transformation.

        **Expected Result**: The result should be a list with 2 elements (one per input segment),
        and a log dictionary. Each result segment should contain the same features as the
        corresponding input segment (identity behavior). This validates that the multi-source
        transform interface works correctly and that the identity transform preserves data
        across multiple segments, which is essential for batch processing scenarios.
        """
        transform = IdentityPassThrough()
        data_segments = [
            NamedTransformInput(features=np.array([[1.0, 2.0]])),
            NamedTransformInput(features=np.array([[3.0, 4.0]])),
        ]
        metadata = {"apply_to_keys": ["features"]}

        result, log = transform.transform_multi_source(data_segments, metadata)

        assert isinstance(result, list)
        # transform_function_to_data_segments processes each segment individually
        assert len(result) == 2  # One result per input segment
        assert isinstance(log, dict)
        # Each result should be the same as input (identity transform)
        assert "features" in result[0]
        np.testing.assert_array_equal(
            result[0]["features"], data_segments[0]["features"]
        )
        np.testing.assert_array_equal(
            result[1]["features"], data_segments[1]["features"]
        )

    def test_with_split_dataset_container(self):
        """Test with SplitDatasetContainer.

        **Assumption**: IdentityPassThrough should work correctly with data extracted from
        SplitDatasetContainer objects, which are used to organize data into train/val/test
        splits. The transform should handle this data format without issues.

        **Action**: Create an IdentityPassThrough transform and a dummy SplitDatasetContainer
        with 2 units. Extract data from the train split's first unit and create a
        NamedTransformInput from it, then apply the transform.

        **Expected Result**: The result should be the same object, and the features should
        match the original train_data. This validates that the identity transform works
        correctly with the framework's standard data container structures, ensuring
        compatibility with real-world data loading and preprocessing pipelines.
        """
        transform = IdentityPassThrough()
        container = create_dummy_split_container(n_units=2)

        # Test on train split
        train_data = container.features.train[0]
        data = NamedTransformInput(features=train_data)
        metadata = {}

        result = transform.transform_data(data, metadata)
        assert result is data
        np.testing.assert_array_equal(result["features"], train_data)

    def test_empty_data(self):
        """Test with empty data.

        **Assumption**: IdentityPassThrough should handle edge cases gracefully, including
        empty NamedTransformInput objects (no keys). This is important for robustness and
        to prevent crashes in edge cases or during pipeline initialization.

        **Action**: Create an IdentityPassThrough transform and an empty NamedTransformInput
        (no keys/data), then call transform_data.

        **Expected Result**: The result should be the same object, and it should remain empty
        (length 0). This validates that the identity transform correctly handles edge cases
        without errors, which is important for defensive programming and ensuring the
        transform doesn't break when encountering unexpected but valid input states.
        """
        transform = IdentityPassThrough()
        data = NamedTransformInput()
        metadata = {}

        result = transform.transform_data(data, metadata)
        assert result is data
        assert len(result) == 0
