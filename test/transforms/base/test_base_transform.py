"""Tests for base transform functionality.

This file tests picid.transforms.base.base_transform module.
All dummy transforms and fixtures are imported from conftest.
"""

import numpy as np
import pytest
from picid.transforms.base.base_transform import (
    BaseTransform,
    DenseTransform,
    RaggedTransform,
    RaggedOrDenseTransform,
)
from picid.data.data_objects import NamedTransformInput

# Import shared fixtures and dummy transforms from conftest
from test.transforms.base.conftest import (
    DummyStatelessTransform,
    DummyRaggedTransform,
    DummyRaggedOrDenseTransform,
)


class TestBaseTransform:
    """Test suite for BaseTransform interface."""

    def test_base_transform_initialization_defaults(self):
        """Test BaseTransform initialization with defaults.

        **Assumption**: BaseTransform should initialize with default values: empty
        exclude_keys list and empty _init_kwargs dict. These defaults allow the transform
        to work out-of-the-box without requiring explicit configuration.

        **Action**: Create a DummyStatelessTransform instance (which inherits from
        BaseTransform) without any initialization parameters.

        **Expected Result**: The transform should have exclude_keys=[] and _init_kwargs={}.
        This validates that default initialization works correctly, ensuring transforms
        can be created and used without mandatory parameters, which is important for
        flexible configuration and backward compatibility.
        """
        transform = DummyStatelessTransform()
        assert transform.exclude_keys == []
        assert transform._init_kwargs == {}

    def test_base_transform_initialization_with_exclude_keys(self):
        """Test BaseTransform with exclude_keys.

        **Assumption**: BaseTransform should accept exclude_keys parameter during initialization
        to specify which keys should be excluded from transformation operations.

        **Action**: Create a DummyStatelessTransform instance with exclude_keys=["target", "metadata"].

        **Expected Result**: The exclude_keys list should contain both "target" and "metadata".
        This is important for transforms that need to skip certain data keys during processing.
        """
        transform = DummyStatelessTransform(exclude_keys=["target", "metadata"])
        assert len(transform.exclude_keys) == 2
        assert "target" in transform.exclude_keys
        assert "metadata" in transform.exclude_keys

    def test_base_transform_initialization_with_kwargs(self):
        """Test BaseTransform with additional kwargs.

        **Assumption**: BaseTransform should store any additional keyword arguments in
        _init_kwargs for later use (e.g., in __repr__).

        **Action**: Create a DummyStatelessTransform with additional kwargs.

        **Expected Result**: The kwargs should be stored in _init_kwargs.
        """
        transform = DummyStatelessTransform(some_param=42, another_param="test")
        assert transform._init_kwargs["some_param"] == 42
        assert transform._init_kwargs["another_param"] == "test"

    def test_base_transform_call(self):
        """Test that __call__ method works.

        **Assumption**: BaseTransform's __call__ method should delegate to transform_data,
        allowing the transform to be called directly like a function. The DummyStatelessTransform
        multiplies all values by 2.

        **Action**: Create a DummyStatelessTransform, provide input data with values [1.0, 2.0, 3.0],
        and call the transform directly using the function call syntax transform(data).

        **Expected Result**: The result should be [2.0, 4.0, 6.0] (each value doubled).
        This validates that the __call__ method properly invokes transform_data and that
        the transform logic works correctly. This is a critical test because __call__ is
        the primary interface for using transforms in the framework.
        """
        transform = DummyStatelessTransform()
        data = NamedTransformInput(features=np.array([1.0, 2.0, 3.0]))
        result = transform(data)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_base_transform_repr(self):
        """Test string representation of transform.

        **Assumption**: The __repr__ method should provide a meaningful string representation
        that includes the class name and important initialization parameters, making debugging
        and logging easier.

        **Action**: Create a DummyStatelessTransform with exclude_keys=["target"] and an additional
        parameter some_param=42, then call repr() on the transform instance.

        **Expected Result**: The string representation should contain "DummyStatelessTransform" (class name)
        and "exclude_keys" (important parameter). This ensures that when transforms are logged
        or debugged, developers can see what parameters were used, which is crucial for
        reproducibility and troubleshooting in complex transformation pipelines.
        """
        transform = DummyStatelessTransform(exclude_keys=["target"], some_param=42)
        repr_str = repr(transform)
        assert "DummyStatelessTransform" in repr_str
        assert "exclude_keys" in repr_str

    def test_base_transform_repr_no_kwargs(self):
        """Test repr with no kwargs."""
        transform = DummyStatelessTransform()
        repr_str = repr(transform)
        assert "DummyStatelessTransform" in repr_str
        assert "exclude_keys" in repr_str

    def test_base_transform_fit_data_default(self):
        """Test default fit_data implementation.

        **Assumption**: BaseTransform's default fit_data implementation should do nothing
        (for stateless transforms). It should not raise errors and should return None.

        **Action**: Create a DummyStatelessTransform and call fit_data with sample data.

        **Expected Result**: The method should complete without errors and return None.
        """
        transform = DummyStatelessTransform()
        data = NamedTransformInput(features=np.array([1.0, 2.0, 3.0]))
        # Should not raise error
        result = transform.fit_data(data, {})
        assert result is None

    def test_base_transform_abstract_method(self):
        """Test that BaseTransform cannot be instantiated directly.

        **Assumption**: BaseTransform is an abstract base class and should raise TypeError
        when instantiated directly, as it requires subclasses to implement transform_data.

        **Action**: Attempt to instantiate BaseTransform directly.

        **Expected Result**: Should raise TypeError.
        """
        with pytest.raises(TypeError):
            BaseTransform()

    def test_dense_transform_marker(self):
        """Test DenseTransform marker class.

        **Assumption**: DenseTransform is a marker class that indicates the transform
        works with dense (regular) arrays. DummyStatelessTransform should be an instance
        of both DenseTransform and BaseTransform, but not RaggedTransform.

        **Action**: Create a DummyStatelessTransform and check isinstance relationships.

        **Expected Result**: Should be instance of DenseTransform and BaseTransform, but not RaggedTransform.
        """
        transform = DummyStatelessTransform()
        assert isinstance(transform, DenseTransform)
        assert isinstance(transform, BaseTransform)
        assert not isinstance(transform, RaggedTransform)

    def test_ragged_transform_marker(self):
        """Test RaggedTransform marker class.

        **Assumption**: RaggedTransform is a marker class that indicates the transform
        works with ragged arrays. DummyRaggedTransform should be an instance of both
        RaggedTransform and BaseTransform, but not DenseTransform.

        **Action**: Create a DummyRaggedTransform and check isinstance relationships.

        **Expected Result**: Should be instance of RaggedTransform and BaseTransform, but not DenseTransform.
        """
        transform = DummyRaggedTransform()
        assert isinstance(transform, RaggedTransform)
        assert isinstance(transform, BaseTransform)
        assert not isinstance(transform, DenseTransform)

    def test_ragged_or_dense_transform_marker(self):
        """Test RaggedOrDenseTransform marker class.

        **Assumption**: RaggedOrDenseTransform is a marker class that indicates the transform
        works with both ragged and dense arrays. DummyRaggedOrDenseTransform should be an
        instance of RaggedOrDenseTransform and BaseTransform.

        **Action**: Create a DummyRaggedOrDenseTransform and check isinstance relationships.

        **Expected Result**: Should be instance of RaggedOrDenseTransform and BaseTransform.
        """
        transform = DummyRaggedOrDenseTransform()
        assert isinstance(transform, RaggedOrDenseTransform)
        assert isinstance(transform, BaseTransform)


class TestDenseTransform:
    """Test suite for DenseTransform marker class."""

    def test_dense_transform_inheritance(self):
        """Test that DenseTransform is a BaseTransform.

        **Assumption**: DenseTransform should inherit from BaseTransform, making it a marker
        class that indicates the transform works with dense (regular) arrays rather than
        ragged arrays. This inheritance relationship is important for type checking and
        for the framework to route data to appropriate transforms.

        **Action**: Create a DummyStatelessTransform instance (which inherits from DenseTransform)
        and check that it is an instance of both BaseTransform and DenseTransform using isinstance().

        **Expected Result**: The transform should be an instance of both BaseTransform and
        DenseTransform. This validates the inheritance hierarchy and ensures that DenseTransform
        properly extends BaseTransform, which is essential for the framework's type system
        and for ensuring transforms are used with compatible data types.
        """
        transform = DummyStatelessTransform()
        assert isinstance(transform, BaseTransform)
        assert isinstance(transform, DenseTransform)
