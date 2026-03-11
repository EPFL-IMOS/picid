"""Tests that transform pipeline failures are wrapped in TransformError with context."""

import pytest

from picid.exceptions import TransformError
from picid.transforms.base.strategy import TransformStrategy

from test.transforms.base.conftest import create_dummy_single_unit_container


def test_strategy_wraps_transform_failure_with_context():
    """When transform_multi_source raises, strategy re-raises TransformError with step_id and class."""

    class FailingTransform:
        def transform_multi_source(self, chunks, metadata=None):
            raise ValueError("expected failure")

    strategy = TransformStrategy()
    container = create_dummy_single_unit_container()

    with pytest.raises(TransformError) as exc_info:
        strategy.apply(
            transform_instance=FailingTransform(),
            data=container,
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
            step_id="failing_transform",
        )

    assert exc_info.value.step_id == "failing_transform"
    assert exc_info.value.transform_class == "FailingTransform"
    assert exc_info.value.cause is not None
    assert isinstance(exc_info.value.cause, ValueError)
    assert "expected failure" in str(exc_info.value.cause)
    assert "Transform failed" in str(exc_info.value)
    assert "failing_transform" in str(exc_info.value)
    assert "FailingTransform" in str(exc_info.value)
