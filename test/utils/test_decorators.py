"""Comprehensive tests for decorator utilities.

This module tests the decorators used throughout the PHM data pipeline
for error handling, validation, and transform context injection.

PHM Context:
-----------
Decorators provide consistent error handling and validation across
transform operations, making debugging easier when processing fails.

Test Coverage Strategy:
----------------------
1. **Transform Context Injection**: Errors are logged with transform context;
   the original exception is re-raised unchanged (type and message).
2. **Output Consistency Checking**: Validation of transform outputs
3. **Exception Handling**: Preservation of original tracebacks
4. **Edge Cases**: Missing attributes, various input types
"""

import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from picid.utils.decorators import (
    inject_transform_context_to_strategy_apply,
    check_transform_output_consistency,
)


class TestInjectTransformContext:
    """Tests for inject_transform_context_to_strategy_apply decorator."""

    def test_successful_execution_passthrough(self):
        """Test that successful execution passes through unchanged.

        **PHM Logic**: Decorator should be transparent for successful calls.

        **Methodology**: Decorate function that succeeds, verify return value.

        **Expected**: Original return value returned.

        Validates: Requirement DEC-1.1 - Transparent on success
        """

        @inject_transform_context_to_strategy_apply
        def successful_func(**kwargs):
            return {"result": "success"}

        transform = SimpleNamespace(transform_name="TestTransform")

        result = successful_func(
            transform_instance=transform, apply_to_keys=["features"]
        )

        assert result == {"result": "success"}

    def test_exception_logs_context_and_reraises_unchanged(self):
        """On failure, ``logger.error`` receives context; exception is bare re-raised.

        **Contract** (``picid.utils.decorators``): The wrapper does not mutate the
        raised exception's type or message; context is only added via logging.
        """

        @inject_transform_context_to_strategy_apply
        def failing_func(**kwargs):
            raise ValueError("Original error")

        transform = SimpleNamespace(transform_name="ScalerTransform")

        with patch("picid.utils.decorators.logger") as mock_logger:
            with pytest.raises(ValueError, match="^Original error$") as exc_info:
                failing_func(
                    transform_instance=transform,
                    apply_to_keys=["features", "target"],
                )

        mock_logger.error.assert_called_once()
        logged = mock_logger.error.call_args[0][0]
        assert "TRANSFORM NAME: **ScalerTransform**" in logged
        assert "APPLY_TO KEY(s): **['features', 'target']**" in logged
        assert "Original Exception: ValueError: Original error" in logged
        assert str(exc_info.value) == "Original error"

    def test_missing_transform_instance_logs_unknown_transform_name(self):
        """Without ``transform_instance``, context uses the literal **Unknown Transform**."""

        @inject_transform_context_to_strategy_apply
        def func_without_transform(**kwargs):
            raise ValueError("Test error")

        with patch("picid.utils.decorators.logger") as mock_logger:
            with pytest.raises(ValueError, match="^Test error$"):
                func_without_transform(apply_to_keys=["features"])

        mock_logger.error.assert_called_once()
        logged = mock_logger.error.call_args[0][0]
        assert "TRANSFORM NAME: **Unknown Transform**" in logged
        assert "APPLY_TO KEY(s): **['features']**" in logged

    def test_uses_class_name_when_transform_name_missing(self):
        """If ``transform_name`` is absent, context uses ``__class__.__name__``."""

        class UnnamedTransform:
            pass

        @inject_transform_context_to_strategy_apply
        def failing_func(**kwargs):
            raise RuntimeError("boom")

        instance = UnnamedTransform()
        with patch("picid.utils.decorators.logger") as mock_logger:
            with pytest.raises(RuntimeError, match="^boom$"):
                failing_func(transform_instance=instance, apply_to_keys=["x"])

        logged = mock_logger.error.call_args[0][0]
        assert "TRANSFORM NAME: **UnnamedTransform**" in logged

    def test_preserves_exception_type_and_message(self):
        """Original exception type and message are preserved (bare ``raise``)."""

        @inject_transform_context_to_strategy_apply
        def func_raises_key_error(**kwargs):
            raise KeyError("missing_key")

        transform = SimpleNamespace(transform_name="Transform")

        with patch("picid.utils.decorators.logger") as mock_logger:
            with pytest.raises(KeyError, match="^'missing_key'$"):
                func_raises_key_error(transform_instance=transform, apply_to_keys=["x"])
        mock_logger.error.assert_called_once()
        assert "TRANSFORM NAME: **Transform**" in mock_logger.error.call_args[0][0]

    def test_apply_to_keys_as_string(self):
        """Test handling of apply_to_keys as string.

        **PHM Logic**: apply_to_keys can be string or list.

        **Methodology**: Pass string instead of list.

        **Expected**: Success path unchanged; string appears in logged context on error.

        Validates: Requirement DEC-1.5 - String apply_to_keys handling
        """

        @inject_transform_context_to_strategy_apply
        def successful_func(**kwargs):
            return "success"

        transform = SimpleNamespace(transform_name="Transform")

        result = successful_func(
            transform_instance=transform,
            apply_to_keys="features",
        )

        assert result == "success"

        @inject_transform_context_to_strategy_apply
        def failing_func(**kwargs):
            raise ValueError("fail")

        with patch("picid.utils.decorators.logger") as mock_logger:
            with pytest.raises(ValueError, match="^fail$"):
                failing_func(
                    transform_instance=transform,
                    apply_to_keys="features",
                )
        logged = mock_logger.error.call_args[0][0]
        assert "APPLY_TO KEY(s): **features**" in logged


class TestCheckTransformOutputConsistency:
    """Tests for check_transform_output_consistency decorator."""

    def test_valid_output_passes(self):
        """Test that valid output passes through.

        **PHM Logic**: When transform output contains expected key, success.

        **Methodology**: Decorate method that returns valid output.

        **Expected**: Output returned unchanged.

        Validates: Requirement DEC-2.1 - Valid output passthrough
        """

        class TestTransform:
            @check_transform_output_consistency
            def transform_data(self, data, metadata):
                return {"expected_key": np.array([1, 2, 3])}

        transform = TestTransform()

        # Metadata needs to be dict-like with .get() method
        metadata = {"assign_to_map": ["expected_key"]}

        result = transform.transform_data({}, metadata)

        assert list(result.keys()) == ["expected_key"]
        assert np.array_equal(result["expected_key"], np.array([1, 2, 3]))

    def test_missing_key_raises_keyerror_with_expected_key_in_message(self):
        """Missing assign_to key raises ``KeyError`` whose message names the key."""

        class TestTransform:
            @check_transform_output_consistency
            def transform_data(self, data, metadata):
                return {"wrong_key": np.array([1, 2, 3])}

        transform = TestTransform()

        metadata = {"assign_to_map": ["expected_key"]}

        with pytest.raises(KeyError, match=r"expected_key") as exc_info:
            transform.transform_data({}, metadata)

        msg = str(exc_info.value)
        assert "expected_key" in msg
        assert "Available keys in output:" in msg
        assert "wrong_key" in msg

    def test_assign_to_map_none_skips_check(self):
        """Test that None assign_to_map skips validation.

        **PHM Logic**: If assign_to_map not specified, no validation needed.

        **Methodology**: Pass metadata with assign_to_map=None.

        **Expected**: Any output accepted.

        Validates: Requirement DEC-2.3 - Optional validation
        """

        class TestTransform:
            @check_transform_output_consistency
            def transform_data(self, data, metadata):
                return {"any_key": np.array([1])}

        transform = TestTransform()

        # Metadata needs to be dict-like with .get() method
        metadata = {"assign_to_map": None}

        result = transform.transform_data({}, metadata)

        assert list(result.keys()) == ["any_key"]
        assert np.array_equal(result["any_key"], np.array([1]))

    def test_assign_to_map_multiple_keys_skips_check(self):
        """Test that multi-key assign_to_map skips validation.

        **PHM Logic**: Multi-key mappings are complex, skip simple validation.

        **Methodology**: Pass assign_to_map with multiple keys.

        **Expected**: Any output accepted (check skipped).

        Validates: Requirement DEC-2.4 - Multi-key handling
        """

        class TestTransform:
            @check_transform_output_consistency
            def transform_data(self, data, metadata):
                return {"key1": np.array([1])}

        transform = TestTransform()

        # Metadata needs to be dict-like with .get() method
        metadata = {"assign_to_map": ["key1", "key2"]}  # Multiple keys

        result = transform.transform_data({}, metadata)
        assert list(result.keys()) == ["key1"]
        assert np.array_equal(result["key1"], np.array([1]))

    def test_non_mapping_output_skips_check(self):
        """Test that non-Mapping output skips check.

        **PHM Logic**: Some transforms return arrays directly, not dicts.

        **Methodology**: Return numpy array instead of dict.

        **Expected**: Array returned unchanged.

        Validates: Requirement DEC-2.5 - Non-dict output handling
        """

        class TestTransform:
            @check_transform_output_consistency
            def transform_data(self, data, metadata):
                return np.array([1, 2, 3])  # Not a dict

        transform = TestTransform()

        # Metadata needs to be dict-like with .get() method
        metadata = {"assign_to_map": ["expected_key"]}

        result = transform.transform_data({}, metadata)

        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_empty_output_dict(self):
        """Test handling of empty output dict.

        **PHM Logic**: Empty output violates expectation if key expected.

        **Methodology**: Return empty dict when key expected.

        **Expected**: KeyError raised.

        Validates: Requirement DEC-2.6 - Empty output detection
        """

        class TestTransform:
            @check_transform_output_consistency
            def transform_data(self, data, metadata):
                return {}  # Empty dict

        transform = TestTransform()

        # Metadata needs to be dict-like with .get() method
        metadata = {"assign_to_map": ["expected_key"]}

        with pytest.raises(KeyError, match=r"expected_key") as exc_info:
            transform.transform_data({}, metadata)

        msg = str(exc_info.value)
        assert "Available keys in output:" in msg
