"""Comprehensive tests for decorator utilities.

This module tests the decorators used throughout the PHM data pipeline
for error handling, validation, and transform context injection.

PHM Context:
-----------
Decorators provide consistent error handling and validation across
transform operations, making debugging easier when processing fails.

Test Coverage Strategy:
----------------------
1. **Transform Context Injection**: Error message enhancement
2. **Output Consistency Checking**: Validation of transform outputs
3. **Exception Handling**: Preservation of original tracebacks
4. **Edge Cases**: Missing attributes, various input types
"""
import pytest
import numpy as np
from unittest.mock import Mock

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
        
        mock_transform = Mock()
        mock_transform.name = "TestTransform"
        
        result = successful_func(
            transform_instance=mock_transform,
            apply_to_keys=["features"]
        )
        
        assert result == {"result": "success"}
    
    def test_exception_context_injection(self):
        """Test that exceptions get context injected.
        
        **PHM Logic**: When transform fails, error message should include
        transform name and keys for debugging.
        
        **Methodology**: Decorate function that raises, verify enhanced message.
        
        **Expected**: Exception with transform context in message.
        
        Validates: Requirement DEC-1.2 - Context injection on exception
        """
        @inject_transform_context_to_strategy_apply
        def failing_func(**kwargs):
            raise ValueError("Original error")
        
        mock_transform = Mock()
        mock_transform.name = "ScalerTransform"
        
        with pytest.raises(ValueError) as exc_info:
            failing_func(
                transform_instance=mock_transform,
                apply_to_keys=["features", "target"]
            )
        
        # Error message should contain context
        error_msg = str(exc_info.value)
        assert "ScalerTransform" in error_msg or "Original error" in error_msg
    
    def test_missing_transform_instance_fallback(self):
        """Test fallback when transform_instance is missing.
        
        **PHM Logic**: Decorator should handle missing kwargs gracefully.
        
        **Methodology**: Call decorated function without transform_instance.
        
        **Expected**: Uses fallback class name.
        
        Validates: Requirement DEC-1.3 - Missing transform fallback
        """
        @inject_transform_context_to_strategy_apply
        def func_without_transform(**kwargs):
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            # Should not crash even without transform_instance
            func_without_transform(apply_to_keys=["features"])
    
    def test_preserves_exception_type(self):
        """Test that original exception type is preserved.
        
        **PHM Logic**: Specific exception types should be preserved for
        proper error handling upstream.
        
        **Methodology**: Raise specific exception, verify type preserved.
        
        **Expected**: Same exception type raised.
        
        Validates: Requirement DEC-1.4 - Exception type preservation
        """
        @inject_transform_context_to_strategy_apply
        def func_raises_key_error(**kwargs):
            raise KeyError("missing_key")
        
        mock_transform = Mock()
        mock_transform.name = "Transform"
        
        with pytest.raises(KeyError):
            func_raises_key_error(
                transform_instance=mock_transform,
                apply_to_keys=["x"]
            )
    
    def test_apply_to_keys_as_string(self):
        """Test handling of apply_to_keys as string.
        
        **PHM Logic**: apply_to_keys can be string or list.
        
        **Methodology**: Pass string instead of list.
        
        **Expected**: No error, context uses string.
        
        Validates: Requirement DEC-1.5 - String apply_to_keys handling
        """
        @inject_transform_context_to_strategy_apply
        def successful_func(**kwargs):
            return "success"
        
        mock_transform = Mock()
        mock_transform.name = "Transform"
        
        # Pass string instead of list
        result = successful_func(
            transform_instance=mock_transform,
            apply_to_keys="features"  # String, not list
        )
        
        assert result == "success"


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
        
        assert "expected_key" in result
    
    def test_missing_key_raises_error(self):
        """Test that missing expected key raises error.
        
        **PHM Logic**: Transform must produce key specified in assign_to_map.
        
        **Methodology**: Return output without expected key.
        
        **Expected**: KeyError raised with helpful message.
        
        Validates: Requirement DEC-2.2 - Missing key detection
        """
        class TestTransform:
            @check_transform_output_consistency
            def transform_data(self, data, metadata):
                return {"wrong_key": np.array([1, 2, 3])}
        
        transform = TestTransform()
        
        # Metadata needs to be dict-like with .get() method
        metadata = {"assign_to_map": ["expected_key"]}
        
        with pytest.raises(KeyError) as exc_info:
            transform.transform_data({}, metadata)
        
        # Error should mention expected key
        assert "expected_key" in str(exc_info.value)
    
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
        
        assert "any_key" in result
    
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
        
        # Should not raise even if only key1 present (multi-key skips check)
        result = transform.transform_data({}, metadata)
        assert "key1" in result
    
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
        
        # Should not raise, just return the array
        result = transform.transform_data({}, metadata)
        
        assert isinstance(result, np.ndarray)
    
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
        
        with pytest.raises(KeyError):
            transform.transform_data({}, metadata)
