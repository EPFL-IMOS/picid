"""Tests for picid/exceptions.py.

Covers TransformError.__str__, PreprocessingDatasourceError.__str__,
and _resolve_datasource_name branch paths.
"""

import pytest

from picid.exceptions import (
    TransformError,
    PreprocessingDatasourceError,
    _resolve_datasource_name,
    build_transform_error,
)


@pytest.mark.unit
class TestTransformError:
    """Tests for TransformError.__str__."""

    def test_str_with_cause_contains_original_error_section(self):
        """
        __str__ with a cause includes 'Original error:' section.

        **Expected**: 'Original error:' header present and cause message indented.
        """
        cause = ValueError("bad value")
        err = TransformError("step failed", step_id="my_step", cause=cause)
        result = str(err)
        assert "Original error:" in result
        assert "bad value" in result

    def test_str_with_cause_none_appends_message(self):
        """
        __str__ with cause=None falls through to the else branch (line 191).

        **Expected**: self.message appended directly, no 'Original error:' section.
        """
        err = TransformError("plain message", step_id="s", cause=None)
        result = str(err)
        assert "plain message" in result
        assert "Original error:" not in result

    def test_str_with_no_metadata_and_no_cause(self):
        """
        __str__ with no step_id/class/keys and no cause → message appears.

        **Expected**: message line present.
        """
        err = TransformError("bare error")
        result = str(err)
        assert "bare error" in result


@pytest.mark.unit
class TestPreprocessingDatasourceError:
    """Tests for PreprocessingDatasourceError.__str__."""

    def test_str_with_cause_contains_datasource_error_section(self):
        """
        __str__ with cause shows 'Original datasource error:' section.

        **Expected**: section header present.
        """
        cause = RuntimeError("ds failed")
        err = PreprocessingDatasourceError(
            "preprocessing failed",
            stage="load_data",
            datasource_type="MyLoader",
            cause=cause,
        )
        result = str(err)
        assert "Original datasource error:" in result
        assert "ds failed" in result

    def test_str_with_cause_none_appends_message(self):
        """
        __str__ with cause=None falls through to else (lines 252-254).

        **Expected**: self.message appears, no 'Original datasource error:' header.
        """
        err = PreprocessingDatasourceError(
            "fallback message",
            stage="split_data",
            cause=None,
        )
        result = str(err)
        assert "fallback message" in result
        assert "Original datasource error:" not in result


@pytest.mark.unit
class TestResolveDatasourceName:
    """Tests for _resolve_datasource_name branch paths."""

    def test_none_datasource_returns_none(self):
        """
        datasource=None returns None immediately (line 104).

        **Expected**: None.
        """
        assert _resolve_datasource_name(None) is None

    def test_data_name_attribute_returned_directly(self):
        """
        Object with string data_name attribute → returned directly.

        **Expected**: the string value of data_name.
        """

        class Ds:
            data_name = "my_ds"

        assert _resolve_datasource_name(Ds()) == "my_ds"

    def test_get_data_names_single_tuple_returns_string(self):
        """
        get_data_names() returning a 1-tuple → unpacked string (line 117).

        **Expected**: "single" (not a tuple).
        """

        class Ds:
            def get_data_names(self):
                return ("single",)

        assert _resolve_datasource_name(Ds()) == "single"

    def test_get_data_names_multi_tuple_returns_list(self):
        """
        get_data_names() returning a multi-string tuple → list (line 119).

        **Expected**: ["a", "b"].
        """

        class Ds:
            def get_data_names(self):
                return ("a", "b")

        assert _resolve_datasource_name(Ds()) == ["a", "b"]

    def test_get_data_name_raises_returns_none(self):
        """
        get_data_name() raising an exception → None returned (line 126).

        **Expected**: None, no exception propagated.
        """

        class Ds:
            def get_data_name(self):
                raise RuntimeError("boom")

        assert _resolve_datasource_name(Ds()) is None

    def test_get_data_name_returns_string(self):
        """
        get_data_name() returning a string → returned directly (line 128).

        **Expected**: "name".
        """

        class Ds:
            def get_data_name(self):
                return "name"

        assert _resolve_datasource_name(Ds()) == "name"

    def test_get_data_name_returns_list_of_strings(self):
        """
        get_data_name() returning a list of strings → list returned (line 132).

        **Expected**: ["x", "y"].
        """

        class Ds:
            def get_data_name(self):
                return ["x", "y"]

        assert _resolve_datasource_name(Ds()) == ["x", "y"]

    def test_no_getter_returns_none(self):
        """
        Object with no name getter → None (line 133).

        **Expected**: None.
        """

        class Ds:
            pass

        assert _resolve_datasource_name(Ds()) is None

    def test_get_data_names_raises_treated_as_none(self):
        """
        get_data_names() raising → names set to None, falls through (lines 114-115).

        **Expected**: None returned (no exception propagated, not matching name tuple).
        """

        class Ds:
            def get_data_names(self):
                raise RuntimeError("exploded")

        assert _resolve_datasource_name(Ds()) is None


@pytest.mark.unit
class TestTransformErrorFormatting:
    """Tests TransformError.__str__ with transform_class and apply_to_keys."""

    def test_str_with_transform_class_includes_class_line(self):
        """
        transform_class set → class name appears in formatted output (line 175).

        **Expected**: transform_class value in str output.
        """
        err = TransformError(
            "failed", step_id="s", transform_class="MyTransform", cause=None
        )
        result = str(err)
        assert "MyTransform" in result

    def test_str_with_apply_to_keys_includes_keys_line(self):
        """
        apply_to_keys set → keys appear in formatted output (line 177).

        **Expected**: apply_to_keys list in str output.
        """
        err = TransformError(
            "failed", step_id="s", apply_to_keys=["features"], cause=None
        )
        result = str(err)
        assert "features" in result


@pytest.mark.unit
class TestBuildTransformError:
    """Tests build_transform_error factory function (lines 26-41)."""

    def test_returns_transform_error_instance(self):
        """
        build_transform_error returns a TransformError (lines 26-41).

        **Expected**: result is TransformError with message and cause set.
        """

        class _FakeContext:
            step_id = "my_step"
            transform_instance = None
            apply_to_keys = ["features"]

        cause = ValueError("raw error")
        result = build_transform_error(_FakeContext(), cause)
        assert isinstance(result, TransformError)
        assert result.cause is cause
        assert "my_step" in str(result)

    def test_context_without_attributes_uses_fallbacks(self):
        """
        Context missing step_id/transform_instance → fallback to None/unknown.

        **Expected**: TransformError returned, no AttributeError.
        """
        result = build_transform_error(object(), RuntimeError("oops"))
        assert isinstance(result, TransformError)


@pytest.mark.unit
class TestPreprocessingDatasourceErrorMultiLine:
    """Tests PreprocessingDatasourceError.__str__ with datasource_name (line 237)."""

    def test_str_with_datasource_name_shows_name(self):
        """
        datasource_name set → 'Datasource name:' appears in output (line 237).

        **Expected**: datasource_name value in formatted string.
        """
        from picid.data.datasources.base.exceptions import DatasourceError

        cause = DatasourceError("ds error")
        err = PreprocessingDatasourceError(
            "failed",
            stage="get_data",
            datasource_type="MyDs",
            datasource_name="my_dataset",
            cause=cause,
        )
        result = str(err)
        assert "my_dataset" in result

    def test_str_with_multiline_cause_indents_continuation(self):
        """
        Multi-line cause → continuation lines indented (line 252).

        **Expected**: second cause line appears with leading spaces.
        """
        from picid.data.datasources.base.exceptions import DatasourceError

        cause = DatasourceError("line1\nline2\nline3")
        err = PreprocessingDatasourceError(
            "failed",
            stage="get_data",
            cause=cause,
        )
        result = str(err)
        assert "line2" in result
        assert "line3" in result
