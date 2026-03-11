"""
Test suite for picid_report.core.validators.

Validates validate_schema, log_modification, validate_seeds, and check_hidden_variations
used by the analysis pipeline. Does not modify picid_report.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from picid_report.core import validators


class TestValidateSchema:
    """Validates validate_schema: required columns check."""

    def test_all_present_passes(self):
        """
        Branch: All required columns present -> no raise, prints success.
        Methodology: DataFrame with columns a, b; required [a, b].
        Expected: No ValueError.
        """
        df = pd.DataFrame({"a": [1], "b": [2]})
        validators.validate_schema(df, ["a", "b"])

    def test_missing_raises(self):
        """
        Branch: Missing column -> ValueError with message.
        Methodology: DataFrame with only "a"; required ["a", "b"].
        Expected: ValueError mentioning "b".
        """
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="were not found"):
            validators.validate_schema(df, ["a", "b"])


class TestValidateSeeds:
    """Validates validate_seeds: filter groups that have all required seeds."""

    def test_no_required_seeds_returns_unchanged(self):
        """
        Branch: required_seeds None or empty -> return df unchanged.
        Methodology: Call with required_seeds=None.
        Expected: Same df returned.
        """
        df = pd.DataFrame({"g": [1, 1], "seed": [1, 2]})
        out = validators.validate_seeds(df, ["g"], "seed", None)
        assert len(out) == 2

    def test_seed_column_missing_returns_unchanged(self):
        """
        Branch: seed_col not in df -> warning, return df unchanged.
        Methodology: Call with seed_col that is not in df.
        Expected: Same df returned.
        """
        df = pd.DataFrame({"g": [1, 1]})
        out = validators.validate_seeds(df, ["g"], "seed", {1, 2})
        assert len(out) == 2

    def test_filters_groups_missing_seed(self):
        """
        Branch: Groups that don't contain all required_seeds are dropped.
        Methodology: Two groups g=1 (seeds 1,2) and g=2 (seed 1); required {1, 2}.
        Expected: Only g=1 rows remain.
        """
        df = pd.DataFrame({"g": [1, 1, 2], "seed": [1, 2, 1]})
        out = validators.validate_seeds(df, ["g"], "seed", {1, 2})
        assert len(out) == 2
        assert out["g"].tolist() == [1, 1]

    def test_allow_fallback_when_filtered_empty(self):
        """Branch: filtered_df.empty and not df.empty and allow_fallback -> return df. (validate_seeds)"""
        df = pd.DataFrame({"g": [1, 2], "seed": [1, 1]})  # no group has both 1 and 2
        out = validators.validate_seeds(df, ["g"], "seed", {1, 2}, allow_fallback=True)
        assert len(out) == 2

    def test_groupby_filter_type_error_returns_df(self):
        """Branch: groupby().filter raises TypeError -> print warning, return df. (validate_seeds 119-121)"""
        df = pd.DataFrame({"g": [1, 1], "seed": [1, 2]})
        mock_gb = MagicMock()
        mock_gb.filter.side_effect = TypeError("mock")
        with patch(
            "picid_report.core.validators.pd.DataFrame.groupby", return_value=mock_gb
        ):
            out = validators.validate_seeds(df, ["g"], "seed", {1, 2})
        assert len(out) == 2
        assert mock_gb.filter.called

    def test_has_required_seeds_exception_returns_false(self):
        """Branch: has_required_seeds group with bad seed value -> except, return False. (validators 110-111)"""
        df = pd.DataFrame({"g": [1, 1, 2, 2], "seed": [1, "not_a_number", 1, 2]})
        out = validators.validate_seeds(df, ["g"], "seed", {1, 2})
        assert len(out) == 2
        assert sorted(out["g"].unique()) == [2]


class TestFilterRowsWithValidSortMetric:
    """Validates filter_rows_with_valid_sort_metric: drop rows where sort metric is missing or not a valid float."""

    def test_no_column_returns_unchanged(self):
        """Branch: sort_metric_col not in df -> warning, return df unchanged."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out = validators.filter_rows_with_valid_sort_metric(df, "missing_col")
        assert len(out) == 2
        assert out.equals(df)

    def test_none_column_returns_unchanged(self):
        """Branch: sort_metric_col is None -> warning, return df unchanged."""
        df = pd.DataFrame({"a": [1, 2]})
        out = validators.filter_rows_with_valid_sort_metric(df, None)
        assert len(out) == 2

    def test_nan_rows_dropped(self):
        """Branch: Rows with NaN in sort metric column are dropped."""
        df = pd.DataFrame({"g": [1, 2, 3], "metric": [1.0, float("nan"), 3.0]})
        out = validators.filter_rows_with_valid_sort_metric(df, "metric")
        assert len(out) == 2
        assert out["g"].tolist() == [1, 3]

    def test_numeric_string_coerced(self):
        """Branch: String values coerced to numeric; invalid -> NaN -> row dropped."""
        df = pd.DataFrame({"x": [1, 2], "m": ["1.0", "not_a_number"]})
        out = validators.filter_rows_with_valid_sort_metric(df, "m")
        assert len(out) == 1
        assert out["x"].iloc[0] == 1


class TestLogModification:
    """Validates log_modification: only logs when shape changes; context printed."""

    def test_no_change_no_print(self):
        """Branch: rows_dropped == 0 and cols_dropped == 0 -> return. (log_modification)"""
        validators.log_modification("x", "y", (2, 3), (2, 3), context=None)

    def test_with_context_and_changes(self):
        """Branch: context provided; rows_dropped and cols_dropped > 0. (log_modification)"""
        validators.log_modification(
            "Drop rows", "test", (5, 4), (3, 2), context="Model=X"
        )

    def test_log_modification_no_context_only_cols_dropped(self):
        """Branch: context None; only cols_dropped > 0. (log_modification 73->75 false branch)"""
        validators.log_modification("Drop cols", "test", (2, 5), (2, 2), context=None)


class TestCheckHiddenVariations:
    """Validates check_hidden_variations: non-HP columns that vary within a group."""

    def test_no_candidates_no_warning(self):
        """
        Branch: All non-group columns numeric or ignored -> no warning.
        Methodology: df with group col and one numeric col.
        Expected: No exception.
        """
        df = pd.DataFrame({"g": [1, 1], "val": [0.1, 0.2]})
        validators.check_hidden_variations(df, ["g"], ["run_name"], None)

    def test_variation_within_group_warns(self):
        """
        Branch: Non-numeric column varies within group -> warning printed.
        Methodology: group g=1 has two rows with different "tag" values.
        Expected: Function runs (warning to stdout).
        """
        df = pd.DataFrame({"g": [1, 1], "tag": ["a", "b"]})
        validators.check_hidden_variations(df, ["g"], [], None)

    def test_additional_ignored_cols(self):
        """Branch: additional_ignored_cols set -> column in ignored. (check_hidden_variations)"""
        df = pd.DataFrame({"g": [1, 1], "tag": ["a", "b"], "ignore_me": [1, 2]})
        validators.check_hidden_variations(df, ["g"], [], ["ignore_me"])

    def test_apply_exception_pass(self):
        """Branch: grouped[col].apply(...) raises -> pass. (check_hidden_variations)"""
        df = pd.DataFrame(
            {"g": [1, 1], "bad": [{"a": 1}, {"b": 2}]}
        )  # dict not comparable in nunique
        validators.check_hidden_variations(df, ["g"], [], None)
