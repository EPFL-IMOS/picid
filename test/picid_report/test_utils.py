"""
Tests for picid_report.utils.

Covers format_mean_std_count and flatten_aggregated_columns.
"""

import pandas as pd

from picid_report import utils


class TestFormatMeanStdCount:
    """format_mean_std_count: display string for aggregate stats."""

    def test_basic(self):
        s = utils.format_mean_std_count(1.2345, 0.0012, 3, precision=4)
        assert "1.2345" in s and "0.0012" in s and "(n=3)" in s and "±" in s

    def test_nan_std_becomes_zero(self):
        s = utils.format_mean_std_count(1.0, float("nan"), 5)
        assert "0.0000" in s and "(n=5)" in s

    def test_count_as_string_passthrough(self):
        s = utils.format_mean_std_count(1.0, 0.0, "n/a")
        assert "(n=n/a)" in s

    def test_precision(self):
        s = utils.format_mean_std_count(1.234567, 0.1, 1, precision=2)
        assert "1.23" in s and "0.10" in s


class TestFlattenAggregatedColumns:
    """flatten_aggregated_columns: MultiIndex columns to suffix form."""

    def test_multiindex_flattened(self):
        df = pd.DataFrame(
            [[1.0, 0.1, 2]],
            columns=pd.MultiIndex.from_tuples(
                [
                    ("val/loss", "mean"),
                    ("val/loss", "std"),
                    ("val/loss", "count"),
                ]
            ),
        )
        out = utils.flatten_aggregated_columns(df)
        assert list(out.columns) == ["val/loss_mean", "val/loss_std", "val/loss_count"]
        assert out.iloc[0].tolist() == [1.0, 0.1, 2]

    def test_plain_columns_unchanged(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        out = utils.flatten_aggregated_columns(df)
        assert list(out.columns) == ["A", "B"]

    def test_mixed_columns(self):
        df = pd.DataFrame(
            [[1]],
            columns=pd.MultiIndex.from_tuples([("m", "mean")]),
        )
        out = utils.flatten_aggregated_columns(df)
        assert list(out.columns) == ["m_mean"]

    def test_returns_copy(self):
        df = pd.DataFrame({"x": [1]})
        out = utils.flatten_aggregated_columns(df)
        assert out is not df
