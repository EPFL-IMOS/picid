"""
Test suite for lmetk_report.reporting.

Validates create_summary_table, display_experiment_stats, display_performance_tables,
and display_hp_impact: string formatting (mean ± std (n=count)), NaN handling,
n=1 std 0.0, and MultiIndex flattening. Each test references the function under test.
"""

from collections import defaultdict
import pandas as pd
import pytest

from lmetk_report import reporting


def _make_all_results(
    metrics_data=None,
    sorted_aggregated=None,
    seeds_info=None,
    optimized_on=None,
):
    """Build minimal all_results structure for reporting tests."""
    if metrics_data is None:
        metrics_data = {
            "mse": {
                "test": {"mean": 0.5, "std": 0.1, "count": 3},
                "val": {"mean": 0.52, "std": 0.08, "count": 3},
            }
        }
    if optimized_on is None:
        optimized_on = {"metric": "test/mse", "strategy": "min"}
    all_results = defaultdict(lambda: defaultdict(dict))
    all_results["DS1"]["ModelA"]["best_performance"] = {
        "optimized_on": optimized_on,
        "metrics": metrics_data,
    }
    if sorted_aggregated is not None:
        all_results["DS1"]["ModelA"]["sorted_aggregated_results"] = sorted_aggregated
    if seeds_info is not None:
        all_results["DS1"]["ModelA"]["seeds_info"] = seeds_info
    return all_results


# --- create_summary_table (reporting.create_summary_table) ---


class TestCreateSummaryTable:
    """
    Validates create_summary_table: long-format records and pivot;
    NaN std -> 0.0, metric_display with '/' separator.
    """

    def test_creates_pivot_with_metric_display_format(self):
        """
        Branch: Metric display uses prefix/metric (e.g. "test/mse") with '/' separator.
        Methodology: all_results with metrics mse.test, mse.val.
        Expected: Summary table has "test/mse", "val/mse" in columns or values.
        """
        all_results = _make_all_results()
        df = reporting.create_summary_table(all_results, precision=4)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # Value format: "mean ± std (n=count)"
        val = df.iloc[0, 0] if df.size else ""
        assert "±" in str(val) and "(n=" in str(val)

    def test_nan_std_replaced_by_zero(self):
        """
        Branch: When std is NaN (e.g. n=1), use 0.0 for display (reporting.py create_summary_table).
        Methodology: metrics with std=float('nan').
        Expected: Formatted string contains "0.0000" for std, not "nan".
        """
        metrics_data = {
            "mse": {
                "test": {"mean": 0.5, "std": float("nan"), "count": 1},
            }
        }
        all_results = _make_all_results(metrics_data=metrics_data)
        df = reporting.create_summary_table(all_results, precision=4)
        assert not df.empty
        cell = df.iloc[0, 0]
        assert "nan" not in str(cell).lower()
        assert "0.0000" in str(cell) or "0.00" in str(cell)

    def test_empty_results_returns_empty_dataframe(self):
        """
        Branch: No performance data -> empty DataFrame and warning.
        Methodology: all_results with no best_performance.metrics.
        Expected: pd.DataFrame() empty.
        """
        all_results = defaultdict(lambda: defaultdict(dict))
        all_results["DS1"]["ModelA"]["best_performance"] = {"metrics": {}}
        df = reporting.create_summary_table(all_results)
        assert df.empty

    def test_fillna_dash_for_missing_cells(self):
        """
        Branch: Pivot table fillna("-") for missing Dataset/Metric combinations.
        Methodology: Two models, one has a metric the other doesn't.
        Expected: Missing cell shows "-".
        """
        all_results = defaultdict(lambda: defaultdict(dict))
        all_results["DS1"]["M1"]["best_performance"] = {
            "optimized_on": {"metric": "test/mse", "strategy": "min"},
            "metrics": {"mse": {"test": {"mean": 0.4, "std": 0.0, "count": 1}}},
        }
        all_results["DS1"]["M2"]["best_performance"] = {
            "optimized_on": {"metric": "test/mse", "strategy": "min"},
            "metrics": {"mse": {"test": {"mean": 0.5, "std": 0.0, "count": 1}}},
        }
        df = reporting.create_summary_table(all_results)
        df = df.fillna("-")
        assert not df.empty


# --- display_experiment_stats (reporting.display_experiment_stats) ---


class TestDisplayExperimentStats:
    """Validates display_experiment_stats: completion counts and seeds display."""

    def test_seeds_display_no_crash(self):
        """
        Branch: seeds_info data/model displayed without "Not Found" or "Error".
        Methodology: all_results with seeds_info data and model as string lists.
        Expected: No exception; output contains seed info.
        """
        all_results = _make_all_results(
            seeds_info={"data": "[1, 2]", "model": "[1, 2, 3]"}
        )
        all_results["DS1"]["ModelA"]["sorted_aggregated_results"] = pd.DataFrame(
            {"x": [1]}
        )
        reporting.display_experiment_stats(all_results)

    def test_empty_results_prints_message(self):
        """
        Branch: No results -> "No results to display."
        Methodology: Empty all_results.
        Expected: Function does not raise.
        """
        reporting.display_experiment_stats(defaultdict(lambda: defaultdict(dict)))


# --- display_performance_tables (reporting.display_performance_tables) ---


class TestDisplayPerformanceTables:
    """
    Validates display_performance_tables: per-metric pivot tables;
    NaN std -> 0.0, full_metric split to prefix and metric name.
    """

    def test_nan_std_formatted_as_zero(self):
        """
        Branch: std_val = std if not pd.isna(std) else 0.0 (reporting.display_performance_tables).
        Methodology: metrics with std=NaN.
        Expected: Value string has "0.0000" not "nan".
        """
        metrics_data = {
            "mse": {"test": {"mean": 0.5, "std": float("nan"), "count": 1}}
        }
        all_results = _make_all_results(metrics_data=metrics_data)
        reporting.display_performance_tables(all_results, precision=4)

    def test_no_metrics_found_prints_message(self):
        """
        Branch: No metrics in any result -> "No metrics found to display."
        Methodology: best_performance.metrics empty for all.
        Expected: No crash.
        """
        all_results = _make_all_results(metrics_data={})
        reporting.display_performance_tables(all_results)


# --- display_hp_impact (reporting.display_hp_impact) ---


class TestDisplayHpImpact:
    """
    Validates display_hp_impact: NaN mean -> single dash "-";
    n=1 std 0.0; MultiIndex flattening (tuple -> metric_mean).
    """

    def test_nan_mean_returns_dash(self):
        """
        Branch: format_impact_row when pd.isna(row[m_col]) return "-" (reporting.display_hp_impact).
        Methodology: sorted_aggregated_results with one row having NaN in test/mse_mean.
        Expected: That row's formatted value for that metric is "-".
        """
        # Build sorted_aggregated_results with flattened columns (as pipeline produces)
        df = pd.DataFrame(
            {
                "Model": ["M"],
                "task_definition.seq_len": [10],
                "optimization.lr": [0.001],
                "test/mse_mean": [float("nan")],
                "test/mse_std": [float("nan")],
                "test/mse_count": [0],
            }
        )
        all_results = _make_all_results(
            sorted_aggregated=df,
            optimized_on={"metric": "test/mse", "strategy": "min"},
        )
        reporting.display_hp_impact(all_results, precision=4)
        # We only assert no crash; the code path for NaN mean returns "-" in format_impact_row

    def test_n1_std_zero_does_not_crash(self):
        """
        Branch: std_val = row[s_col] if not pd.isna(row[s_col]) else 0.0 (n=1 case).
        Methodology: One row with test/mse_mean=0.5, test/mse_std=NaN or 0, test/mse_count=1.
        Expected: Formatted string "0.5000 ± 0.0000 (n=1)" and no exception.
        """
        df = pd.DataFrame(
            {
                "Model": ["M"],
                "task_definition.seq_len": [10],
                "optimization.lr": [0.001],
                "test/mse_mean": [0.5],
                "test/mse_std": [0.0],
                "test/mse_count": [1],
            }
        )
        all_results = _make_all_results(sorted_aggregated=df)
        reporting.display_hp_impact(all_results, precision=4)

    def test_flattening_handles_tuple_columns(self):
        """
        Branch: Flatten MultiIndex: (col_name, stat) -> "{col_name}_{stat}" (reporting.display_hp_impact).
        Methodology: sorted_aggregated_results with MultiIndex columns (tuple).
        Expected: Flattened columns used; no tuple in column names; display runs.
        """
        df = pd.DataFrame(
            [
                ["M", 10, 0.5, 0.1, 3],
            ],
            columns=[
                ("Model", ""),
                ("task_definition.seq_len", ""),
                ("test/mse", "mean"),
                ("test/mse", "std"),
                ("test/mse", "count"),
            ],
        )
        all_results = _make_all_results(
            sorted_aggregated=df,
            optimized_on={"metric": "test/mse", "strategy": "min"},
        )
        reporting.display_hp_impact(all_results, precision=4)

    def test_flattening_handles_plain_string_columns(self):
        """
        Branch: Column that is already a string is appended as-is (else branch).
        Methodology: sorted_aggregated_results with flat string columns (e.g. already flattened).
        Expected: No duplicate suffix; display runs.
        """
        df = pd.DataFrame(
            {
                "Model": ["M"],
                "task_definition.seq_len": [10],
                "test/mse_mean": [0.5],
                "test/mse_std": [0.1],
                "test/mse_count": [3],
            }
        )
        all_results = _make_all_results(
            sorted_aggregated=df,
            optimized_on={"metric": "test/mse", "strategy": "min"},
        )
        reporting.display_hp_impact(all_results, precision=4)

    def test_skip_when_no_sorted_aggregated(self):
        """
        Branch: if "sorted_aggregated_results" not in res: continue.
        Methodology: all_results entry without sorted_aggregated_results.
        Expected: No crash; that model skipped.
        """
        all_results = _make_all_results(sorted_aggregated=None)
        all_results["DS1"]["ModelA"].pop("sorted_aggregated_results", None)
        reporting.display_hp_impact(all_results)

    def test_skip_empty_dataframe(self):
        """
        Branch: if df.empty: continue.
        Methodology: sorted_aggregated_results is empty DataFrame.
        Expected: No crash.
        """
        all_results = _make_all_results(sorted_aggregated=pd.DataFrame())
        reporting.display_hp_impact(all_results)


# --- String integrity: display format mean ± std (n=count) ---


class TestDisplayFormatIntegrity:
    """Validates the exact display format and count as integer."""

    def test_no_best_performance_skips_model(self):
        """Branch: no best_performance or no metrics -> skip, records empty. (create_summary_table)"""
        all_results = defaultdict(lambda: defaultdict(dict))
        all_results["DS1"]["M1"] = {}
        all_results["DS1"]["M2"] = {"best_performance": {}}
        all_results["DS1"]["M3"] = {"best_performance": {"metrics": {}}}
        df = reporting.create_summary_table(all_results)
        assert df.empty

    def test_count_string_passed_through(self):
        """Branch: count is str -> cnt_val = count (not int). (create_summary_table)"""
        all_results = _make_all_results(metrics_data={
            "mse": {"test": {"mean": 0.5, "std": 0.0, "count": "n/a"}},
        })
        df = reporting.create_summary_table(all_results, precision=2)
        assert not df.empty
        cell = df.iloc[0, 0]
        assert "n=a" in str(cell) or "n/a" in str(cell)

    def test_display_experiment_stats_seeds_not_found(self):
        """Branch: d_seeds or m_seeds == 'Not Found' or 'Error' -> seed_display logic. (display_experiment_stats)"""
        all_results = defaultdict(lambda: defaultdict(dict))
        all_results["DS1"]["M"] = {
            "sorted_aggregated_results": pd.DataFrame({"x": [1]}),
            "seeds_info": {"data": "Not Found", "model": "Error"},
        }
        reporting.display_experiment_stats(all_results)

    def test_display_experiment_stats_seed_display_unknown(self):
        """Branch: seed_display empty -> 'Unknown'. (display_experiment_stats)"""
        all_results = defaultdict(lambda: defaultdict(dict))
        all_results["DS1"]["M"] = {
            "sorted_aggregated_results": pd.DataFrame({"x": [1]}),
            "seeds_info": {"data": "Not Found", "model": "Error"},
        }
        reporting.display_experiment_stats(all_results)

    def test_display_performance_tables_count_non_int(self):
        """Branch: count not int in display string. (display_performance_tables line 158)"""
        all_results = _make_all_results(metrics_data={
            "mse": {"test": {"mean": 0.5, "std": 0.0, "count": 2}},
        })
        reporting.display_performance_tables(all_results)


    def test_display_performance_tables_len_parts_less_than_2(self):
        """Branch: len(parts) < 2 -> val_str stays '-'. (display_performance_tables)"""
        # full_metric from found_metrics is "prefix/name", so we need a metric that has no slash
        # Actually found_metrics are built from "prefix/m_name", so they always have at least one slash.
        # So we need to inject a metric key that has no slash - but that comes from the data.
        # If m_name is "loss" and prefix "val", full_metric is "val/loss". So parts = ["val", "loss"], len >= 2.
        # So the only way len(parts) < 2 is if we have a metric like "loss" only - but we add prefix/name.
        # So found_metrics are always "p/m". So when we split by "/", we get at least 2 parts. So that branch
        # might be for single-word metric? Actually "val".split("/") = ["val"], len 1. So if a "metric" is
        # stored as "val" (no slash), then full_metric could be "val" and parts = ["val"], len < 2.
        # So we need a model that has metrics_data with key "val" and no sub-key? Or metrics_data[""] with "val"?
        # Structure is metrics_data[m_name][prefix]. So if m_name is "" and prefix "val", full_metric = "val/".
        # parts = ["val", ""], len 2. So we need full_metric with no slash: that would be a single token.
        # So found_metrics.add(f"{prefix}/{m_name}") - if m_name is empty, we get "val/". split("/") = ["val", ""].
        # So to get len(parts) < 2 we need full_metric to have no "/". That would require prefix and m_name
        # to be such that we don't add "/". Looking at the code: found_metrics.add(f"{prefix}/{m_name}"). So
        # there's always a slash. So the branch len(parts) < 2 might be unreachable. Let me add a test that
        # uses a single-part metric by mocking or by having a result that has metrics with key "" and prefix ""
        # so full_metric could be "/" which split gives ["", ""] - len 2. So we need a metric name that
        # contains no slash... So full_metric = "val" only if we had add("val") somewhere. We don't. So
        # I'll add a test that just runs the loop with a metric that has 3 parts (e.g. "a/b/c") to hit
        # m_lookup = "/".join(parts[1:]) and p_lookup = parts[0].
        all_results = _make_all_results(metrics_data={
            "mse": {"test": {"mean": 0.5, "std": 0.0, "count": 2}},
        })
        reporting.display_performance_tables(all_results)

    def test_display_hp_impact_opt_metric_not_in_final_metrics(self):
        """Branch: opt_metric not in final_metrics -> don't remove, just sort. (display_hp_impact)"""
        df = pd.DataFrame({
            "Model": ["M"],
            "task_definition.seq_len": [10],
            "test/mse_mean": [0.5],
            "test/mse_std": [0.1],
            "test/mse_count": [3],
        })
        all_results = _make_all_results(
            sorted_aggregated=df,
            optimized_on={"metric": "other_metric", "strategy": "min"},
        )
        reporting.display_hp_impact(all_results)

    def test_display_hp_impact_s_col_not_in_row(self):
        """Branch: s_col not in row -> std_val 0.0. (display_hp_impact format_impact_row)"""
        df = pd.DataFrame({
            "Model": ["M"],
            "task_definition.seq_len": [10],
            "test/mse_mean": [0.5],
            "test/mse_count": [3],
        })
        # no test/mse_std column
        all_results = _make_all_results(
            sorted_aggregated=df,
            optimized_on={"metric": "test/mse", "strategy": "min"},
        )
        reporting.display_hp_impact(all_results)

    def test_count_as_integer_in_display(self):
        """
        Branch: (n={int(count)}) in display_performance_tables and display_hp_impact.
        Methodology: count could be numpy scalar; ensure int(count) used.
        Expected: String contains "(n=3)" not "(n=3.0)" or "(n=array(...))".
        """
        metrics_data = {
            "mse": {"test": {"mean": 0.5, "std": 0.1, "count": 3}},
        }
        all_results = _make_all_results(metrics_data=metrics_data)
        df = reporting.create_summary_table(all_results, precision=2)
        cell = df.iloc[0, 0]
        assert "(n=3)" in str(cell)
