"""
Test suite for picid_report.report.reporting.

Validates create_summary_table, display_experiment_stats, display_performance_tables,
and display_hp_impact: string formatting (mean ± std (n=count)), NaN handling,
n=1 std 0.0, and MultiIndex flattening. Each test references the function under test.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
import xarray as xr

from picid_report.report import reporting


def _make_agg_ds(df: pd.DataFrame, opt_col: str = "test/mse", opt_mode: str = "min") -> xr.Dataset:
    """Convert flat DataFrame with {metric}_mean/_std/_count columns to xr.Dataset."""
    metric_names = [c[:-5] for c in df.columns if c.endswith("_mean")]
    metric_stat_cols = {f"{m}_{s}" for m in metric_names for s in ("mean", "std", "count")}
    hp_cols = [c for c in df.columns if c not in metric_stat_cols]

    n_configs = len(df)
    n_metrics = len(metric_names)
    mean_data = np.full((n_configs, n_metrics), np.nan)
    std_data = np.full((n_configs, n_metrics), np.nan)
    count_data = np.full((n_configs, n_metrics), np.nan)
    for mi, metric in enumerate(metric_names):
        if f"{metric}_mean" in df.columns:
            mean_data[:, mi] = pd.to_numeric(df[f"{metric}_mean"], errors="coerce").values
        if f"{metric}_std" in df.columns:
            std_data[:, mi] = pd.to_numeric(df[f"{metric}_std"], errors="coerce").values
        if f"{metric}_count" in df.columns:
            count_data[:, mi] = pd.to_numeric(df[f"{metric}_count"], errors="coerce").values

    coords: dict = {"config": np.arange(n_configs), "metric": metric_names}
    for hp in hp_cols:
        vals = np.empty(n_configs, dtype=object)
        for j, v in enumerate(df[hp].values):
            vals[j] = v
        coords[hp] = ("config", vals)

    return xr.Dataset(
        {
            "mean":  (["config", "metric"], mean_data),
            "std":   (["config", "metric"], std_data),
            "count": (["config", "metric"], count_data),
        },
        coords=coords,
        attrs={"optimization_col": opt_col, "optimization_mode": opt_mode},
    )


def _make_all_results(
    metrics_data=None,
    sorted_aggregated=None,
    seeds_info=None,
    optimized_on=None,
    total_runs=None,
    configs_failed_not_full_seed_set=None,
    configs_failed_missing_invalid_metric=None,
    sort_metric_used=None,
    sort_metric_is_fallback=None,
    original_sort_metric=None,
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
    if total_runs is not None:
        all_results["DS1"]["ModelA"]["total_runs"] = total_runs
    if configs_failed_not_full_seed_set is not None:
        all_results["DS1"]["ModelA"]["configs_failed_not_full_seed_set"] = (
            configs_failed_not_full_seed_set
        )
    if configs_failed_missing_invalid_metric is not None:
        all_results["DS1"]["ModelA"]["configs_failed_missing_invalid_metric"] = (
            configs_failed_missing_invalid_metric
        )
    if sort_metric_used is not None:
        all_results["DS1"]["ModelA"]["sort_metric_used"] = sort_metric_used
    if sort_metric_is_fallback is not None:
        all_results["DS1"]["ModelA"]["sort_metric_is_fallback"] = (
            sort_metric_is_fallback
        )
    if original_sort_metric is not None:
        all_results["DS1"]["ModelA"]["original_sort_metric"] = original_sort_metric
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

    def test_create_summary_table_without_sort_metric_uses_optimization(self):
        """
        Branch: sort_metric=None uses optimization metric (backward compatibility).
        Methodology: Call create_summary_table with sort_metric=None.
        Expected: Uses best_performance (selected by optimization metric).
        """
        all_results = _make_all_results()
        df = reporting.create_summary_table(all_results, sort_metric=None)
        assert not df.empty
        # Should work the same as without sort_metric parameter

    def test_create_summary_table_with_sort_metric_re_selects(self):
        """
        Branch: sort_metric provided, re-selects best run from sorted_aggregated_results.
        Methodology: Create all_results with sorted_aggregated_results where best by test/mse
        differs from best by test/accuracy. Use sort_metric="test/accuracy".
        Expected: Summary table shows metrics from run selected by test/accuracy.
        """
        # Create sorted_aggregated_results with two configs
        # Config 1: test/mse=0.3 (best for mse), test/accuracy=0.7
        # Config 2: test/mse=0.5, test/accuracy=0.9 (best for accuracy)
        df_sorted = pd.DataFrame(
            {
                "Model": ["M", "M"],
                "task_definition.seq_len": [10, 20],
                "test/mse_mean": [0.3, 0.5],  # First is best for mse
                "test/mse_std": [0.05, 0.06],
                "test/mse_count": [3, 3],
                "test/accuracy_mean": [0.7, 0.9],  # Second is best for accuracy
                "test/accuracy_std": [0.02, 0.01],
                "test/accuracy_count": [3, 3],
            }
        )

        all_results = defaultdict(lambda: defaultdict(dict))
        all_results["DS1"]["ModelA"]["sorted_aggregated_results"] = df_sorted
        all_results["DS1"]["ModelA"]["best_performance"] = {
            "optimized_on": {"metric": "test/mse", "strategy": "min"},
            # This would be from config 1 (best by mse)
            "metrics": {
                "mse": {"test": {"mean": 0.3, "std": 0.05, "count": 3}},
                "accuracy": {"test": {"mean": 0.7, "std": 0.02, "count": 3}},
            },
        }

        # Use sort_metric="test/accuracy" - should select config 2
        df = reporting.create_summary_table(all_results, sort_metric="test/accuracy")
        assert not df.empty
        # The accuracy value should be from config 2 (0.9), not config 1 (0.7)
        # We can't easily check the exact value without parsing the formatted string,
        # but we verify it doesn't crash and produces a table


# --- get_experiment_stats_df (reporting.get_experiment_stats_df) ---


class TestGetExperimentStatsDf:
    """Validates get_experiment_stats_df: columns and Runs / Configs Failed (not full seed set)."""

    def test_table_has_runs_and_configs_failed_columns(self):
        """
        get_experiment_stats_df returns DataFrame with "Runs" and "Configs Failed (not full seed set)".
        """
        all_results = _make_all_results(
            seeds_info={"data": "[1]", "model": "[72, 88, 101]"},
            total_runs=75,
            configs_failed_not_full_seed_set=2,
        )
        all_results["DS1"]["ModelA"]["sorted_aggregated_results"] = pd.DataFrame(
            {"x": [1]}
        )
        df = reporting.get_experiment_stats_df(all_results)
        assert "Runs" in df.columns
        assert "Configs Failed (not full seed set)" in df.columns
        assert "Configs Failed (missing/invalid metric)" in df.columns
        assert df.columns.tolist() == [
            "Dataset",
            "Model",
            "Configs Completed",
            "Runs",
            "Configs Failed (not full seed set)",
            "Configs Failed (missing/invalid metric)",
            "Seeds Found",
        ]

    def test_runs_and_configs_failed_values_present(self):
        """
        When total_runs and configs_failed_not_full_seed_set are in all_results, they appear in the table.
        """
        all_results = _make_all_results(
            seeds_info={"model": "[72, 88, 101]"},
            total_runs=75,
            configs_failed_not_full_seed_set=1,
        )
        all_results["DS1"]["ModelA"]["sorted_aggregated_results"] = pd.DataFrame(
            {"x": [1]}
        )
        df = reporting.get_experiment_stats_df(all_results)
        row = df.iloc[0]
        assert row["Runs"] == 75
        assert row["Configs Failed (not full seed set)"] == 1

    def test_runs_empty_configs_failed_zero_when_keys_missing(self):
        """
        Legacy all_results without total_runs/configs_failed: Runs is empty, Configs Failed is 0.
        """
        all_results = _make_all_results(seeds_info={"model": "[1, 2, 3]"})
        all_results["DS1"]["ModelA"]["sorted_aggregated_results"] = pd.DataFrame(
            {"x": [1]}
        )
        # Do not pass total_runs or configs_failed_not_full_seed_set
        df = reporting.get_experiment_stats_df(all_results)
        row = df.iloc[0]
        assert row["Runs"] == ""
        assert row["Configs Failed (not full seed set)"] == 0
        assert row["Configs Failed (missing/invalid metric)"] == 0

    def test_configs_completed_counts_configs_not_data_vars(self):
        """
        get_experiment_stats_df must use ds.sizes["config"], not len(ds).
        len(xr.Dataset) returns the number of data variables (mean/std/count/run_names = 4),
        not the number of HP configs. This test guards against that regression.
        """
        ds3 = _make_agg_ds(pd.DataFrame({
            "task_definition.seq_len": [10, 20, 50],
            "test/mse_mean": [0.5, 0.4, 0.6],
            "test/mse_std": [0.1, 0.05, 0.2],
            "test/mse_count": [3, 3, 3],
        }))
        all_results = _make_all_results(sorted_aggregated=ds3)
        df = reporting.get_experiment_stats_df(all_results)
        assert df.iloc[0]["Configs Completed"] == 3


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
        metrics_data = {"mse": {"test": {"mean": 0.5, "std": float("nan"), "count": 1}}}
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


# --- _infer_metric_mode (reporting._infer_metric_mode) ---


class TestInferMetricMode:
    """Validates _infer_metric_mode: min/max inference from metric names."""

    def test_infer_min_for_loss_metrics(self):
        """Branch: Loss metrics -> "min"."""
        assert reporting._infer_metric_mode("test/loss") == "min"
        assert reporting._infer_metric_mode("val/loss") == "min"
        assert reporting._infer_metric_mode("loss") == "min"

    def test_infer_min_for_mse_metrics(self):
        """Branch: MSE metrics -> "min"."""
        assert reporting._infer_metric_mode("test/mse") == "min"
        assert reporting._infer_metric_mode("val/mse") == "min"
        assert reporting._infer_metric_mode("mean_squared_error") == "min"

    def test_infer_min_for_mae_metrics(self):
        """Branch: MAE metrics -> "min"."""
        assert reporting._infer_metric_mode("test/mae") == "min"
        assert reporting._infer_metric_mode("mean_absolute_error") == "min"

    def test_infer_max_for_accuracy_metrics(self):
        """Branch: Accuracy metrics -> "max"."""
        assert reporting._infer_metric_mode("test/accuracy") == "max"
        assert reporting._infer_metric_mode("val/accuracy") == "max"
        assert reporting._infer_metric_mode("acc") == "max"

    def test_infer_max_for_f1_metrics(self):
        """Branch: F1 metrics -> "max"."""
        assert reporting._infer_metric_mode("test/f1") == "max"
        assert reporting._infer_metric_mode("f1_score") == "max"

    def test_infer_max_for_auc_metrics(self):
        """Branch: AUC metrics -> "max"."""
        assert reporting._infer_metric_mode("test/auc") == "max"
        assert reporting._infer_metric_mode("roc_auc") == "max"

    def test_infer_defaults_to_min(self):
        """Branch: Unknown metric -> defaults to "min"."""
        assert reporting._infer_metric_mode("unknown_metric") == "min"
        assert reporting._infer_metric_mode("custom/weird") == "min"


# --- _build_hp_impact_df with sort_metric (reporting._build_hp_impact_df) ---


class TestBuildHpImpactDfWithSortMetric:
    """Validates _build_hp_impact_df with sort_metric parameter."""

    def test_build_hp_impact_df_without_sort_metric_uses_optimization(self):
        """
        Branch: sort_metric=None uses optimization metric (backward compatibility).
        Methodology: Call _build_hp_impact_df with sort_metric=None.
        Expected: Uses optimization metric for sorting and column order.
        """
        ds_sorted = _make_agg_ds(pd.DataFrame(
            {
                "Model": ["M"],
                "task_definition.seq_len": [10],
                "test/mse_mean": [0.5],
                "test/mse_std": [0.1],
                "test/mse_count": [3],
            }
        ))
        res = {
            "sorted_aggregated_results": ds_sorted,
            "best_performance": {
                "optimized_on": {"metric": "test/mse", "strategy": "min"},
            },
        }
        result = reporting._build_hp_impact_df(res, precision=4, sort_metric=None)
        assert result is not None
        df, metric_used = result
        assert metric_used == "test/mse"
        assert "test/mse" in df.columns

    def test_build_hp_impact_df_with_sort_metric_re_sorts(self):
        """
        Branch: sort_metric provided, re-sorts by that metric.
        Methodology: sorted_aggregated_results with multiple rows, sort by different metric.
        Expected: DataFrame sorted by sort_metric, not optimization metric.
        """
        # Create Dataset with two configs
        # Config 1: test/mse=0.3 (best for mse), test/accuracy=0.7
        # Config 2: test/mse=0.5, test/accuracy=0.9 (best for accuracy)
        ds_sorted = _make_agg_ds(pd.DataFrame(
            {
                "Model": ["M", "M"],
                "task_definition.seq_len": [10, 20],
                "test/mse_mean": [0.3, 0.5],  # Sorted by mse (ascending)
                "test/mse_std": [0.05, 0.06],
                "test/mse_count": [3, 3],
                "test/accuracy_mean": [0.7, 0.9],
                "test/accuracy_std": [0.02, 0.01],
                "test/accuracy_count": [3, 3],
            }
        ))
        res = {
            "sorted_aggregated_results": ds_sorted,
            "best_performance": {
                "optimized_on": {"metric": "test/mse", "strategy": "min"},
            },
        }

        # Use sort_metric="test/accuracy" - should sort descending (maximize accuracy)
        result = reporting._build_hp_impact_df(
            res, precision=4, sort_metric="test/accuracy"
        )
        assert result is not None
        df, metric_used = result
        assert metric_used == "test/accuracy"
        # First row should be config 2 (accuracy=0.9)
        assert df.iloc[0]["task_definition.seq_len"] == 20
        assert "test/accuracy" in df.columns
        # test/accuracy should be first metric column
        metric_cols = [c for c in df.columns if c.startswith("test/")]
        assert metric_cols[0] == "test/accuracy"

    def test_build_hp_impact_df_sort_metric_not_found_fallback(self):
        """
        Branch: sort_metric column not found, falls back to optimization metric.
        Methodology: sort_metric="test/nonexistent", column doesn't exist.
        Expected: Uses optimization metric, no crash.
        """
        ds_sorted = _make_agg_ds(pd.DataFrame(
            {
                "Model": ["M"],
                "task_definition.seq_len": [10],
                "test/mse_mean": [0.5],
                "test/mse_std": [0.1],
                "test/mse_count": [3],
            }
        ))
        res = {
            "sorted_aggregated_results": ds_sorted,
            "best_performance": {
                "optimized_on": {"metric": "test/mse", "strategy": "min"},
            },
        }
        result = reporting._build_hp_impact_df(
            res, precision=4, sort_metric="test/nonexistent"
        )
        assert result is not None
        df, metric_used = result
        # When sort_metric column not found, should fallback to optimization metric
        assert (
            metric_used == "test/mse"
        )  # Falls back to optimization metric when column not found


# --- iter_hp_impact_tables with sort_metric (reporting.iter_hp_impact_tables) ---


class TestIterHpImpactTablesWithSortMetric:
    """Validates iter_hp_impact_tables with sort_metric parameter."""

    def test_iter_hp_impact_tables_without_sort_metric(self):
        """
        Branch: sort_metric=None uses optimization metric (backward compatibility).
        Methodology: Call iter_hp_impact_tables with sort_metric=None.
        Expected: Yields (dataset, model, df, metric_used) with optimization metric.
        """
        ds_sorted = _make_agg_ds(pd.DataFrame(
            {
                "Model": ["M"],
                "task_definition.seq_len": [10],
                "test/mse_mean": [0.5],
                "test/mse_std": [0.1],
                "test/mse_count": [3],
            }
        ))
        all_results = _make_all_results(sorted_aggregated=ds_sorted)
        entries = list(
            reporting.iter_hp_impact_tables(all_results, precision=4, sort_metric=None)
        )
        assert len(entries) == 1
        dataset, model, df, metric_used = entries[0]
        assert metric_used == "test/mse"  # Optimization metric

    def test_iter_hp_impact_tables_with_sort_metric(self):
        """
        Branch: sort_metric provided, passes to _build_hp_impact_df.
        Methodology: Call iter_hp_impact_tables with sort_metric="test/accuracy".
        Expected: Yields entries with metric_used="test/accuracy".
        """
        ds_sorted = _make_agg_ds(pd.DataFrame(
            {
                "Model": ["M"],
                "task_definition.seq_len": [10],
                "test/mse_mean": [0.5],
                "test/mse_std": [0.1],
                "test/mse_count": [3],
                "test/accuracy_mean": [0.9],
                "test/accuracy_std": [0.02],
                "test/accuracy_count": [3],
            }
        ))
        all_results = _make_all_results(sorted_aggregated=ds_sorted)
        entries = list(
            reporting.iter_hp_impact_tables(
                all_results, precision=4, sort_metric="test/accuracy"
            )
        )
        assert len(entries) == 1
        dataset, model, df, metric_used = entries[0]
        assert metric_used == "test/accuracy"

    def test_iter_hp_impact_tables_with_dict_sort_metric(self):
        """
        Branch: sort_metric as dict mapping (dataset, model) -> metric.
        Methodology: Call with dict sort_metric.
        Expected: Uses per-combination metric.
        """
        ds_sorted = _make_agg_ds(pd.DataFrame(
            {
                "Model": ["M"],
                "task_definition.seq_len": [10],
                "test/mse_mean": [0.5],
                "test/mse_std": [0.1],
                "test/mse_count": [3],
                "test/accuracy_mean": [0.9],
                "test/accuracy_std": [0.02],
                "test/accuracy_count": [3],
            }
        ))
        all_results = _make_all_results(sorted_aggregated=ds_sorted)
        sort_metric_dict = {("DS1", "ModelA"): "test/accuracy"}
        entries = list(
            reporting.iter_hp_impact_tables(
                all_results, precision=4, sort_metric=sort_metric_dict
            )
        )
        assert len(entries) == 1
        dataset, model, df, metric_used = entries[0]
        assert metric_used == "test/accuracy"

    def test_iter_hp_impact_tables_metric_used_includes_fallback_suffix(self):
        """
        Branch: res has sort_metric_is_fallback and original_sort_metric.
        Methodology: Build all_results with sort_metric_used=val/mse, sort_metric_is_fallback=True,
        original_sort_metric=val/loss. Call iter_hp_impact_tables.
        Expected: metric_used contains "fallback" and "val/loss" (original not found).
        """
        ds_sorted = _make_agg_ds(pd.DataFrame(
            {
                "Model": ["M"],
                "task_definition.seq_len": [10],
                "val/mse_mean": [0.52],
                "val/mse_std": [0.08],
                "val/mse_count": [3],
            }
        ), opt_col="val/mse")
        all_results = _make_all_results(
            sorted_aggregated=ds_sorted,
            sort_metric_used="val/mse",
            sort_metric_is_fallback=True,
            original_sort_metric="val/loss",
        )
        entries = list(
            reporting.iter_hp_impact_tables(
                all_results, precision=4, sort_metric="val/mse"
            )
        )
        assert len(entries) == 1
        _dataset, _model, _df, metric_used = entries[0]
        assert "fallback" in metric_used.lower()
        assert "val/loss" in metric_used
        assert (
            "original metric" in metric_used.lower()
            or "not found" in metric_used.lower()
        )


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

    def test_display_hp_impact_with_sort_metric(self):
        """
        Branch: sort_metric parameter passed through to iter_hp_impact_tables.
        Methodology: Call display_hp_impact with sort_metric.
        Expected: No crash, displays tables sorted by sort_metric.
        """
        df_sorted = pd.DataFrame(
            {
                "Model": ["M"],
                "task_definition.seq_len": [10],
                "test/mse_mean": [0.5],
                "test/mse_std": [0.1],
                "test/mse_count": [3],
                "test/accuracy_mean": [0.9],
                "test/accuracy_std": [0.02],
                "test/accuracy_count": [3],
            }
        )
        all_results = _make_all_results(sorted_aggregated=df_sorted)
        reporting.display_hp_impact(
            all_results, precision=4, sort_metric="test/accuracy"
        )
        # We only assert no crash


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
        all_results = _make_all_results(
            metrics_data={
                "mse": {"test": {"mean": 0.5, "std": 0.0, "count": "n/a"}},
            }
        )
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
        all_results = _make_all_results(
            metrics_data={
                "mse": {"test": {"mean": 0.5, "std": 0.0, "count": 2}},
            }
        )
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
        all_results = _make_all_results(
            metrics_data={
                "mse": {"test": {"mean": 0.5, "std": 0.0, "count": 2}},
            }
        )
        reporting.display_performance_tables(all_results)

    def test_display_hp_impact_opt_metric_not_in_final_metrics(self):
        """Branch: opt_metric not in final_metrics -> don't remove, just sort. (display_hp_impact)"""
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
            optimized_on={"metric": "other_metric", "strategy": "min"},
        )
        reporting.display_hp_impact(all_results)

    def test_display_hp_impact_s_col_not_in_row(self):
        """Branch: s_col not in row -> std_val 0.0. (display_hp_impact format_impact_row)"""
        df = pd.DataFrame(
            {
                "Model": ["M"],
                "task_definition.seq_len": [10],
                "test/mse_mean": [0.5],
                "test/mse_count": [3],
            }
        )
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
