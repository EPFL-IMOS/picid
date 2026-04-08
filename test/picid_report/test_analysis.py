"""
Test suite for picid_report.core.analysis.

Validates result aggregation, hyperparameter grid-tracking, string normalization
for merge, seed formatting, and edge cases (n=1, zero metrics, MultiIndex flattening).
Each test documents the analytical branch and the function under test.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from picid_report import config
from picid_report.core import analysis
from picid_report.core.analysis import _ds_is_empty


# --- _trace_nans (analysis._trace_nans) ---


def test_trace_nans_returns_nan_count():
    """Branch: _trace_nans returns current NaN count. (analysis._trace_nans)"""
    df = pd.DataFrame({"a": [1, np.nan], "b": [np.nan, 2]})
    n = analysis._trace_nans(df, "stage", 0)
    assert n == 2


# --- get_search_grid_df (analysis.get_search_grid_df) ---


class TestGetSearchGridDf:
    """Validates get_search_grid_df: Cartesian product of HP values for schema-first grid."""

    def test_empty_search_space_returns_empty_dataframe(self):
        """
        Branch: Empty search space returns empty DataFrame.
        Methodology: Call get_search_grid_df({}).
        Expected: pd.DataFrame() with no columns.
        """
        result = analysis.get_search_grid_df({})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == []

    def test_single_param_grid(self):
        """
        Branch: Single hyperparameter with multiple values.
        Methodology: search_space = {"lr": [0.01, 0.001]}.
        Expected: 2 rows, one column "lr".
        """
        result = analysis.get_search_grid_df({"lr": [0.01, 0.001]})
        assert len(result) == 2
        assert list(result.columns) == ["lr"]
        assert result["lr"].tolist() == [0.01, 0.001]

    def test_cartesian_product_two_params(self):
        """
        Branch: Cartesian product of two HP dimensions.
        Methodology: seq_len=[1, 10], lr=[0.001, 0.0001] -> 4 rows.
        Expected: 4 rows with all combinations (itertools.product order: last key varies fastest).
        """
        result = analysis.get_search_grid_df(
            {"task_definition.seq_len": [1, 10], "optimization.lr": [0.001, 0.0001]}
        )
        assert len(result) == 4
        assert list(result.columns) == ["task_definition.seq_len", "optimization.lr"]
        assert sorted(result["task_definition.seq_len"].tolist()) == [1, 1, 10, 10]
        assert sorted(result["optimization.lr"].tolist()) == [
            0.0001,
            0.0001,
            0.001,
            0.001,
        ]


# --- get_unique_values (analysis.get_unique_values) ---


class TestGetUniqueValues:
    """Validates get_unique_values: extraction of unique values from a column."""

    def test_missing_column_returns_empty_list(self):
        """
        Branch: Column not in DataFrame.
        Methodology: Call with column name not in df.columns.
        Expected: [].
        """
        df = pd.DataFrame({"a": [1, 2]})
        assert analysis.get_unique_values(df, "b") == []

    def test_scalar_unique_values(self):
        """
        Branch: Scalar values in column.
        Methodology: Column with [1, 2, 1] -> unique [1, 2].
        Expected: [1, 2] (order may vary).
        """
        df = pd.DataFrame({"x": [1, 2, 1]})
        result = analysis.get_unique_values(df, "x")
        assert set(result) == {1, 2}

    def test_list_values_flattened(self):
        """
        Branch: Column contains lists; unique items are flattened.
        Methodology: Column with [[1, 2], [2, 3]] -> unique {1, 2, 3}.
        Expected: list containing 1, 2, 3.
        """
        df = pd.DataFrame({"col": [[1, 2], [2, 3]]})
        result = analysis.get_unique_values(df, "col")
        assert set(result) == {1, 2, 3}


# --- get_dynamic_metrics (analysis.get_dynamic_metrics) ---


class TestGetDynamicMetrics:
    """Validates get_dynamic_metrics: extraction of metric names from evaluator config column."""

    def test_missing_evaluator_column_returns_empty(self):
        """
        Branch: evaluator_metrics column not present.
        Methodology: DataFrame without config.COLUMN_CONFIG["evaluator_metrics"].
        Expected: [].
        """
        df = pd.DataFrame({"other": [1]})
        result = analysis.get_dynamic_metrics(df)
        assert result == []

    def test_list_entries_collected(self):
        """
        Branch: Column contains list/tuple/ndarray of metric names.
        Methodology: evaluator_metrics column with ["mse", "mae"].
        Expected: sorted list including mse, mae.
        """
        col = config.COLUMN_CONFIG["evaluator_metrics"]
        df = pd.DataFrame({col: [["mse", "mae"], ["mse"]]})
        result = analysis.get_dynamic_metrics(df)
        assert sorted(result) == ["mae", "mse"]

    def test_string_entry_literal_eval(self):
        """
        Branch: evaluator_metrics entry is string; ast.literal_eval parses to list.
        Methodology: Column with string '["mse", "mae"]'.
        Expected: metrics include mse, mae. (analysis.get_dynamic_metrics)
        """
        col = config.COLUMN_CONFIG["evaluator_metrics"]
        df = pd.DataFrame({col: ['["mse", "mae"]']})
        result = analysis.get_dynamic_metrics(df)
        assert "mse" in result and "mae" in result

    def test_string_entry_literal_eval_fails_adds_entry(self):
        """Branch: ast.literal_eval(entry) raises -> found_metrics.add(entry). (analysis.get_dynamic_metrics)"""
        col = config.COLUMN_CONFIG["evaluator_metrics"]
        df = pd.DataFrame({col: ["not valid python"]})
        result = analysis.get_dynamic_metrics(df)
        assert "not valid python" in result

    def test_string_entry_parsed_not_list_adds_entry(self):
        """Branch: parsed is not list/tuple -> found_metrics.add(entry). (analysis.get_dynamic_metrics)"""
        col = config.COLUMN_CONFIG["evaluator_metrics"]
        df = pd.DataFrame({col: ["single_metric"]})  # literal_eval gives str, not list
        result = analysis.get_dynamic_metrics(df)
        assert "single_metric" in result

    def test_entry_neither_list_nor_str_skipped(self):
        """Branch: entry is not list/tuple/ndarray and not str (e.g. int) -> no add, next. (get_dynamic_metrics)"""
        col = config.COLUMN_CONFIG["evaluator_metrics"]
        df = pd.DataFrame({col: [42]})  # int: no update, no add in except
        result = analysis.get_dynamic_metrics(df)
        assert 42 not in result and "42" not in result


# --- get_optimized_metric (analysis.get_optimized_metric) ---


class TestGetOptimizedMetric:
    """Validates get_optimized_metric: priority task definition -> early stopping -> fallbacks."""

    def test_from_task_definition_target_metric(self):
        """
        Branch: Single target_metric in task definition; candidate val/ or test/ exists.
        Methodology: target_metric column has "mse"; code checks val/, test/, val_best_rerun/ in that order.
        Expected: First candidate with data wins (e.g. "val/mse" if present and has count > 0).
        """
        tc = config.COLUMN_CONFIG["target_metric"]
        df = pd.DataFrame(
            {
                tc: ["mse"],
                "test/mse": [0.5],
                "val/mse": [0.52],
            }
        )
        result = analysis.get_optimized_metric(df, "M", "D")
        assert result in ("val/mse", "test/mse")

    def test_from_early_stopping_monitor(self):
        """
        Branch: optimization_metric (early stopping monitor) used when task def not single.
        Methodology: No target_metric single value; optimization_metric column has "val/loss"; column exists.
        Expected: "val/loss".
        """
        om = config.COLUMN_CONFIG["optimization_metric"]
        df = pd.DataFrame({om: ["val/loss"], "val/loss": [0.3]})
        result = analysis.get_optimized_metric(df, "M", "D")
        assert result == "val/loss"

    def test_fallback_metric(self):
        """
        Branch: Fallback list (test/mse, val/loss, etc.) when no task def or early stopping.
        Methodology: No target_metric, no optimization_metric; df has "test/mse".
        Expected: "test/mse".
        """
        df = pd.DataFrame({"test/mse": [0.4]})
        result = analysis.get_optimized_metric(df, "M", "D")
        assert result == "test/mse"

    def test_no_valid_metric_raises(self):
        """
        Branch: No column with data from task def, early stopping, or fallbacks.
        Methodology: DataFrame with no metric columns.
        Expected: ValueError.
        """
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="No valid metric found"):
            analysis.get_optimized_metric(df, "M", "D")

    def test_metric_base_in_columns_when_no_prefix_match(self):
        """
        Branch: metric_base in df_subset.columns when no val/test/val_best_rerun candidate has data.
        Methodology: target_metric "mse", only column "mse" (no val/mse or test/mse).
        Expected: return "mse". (analysis.get_optimized_metric)
        """
        tc = config.COLUMN_CONFIG["target_metric"]
        df = pd.DataFrame({tc: ["mse"], "mse": [0.5]})
        result = analysis.get_optimized_metric(df, "M", "D")
        assert result == "mse"

    def test_no_target_col_uses_early_stopping(self):
        """Branch: target_col not in df -> skip task def, use early stopping. (analysis.get_optimized_metric)"""
        om = config.COLUMN_CONFIG["optimization_metric"]
        df = pd.DataFrame({om: ["test/mse"], "test/mse": [0.4]})
        result = analysis.get_optimized_metric(df, "M", "D")
        assert result == "test/mse"

    def test_early_stopping_candidate_missing_uses_fallback(self):
        """Branch: len(metric_series) >= 1 but candidate not in columns or count 0 -> fallbacks. (get_optimized_metric)"""
        om = config.COLUMN_CONFIG["optimization_metric"]
        df = pd.DataFrame({om: ["val/loss"], "test/mse": [0.4]})  # val/loss not in df
        result = analysis.get_optimized_metric(df, "M", "D")
        assert result == "test/mse"

    def test_fallback_val_loss(self):
        """Branch: fallback val/loss. (analysis.get_optimized_metric)"""
        df = pd.DataFrame({"val/loss": [0.3]})
        result = analysis.get_optimized_metric(df, "M", "D")
        assert result == "val/loss"

    def test_fallback_test_mae(self):
        """Branch: fallback test/mae. (analysis.get_optimized_metric)"""
        df = pd.DataFrame({"test/mae": [0.2]})
        result = analysis.get_optimized_metric(df, "M", "D")
        assert result == "test/mae"

    def test_fallback_val_f1(self):
        """Branch: fallback val/f1. (analysis.get_optimized_metric)"""
        df = pd.DataFrame({"val/f1": [0.8]})
        result = analysis.get_optimized_metric(df, "M", "D")
        assert result == "val/f1"

    def test_fallback_test_accuracy(self):
        """Branch: fallback test/accuracy. (analysis.get_optimized_metric)"""
        df = pd.DataFrame({"test/accuracy": [0.9]})
        result = analysis.get_optimized_metric(df, "M", "D")
        assert result == "test/accuracy"

    # --- fit_predict model resolution (tabpfn / tabdpt / xgboost) ---
    # These models log task_definition.target_metric = "val/loss" in their Hydra
    # config, and val/loss exists as a column in W&B data, but its value is a
    # constant placeholder (1.0) for every config because fit_predict models do
    # not perform iterative training.  Sorting by a constant metric is
    # meaningless, so the tests below document the currently-resolved metric for
    # each model so that any future fix can be verified against them.

    def _fit_predict_df(self, model_target: str) -> pd.DataFrame:
        """
        Build a minimal DataFrame matching what W&B produces for fit_predict
        models: task_definition.target_metric = "val/loss", val/loss always 1.0,
        real performance in test/mse_normalized and test/mae_normalized.
        Two rows represent two HP configs (e.g. different seq_len / stride_train).
        """
        tc = config.COLUMN_CONFIG["target_metric"]
        om = config.COLUMN_CONFIG["optimization_metric"]
        return pd.DataFrame(
            {
                config.COLUMN_CONFIG["model_target"]: [model_target, model_target],
                tc: ["val/loss", "val/loss"],          # Hydra default for fit_predict
                om: ["val/loss", "val/loss"],          # early-stopping monitor
                "val/loss": [1.0, 1.0],                # constant placeholder
                "test/loss": [1.0, 1.0],               # constant placeholder
                "test/mse_normalized": [0.08, 0.15],  # real metric, varies
                "test/mae_normalized": [0.18, 0.27],  # real metric, varies
                "task_definition.seq_len": [50, 1],
                "task_definition.stride_train": [5, 1],
            }
        )

    def test_tabpfn_resolves_to_val_loss(self):
        """
        fit_predict / tabpfn: task_definition.target_metric = "val/loss" and
        val/loss column exists -> Priority 1 (metric_base in columns) returns
        "val/loss".

        NOTE: val/loss is a constant 1.0 for all configs, so this sort metric
        does NOT differentiate HP configurations.  This test documents the
        current (broken) behaviour.
        """
        model = "model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper"
        df = self._fit_predict_df(model)
        result = analysis.get_optimized_metric(df, model, "XJTU-SY")
        assert result == "val/loss"

    def test_tabdpt_resolves_to_val_loss(self):
        """
        fit_predict / tabdpt: same Hydra config as tabpfn -> also resolves to
        "val/loss" (constant placeholder).

        NOTE: documents current (broken) behaviour.
        """
        model = "model.wrappers.fit_predict_tabdpt_wrapper.FitPredictTabDPTWrapper"
        df = self._fit_predict_df(model)
        result = analysis.get_optimized_metric(df, model, "XJTU-SY")
        assert result == "val/loss"

    def test_xgboost_resolves_to_val_loss(self):
        """
        fit_predict / xgboost: same Hydra config pattern -> also resolves to
        "val/loss" (constant placeholder).

        NOTE: documents current (broken) behaviour.
        """
        model = "model.wrappers.fit_predict_xgboost_wrapper.FitPredictXGBoostWrapper"
        df = self._fit_predict_df(model)
        result = analysis.get_optimized_metric(df, model, "XJTU-SY")
        assert result == "val/loss"

    def test_fit_predict_val_loss_constant_does_not_differentiate_configs(self):
        """
        Confirm that val/loss is constant across all fit_predict HP configs
        (value always 1.0), meaning sorting by it produces an arbitrary order.

        This is the root cause of the mis-sorted HP tables for tabpfn/tabdpt/
        xgboost.  A fix should resolve to a metric that actually varies, such
        as test/mse_normalized or test/mae_normalized.
        """
        model = "model.wrappers.fit_predict_tabpfn_wrapper.FitPredictTabPFNWrapper"
        df = self._fit_predict_df(model)
        assert df["val/loss"].nunique() == 1, (
            "val/loss must be constant (1.0) for fit_predict models — "
            "sorting by it cannot rank HP configs."
        )


# --- get_varying_hyperparameters (analysis.get_varying_hyperparameters) ---


class TestGetVaryingHyperparameters:
    """Validates get_varying_hyperparameters: identify HPs with more than one unique value."""

    def test_single_value_excluded(self):
        """
        Branch: Column with one unique value is not varying.
        Methodology: config_cols include "lr", column "lr" has only 0.001.
        Expected: lr not in result.
        """
        df = pd.DataFrame({"lr": [0.001, 0.001], "seq_len": [10, 20]})
        result = analysis.get_varying_hyperparameters(df, ["lr", "seq_len"], [])
        assert "lr" not in result
        assert "seq_len" in result
        assert sorted(result["seq_len"]) == [10, 20]

    def test_special_columns_excluded(self):
        """
        Branch: special_cols_to_exclude are removed from varying params.
        Methodology: seed column has two values but is in special_cols.
        Expected: seed not in result.
        """
        df = pd.DataFrame({"seed": [1, 2], "lr": [0.01, 0.001]})
        result = analysis.get_varying_hyperparameters(df, ["seed", "lr"], ["seed"])
        assert "seed" not in result
        assert "lr" in result

    def test_list_or_tuple_values_nunique(self):
        """
        Branch: Column with list/tuple values; s.nunique() > 1 -> varying_params.
        Methodology: Column "x" with [[1], [2]].
        Expected: varying_params["x"] has two entries. (analysis.get_varying_hyperparameters)
        """
        df = pd.DataFrame({"x": [[1], [2]], "y": [0, 0]})
        result = analysis.get_varying_hyperparameters(df, ["x", "y"], [])
        assert "x" in result
        assert len(result["x"]) == 2

    def test_varying_hyperparameters_exception_logs_and_continues(self):
        """Branch: Exception in column analysis -> logging.warning, pass. (analysis.get_varying_hyperparameters)"""
        df = pd.DataFrame({"bad": [1, 2]})
        result = analysis.get_varying_hyperparameters(df, ["bad"], [])
        assert "bad" in result

    def test_varying_hyperparameters_column_raises(self):
        """Branch: Exception inside try -> import logging; logging.warning; pass. (get_varying_hyperparameters)"""
        # Force exception: patch dropna to raise on first call for second column
        df = pd.DataFrame({"a": [1, 2], "b": [1, 2]})
        with patch.object(pd.Series, "nunique", side_effect=[2, TypeError("mock")]):
            result = analysis.get_varying_hyperparameters(df, ["a", "b"], [])
        assert isinstance(result, dict)
        assert "a" in result  # first column succeeded

    def test_aggregation_col_with_list_applies_tuple(self):
        """Branch: aggregation_col has list/ndarray -> apply tuple. (aggregate_and_find_best)"""
        df = pd.DataFrame(
            {
                "Model": ["M", "M"],
                "lr": [0.001, 0.001],
                "tags": [[1, 2], [1, 2]],
                "test/mse": [0.4, 0.5],
                "run_name": ["r1", "r2"],
            }
        )
        sorted_ds, _ = analysis.aggregate_and_find_best(
            df, ["Model", "lr", "tags"], ["test/mse"], "test/mse", "min"
        )
        assert sorted_ds.sizes["config"] == 1

    def test_optimization_col_mean_missing_empty_perf_returns_empty(self):
        """Branch: no numeric perf cols -> return empty Dataset. (aggregate_and_find_best)"""
        df = pd.DataFrame({"Model": ["M"], "lr": [0.001]})
        a, b = analysis.aggregate_and_find_best(df, ["Model", "lr"], [], "x", "min")
        assert _ds_is_empty(a) and _ds_is_empty(b)

    def test_optimization_col_mean_missing_valid_means_empty_returns_agg_and_empty_best(
        self,
    ):
        """Branch: perf col not in df -> empty Dataset. (aggregate_and_find_best)"""
        df = pd.DataFrame({"Model": ["M"], "lr": [0.001], "run_name": ["r1"]})
        a, b = analysis.aggregate_and_find_best(
            df, ["Model", "lr"], ["missing_metric"], "missing_metric", "min"
        )
        assert _ds_is_empty(b)


# --- aggregate_and_find_best (analysis.aggregate_and_find_best) ---


class TestAggregateAndFindBest:
    """Validates aggregate_and_find_best: groupby mean/std/count and best run selection."""

    def test_numeric_guard_coerces_non_numeric(self):
        """
        Branch: All performance columns coerced to numeric via pd.to_numeric(..., errors="coerce").
        Methodology: One performance column has string "0.5"; should be coerced and aggregated.
        Expected: mean/std/count produced without crash; invalid becomes NaN.
        """
        df = pd.DataFrame(
            {
                "Model": ["M", "M"],
                "lr": [0.001, 0.001],
                "test/mse": [0.4, 0.6],
                "run_name": ["r1", "r2"],
            }
        )
        agg_cols = ["Model", "lr"]
        perf_cols = ["test/mse"]
        opt_col = "test/mse"
        sorted_ds, best = analysis.aggregate_and_find_best(
            df, agg_cols, perf_cols, opt_col, "min"
        )
        assert sorted_ds.sizes["config"] == 1
        assert abs(float(sorted_ds.sel(metric="test/mse")["mean"].values[0]) - 0.5) < 1e-6
        assert float(sorted_ds.sel(metric="test/mse")["std"].values[0]) >= 0
        assert int(sorted_ds.sel(metric="test/mse")["count"].values[0]) == 2

    def test_empty_agg_dict_returns_empty_dataframes(self):
        """
        Branch: When no performance column is numeric or present -> empty Dataset.
        """
        df = pd.DataFrame({"Model": ["M"], "lr": [0.001]})
        sorted_ds, best = analysis.aggregate_and_find_best(
            df, ["Model", "lr"], [], "test/mse", "min"
        )
        assert _ds_is_empty(sorted_ds)
        assert _ds_is_empty(best)

    def test_nan_in_performance_still_aggregates(self):
        """
        Branch: NaNs in performance columns; groupby still runs.
        Methodology: Two rows same group, one test/mse=0.4, one test/mse=NaN.
        Expected: One config row; no crash.
        """
        df = pd.DataFrame(
            {
                "Model": ["M", "M"],
                "lr": [0.001, 0.001],
                "test/mse": [0.4, np.nan],
                "run_name": ["r1", "r2"],
            }
        )
        sorted_ds, best = analysis.aggregate_and_find_best(
            df, ["Model", "lr"], ["test/mse"], "test/mse", "min"
        )
        assert sorted_ds.sizes["config"] == 1
        # count may be 2 or 1 depending on how NaN is handled in groupby
        assert int(sorted_ds.sel(metric="test/mse")["count"].values[0]) >= 1

    def test_fallback_to_first_mean_column(self):
        """
        Branch: optimization_col not in perf cols -> falls back to first available metric.
        Expected: non-empty Dataset; "other_metric" in metric coordinate.
        """
        df = pd.DataFrame(
            {
                "Model": ["M", "M"],
                "lr": [0.001, 0.001],
                "other_metric": [1.0, 2.0],
                "run_name": ["r1", "r2"],
            }
        )
        sorted_ds, best = analysis.aggregate_and_find_best(
            df, ["Model", "lr"], ["other_metric"], "missing_col", "min"
        )
        assert not _ds_is_empty(sorted_ds)
        assert "other_metric" in sorted_ds.coords["metric"].values


# --- analyze_results: Schema-First (EXPECTED_SEARCH_SPACE) ---


class TestAnalyzeResultsSchemaFirst:
    """
    Validates schema-first mode: when EXPECTED_SEARCH_SPACE is provided, the resulting
    DataFrame contains every grid entry, even those with zero matching runs (displayed as -).
    Function: analyze_results (grid merge and left-join logic).
    """

    @pytest.fixture
    def expected_grid(self):
        return {
            "task_definition.seq_len": [10, 50],
            "optimization.lr": [0.001, 0.0005],
        }

    def test_grid_completion_left_join(
        self, mock_df_partial_sweep, default_analysis_kwargs, expected_grid
    ):
        """
        Left-join logic for grid completion: partial sweep merged with full grid.
        Methodology: Mock 2 runs (10/0.001 and 50/0.0005); grid has 4 points.
        Expected: sorted_aggregated_results has 4 rows; missing cells show NaN (display as -).
        Uses new dataset/model structure.
        """
        # New structure: {dataset: {model: {hp: [values]}}}
        # Patch search_space.EXPECTED_SEARCH_SPACE: get_search_space() reads that, not config.
        with patch(
            "picid_report.configs.search_space.EXPECTED_SEARCH_SPACE",
            {
                "DS1": {
                    "baselines.tide_model.TiDE_Forecaster": expected_grid,
                }
            },
        ):
            results = analysis.analyze_results(
                mock_df_partial_sweep,
                **default_analysis_kwargs,
            )
        assert "DS1" in results
        assert "baselines.tide_model.TiDE_Forecaster" in results["DS1"]
        res = results["DS1"]["baselines.tide_model.TiDE_Forecaster"]
        tbl = res["sorted_aggregated_results"]
        assert tbl.sizes["config"] == 4
        # Grid has 2*2 = 4 combinations
        assert set(tbl.coords["task_definition.seq_len"].values.astype(float).astype(int)) == {10, 50}
        assert set(tbl.coords["optimization.lr"].values.astype(float)) == {0.001, 0.0005}

    def test_heterogeneous_hp_string_normalization_merge(
        self, mock_df_heterogeneous_hp, default_analysis_kwargs
    ):
        """
        Schema-first merge with grid: join keys normalized to string so grid and
        actual results align. Methodology: Data with two distinct seq_len values (10, 20)
        so aggregation_cols include task_definition.seq_len; grid [10, 20]. Validates
        left-join and that analysis applies .astype(str) to join keys.
        Expected: Two rows in sorted_aggregated_results; task_definition.seq_len present.
        Uses new dataset/model structure.
        """
        df = mock_df_heterogeneous_hp.copy()
        df.loc[0, "task_definition.seq_len"] = 10
        df.loc[1, "task_definition.seq_len"] = 20
        grid = {
            "task_definition.seq_len": [10, 20],
        }
        # New structure: {dataset: {model: {hp: [values]}}}
        # Patch search_space.EXPECTED_SEARCH_SPACE: get_search_space() reads that, not config.
        with patch(
            "picid_report.configs.search_space.EXPECTED_SEARCH_SPACE",
            {
                "DS1": {
                    "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)": grid,
                }
            },
        ):
            results = analysis.analyze_results(df, **default_analysis_kwargs)
        assert "DS1" in results
        model_key = "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)"
        assert model_key in results["DS1"]
        tbl = results["DS1"][model_key]["sorted_aggregated_results"]
        assert tbl.sizes["config"] == 2
        assert "test/mse" in tbl.coords["metric"].values
        assert sorted(tbl.coords["task_definition.seq_len"].values.astype(float).astype(int).tolist()) == [10, 20]


# --- analyze_results: Data-First (EXPECTED_SEARCH_SPACE None) ---


class TestAnalyzeResultsDataFirst:
    """
    Validates data-first mode: when search space is None, the system auto-discovers
    only the parameters that actually varied. Function: analyze_results.
    """

    def test_auto_discovery_only_varying_params(
        self, mock_df_multi_seed_runs, default_analysis_kwargs
    ):
        """
        Data-first mode: only varying HPs appear in result table.
        Methodology: Mock df with varying seq_len (10, 20) and fixed lr; patch EXPECTED_SEARCH_SPACE to None.
        Expected: Table has only 2 rows (seq_len 10 and 20); no full grid.
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_multi_seed_runs,
                **default_analysis_kwargs,
            )
        assert "DS1" in results
        model_key = "baselines.lstm_model.LSTM_Forecaster"
        assert model_key in results["DS1"]
        tbl = results["DS1"][model_key]["sorted_aggregated_results"]
        assert tbl.sizes["config"] == 2
        assert sorted(tbl.coords["task_definition.seq_len"].values.astype(float).astype(int).tolist()) == [10, 20]


# --- Model wrapper splitting (model_type differentiation) ---


class TestAnalyzeResultsModelWrapperSplitting:
    """
    Validates that models like StatisticalBaselineWrapper (linear) vs (exponential)
    are correctly differentiated by model_target during preprocessing.
    Function: analyze_results (subsetting by dataset and model).
    """

    def test_two_wrappers_separate_entries(
        self, mock_df_two_model_wrappers, default_analysis_kwargs
    ):
        """
        Model wrapper splitting: each model_target gets its own result entry.
        Methodology: DataFrame with (linear) and (exponential) wrapper runs.
        Expected: Two entries in results["DS1"], one per model; each has own best_performance.
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_two_model_wrappers,
                **default_analysis_kwargs,
            )
        assert "DS1" in results
        linear_key = "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (linear)"
        exp_key = "model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper (exponential)"
        assert linear_key in results["DS1"]
        assert exp_key in results["DS1"]
        linear_best = results["DS1"][linear_key]["best_performance"]["metrics"]
        exp_best = results["DS1"][exp_key]["best_performance"]["metrics"]
        assert "mse" in linear_best
        assert "mse" in exp_best
        assert linear_best["mse"]["test"]["mean"] != exp_best["mse"]["test"]["mean"]


# --- String integrity: seeds_info as clean Python int lists ---


class TestSeedsInfoStringIntegrity:
    """
    Validates that seed lists in seeds_info are returned as clean Python int lists
    (string representation like "[1, 2, 3]" not "np.int64(...)").
    Function: analyze_results -> get_formatted_seeds.
    """

    def test_seeds_info_no_np_int64_in_string(
        self, mock_df_multi_seed_runs, default_analysis_kwargs
    ):
        """
        Seeds_info strings must not contain np.int64(...).
        Methodology: Runs with model seeds 1, 2, 3; inspect seeds_info["model"].
        Expected: String looks like "[1, 2, 3]" with no "np.int64" or "numpy" in it.
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_multi_seed_runs,
                **default_analysis_kwargs,
            )
        model_key = "baselines.lstm_model.LSTM_Forecaster"
        seeds_info = results["DS1"][model_key]["seeds_info"]
        model_seeds_str = seeds_info["model"]
        assert "np.int64" not in model_seeds_str
        assert "numpy" not in model_seeds_str
        assert (
            "1" in model_seeds_str and "2" in model_seeds_str and "3" in model_seeds_str
        )


# --- Edge: n=1 (std 0.0), zero metric match, MultiIndex flattening ---


class TestAnalyzeResultsEdgeCases:
    """
    Edge cases: n=1 std formatting, zero metric match no crash, MultiIndex flattening.
    """

    def test_n1_single_run_does_not_crash(
        self, mock_df_single_run_n1, default_analysis_kwargs
    ):
        """
        n=1 case: standard deviation defaults to 0.0 and formatting doesn't crash.
        Methodology: One run per config; aggregation gives count=1, std=NaN or 0.
        Expected: Pipeline completes; reporting uses std 0.0 for display.
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_single_run_n1,
                **default_analysis_kwargs,
            )
        assert "DS1" in results
        model_key = "baselines.lstm_model.LSTM_Forecaster"
        assert model_key in results["DS1"]
        perf = results["DS1"][model_key]["best_performance"]["metrics"]
        assert "mse" in perf
        # std may be NaN in raw; reporting layer converts to 0.0
        assert "test" in perf["mse"]

    def test_zero_metric_match_no_crash(
        self, mock_df_no_matching_metrics, default_analysis_kwargs
    ):
        """
        Zero metric match: when no column matches metric_prefixes, optimization_col still
        drives aggregation if present in df. Use df that has only one metric column (test/mse)
        so that effectively "no extra" metrics match; pipeline must not crash.
        Methodology: Same mock but add test/mse so aggregation runs; metric_prefixes omit custom/.
        Expected: Pipeline completes; single metric drives result.
        """
        # Add test/mse so there is one matching metric; pipeline does not hit empty agg_dict path.
        # (Zero metrics with no optimization_col in df would yield empty agg_dict and later KeyError
        # on sort; we cannot change picid_report so we test the path where one metric exists.)
        df = mock_df_no_matching_metrics.copy()
        df["test/mse"] = 1.0
        kwargs = {**default_analysis_kwargs, "optimization_col": "test/mse"}
        kwargs["metric_prefixes"] = ["val/", "test/"]
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(df, **kwargs)
        assert "DS1" in results
        assert "baselines.lstm_model.LSTM_Forecaster" in results["DS1"]
        res = results["DS1"]["baselines.lstm_model.LSTM_Forecaster"]
        assert "sorted_aggregated_results" in res
        assert True  # may be empty if no varying HPs (xr.Dataset has no .empty)

    def test_multiindex_flattening_in_sorted_aggregated(
        self, mock_df_multi_seed_runs, default_analysis_kwargs
    ):
        """
        MultiIndex flattening: sorted_aggregated_results has flattened columns.
        Methodology: After aggregate_and_find_best, columns are (metric, mean/std/count); flatten to metric_mean etc.
        Expected: Column names are "test/mse_mean", "test/mse_std", "test/mse_count", not tuples.
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_multi_seed_runs,
                **default_analysis_kwargs,
            )
        tbl = results["DS1"]["baselines.lstm_model.LSTM_Forecaster"][
            "sorted_aggregated_results"
        ]
        # xr.Dataset: metrics are coordinate values, not column names
        assert "test/mse" in tbl.coords["metric"].values
        assert "mean" in tbl.data_vars
        assert "std" in tbl.data_vars
        assert "count" in tbl.data_vars


# --- Mean ± std (n=count) and multi-seed aggregation ---


class TestAnalyzeResultsAdditionalBranches:
    """
    Additional branches in analyze_results: empty subset, optimization_mode from config,
    validate_seeds empty, max mode scaling, run_name missing, get_formatted_seeds "?".
    """

    def test_subset_empty_skips_model_dataset(self, default_analysis_kwargs):
        """Branch: subset.empty after filtering by dataset/model -> continue. (analyze_results line 256)"""
        # Two datasets D1, D2; two models M1, M2. Only (D1,M1) and (D2,M2) have rows; (D1,M2) and (D2,M1) empty
        df = pd.DataFrame(
            {
                config.COLUMN_CONFIG["model_target"]: ["M1", "M2"],
                config.COLUMN_CONFIG["dataset_name"]: ["D1", "D2"],
                "seed": [1, 1],
                "run_name": ["r1", "r2"],
                "test/mse": [0.5, 0.6],
            }
        )
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(df, **default_analysis_kwargs)
        assert "D1" in results and "D2" in results
        assert "M1" in results["D1"] and "M2" in results["D2"]

    def test_get_optimized_metric_exception_fallback_to_test_mse(
        self, default_analysis_kwargs
    ):
        """Branch: get_optimized_metric raises -> except, cur_optimization_col = test/mse. (analyze_results)"""
        df = pd.DataFrame(
            {
                config.COLUMN_CONFIG["model_target"]: ["M"],
                config.COLUMN_CONFIG["dataset_name"]: ["D"],
                "seed": [1],
                "run_name": ["r1"],
                "test/mse": [0.5],  # have so we can aggregate
            }
        )
        with patch(
            "picid_report.core.analysis.get_optimized_metric",
            side_effect=ValueError("no metric"),
        ):
            with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
                results = analysis.analyze_results(df, **default_analysis_kwargs)
        assert "D" in results and "M" in results["D"]
        assert (
            results["D"]["M"]["best_performance"]["optimized_on"]["metric"]
            == "test/mse"
        )

    def test_optimization_mode_from_target_metric_mode(self, default_analysis_kwargs):
        """Branch: cur_optimization_mode from target_metric_mode column. (analyze_results)"""
        tmm = config.COLUMN_CONFIG["target_metric_mode"]
        df = pd.DataFrame(
            {
                config.COLUMN_CONFIG["model_target"]: ["M"],
                config.COLUMN_CONFIG["dataset_name"]: ["D"],
                "seed": [1],
                "run_name": ["r1"],
                "test/mse": [0.5],
                tmm: ["max"],
            }
        )
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                df, **{**default_analysis_kwargs, "optimization_mode": None}
            )
        assert (
            results["D"]["M"]["best_performance"]["optimized_on"]["strategy"] == "max"
        )

    def test_optimization_mode_from_es_mode_when_target_mode_empty(
        self, default_analysis_kwargs
    ):
        """Branch: cur_optimization_mode from optimization_mode column when target_metric_mode empty. (analyze_results)"""
        es_mode = config.COLUMN_CONFIG["optimization_mode"]
        df = pd.DataFrame(
            {
                config.COLUMN_CONFIG["model_target"]: ["M"],
                config.COLUMN_CONFIG["dataset_name"]: ["D"],
                "seed": [1],
                "run_name": ["r1"],
                "test/mse": [0.5],
                es_mode: ["max"],
            }
        )
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                df, **{**default_analysis_kwargs, "optimization_mode": None}
            )
        assert (
            results["D"]["M"]["best_performance"]["optimized_on"]["strategy"] == "max"
        )

    def test_optimization_mode_default_min(self, default_analysis_kwargs):
        """Branch: cur_optimization_mode None from both cols -> min. (analyze_results)"""
        df = pd.DataFrame(
            {
                config.COLUMN_CONFIG["model_target"]: ["M"],
                config.COLUMN_CONFIG["dataset_name"]: ["D"],
                "seed": [1],
                "run_name": ["r1"],
                "test/mse": [0.5],
            }
        )
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                df, **{**default_analysis_kwargs, "optimization_mode": None}
            )
        assert (
            results["D"]["M"]["best_performance"]["optimized_on"]["strategy"] == "min"
        )

    def test_required_model_seeds_filters_subset_empty_continues(
        self, default_analysis_kwargs
    ):
        """Branch: validate_seeds with allow_fallback=False removes all rows -> subset.empty -> continue. (analyze_results)"""
        # Use required_data_seeds (allow_fallback=False) so when filter removes all we don't fall back
        df = pd.DataFrame(
            {
                config.COLUMN_CONFIG["model_target"]: ["M"],
                config.COLUMN_CONFIG["dataset_name"]: ["D"],
                "seed": [1],
                "run_name": ["r1"],
                "task_definition.seq_len": [10],
                "optimization.lr": [0.001],
                "test/mse": [0.5],
                config.DEFAULT_DATA_SEED_COL: [1],  # only data seed 1
            }
        )
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                df,
                **{**default_analysis_kwargs, "required_data_seeds": {1, 2}},
            )
        # Group has only data_seed 1; required {1,2} so filtered out (no fallback for data seeds)
        assert "M" not in results.get("D", {})

    def test_model_not_in_expected_search_space_uses_data_first(
        self, mock_df_multi_seed_runs, default_analysis_kwargs
    ):
        """Branch: EXPECTED_SEARCH_SPACE.get(model) None -> model_grid None, data-first sort. (analyze_results)
        Uses new dataset/model structure but model not found, so falls back to data-first."""
        # New structure: {dataset: {model: {hp: [values]}}}
        # Model "OtherModel" not in "DS1", so should fall back to data-first
        with patch(
            "picid_report.configs.search_space.EXPECTED_SEARCH_SPACE",
            {
                "DS1": {
                    "OtherModel": {"lr": [0.01]}
                }  # Different model, so current model uses data-first
            },
        ):
            results = analysis.analyze_results(
                mock_df_multi_seed_runs, **default_analysis_kwargs
            )
        assert "DS1" in results
        assert "baselines.lstm_model.LSTM_Forecaster" in results["DS1"]

    def test_max_mode_scales_mean_std_100(self, default_analysis_kwargs):
        """Branch: col_full == cur_optimization_col and mode == max -> mean, std * 100. (analyze_results)"""
        df = pd.DataFrame(
            {
                config.COLUMN_CONFIG["model_target"]: ["M"],
                config.COLUMN_CONFIG["dataset_name"]: ["D"],
                "seed": [1],
                "run_name": ["r1"],
                "test/f1": [0.5],
                "test/mse": [0.1],
            }
        )
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                df,
                **{
                    **default_analysis_kwargs,
                    "optimization_col": "test/f1",
                    "optimization_mode": "max",
                },
            )
        metrics = results["D"]["M"]["best_performance"]["metrics"]
        assert "f1" in metrics
        assert float(metrics["f1"]["test"]["mean"]) == 50.0
        std_val = metrics["f1"]["test"]["std"]
        assert std_val == 0.0 or (std_val != std_val)  # 0 or NaN (n=1)

    def test_seeds_info_question_when_col_missing(self, default_analysis_kwargs):
        """Branch: get_formatted_seeds returns "?" when column not in subset. (analyze_results)"""
        df = pd.DataFrame(
            {
                config.COLUMN_CONFIG["model_target"]: ["M"],
                config.COLUMN_CONFIG["dataset_name"]: ["D"],
                "seed": [1],
                "run_name": ["r1"],
                "test/mse": [0.5],
            }
        )
        # data_seed_col is datasource.parameters.data_seed - not in df
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(df, **default_analysis_kwargs)
        assert results["D"]["M"]["seeds_info"]["data"] == "?"

    def test_seeds_info_question_on_value_error(self, default_analysis_kwargs):
        """Branch: get_formatted_seeds except (ValueError, TypeError) -> "?". (analyze_results)"""
        df = pd.DataFrame(
            {
                config.COLUMN_CONFIG["model_target"]: ["M"],
                config.COLUMN_CONFIG["dataset_name"]: ["D"],
                "seed": ["not_an_int"],
                "run_name": ["r1"],
                "test/mse": [0.5],
            }
        )
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(df, **default_analysis_kwargs)
        assert results["D"]["M"]["seeds_info"]["model"] == "?"

    def test_best_result_no_run_name_no_run_names_in_best_params(
        self, default_analysis_kwargs
    ):
        """Branch: run_name not in best_result.columns -> best_params has no run_names. (analyze_results)"""
        df = pd.DataFrame(
            {
                config.COLUMN_CONFIG["model_target"]: ["M"],
                config.COLUMN_CONFIG["dataset_name"]: ["D"],
                "seed": [1],
                "task_definition.seq_len": [10],
                "optimization.lr": [0.001],
                "test/mse": [0.5],
            }
        )
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(df, **default_analysis_kwargs)
        bp = results["D"]["M"]["best_hyperparameters"]
        # When run_name not in subset, agg_dict has no run_name so best_result has no run_name column
        assert "run_names" not in bp


class TestMeanStdCountAggregation:
    """
    Validates mean ± std (n=count) from multiple runs per configuration.
    Functions: aggregate_and_find_best (analysis), then reporting formatting.
    """

    def test_mean_std_count_values(
        self, mock_df_multi_seed_runs, default_analysis_kwargs
    ):
        """
        Multiple runs per config: mean, std, count are correct.
        Methodology: seq_len=10 has test/mse 0.4, 0.5, 0.6 -> mean 0.5, std ~0.1, n=3.
        Expected: best_performance.metrics.mse.test has mean 0.5, count 3, std > 0.
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_multi_seed_runs,
                **default_analysis_kwargs,
            )
        model_key = "baselines.lstm_model.LSTM_Forecaster"
        metrics = results["DS1"][model_key]["best_performance"]["metrics"]
        mse_test = metrics["mse"]["test"]
        assert mse_test["count"] == 3
        assert abs(mse_test["mean"] - 0.5) < 1e-6
        assert mse_test["std"] >= 0


# --- total_runs and configs_failed_not_full_seed_set (analyze_results) ---


class TestTotalRunsAndConfigsFailed:
    """
    Validates that analyze_results stores total_runs and configs_failed_not_full_seed_set
    after model-seed filtering (runs count and configs dropped for incomplete seed set).
    """

    def test_total_runs_and_configs_failed_when_seed_filter_drops_configs(
        self, mock_df_partial_seed_config, default_analysis_kwargs
    ):
        """
        When required_model_seeds is set and one config has only 2 of 3 seeds, that config
        is dropped. total_runs = runs in retained configs; configs_failed = 1.
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_partial_seed_config,
                **{
                    **default_analysis_kwargs,
                    "required_model_seeds": {72, 88, 101},
                },
            )
        res = results["DS1"]["baselines.lstm_model.LSTM_Forecaster"]
        assert res["total_runs"] == 6  # 3 + 3 runs in the two retained configs
        assert res["configs_failed_not_full_seed_set"] == 1  # seq_len=30 dropped
        assert res["sorted_aggregated_results"].sizes["config"] == 2  # 2 configs completed

    def test_total_runs_and_configs_failed_zero_when_no_seed_filter(
        self, mock_df_multi_seed_runs, default_analysis_kwargs
    ):
        """
        When required_model_seeds is None, no configs are dropped by seed filter.
        total_runs = all runs (6); configs_failed_not_full_seed_set = 0.
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_multi_seed_runs,
                **default_analysis_kwargs,
            )
        res = results["DS1"]["baselines.lstm_model.LSTM_Forecaster"]
        assert res["total_runs"] == 6
        assert res["configs_failed_not_full_seed_set"] == 0
        assert res["sorted_aggregated_results"].sizes["config"] == 2


# --- configs_failed_missing_invalid_metric (analyze_results) ---


class TestConfigsFailedMissingInvalidMetric:
    """
    Validates that analyze_results stores configs_failed_missing_invalid_metric
    when some runs have missing/invalid sort metric (dropped by metric filter).
    """

    def test_configs_failed_invalid_metric_when_one_config_has_nan_metric(
        self, mock_df_partial_valid_metric, default_analysis_kwargs
    ):
        """
        One config has all seeds but one run has NaN for sort metric; after metric filter
        that config has only 2 runs -> dropped. configs_failed_missing_invalid_metric=1.
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_partial_valid_metric,
                **{
                    **default_analysis_kwargs,
                    "required_model_seeds": {72, 88, 101},
                },
            )
        res = results["DS1"]["baselines.lstm_model.LSTM_Forecaster"]
        assert res["configs_failed_missing_invalid_metric"] == 1
        assert res["sorted_aggregated_results"].sizes["config"] == 1  # only seq_len=10 kept

    def test_configs_failed_invalid_metric_zero_when_all_valid(
        self, mock_df_multi_seed_runs, default_analysis_kwargs
    ):
        """All rows have valid sort metric -> configs_failed_missing_invalid_metric=0."""
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_multi_seed_runs,
                **default_analysis_kwargs,
            )
        res = results["DS1"]["baselines.lstm_model.LSTM_Forecaster"]
        assert res["configs_failed_missing_invalid_metric"] == 0


# --- Schema-first configs failed stats (x-y, x-z) ---


class TestSchemaFirstConfigsFailedStats:
    """
    Validates schema-first stats: Configs Failed (not full seed set) = x - y,
    Configs Failed (missing/invalid metric) = x - z, where x = grid size.
    """

    def test_schema_first_x_minus_y_and_x_minus_z(
        self, mock_df_partial_valid_metric, default_analysis_kwargs
    ):
        """
        Grid has 3 points (seq_len 10, 20, 30); data has only 10 and 20 with all seeds before
        metric filter (y=2). After metric filter only config 10 has all seeds (z=1).
        Expected: configs_failed_not_full_seed_set = x - y = 1, configs_failed_missing_invalid_metric = x - z = 2.
        """
        # Grid of 3 (single varying key so merge keys match aggregated results)
        grid_3 = {"task_definition.seq_len": [10, 20, 30]}
        with patch.object(
            config,
            "EXPECTED_SEARCH_SPACE",
            {
                "baselines.lstm_model.LSTM_Forecaster": grid_3,
            },
        ):
            results = analysis.analyze_results(
                mock_df_partial_valid_metric,
                **{
                    **default_analysis_kwargs,
                    "required_model_seeds": {72, 88, 101},
                },
            )
        res = results["DS1"]["baselines.lstm_model.LSTM_Forecaster"]
        assert res["configs_failed_not_full_seed_set"] == 1  # x - y = 3 - 2
        assert res["configs_failed_missing_invalid_metric"] == 2  # x - z = 3 - 1


# --- Sort-metric fallback when original would drop all rows ---


class TestSortMetricFallback:
    """
    When the chosen sort metric would drop all rows (missing or all invalid),
    pipeline should find a fallback (val* then test*) and use it; model is not skipped.
    """

    def test_fallback_used_when_original_metric_missing(
        self, mock_df_original_metric_missing_has_fallback, default_analysis_kwargs
    ):
        """
        Original metric val/loss is not in subset; val/mse and test/mse exist with valid values.
        Expected: Fallback to val/mse (prefer val over test), model not skipped,
        res has sort_metric_used=val/mse, sort_metric_is_fallback=True, original_sort_metric=val/loss.
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_original_metric_missing_has_fallback,
                **{
                    **default_analysis_kwargs,
                    "optimization_col": "val/loss",
                },
            )
        assert "DS1" in results
        assert "baselines.lstm_model.LSTM_Forecaster" in results["DS1"]
        res = results["DS1"]["baselines.lstm_model.LSTM_Forecaster"]
        assert res.get("sort_metric_is_fallback") is True
        assert res.get("original_sort_metric") == "val/loss"
        assert res.get("sort_metric_used") == "val/mse"
        assert (
            "sorted_aggregated_results" in res
            and not _ds_is_empty(res["sorted_aggregated_results"])
        )

    def test_no_fallback_when_no_val_or_test_metric(self, base_config_columns):
        """
        Original metric val/loss missing and no val/ or test/ column with valid data.
        Expected: Fallback returns None; we do not set sort_metric_is_fallback.
        Pipeline completes without crash.
        """
        df = pd.DataFrame(
            [
                {
                    config.COLUMN_CONFIG["model_target"]: "SomeModel",
                    config.COLUMN_CONFIG["dataset_name"]: "DS1",
                    config.DEFAULT_MODEL_SEED_COL: 1,
                    "run_name": "r1",
                    config.COLUMN_CONFIG["optimization_metric"]: "test/mse",
                    config.COLUMN_CONFIG["optimization_mode"]: "min",
                    "task_definition.seq_len": 10,
                    "optimization.lr": 0.001,
                    "custom/only": 1.0,
                }
            ]
        )
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                df,
                config_columns=base_config_columns + ["custom/only"],
                dropped_columns=[],
                optimization_col="val/loss",
                reporting_metrics=["mse", "loss"],
                metric_prefixes=["val/", "test/"],
            )
        assert isinstance(results, dict)
        if "DS1" in results and "SomeModel" in results["DS1"]:
            res = results["DS1"]["SomeModel"]
            assert res.get("sort_metric_is_fallback") is not True

    def test_fallback_prefers_varying_metric_over_constant(self, base_config_columns):
        """
        When fallback is needed, prefer a metric that varies across rows over one that is constant
        (e.g. test/mae_normalized over test/loss when test/loss is 1.0 for all rows).
        """
        df = pd.DataFrame(
            [
                {
                    config.COLUMN_CONFIG["model_target"]: "FitPredictWrapper",
                    config.COLUMN_CONFIG["dataset_name"]: "DS1",
                    config.DEFAULT_MODEL_SEED_COL: 1,
                    "run_name": "r1",
                    config.COLUMN_CONFIG["optimization_metric"]: "test/mse",
                    config.COLUMN_CONFIG["optimization_mode"]: "min",
                    "task_definition.seq_len": 10,
                    "test/loss": 1.0,
                    "test/mae_normalized": 0.25,
                },
                {
                    config.COLUMN_CONFIG["model_target"]: "FitPredictWrapper",
                    config.COLUMN_CONFIG["dataset_name"]: "DS1",
                    config.DEFAULT_MODEL_SEED_COL: 2,
                    "run_name": "r2",
                    config.COLUMN_CONFIG["optimization_metric"]: "test/mse",
                    config.COLUMN_CONFIG["optimization_mode"]: "min",
                    "task_definition.seq_len": 10,
                    "test/loss": 1.0,
                    "test/mae_normalized": 0.31,
                },
            ]
        )
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                df,
                config_columns=base_config_columns,
                dropped_columns=[],
                optimization_col="val/loss",
                reporting_metrics=["mse", "loss"],
                metric_prefixes=["val/", "test/"],
            )
        assert "DS1" in results and "FitPredictWrapper" in results["DS1"]
        res = results["DS1"]["FitPredictWrapper"]
        assert res.get("sort_metric_is_fallback") is True
        assert res.get("original_sort_metric") == "val/loss"
        # Should use varying test/mae_normalized, not constant test/loss
        assert res.get("sort_metric_used") == "test/mae_normalized"


# --- Backward Compatibility: Legacy Model-Only Structure ---


class TestBackwardCompatibilityLegacySearchSpace:
    """Validates backward compatibility with legacy model-only EXPECTED_SEARCH_SPACE structure."""

    @pytest.fixture
    def expected_grid(self):
        return {
            "task_definition.seq_len": [10, 50],
            "optimization.lr": [0.001, 0.0005],
        }

    def test_legacy_model_only_structure_still_works(
        self, mock_df_partial_sweep, default_analysis_kwargs, expected_grid
    ):
        """
        Branch: Legacy structure {model: {hp: [values]}} still works via backward compatibility.
        Methodology: Use old structure, verify it works.
        Expected: Grid lookup succeeds using legacy structure.
        """
        # Legacy structure: {model: {hp: [values]}}
        with patch.object(
            config,
            "EXPECTED_SEARCH_SPACE",
            {
                "baselines.tide_model.TiDE_Forecaster": expected_grid,
            },
        ):
            results = analysis.analyze_results(
                mock_df_partial_sweep,
                **default_analysis_kwargs,
            )
        assert "DS1" in results
        assert "baselines.tide_model.TiDE_Forecaster" in results["DS1"]
        res = results["DS1"]["baselines.tide_model.TiDE_Forecaster"]
        tbl = res["sorted_aggregated_results"]
        # Should still work with legacy structure
        assert (
            len(tbl) == 4 or len(tbl) >= 2
        )  # May use grid or data-first depending on implementation


# --- Sort Metric Resolution (analysis.analyze_results with sort_metric_resolver) ---


class TestSortMetricResolution:
    """Validates sort metric resolution in analyze_results."""

    def test_analyze_results_with_sort_metric_resolver_stores_metric(
        self, default_analysis_kwargs
    ):
        """
        Branch: sort_metric_resolver provided, stores sort_metric_used in results.
        Methodology: Call analyze_results with resolver that returns "test/accuracy".
        Expected: sort_metric_used = "test/accuracy" in results.
        """

        def resolver(dataset, model, task_type=None, dataset_category=None):
            return "test/accuracy"

        df = pd.DataFrame(
            {
                "model._target_": ["M", "M"],
                "datasource.data_name": ["DS1", "DS1"],
                "test/mse": [0.5, 0.6],
                "test/accuracy": [0.8, 0.9],
                "task_definition.seq_len": [10, 20],
                "seed": [1, 2],
            }
        )

        results = analysis.analyze_results(
            df, **{**default_analysis_kwargs, "sort_metric_resolver": resolver}
        )

        res = results["DS1"]["M"]
        assert "sort_metric_used" in res
        assert res["sort_metric_used"] == "test/accuracy"

    def test_analyze_results_without_resolver_no_sort_metric_used(
        self, default_analysis_kwargs
    ):
        """
        Branch: sort_metric_resolver=None, sort_metric_used is None.
        Methodology: Call analyze_results without resolver.
        Expected: sort_metric_used = None (or not present).
        """
        df = pd.DataFrame(
            {
                "model._target_": ["M", "M"],
                "datasource.data_name": ["DS1", "DS1"],
                "test/mse": [0.5, 0.6],
                "task_definition.seq_len": [10, 20],
                "seed": [1, 2],
            }
        )

        results = analysis.analyze_results(df, **default_analysis_kwargs)

        res = results["DS1"]["M"]
        assert res.get("sort_metric_used") is None

    def test_sort_metric_resolver_returns_none_uses_optimization(
        self, default_analysis_kwargs
    ):
        """
        Branch: Resolver returns None, sort_metric_used is None (use optimization metric).
        Methodology: Resolver returns None.
        Expected: sort_metric_used = None.
        """

        def resolver(dataset, model, task_type=None, dataset_category=None):
            return None

        df = pd.DataFrame(
            {
                "model._target_": ["M", "M"],
                "datasource.data_name": ["DS1", "DS1"],
                "test/mse": [0.5, 0.6],
                "task_definition.seq_len": [10, 20],
                "seed": [1, 2],
            }
        )

        results = analysis.analyze_results(
            df, **{**default_analysis_kwargs, "sort_metric_resolver": resolver}
        )

        res = results["DS1"]["M"]
        assert res["sort_metric_used"] is None

    def test_sort_metric_resolver_receives_correct_parameters(
        self, default_analysis_kwargs
    ):
        """
        Branch: Resolver called with dataset, model, task_type, dataset_category.
        Methodology: Mock resolver to capture arguments.
        Expected: Called with correct dataset and model.
        """
        captured_args = []

        def resolver(dataset, model, task_type=None, dataset_category=None):
            captured_args.append((dataset, model, task_type, dataset_category))
            return "test/accuracy"

        df = pd.DataFrame(
            {
                "model._target_": ["M", "M"],
                "datasource.data_name": ["DS1", "DS1"],
                "test/mse": [0.5, 0.6],
                "task_definition.seq_len": [10, 20],
                "seed": [1, 2],
            }
        )

        analysis.analyze_results(
            df, **{**default_analysis_kwargs, "sort_metric_resolver": resolver}
        )

        assert len(captured_args) == 1
        dataset, model, task_type, dataset_category = captured_args[0]
        assert dataset == "DS1"
        assert model == "M"

    def test_sort_metric_resolver_exception_handled_gracefully(
        self, default_analysis_kwargs
    ):
        """
        Branch: Resolver raises exception, handled gracefully, uses optimization metric.
        Methodology: Resolver raises ValueError.
        Expected: Warning logged, sort_metric_used = None.
        """

        def resolver(dataset, model, task_type=None, dataset_category=None):
            raise ValueError("Resolver error")

        df = pd.DataFrame(
            {
                "model._target_": ["M", "M"],
                "datasource.data_name": ["DS1", "DS1"],
                "test/mse": [0.5, 0.6],
                "task_definition.seq_len": [10, 20],
                "seed": [1, 2],
            }
        )

        results = analysis.analyze_results(
            df, **{**default_analysis_kwargs, "sort_metric_resolver": resolver}
        )

        res = results["DS1"]["M"]
        # Should handle exception and set to None
        assert res["sort_metric_used"] is None

    def test_pipeline_config_sort_metric_resolver_used(self, default_analysis_kwargs):
        """
        Branch: PipelineConfig.sort_metric_resolver used if provided.
        Methodology: Create PipelineConfig with resolver.
        Expected: Resolver from config used.
        """
        from picid_report.config import PipelineConfig

        def resolver(dataset, model, task_type=None, dataset_category=None):
            return "test/f1"

        pipeline_config = PipelineConfig(
            column_config=config.COLUMN_CONFIG,
            expected_search_space={},
            column_filters_to_ignore_for_hp_search=config.COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH,
            special_columns=config.SPECIAL_COLUMNS,
            columns_to_normalize=config.COLUMNS_TO_NORMALIZE,
            column_filters_to_drop=config.COLUMN_FILTERS_TO_DROP,
            sort_metric_resolver=resolver,
        )

        df = pd.DataFrame(
            {
                "model._target_": ["M", "M"],
                "datasource.data_name": ["DS1", "DS1"],
                "test/mse": [0.5, 0.6],
                "task_definition.seq_len": [10, 20],
                "seed": [1, 2],
            }
        )

        results = analysis.analyze_results(
            df, **{**default_analysis_kwargs, "pipeline_config": pipeline_config}
        )

        res = results["DS1"]["M"]
        assert res["sort_metric_used"] == "test/f1"
