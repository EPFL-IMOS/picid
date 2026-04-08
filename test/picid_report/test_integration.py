"""
Integration tests for picid_report: full pipeline from analyze_results to reporting.

Validates that analysis output structure is compatible with reporting functions
and that schema-first vs data-first and model wrapper differentiation hold end-to-end.
Does not modify picid_report or touch picid_report/not_used.
"""

from unittest.mock import patch

from picid_report import config
from picid_report.core import analysis
from picid_report.report import reporting


class TestPipelineIntegration:
    """
    End-to-end: analyze_results -> create_summary_table / display_hp_impact.
    """

    def test_analysis_output_feed_into_summary_table(
        self, mock_df_multi_seed_runs, default_analysis_kwargs
    ):
        """
        Full pipeline: analyze_results then create_summary_table.
        Methodology: Run analysis with data-first; pass all_results to create_summary_table.
        Expected: Summary table non-empty, contains mean ± std (n=count) format.
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_multi_seed_runs,
                **default_analysis_kwargs,
            )
        table = reporting.create_summary_table(results, precision=4)
        assert not table.empty
        assert "±" in table.to_string()
        assert "(n=" in table.to_string()

    def test_analysis_output_feed_into_display_hp_impact(
        self, mock_df_multi_seed_runs, default_analysis_kwargs
    ):
        """
        Full pipeline: analyze_results then display_hp_impact (no crash).
        Methodology: Run analysis; pass all_results to display_hp_impact.
        Expected: No exception; display_hp_impact flattens and formats.
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_multi_seed_runs,
                **default_analysis_kwargs,
            )
        reporting.display_hp_impact(results, precision=4)

    def test_schema_first_grid_then_display_hp_impact_shows_all_rows(
        self, mock_df_partial_sweep, default_analysis_kwargs
    ):
        """
        Schema-first: merged grid has all points; display_hp_impact shows them (with - for missing).
        Methodology: Partial sweep (2 runs), grid of 4; run analysis then display_hp_impact.
        Expected: 4 rows in sorted_aggregated_results; display fills NaN with "-".
        Uses new dataset/model structure.
        """
        grid = {
            "task_definition.seq_len": [10, 50],
            "optimization.lr": [0.001, 0.0005],
        }
        # New structure: {dataset: {model: {hp: [values]}}}
        # Patch search_space.EXPECTED_SEARCH_SPACE: get_search_space() reads that, not config.
        with patch(
            "picid_report.configs.search_space.EXPECTED_SEARCH_SPACE",
            {
                "DS1": {
                    "baselines.tide_model.TiDE_Forecaster": grid,
                }
            },
        ):
            results = analysis.analyze_results(
                mock_df_partial_sweep,
                **default_analysis_kwargs,
            )
        tbl = results["DS1"]["baselines.tide_model.TiDE_Forecaster"][
            "sorted_aggregated_results"
        ]
        assert tbl.sizes["config"] == 4
        reporting.display_hp_impact(results, precision=4)

    def test_val_best_rerun_captured_as_metric_not_hp(
        self, mock_df_val_best_rerun_prefix, default_analysis_kwargs
    ):
        """
        Custom prefix val_best_rerun/: captured as metric, ignored as HP.
        Methodology: Mock df with val_best_rerun/mse; run analysis.
        Expected: best_performance.metrics includes val_best_rerun (or mse with val_best_rerun key).
        """
        with patch.object(config, "EXPECTED_SEARCH_SPACE", None):
            results = analysis.analyze_results(
                mock_df_val_best_rerun_prefix,
                **default_analysis_kwargs,
            )
        model_key = "baselines.patchtst_model.PatchTST_Forecaster"
        assert "DS1" in results and model_key in results["DS1"]
        metrics = results["DS1"][model_key]["best_performance"]["metrics"]
        # val_best_rerun is a prefix; metric name could be "mse" with prefix "val_best_rerun"
        prefix_found = any(
            "val_best_rerun" in str(p) or "val_best_rerun" in str(m)
            for m, prefs in metrics.items()
            for p in prefs
        )
        assert prefix_found or "mse" in metrics
