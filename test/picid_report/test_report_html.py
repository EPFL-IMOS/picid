"""
Test suite for picid_report.report.report_html.

Validates HTML report generation with metric information display.
"""

from collections import defaultdict
import os
import tempfile

from picid_report.report import report_html, reporting


def _make_all_results(
    metrics_data=None,
    sorted_aggregated=None,
    seeds_info=None,
    optimized_on=None,
):
    """Build minimal all_results structure for HTML report tests."""
    if metrics_data is None:
        metrics_data = {
            "mse": {
                "test": {"mean": 0.5, "std": 0.1, "count": 3},
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


# --- write_report_html (report_html.write_report_html) ---


class TestWriteReportHtml:
    """Validates write_report_html: HTML generation with metric information."""

    def test_write_report_html_basic(self):
        """
        Branch: Basic HTML report generation without metric info.
        Methodology: Call write_report_html with minimal data.
        Expected: HTML file created successfully.
        """
        all_results = _make_all_results()
        summary_df = reporting.create_summary_table(all_results)
        stats_df = reporting.get_experiment_stats_df(all_results)

        with tempfile.TemporaryDirectory() as tmpdir:
            hp_impact_entries = []
            global_plots = []
            path = report_html.write_report_html(
                output_dir=tmpdir,
                summary_df=summary_df,
                stats_df=stats_df,
                hp_impact_entries=hp_impact_entries,
                global_plots=global_plots,
            )
            assert os.path.exists(path)
            assert path.endswith("report.html")
            with open(path, "r") as f:
                content = f.read()
                assert "Summary table" in content

    def test_metric_info_display_in_summary_table_single_metric(self):
        """
        Branch: Single sort metric used for all model/dataset combinations.
        Methodology: all_results with same optimization metric, sort_metric="test/accuracy".
        Expected: HTML shows "Metric used to select best results: test/accuracy".
        """
        all_results = _make_all_results()
        summary_df = reporting.create_summary_table(
            all_results, sort_metric="test/accuracy"
        )
        stats_df = reporting.get_experiment_stats_df(all_results)

        with tempfile.TemporaryDirectory() as tmpdir:
            hp_impact_entries = []
            global_plots = []
            path = report_html.write_report_html(
                output_dir=tmpdir,
                summary_df=summary_df,
                stats_df=stats_df,
                hp_impact_entries=hp_impact_entries,
                global_plots=global_plots,
                all_results=all_results,
                sort_metric="test/accuracy",
            )
            with open(path, "r") as f:
                content = f.read()
                assert "Metric used to select best results" in content
                assert "test/accuracy" in content

    def test_metric_info_display_in_summary_table_multiple_metrics(self):
        """
        Branch: Multiple sort metrics used (varies by model/dataset).
        Methodology: all_results with different metrics, sort_metric as dict.
        Expected: HTML shows "varies by model/dataset".
        """
        all_results = defaultdict(lambda: defaultdict(dict))
        all_results["DS1"]["ModelA"]["best_performance"] = {
            "optimized_on": {"metric": "test/mse", "strategy": "min"},
            "metrics": {"mse": {"test": {"mean": 0.5, "std": 0.1, "count": 3}}},
        }
        all_results["DS2"]["ModelB"]["best_performance"] = {
            "optimized_on": {"metric": "test/accuracy", "strategy": "max"},
            "metrics": {"accuracy": {"test": {"mean": 0.9, "std": 0.02, "count": 3}}},
        }

        summary_df = reporting.create_summary_table(all_results)
        stats_df = reporting.get_experiment_stats_df(all_results)
        sort_metric_dict = {
            ("DS1", "ModelA"): "test/mse",
            ("DS2", "ModelB"): "test/accuracy",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            hp_impact_entries = []
            global_plots = []
            path = report_html.write_report_html(
                output_dir=tmpdir,
                summary_df=summary_df,
                stats_df=stats_df,
                hp_impact_entries=hp_impact_entries,
                global_plots=global_plots,
                all_results=all_results,
                sort_metric=sort_metric_dict,
            )
            with open(path, "r") as f:
                content = f.read()
                assert "Metric used to select best results" in content
                # Should mention that it varies
                assert "varies" in content.lower() or "test/mse" in content

    def test_metric_info_display_in_hp_impact(self):
        """
        Branch: Metric info displayed for each HP impact table.
        Methodology: hp_impact_entries includes metric_used.
        Expected: HTML shows "Dataset name: ..., Model name: ..., Metric used to select best results: ...".
        """
        import pandas as pd

        df_sorted = pd.DataFrame(
            {
                "Model": ["M"],
                "task_definition.seq_len": [10],
                "test/mse_mean": [0.5],
                "test/mse_std": [0.1],
                "test/mse_count": [3],
            }
        )
        all_results = _make_all_results(sorted_aggregated=df_sorted)
        summary_df = reporting.create_summary_table(all_results)
        stats_df = reporting.get_experiment_stats_df(all_results)

        # Create hp_impact_entries with metric_used
        hp_df = pd.DataFrame({"x": [1]})
        hp_impact_entries = [
            ("DS1", "ModelA", hp_df, "plots/test.png", "test/accuracy")
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            global_plots = []
            path = report_html.write_report_html(
                output_dir=tmpdir,
                summary_df=summary_df,
                stats_df=stats_df,
                hp_impact_entries=hp_impact_entries,
                global_plots=global_plots,
                all_results=all_results,
                sort_metric="test/accuracy",
            )
            with open(path, "r") as f:
                content = f.read()
                assert "Dataset name:" in content
                assert "Model name:" in content
                assert "Metric used to select best results:" in content
                assert "DS1" in content
                assert "ModelA" in content
                assert "test/accuracy" in content

    def test_write_report_html_with_all_results_parameter(self):
        """
        Branch: all_results parameter works correctly.
        Methodology: Pass all_results to write_report_html.
        Expected: HTML generated successfully, metric info extracted.
        """
        all_results = _make_all_results()
        summary_df = reporting.create_summary_table(all_results)
        stats_df = reporting.get_experiment_stats_df(all_results)

        with tempfile.TemporaryDirectory() as tmpdir:
            hp_impact_entries = []
            global_plots = []
            path = report_html.write_report_html(
                output_dir=tmpdir,
                summary_df=summary_df,
                stats_df=stats_df,
                hp_impact_entries=hp_impact_entries,
                global_plots=global_plots,
                all_results=all_results,
                sort_metric=None,
            )
            assert os.path.exists(path)
            # Should work without errors
