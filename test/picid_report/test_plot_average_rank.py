"""
Test suite for ranking functionality in picid_report.scripts.plot_average_rank.

Validates parsing of summary CSVs, task-type inference, metric selection (with fallback),
rank computation (average rank across datasets and per-dataset), and filename/label helpers.
"""

from pathlib import Path

import pandas as pd

from picid_report.scripts import plot_average_rank as pa


# --- _parse_value_cell ---


class TestParseValueCell:
    """Tests for _parse_value_cell: extract numeric mean from summary table cells."""

    def test_valid_mean_std_count(self):
        assert pa._parse_value_cell("0.0657 ± 0.0000 (n=1)") == 0.0657
        assert pa._parse_value_cell("1.5 ± 0.2 (n=5)") == 1.5

    def test_number_only(self):
        assert pa._parse_value_cell("0.42") == 0.42
        assert pa._parse_value_cell("  -3.14  ") == -3.14

    def test_scientific_notation(self):
        assert pa._parse_value_cell("1e-3 ± 0 (n=1)") == 0.001
        assert pa._parse_value_cell("2.5E+2") == 250.0

    def test_dash_returns_none(self):
        assert pa._parse_value_cell("-") is None
        assert pa._parse_value_cell("  -  ") is None

    def test_nan_returns_none(self):
        assert pa._parse_value_cell("nan") is None
        assert pa._parse_value_cell("NaN") is None

    def test_empty_or_none_returns_none(self):
        assert pa._parse_value_cell("") is None
        assert pa._parse_value_cell(None) is None

    def test_invalid_returns_none(self):
        assert pa._parse_value_cell("n/a") is None
        assert pa._parse_value_cell("missing") is None


# --- _shorten_model_name ---


class TestShortenModelName:
    """Tests for _shorten_model_name: last segment after '.', keep (linear) suffix."""

    def test_last_segment_used(self):
        assert pa._shorten_model_name("a.b.c.ModelName") == "ModelName"

    def test_single_part_unchanged(self):
        assert pa._shorten_model_name("ModelA") == "ModelA"

    def test_suffix_preserved(self):
        assert (
            pa._shorten_model_name("a.b.StatisticalBaselineWrapper (linear)")
            == "StatisticalBaselineWrapper (linear)"
        )

    def test_empty_returns_empty(self):
        assert pa._shorten_model_name("") == ""


# --- _format_metrics_label ---


class TestFormatMetricsLabel:
    """Tests for _format_metrics_label: plot title metric summary."""

    def test_empty_returns_auto(self):
        assert pa._format_metrics_label([]) == "test metric (auto)"

    def test_single_metric_no_count(self):
        assert pa._format_metrics_label(["mae_normalized"]) == "mae_normalized"
        assert pa._format_metrics_label(["accuracy"]) == "accuracy"

    def test_multiple_with_counts_sorted_by_count_desc(self):
        out = pa._format_metrics_label(
            ["loss", "loss", "loss", "mae_normalized", "mae_normalized"]
        )
        assert "loss (3)" in out
        assert "mae_normalized (2)" in out
        assert out.startswith("loss (3)")  # higher count first


# --- _safe_filename ---


class TestSafeFilename:
    """Tests for _safe_filename: safe for filesystem."""

    def test_colon_replaced(self):
        assert "::" not in pa._safe_filename("run_id::PHME20")
        assert pa._safe_filename("run_id::PHME20") == "run_id__PHME20"

    def test_safe_unchanged(self):
        assert pa._safe_filename("29_01_2026_phme20") == "29_01_2026_phme20"

    def test_empty_returns_unnamed(self):
        assert pa._safe_filename("") == "unnamed"
        assert pa._safe_filename(":::") == "unnamed"


# --- _dataset_key ---


class TestDatasetKey:
    """Tests for _dataset_key: unique key per (file, dataset)."""

    def test_run_id_and_dataset(self):
        path = Path("/some/report_output/29_01_2026_phme20/tables/summary.csv")
        assert pa._dataset_key(path, "PHME20") == "29_01_2026_phme20::PHME20"

    def test_no_parent_returns_dataset_only(self):
        assert pa._dataset_key(Path("summary.csv"), "DS1") == "DS1"


# --- compute_average_ranks ---


class TestComputeAverageRanks:
    """Tests for compute_average_ranks: rank per dataset, then average per model."""

    def test_empty_returns_empty_df(self):
        df = pa.compute_average_ranks([])
        assert df.empty

    def test_single_dataset_two_models_min(self):
        records = [("D1", "M1", 0.3), ("D1", "M2", 0.5)]
        df = pa.compute_average_ranks(records, rank_mode="min")
        assert len(df) == 2
        assert set(df["model_full"]) == {"M1", "M2"}
        assert df[df["model_full"] == "M1"]["average_rank"].iloc[0] == 1.0
        assert df[df["model_full"] == "M2"]["average_rank"].iloc[0] == 2.0
        assert df["n_datasets"].iloc[0] == 1

    def test_single_dataset_two_models_max(self):
        records = [("D1", "M1", 0.3), ("D1", "M2", 0.5)]  # higher is better
        df = pa.compute_average_ranks(records, rank_mode="max")
        assert df[df["model_full"] == "M2"]["average_rank"].iloc[0] == 1.0
        assert df[df["model_full"] == "M1"]["average_rank"].iloc[0] == 2.0

    def test_two_datasets_average_rank(self):
        records = [
            ("D1", "M1", 0.1),
            ("D1", "M2", 0.2),
            ("D1", "M3", 0.3),
            ("D2", "M2", 0.1),
            ("D2", "M1", 0.2),
            ("D2", "M3", 0.3),
        ]
        df = pa.compute_average_ranks(records, rank_mode="min")
        assert len(df) == 3
        # M1: rank 1 in D1, rank 2 in D2 -> avg 1.5
        # M2: rank 2 in D1, rank 1 in D2 -> avg 1.5
        # M3: rank 3 in both -> avg 3
        m1_avg = df[df["model_full"] == "M1"]["average_rank"].iloc[0]
        m2_avg = df[df["model_full"] == "M2"]["average_rank"].iloc[0]
        m3_avg = df[df["model_full"] == "M3"]["average_rank"].iloc[0]
        assert m1_avg == 1.5
        assert m2_avg == 1.5
        assert m3_avg == 3.0
        assert df["n_datasets"].iloc[0] == 2

    def test_has_model_short_column(self):
        records = [("D1", "baselines.lstm.LSTM", 0.5)]
        df = pa.compute_average_ranks(records, rank_mode="min")
        assert "model_short" in df.columns
        assert df["model_short"].iloc[0] == "LSTM"


# --- compute_rank_one_dataset ---


class TestComputeRankOneDataset:
    """Tests for compute_rank_one_dataset: rank for a single dataset."""

    def test_empty_returns_empty_df(self):
        df = pa.compute_rank_one_dataset([])
        assert df.empty

    def test_two_models_min(self):
        records = [("M1", 0.4), ("M2", 0.2)]
        df = pa.compute_rank_one_dataset(records, rank_mode="min")
        assert len(df) == 2
        assert df[df["model_full"] == "M2"]["average_rank"].iloc[0] == 1.0
        assert df[df["model_full"] == "M1"]["average_rank"].iloc[0] == 2.0
        assert list(df["n_datasets"]) == [1, 1]

    def test_two_models_max(self):
        records = [("M1", 0.9), ("M2", 0.7)]
        df = pa.compute_rank_one_dataset(records, rank_mode="max")
        assert df[df["model_full"] == "M1"]["average_rank"].iloc[0] == 1.0
        assert df[df["model_full"] == "M2"]["average_rank"].iloc[0] == 2.0


# --- parse_summary_csv (with temp file) ---


class TestParseSummaryCsv:
    """Tests for parse_summary_csv: parse pivot summary, extract (dataset, model, value) for metric."""

    def test_too_few_lines_returns_empty(self, tmp_path):
        f = tmp_path / "summary.csv"
        f.write_text("Dataset\nMetric\nModel\n")
        assert pa.parse_summary_csv(f, "test/mse") == []

    def test_metric_match_extracts_values(self, tmp_path):
        content = (
            "Dataset,DS1,DS1\n"
            "Metric,test/mse,test/loss\n"
            "Model,,,\n"
            "M1,0.5 ± 0.1 (n=3),1.0\n"
            "M2,0.3 ± 0.0 (n=3),0.8\n"
        )
        (tmp_path / "summary.csv").write_text(content)
        records = pa.parse_summary_csv(tmp_path / "summary.csv", "test/mse")
        assert len(records) == 2
        assert records[0][0] == "DS1" and records[0][1] == "M1" and records[0][2] == 0.5
        assert records[1][1] == "M2" and records[1][2] == 0.3

    def test_metric_missing_returns_empty(self, tmp_path):
        content = "Dataset,DS1\nMetric,test/loss\nModel,\nM1,0.5\n"
        (tmp_path / "summary.csv").write_text(content)
        assert pa.parse_summary_csv(tmp_path / "summary.csv", "test/mse") == []


# --- infer_task_type ---


class TestInferTaskType:
    """Tests for infer_task_type: classification vs regression from metric row."""

    def test_classification_when_accuracy_present(self, tmp_path):
        content = "Dataset,DS1\nMetric,test/accuracy\nModel,\nM1,0.9\n"
        (tmp_path / "s.csv").write_text(content)
        assert pa.infer_task_type(tmp_path / "s.csv") == "classification"

    def test_regression_when_mse_present(self, tmp_path):
        content = "Dataset,DS1\nMetric,test/mse\nModel,\nM1,0.5\n"
        (tmp_path / "s.csv").write_text(content)
        assert pa.infer_task_type(tmp_path / "s.csv") == "regression"

    def test_regression_when_loss_present(self, tmp_path):
        content = "Dataset,DS1\nMetric,test/loss\nModel,\nM1,0.3\n"
        (tmp_path / "s.csv").write_text(content)
        assert pa.infer_task_type(tmp_path / "s.csv") == "regression"

    def test_classification_preferred_when_both(self, tmp_path):
        content = (
            "Dataset,DS1,DS1\nMetric,test/accuracy,test/mse\nModel,,\nM1,0.9,0.5\n"
        )
        (tmp_path / "s.csv").write_text(content)
        assert pa.infer_task_type(tmp_path / "s.csv") == "classification"

    def test_too_few_lines_returns_none(self, tmp_path):
        (tmp_path / "s.csv").write_text("Dataset\nMetric\n")
        assert pa.infer_task_type(tmp_path / "s.csv") is None


# --- pick_metric_and_collect_records_with_fallback ---


class TestPickMetricAndCollectRecordsWithFallback:
    """Tests for pick_metric_and_collect_records_with_fallback: per-model fallback, logical metrics."""

    def test_returns_none_when_no_match(self, tmp_path):
        content = "Dataset,DS1\nMetric,custom/only\nModel,\nM1,1.0\n"
        (tmp_path / "s.csv").write_text(content)
        logical = [
            ("mae_normalized", ["test/mae_normalized", "test/mae_normalized_mean"])
        ]
        chosen, recs = pa.pick_metric_and_collect_records_with_fallback(
            tmp_path / "s.csv", logical
        )
        assert chosen is None
        assert recs == []

    def test_picks_first_logical_with_data(self, tmp_path):
        content = (
            "Dataset,DS1,DS1\n"
            "Metric,test/mae_normalized,test/loss\n"
            "Model,,\n"
            "M1,0.25 ± 0.0 (n=1),1.0\n"
            "M2,0.31 ± 0.0 (n=1),1.0\n"
        )
        (tmp_path / "s.csv").write_text(content)
        logical = [
            (
                "mae_normalized",
                ["test_best_rerun/mae_normalized", "test/mae_normalized"],
            ),
            ("loss", ["test_best_rerun/loss", "test/loss"]),
        ]
        chosen, recs = pa.pick_metric_and_collect_records_with_fallback(
            tmp_path / "s.csv", logical
        )
        assert chosen == "mae_normalized"
        assert len(recs) == 2
        assert recs[0][1] == "M1" and recs[0][2] == 0.25
        assert recs[1][1] == "M2" and recs[1][2] == 0.31

    def test_dataset_key_in_records_uses_path(self, tmp_path):
        content = "Dataset,DS1\n" "Metric,test/mae_normalized\n" "Model,\n" "M1,0.5\n"
        sub = tmp_path / "run_phme20" / "tables"
        sub.mkdir(parents=True)
        (sub / "summary.csv").write_text(content)
        logical = [("mae_normalized", ["test/mae_normalized"])]
        chosen, recs = pa.pick_metric_and_collect_records_with_fallback(
            sub / "summary.csv", logical
        )
        assert chosen == "mae_normalized"
        assert len(recs) == 1
        assert recs[0][0] == "run_phme20::DS1"  # _dataset_key applied


# --- discover_summary_csvs ---


class TestDiscoverSummaryCsvs:
    """Tests for discover_summary_csvs: rglob tables/summary.csv."""

    def test_finds_nested(self, tmp_path):
        (tmp_path / "a" / "tables" / "summary.csv").parent.mkdir(parents=True)
        (tmp_path / "a" / "tables" / "summary.csv").write_text("x")
        found = pa.discover_summary_csvs(tmp_path)
        assert len(found) == 1
        assert found[0].name == "summary.csv"
        assert "tables" in str(found[0])

    def test_empty_dir_returns_empty(self, tmp_path):
        assert pa.discover_summary_csvs(tmp_path) == []


# --- collect_all_records ---


class TestCollectAllRecords:
    """Tests for collect_all_records: parse multiple CSVs with one metric."""

    def test_aggregates_from_multiple_files(self, tmp_path):
        (tmp_path / "p1" / "tables").mkdir(parents=True)
        (tmp_path / "p2" / "tables").mkdir(parents=True)
        (tmp_path / "p1" / "tables" / "summary.csv").write_text(
            "Dataset,D1\nMetric,test/mse\nModel,\nM1,0.5\n"
        )
        (tmp_path / "p2" / "tables" / "summary.csv").write_text(
            "Dataset,D2\nMetric,test/mse\nModel,\nM1,0.6\n"
        )
        paths = [
            tmp_path / "p1" / "tables" / "summary.csv",
            tmp_path / "p2" / "tables" / "summary.csv",
        ]
        records = pa.collect_all_records(paths, "test/mse")
        assert len(records) == 2
        datasets = {r[0] for r in records}
        assert "D1" in datasets and "D2" in datasets


# --- plot_average_rank (bar plot figure) ---


class TestPlotAverageRankFigure:
    """Tests for plot_average_rank: bar plot from rank DataFrame."""

    def test_empty_df_returns_none(self):
        fig = pa.plot_average_rank(pd.DataFrame(), save_path=None)
        assert fig is None

    def test_returns_figure_with_bars(self):
        df = pd.DataFrame(
            {
                "model_full": ["M1", "M2"],
                "model_short": ["M1", "M2"],
                "average_rank": [1.0, 2.0],
                "n_datasets": [1, 1],
            }
        )
        fig = pa.plot_average_rank(df, save_path=None, title="Test")
        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.patches) == 2
        assert ax.get_ylabel() == "Average rank"
        assert ax.get_title() == "Test"

    def test_custom_ylabel(self):
        df = pd.DataFrame(
            {
                "model_full": ["M1"],
                "model_short": ["M1"],
                "average_rank": [1.0],
                "n_datasets": [1],
            }
        )
        fig = pa.plot_average_rank(df, save_path=None, ylabel="Rank")
        assert fig.axes[0].get_ylabel() == "Rank"

    def test_save_path_creates_file(self, tmp_path):
        df = pd.DataFrame(
            {
                "model_full": ["M1"],
                "model_short": ["M1"],
                "average_rank": [1.0],
                "n_datasets": [1],
            }
        )
        out = tmp_path / "rank.png"
        pa.plot_average_rank(df, save_path=out, title="Test")
        assert out.is_file()


# --- _save_per_dataset_rank_plots ---


class TestSavePerDatasetRankPlots:
    """Tests for _save_per_dataset_rank_plots: one PNG per dataset."""

    def test_empty_records_does_nothing(self, tmp_path):
        pa._save_per_dataset_rank_plots([], tmp_path, "min", "regression")
        assert list(tmp_path.iterdir()) == []

    def test_saves_one_png_per_dataset(self, tmp_path):
        records_with_metric = [
            ("run1::DS1", "M1", 0.3, None, None, "mae_normalized"),
            ("run1::DS1", "M2", 0.5, None, None, "mae_normalized"),
        ]
        pa._save_per_dataset_rank_plots(
            records_with_metric, tmp_path, rank_mode="min", task_label="regression"
        )
        files = list(tmp_path.glob("*.png"))
        assert len(files) == 1
        assert "mae_normalized" in files[0].name
        assert "DS1" in files[0].name or "run1" in files[0].name

    def test_two_datasets_two_files(self, tmp_path):
        records_with_metric = [
            ("r1::D1", "M1", 0.4, None, None, "loss"),
            ("r1::D1", "M2", 0.6, None, None, "loss"),
            ("r2::D2", "M1", 0.5, None, None, "loss"),
        ]
        pa._save_per_dataset_rank_plots(
            records_with_metric, tmp_path, rank_mode="min", task_label="regression"
        )
        files = list(tmp_path.glob("*.png"))
        assert len(files) == 2
