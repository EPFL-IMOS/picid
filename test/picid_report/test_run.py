"""
Tests for picid_report.run.

Covers _safe_basename, _save_outputs (mocked I/O), and main (CLI parsing).
"""

import os
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pandas as pd

from picid_report import run


class TestSafeBasename:
    """_safe_basename: sanitize dataset and model for filenames."""

    def test_sanitizes_special_chars(self):
        out = run._safe_basename("ds/with/slash", "model.with.dots")
        assert "/" not in out
        assert out == "ds_with_slash_model.with.dots" or "_" in out

    def test_truncates_to_max_len(self):
        long_ds = "a" * 100
        long_m = "b" * 100
        out = run._safe_basename(long_ds, long_m, max_len=10)
        assert len(out) <= 10 + 1 + 10  # two parts plus separator

    def test_typical_names_unchanged(self):
        out = run._safe_basename("UNIBO21", "model.wrappers.mlp_wrapper.MLPWrapper")
        assert "UNIBO21" in out
        assert "mlp" in out.lower() or "MLP" in out


class TestSaveOutputs:
    """_save_outputs: writes tables, optional plots, and report.html."""

    def test_creates_dirs_and_files(self):
        all_results = defaultdict(lambda: defaultdict(dict))
        all_results["DS1"]["M1"]["best_performance"] = {
            "optimized_on": {"metric": "test/mse"},
            "metrics": {"mse": {"test": {"mean": 0.5, "std": 0.1, "count": 3}}},
        }
        summary_df = pd.DataFrame({"A": [1]})
        stats_df = pd.DataFrame({"total_runs": [10]})
        with tempfile.TemporaryDirectory() as tmpdir:
            run._save_outputs(
                output_dir=tmpdir,
                all_results=all_results,
                summary_df=summary_df,
                stats_df=stats_df,
                show_plots=False,
                precision=4,
                export_laTeX=False,
                plot_metric="test/mse",
                sort_metric=None,
            )
            assert os.path.isdir(os.path.join(tmpdir, "tables"))
            assert os.path.isdir(os.path.join(tmpdir, "plots"))
            assert os.path.isfile(os.path.join(tmpdir, "report.html"))
            summary_csv = os.path.join(tmpdir, "tables", "summary.csv")
            assert os.path.isfile(summary_csv)
            stats_csv = os.path.join(tmpdir, "tables", "experiment_stats.csv")
            assert os.path.isfile(stats_csv)


class TestMain:
    """main: CLI entry point."""

    def test_parses_args_and_calls_run_pipeline(self):
        with patch.object(run, "run_pipeline", MagicMock()) as mock_run:
            with patch("sys.argv", ["run", "--output-dir", "/tmp/out", "--no-plots"]):
                run.main()
            mock_run.assert_called_once()
            call_kw = mock_run.call_args[1]
            assert call_kw.get("output_dir") == "/tmp/out"
            assert call_kw.get("show_plots") is False

    def test_default_output_dir_none(self):
        with patch.object(run, "run_pipeline", MagicMock()) as mock_run:
            with patch("sys.argv", ["run"]):
                run.main()
            call_kw = mock_run.call_args[1]
            assert call_kw.get("output_dir") is None
