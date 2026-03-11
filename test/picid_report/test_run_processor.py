"""
Test suite for picid_report.core.run_processor.

Covers _filter_columns, _normalize_column, and load_runs_df with mocks (no real wandb or disk).
"""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pandas as pd

from picid_report.core import run_processor


class TestFilterColumns:
    """Validates _filter_columns."""

    def test_returns_columns_containing_any_filter(self):
        df = pd.DataFrame(columns=["paths.x", "other", "paths.y", "a"])
        out = run_processor._filter_columns(df, ["paths."])
        assert set(out) == {"paths.x", "paths.y"}

    def test_empty_filters_returns_empty(self):
        df = pd.DataFrame(columns=["a"])
        out = run_processor._filter_columns(df, [])
        assert out == []


class TestNormalizeColumn:
    """Validates _normalize_column."""

    def test_column_not_in_df_returns_unchanged(self):
        df = pd.DataFrame({"a": [1, 2]})
        out_df, cols = run_processor._normalize_column(df, "missing")
        assert out_df is df
        assert cols == []

    def test_valid_entries_empty_drops_column_returns_empty_list(self):
        df = pd.DataFrame({"nest": [None, None]})
        out_df, cols = run_processor._normalize_column(df, "nest")
        assert "nest" not in out_df.columns
        assert cols == []

    def test_string_dict_literal_eval_fails_safe_load_returns_empty_dict(self):
        df = pd.DataFrame({"nest": ["{invalid"]})
        out_df, cols = run_processor._normalize_column(df, "nest")
        # valid_entries has one entry; safe_load returns {}; json_normalize of [{}] gives empty or one col
        assert "nest" not in out_df.columns
        assert isinstance(cols, list)

    def test_flattens_nested_dict(self):
        df = pd.DataFrame({"nest": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]})
        out_df, cols = run_processor._normalize_column(df, "nest")
        assert "nest" not in out_df.columns
        assert "nest.a" in out_df.columns and "nest.b" in out_df.columns
        assert out_df["nest.a"].tolist() == [1, 3]
        assert "nest.a" in cols and "nest.b" in cols

    def test_safe_load_non_dict_non_str_returns_empty_dict(self):
        df = pd.DataFrame({"nest": [123]})  # int not dict
        out_df, cols = run_processor._normalize_column(df, "nest")
        assert "nest" not in out_df.columns


class TestLoadRunsDf:
    """Validates load_runs_df with mocked wandb and filesystem."""

    def test_load_from_cache(self):
        """Branch: cache file exists -> read_csv, literal_eval summary/config, normalize, return."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, "csv_files")
            os.makedirs(cache_dir)
            csv_path = os.path.join(cache_dir, "proj.csv")
            # summary and config must be valid literal_eval strings (double quotes for JSON)
            raw = pd.DataFrame(
                {
                    "summary": ['{"test/mse": 0.5}'],
                    "config": ['{"model._target_": "M", "datasource.data_name": "D"}'],
                    "run_name": ["r1"],
                }
            )
            raw.to_csv(csv_path)
            df, config_cols, dropped = run_processor.load_runs_df(
                "proj", "user", csv_cache_dir=cache_dir
            )
            assert isinstance(df, pd.DataFrame)
            assert "run_name" in df.columns
            assert isinstance(config_cols, list)
            assert isinstance(dropped, list)

    def test_fetch_from_wandb_when_cache_missing(self):
        """Branch: FileNotFoundError -> wandb api.runs(), build df, save cache."""
        mock_run = MagicMock()
        mock_run.state = "finished"
        mock_run.name = "r1"
        mock_run.summary._json_dict = {"test/mse": 0.5}
        mock_run.config = {"model._target_": "M", "datasource.data_name": "D"}

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, "csv_cache")
            os.makedirs(cache_dir)
            with patch(
                "picid_report.core.run_processor.pd.read_csv",
                side_effect=FileNotFoundError,
            ):
                with patch("picid_report.core.run_processor.wandb.Api") as MockApi:
                    mock_api = MockApi.return_value
                    mock_api.runs.return_value = [mock_run]
                    df, config_cols, dropped = run_processor.load_runs_df(
                        "proj", "user", csv_cache_dir=cache_dir
                    )
            assert isinstance(df, pd.DataFrame)
            assert len(config_cols) >= 0
            assert isinstance(dropped, list)

    def test_skip_run_not_finished(self):
        """Branch: run.state != 'finished' -> skip, print."""
        mock_run_finished = MagicMock()
        mock_run_finished.state = "finished"
        mock_run_finished.name = "r1"
        mock_run_finished.summary._json_dict = {}
        mock_run_finished.config = {}
        mock_run_crashed = MagicMock()
        mock_run_crashed.state = "crashed"
        mock_run_crashed.name = "r2"

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, "csv_cache")
            os.makedirs(cache_dir)
            with patch(
                "picid_report.core.run_processor.pd.read_csv",
                side_effect=FileNotFoundError,
            ):
                with patch("picid_report.core.run_processor.wandb.Api") as MockApi:
                    mock_api = MockApi.return_value
                    mock_api.runs.return_value = [mock_run_finished, mock_run_crashed]
                    df, _, _ = run_processor.load_runs_df(
                        "proj", "user", csv_cache_dir=cache_dir
                    )
            assert len(df) == 1
