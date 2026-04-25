"""
Tests for picid_report.core.preprocess.

Covers clean_and_rename_models.
"""

import pandas as pd

from picid_report import config
from picid_report.core import preprocess


class TestCleanAndRenameModels:
    """clean_and_rename_models: strip common prefix and append model type."""

    def test_strips_common_prefix(self):
        df = pd.DataFrame(
            {
                "model._target_": [
                    "picid.model.forecasters.patchtst_model.PatchTST_Forecaster",
                    "picid.model.forecasters.lstm_model.LSTM_Forecaster",
                ],
            }
        )
        out = preprocess.clean_and_rename_models(df, model_target_col="model._target_")
        # common prefix up to last dot before final segment
        assert out["model._target_"].iloc[0] == "patchtst_model.PatchTST_Forecaster"
        assert out["model._target_"].iloc[1] == "lstm_model.LSTM_Forecaster"

    def test_appends_model_type_when_present(self):
        df = pd.DataFrame(
            {
                "model._target_": [
                    "pkg.StatisticalBaselineWrapper",
                    "pkg.StatisticalBaselineWrapper",
                ],
                "model.model_type": ["linear", "exponential"],
            }
        )
        out = preprocess.clean_and_rename_models(
            df, model_target_col="model._target_", model_type_col="model.model_type"
        )
        assert " (linear)" in out["model._target_"].iloc[0]
        assert " (exponential)" in out["model._target_"].iloc[1]

    def test_single_model_no_prefix_stripped(self):
        df = pd.DataFrame({"model._target_": ["only.model.Class"]})
        out = preprocess.clean_and_rename_models(df, model_target_col="model._target_")
        assert out["model._target_"].iloc[0] == "only.model.Class"

    def test_returns_copy(self):
        df = pd.DataFrame({"model._target_": ["a.b.C"]})
        out = preprocess.clean_and_rename_models(df, model_target_col="model._target_")
        assert out is not df

    def test_missing_type_col_no_error(self):
        df = pd.DataFrame({"model._target_": ["a.b.C"]})
        out = preprocess.clean_and_rename_models(df, model_target_col="model._target_")
        assert out["model._target_"].iloc[0] == "a.b.C"

    def test_uses_config_defaults_when_cols_not_passed(self):
        """When model_target_col is None, uses config.COLUMN_CONFIG['model_target']."""
        df = pd.DataFrame(
            {
                config.COLUMN_CONFIG["model_target"]: [
                    "picid.model.forecasters.patchtst_model.PatchTST_Forecaster",
                    "picid.model.forecasters.lstm_model.LSTM_Forecaster",
                ],
            }
        )
        out = preprocess.clean_and_rename_models(df)
        assert (
            out[config.COLUMN_CONFIG["model_target"]].iloc[0]
            == "patchtst_model.PatchTST_Forecaster"
        )
