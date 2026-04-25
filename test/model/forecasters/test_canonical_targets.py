from pathlib import Path

from omegaconf import OmegaConf

from picid.interface.schemas.model.crossformer import CrossformerConfig
from picid.interface.schemas.model.linear_forecaster import LinearForecasterConfig
from picid.interface.schemas.model.lstm import LSTMConfig


def test_model_schema_defaults_use_canonical_forecasters_namespace():
    assert (
        LSTMConfig().model_class == "picid.model.forecasters.lstm_model.LSTM_Forecaster"
    )
    assert (
        LinearForecasterConfig().model_class
        == "picid.model.forecasters.linear_model.linear_model.Linear_Forecaster"
    )
    assert (
        CrossformerConfig(dropout=0.1).model_class
        == "picid.model.forecasters.crossformer_model.Crossformer_Forecaster"
    )


def test_forecaster_yaml_targets_use_canonical_namespace():
    repo_root = Path(__file__).resolve().parents[3]
    expected_targets = {
        "lstm.yaml": "picid.model.forecasters.lstm_model.LSTM_Forecaster",
        "linear_forecaster.yaml": (
            "picid.model.forecasters.linear_model.linear_model.Linear_Forecaster"
        ),
        "patchtst.yaml": ("picid.model.forecasters.patchtst_model.PatchTST_Forecaster"),
        "crossformer.yaml": (
            "picid.model.forecasters.crossformer_model.Crossformer_Forecaster"
        ),
        "tide.yaml": "picid.model.forecasters.tide_model.TiDE_Forecaster",
        "timeseries_transformer.yaml": (
            "picid.model.forecasters.timeseries_transformer_model."
            "Timeseries_Transformer_Forecaster"
        ),
        "stf.yaml": (
            "picid.model.forecasters.spacetimeformer_model."
            "Spacetimeformer_Forecaster"
        ),
    }

    for filename, expected_target in expected_targets.items():
        cfg = OmegaConf.load(repo_root / "configs" / "model" / filename)
        assert cfg._target_ == expected_target
