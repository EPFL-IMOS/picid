from importlib import import_module

import pytest


MODULE_PATHS = [
    "picid.model.forecasters",
    "picid.model.forecasters.forecaster",
    "picid.model.forecasters.linear_model",
    "picid.model.forecasters.linear_model.linear_ar",
    "picid.model.forecasters.linear_model.linear_model",
    "picid.model.forecasters.lstm_model",
    "picid.model.forecasters.lstm_model.lstm_model",
    "picid.model.forecasters.patchtst_model",
    "picid.model.forecasters.patchtst_model.thuml_patchtst",
    "picid.model.forecasters.patchtst_model.thuml_patchtst_model",
    "picid.model.forecasters.crossformer_model",
    "picid.model.forecasters.crossformer_model.cross_attn",
    "picid.model.forecasters.crossformer_model.cross_decoder",
    "picid.model.forecasters.crossformer_model.cross_embed",
    "picid.model.forecasters.crossformer_model.cross_encoder",
    "picid.model.forecasters.crossformer_model.cross_former",
    "picid.model.forecasters.crossformer_model.crossformer_model",
    "picid.model.forecasters.tide_model",
    "picid.model.forecasters.tide_model.tide",
    "picid.model.forecasters.tide_model.tide_model",
    "picid.model.forecasters.timeseries_transformer_model",
    "picid.model.forecasters.timeseries_transformer_model.timeseries_transformer",
    "picid.model.forecasters.timeseries_transformer_model.timeseries_transformer_model",
    "picid.model.forecasters.shared_layers",
    "picid.model.forecasters.shared_layers.AutoCorrelation",
    "picid.model.forecasters.shared_layers.Autoformer_EncDec",
    "picid.model.forecasters.shared_layers.Embed_DLinear",
    "picid.model.forecasters.shared_layers.Embed_PatchTST",
    "picid.model.forecasters.shared_layers.Embed_iTransformer",
    "picid.model.forecasters.shared_layers.SelfAttention_Family",
    "picid.model.forecasters.shared_layers.Transformer_EncDec",
    "picid.model.forecasters.shared_layers.masking",
    "picid.model.forecasters.shared_layers.siren",
    "picid.model.forecasters.spacetimeformer_model",
    "picid.model.forecasters.spacetimeformer_model.spacetimeformer_model",
    "picid.model.forecasters.spacetimeformer_model.nn",
    "picid.model.forecasters.spacetimeformer_model.nn.attn",
    "picid.model.forecasters.spacetimeformer_model.nn.data_dropout",
    "picid.model.forecasters.spacetimeformer_model.nn.decoder",
    "picid.model.forecasters.spacetimeformer_model.nn.embed",
    "picid.model.forecasters.spacetimeformer_model.nn.encoder",
    "picid.model.forecasters.spacetimeformer_model.nn.extra_layers",
    "picid.model.forecasters.spacetimeformer_model.nn.model",
    "picid.model.forecasters.spacetimeformer_model.nn.powernorm",
    "picid.model.forecasters.spacetimeformer_model.nn.scalenorm",
    "picid.model.forecasters.spacetimeformer_model.nn.time2vec",
    "picid.model.forecasters.spacetimeformer_model.utils",
    "picid.model.forecasters.spacetimeformer_model.utils.masking",
]


SYMBOL_PARITY_CASES = [
    (
        "picid.baselines.forecaster",
        "picid.model.forecasters.forecaster",
        ["Forecaster", "TransformerForecaster"],
    ),
    (
        "picid.baselines.linear_model.linear_ar",
        "picid.model.forecasters.linear_model.linear_ar",
        ["LinearModel"],
    ),
    (
        "picid.baselines.linear_model.linear_model",
        "picid.model.forecasters.linear_model.linear_model",
        ["Linear_Forecaster"],
    ),
    (
        "picid.baselines.lstm_model",
        "picid.model.forecasters.lstm_model",
        ["LSTM_Forecaster"],
    ),
    (
        "picid.baselines.patchtst_model",
        "picid.model.forecasters.patchtst_model",
        ["PatchTST_Forecaster"],
    ),
    (
        "picid.baselines.patchtst_model.thuml_patchtst",
        "picid.model.forecasters.patchtst_model.thuml_patchtst",
        ["Model", "Transpose", "FlattenHead"],
    ),
    (
        "picid.baselines.crossformer_model",
        "picid.model.forecasters.crossformer_model",
        ["Crossformer_Forecaster"],
    ),
    (
        "picid.baselines.crossformer_model.cross_former",
        "picid.model.forecasters.crossformer_model.cross_former",
        ["Crossformer"],
    ),
    (
        "picid.baselines.crossformer_model.cross_attn",
        "picid.model.forecasters.crossformer_model.cross_attn",
        ["FullAttention", "AttentionLayer", "TwoStageAttentionLayer"],
    ),
    (
        "picid.baselines.tide_model",
        "picid.model.forecasters.tide_model",
        ["TiDE_Forecaster"],
    ),
    (
        "picid.baselines.tide_model.tide",
        "picid.model.forecasters.tide_model.tide",
        ["Model"],
    ),
    (
        "picid.baselines.timeseries_transformer_model",
        "picid.model.forecasters.timeseries_transformer_model",
        ["Timeseries_Transformer_Forecaster"],
    ),
    (
        "picid.baselines.timeseries_transformer_model.timeseries_transformer",
        "picid.model.forecasters.timeseries_transformer_model.timeseries_transformer",
        ["Model"],
    ),
    (
        "picid.baselines.shared_layers.SelfAttention_Family",
        "picid.model.forecasters.shared_layers.SelfAttention_Family",
        ["FullAttention", "AttentionLayer", "ProbAttention"],
    ),
    (
        "picid.baselines.shared_layers.Transformer_EncDec",
        "picid.model.forecasters.shared_layers.Transformer_EncDec",
        ["Encoder", "Decoder", "EncoderLayer", "DecoderLayer"],
    ),
    (
        "picid.baselines.shared_layers.Embed_PatchTST",
        "picid.model.forecasters.shared_layers.Embed_PatchTST",
        ["PatchEmbedding", "PositionalEmbedding"],
    ),
    (
        "picid.baselines.shared_layers.masking",
        "picid.model.forecasters.shared_layers.masking",
        ["TriangularCausalMask", "ProbMask"],
    ),
    (
        "picid.baselines.spacetimeformer_model",
        "picid.model.forecasters.spacetimeformer_model",
        ["Spacetimeformer_Forecaster"],
    ),
    (
        "picid.baselines.spacetimeformer_model.nn",
        "picid.model.forecasters.spacetimeformer_model.nn",
        ["Spacetimeformer"],
    ),
    (
        "picid.baselines.spacetimeformer_model.nn.attn",
        "picid.model.forecasters.spacetimeformer_model.nn.attn",
        ["FullAttention", "ProbAttention", "AttentionLayer"],
    ),
    (
        "picid.baselines.spacetimeformer_model.nn.embed",
        "picid.model.forecasters.spacetimeformer_model.nn.embed",
        ["Embedding"],
    ),
    (
        "picid.baselines.spacetimeformer_model.utils.masking",
        "picid.model.forecasters.spacetimeformer_model.utils.masking",
        ["TriangularCausalMask", "ProbMask"],
    ),
]


@pytest.mark.parametrize("module_path", MODULE_PATHS)
def test_forecasters_namespace_modules_are_importable(module_path):
    assert import_module(module_path) is not None


@pytest.mark.parametrize(
    ("legacy_module_path", "new_module_path", "symbol_names"),
    SYMBOL_PARITY_CASES,
)
def test_forecasters_namespace_reexports_legacy_symbols(
    legacy_module_path, new_module_path, symbol_names
):
    legacy_module = import_module(legacy_module_path)
    new_module = import_module(new_module_path)

    for symbol_name in symbol_names:
        assert getattr(new_module, symbol_name) is getattr(legacy_module, symbol_name)
