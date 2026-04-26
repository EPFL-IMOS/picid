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


@pytest.mark.parametrize("module_path", MODULE_PATHS)
def test_forecasters_namespace_modules_are_importable(module_path):
    assert import_module(module_path) is not None
