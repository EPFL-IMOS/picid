# Baselines and Backbones

Backbone implementations live under the canonical
`picid/model/forecasters/` namespace (LSTM, PatchTST, Crossformer, TiDE, etc.).

Use `picid.model.forecasters` for all new code, configs, and extension work.

Wrappers adapt these heterogeneous models to a unified pipeline contract.

API:

- [picid.model.forecasters](../../reference/api/picid_model_forecasters.md)
