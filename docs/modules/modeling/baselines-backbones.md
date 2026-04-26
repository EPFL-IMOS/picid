# Baselines and Backbones

Temporal backbone implementations live under
`picid/model/forecasters/` (LSTM, PatchTST, Crossformer, TiDE, etc.).

Non-forecaster families now live under
`picid/model/estimators/` with one directory per family
(`mlp`, `cnn1d`, `window_average`, `statistical`, and fit-predict families).

Use `picid.model.forecasters` and `picid.model.estimators` for all new code,
configs, and extension work.

Wrappers adapt these heterogeneous models to a unified pipeline contract.

API:

- [picid.model.estimators](../../reference/api/picid_model_estimators.md)
- [picid.model.forecasters](../../reference/api/picid_model_forecasters.md)
