# Baselines and Backbones

Backbone implementations now live under the canonical `picid/model/forecasters/`
namespace (LSTM, PatchTST, Crossformer, TiDE, etc.), with `picid.baselines`
retained only as a legacy compatibility layer during migration.

Use `picid.model.forecasters` for all new code, configs, and extension work.
`picid.baselines` should be treated as deprecated and read-only.

Wrappers adapt these heterogeneous models to a unified pipeline contract.

API:

- [picid.model.forecasters](../../reference/api/picid_model_forecasters.md)
- [picid.baselines (legacy)](../../reference/api/picid_baselines.md)
