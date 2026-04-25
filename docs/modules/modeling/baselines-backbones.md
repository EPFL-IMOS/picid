# Baselines and Backbones

Backbone implementations now live under the canonical `picid/model/forecasters/`
namespace (LSTM, PatchTST, Crossformer, TiDE, etc.), with `picid.baselines`
retained temporarily as a compatibility layer during migration.

Wrappers adapt these heterogeneous models to a unified pipeline contract.

API:

- [picid.model.forecasters](../../reference/api/picid_model_forecasters.md)
