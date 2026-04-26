# Modeling Module

Modeling now uses a family-first layout:

- `picid.model.forecasters` for forecasting families
- `picid.model.estimators` for non-forecaster families
- `picid.model.adapters.base` for shared adapter interfaces

Pages:

- [Wrapper Interfaces](wrapper-interfaces.md)
- [Fit-Predict vs Feed-Forward](fitpredict-vs-feedforward.md)
- [Baselines and Backbones](baselines-backbones.md)
- [Model Capabilities](model-capabilities.md)
- [Loss/Optimizer/LR Scheduler](loss-optimizer-lr-scheduler.md)
- [Extending Models](extending-models.md)

API:

- [picid.model](../../reference/api/picid_model.md)
- [picid.model.adapters](../../reference/api/picid_model_adapters.md)
- [picid.model.estimators](../../reference/api/picid_model_estimators.md)
- [picid.model.forecasters](../../reference/api/picid_model_forecasters.md)
