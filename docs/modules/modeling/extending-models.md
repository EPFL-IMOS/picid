# Extending Models

1. Choose the right canonical home:
   `picid/model/forecasters/<family>/` for forecasting families or
   `picid/model/estimators/<family>/` for non-forecaster families.
2. Reuse `picid.model.adapters.base` when you need one of the shared adapter
   contracts.
3. Ensure the output contract stays evaluator-compatible.
4. Add model config under `configs/model/`.
5. Add experiment config wiring datasource, task, model, and evaluator.
6. Validate both training and evaluation paths.

See existing guide: [Adding a New Model](../../guides/how_to_add_a_new_model.md)
