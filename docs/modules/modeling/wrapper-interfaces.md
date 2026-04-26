# Wrapper Interfaces

Core adapter interfaces live in `picid.model.adapters.base`.
`picid.model.wrappers.base` currently re-exports the same classes for
compatibility, but new code should import from the canonical adapters module.

- `AbstractFitPredictWrapper`
- `AbstractFeedForwardWrapper`
- `AbstractFeedForwardTrainingWrapper`

Typical ownership:

- Put shared adapter abstractions in `picid.model.adapters.base`.
- Put non-forecaster families in `picid.model.estimators.<family>`.
- Put forecasting families in `picid.model.forecasters.<family>`.

Standard output contract for evaluator compatibility:

- `predictions`
- `targets`
- `loss`
- optional `unit_id`
