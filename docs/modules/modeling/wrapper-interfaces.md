# Wrapper Interfaces

Core adapter interfaces live in `picid.model.adapters.base` and re-export the legacy wrapper base classes:

- `AbstractFitPredictWrapper`
- `AbstractFeedForwardWrapper`
- `AbstractFeedForwardTrainingWrapper`

Standard output contract for evaluator compatibility:

- `predictions`
- `targets`
- `loss`
- optional `unit_id`
