# Wrapper Interfaces

Core wrapper interfaces live in `picid.model.wrappers.base`:

- `AbstractFitPredictWrapper`
- `AbstractFeedForwardWrapper`
- `AbstractFeedForwardTrainingWrapper`

Standard output contract for evaluator compatibility:

- `predictions`
- `targets`
- `loss`
- optional `unit_id`
