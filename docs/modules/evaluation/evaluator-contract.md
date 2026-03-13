# Evaluator Contract

Evaluators consume standardized model outputs:

- `predictions`
- `targets`
- optional `unit_id`

Lifecycle methods:

- `update(model_out)` per batch
- `compute(mode, epoch, step)` per phase
- `reset()` between phases

Contract base: `AbstractEvaluator`.
