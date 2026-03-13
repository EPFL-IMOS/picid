# Add a Model Wrapper

1. Pick wrapper base type.
2. Implement wrapper in `picid/model/wrappers/`.
3. Preserve output contract (`predictions`, `targets`, `loss`, optional `unit_id`).
4. Add config entries and experiment wiring.
5. Validate through full run with evaluator metrics.

See [Modeling Module](../modules/modeling/index.md).
