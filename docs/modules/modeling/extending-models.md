# Extending Models

1. Implement wrapper in `picid/model/wrappers/`.
2. Ensure output contract is evaluator-compatible.
3. Add model config under `configs/model*` groups.
4. Add experiment config wiring datasource/task/model/evaluator.
5. Validate train + test path.

See existing guide: [Adding a New Model](../../guides/how_to_add_a_new_model.md)
