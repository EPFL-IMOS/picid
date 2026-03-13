# End-to-End Flow

1. Instantiate datasource from config.
2. Run preprocessor (direct or cached path).
3. Build task-specific datasets.
4. Build datamodule loaders.
5. Instantiate model wrapper and Lightning module.
6. Run trainer (`fit` and `test` or direct `test`).
7. Compute evaluator metrics and persist run artifacts.

See [Run Lifecycle](../modules/orchestration/run-lifecycle.md) for details.
