# Add a Dataset

1. Implement dataset class under `picid/data/datasets/`.
2. Define output batch contract and key names.
3. Wire into dataset config (`configs/dataset/` or related model config groups).
4. Validate compatibility with `BaseDataModule`.
5. Add tests for edge cases (empty units, shape mismatch).

Related docs:

- [Task Formats](../modules/data/datasets/task-formats-rul-forecast-fitpredict.md)
- [Batch Contracts](../modules/data/datamodules/batch-contracts.md)
