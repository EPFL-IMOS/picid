# Add a Datasource

1. Implement loader in `picid/data/datasources/`.
2. Ensure `load_data()`, `split_data()`, and `get_data()` are implemented.
3. Return `SplitDatasetContainer` with list-per-unit split values.
4. Add Hydra config under `configs/datasource/`.
5. Test with a small debug experiment.

Related APIs:

- [picid.data.datasources](../reference/api/picid_data_datasources.md)
- [SplitDatasetContainer](../modules/data/data-objects/split-dataset-container.md)
