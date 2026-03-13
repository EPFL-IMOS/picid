# Datasources

Datasources are data-centric loaders in `picid.data.datasources`. They own raw I/O, split policy, and metadata extraction.

## Contract

`get_data()` must return a `SplitDatasetContainer` where each split key contains per-unit lists:

```text
data[key][split] = List[ndarray | ak.Array]
```

Single-unit sources should still return lists (length 1) for consistency.

See full API: [picid.data.datasources](../../reference/api/picid_data_datasources.md).
