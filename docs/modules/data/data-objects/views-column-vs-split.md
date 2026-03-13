# Views: Column-Oriented vs Split-Keyed

`SplitDatasetContainer` supports two common views.

## Column-oriented view

Natural object access:

```python
container.features.train
container.target.val
```

Useful for internal pipeline operations.

## Split-keyed view

`to_split_dict()` returns:

```text
{split: {field: list_of_unit_arrays}}
```

Useful for Hydra-instantiated datasets and datamodules.

## Per-split grouped view

`group_by_split()` provides split-grouped objects for inspection/debugging workflows.
