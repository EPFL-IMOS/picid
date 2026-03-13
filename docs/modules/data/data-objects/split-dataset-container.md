# SplitDatasetContainer

`SplitDatasetContainer` (in `picid.data.data_objects.data`) is the canonical container for split-aware PHM workflows.

## Why it is the standard

- **Type-safe transport** across the preprocessing pipeline.
- **Consistent split shape** for train/val/test.
- **Metadata-compatible** with manifest tracking.
- **View conversion helpers** for downstream consumers.

## Expected structure

```text
container.<field>.<split> -> List[ndarray | ak.Array]
```

Typical fields include `features`, `target`, and optional `unit_id`.

## Validation semantics

For each split, unit counts must match across required fields (for example `features.train` and `target.train`).

If lengths diverge, validation fails early to prevent silent misalignment.

API: [picid.data.data_objects](../../../reference/api/picid_data_data_objects.md)
