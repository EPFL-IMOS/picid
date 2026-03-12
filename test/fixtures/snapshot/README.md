# Snapshot Fixtures

## Layout

- **data/** — Input fixtures (source data for pipeline). From `generate_snapshot_fixtures.py` and `generate_snapshot_ragged_fixtures.py`.
- **reference/** — Reference metrics. Per-experiment files (`reference/<task>/<model>.json`) are source of truth. `reference.json` is derived by merging them; tests use it.
