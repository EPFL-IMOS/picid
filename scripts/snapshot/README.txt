Pipeline Snapshot Scripts
=========================

This folder contains scripts for the pipeline snapshot test suite. Snapshot tests
run the full pipeline on synthetic data and compare metrics to a committed
reference to detect unintended behavior changes (e.g. transforms, evaluators).


generate_snapshot_fixtures.py
-----------------------------

What it does:
  Creates synthetic .npz data files for prognostics, diagnostics, anomaly
  detection, and forecasting. Uses a fixed seed (42) for reproducibility.
  Output: test/fixtures/snapshot/*.npz

How to use:
  uv run python scripts/snapshot/generate_snapshot_fixtures.py

When to use:
  - Once when setting up the snapshot test suite
  - If you change the expected data format (keys, shapes) for snapshot tests
  - If fixtures are missing or corrupted


generate_snapshot_ragged_fixtures.py
------------------------------------

What it does:
  Creates synthetic ragged .pkl data for prognostics_ragged snapshot tests.
  Uses a fixed seed (42) for reproducibility across machines.
  Output: test/fixtures/snapshot/ragged_prognostics.pkl

How to use:
  uv run python scripts/snapshot/generate_snapshot_ragged_fixtures.py

When to use:
  - Once when setting up the ragged snapshot tests
  - If ragged fixtures are missing or corrupted


generate_pipeline_snapshots.py
------------------------------

What it does:
  Generates real-data pipeline snapshots for phme20 (loaded + transformed slices).
  Used by test_pipeline_real_data.py. Requires datasets and --data-dir.

How to use:
  uv run python scripts/snapshot/generate_pipeline_snapshots.py
  uv run python scripts/snapshot/generate_pipeline_snapshots.py --data-dir "$(pwd)/datasets"

When to use:
  - When setting up real-data pipeline tests (requires_snapshots)
  - After changing phme20 or transforms that affect output


generate_snapshot_reference.py
------------------------------

What it does:
  Runs each snapshot experiment (linear_regression, exponential_regression,
  lstm, patchtst, crossformer, linear_classifier) and captures test metrics.
  Rounds to 4 decimals and writes test/fixtures/snapshot/reference.json.
  The test suite compares current runs against this reference.

How to use:
  uv run python scripts/snapshot/generate_snapshot_reference.py

When to use:
  - After pipeline changes you intend to keep (transforms, evaluators, models)
  - After adding or removing snapshot experiments
  - After changing trainer.max_epochs or other snapshot-affecting config
  - When reference.json is missing (tests will skip with a hint)

Note: Regenerating the reference updates the baseline. Commit the new
reference.json only when the new behavior is correct.


Order of operations
-------------------

1. Run generate_snapshot_fixtures.py (creates .npz files)
2. Run generate_snapshot_ragged_fixtures.py (creates ragged_prognostics.pkl)
3. Run generate_snapshot_reference.py (creates reference.json)
4. Run tests: uv run pytest test/pipeline/test_snapshot.py -v

For real-data pipeline tests (phme20):
  - Run generate_pipeline_snapshots.py --data-dir "$(pwd)/datasets"
  - Then: uv run pytest test/pipeline/test_pipeline_real_data.py -v
