Pipeline Snapshot Scripts
=========================

This folder contains scripts for the pipeline snapshot test suite. Snapshot tests
run the full pipeline on synthetic data and compare metrics to a committed
reference to detect unintended behavior changes (e.g. transforms, evaluators).

Fixture layout (test/fixtures/snapshot/):
  data/                    — Input fixtures (.npz, .pkl) from generate_snapshot_fixtures, generate_snapshot_ragged_fixtures
  reference/<task>/<model>.json — Per-experiment metrics (source of truth)
  reference/reference.json — Merged view for tests (derived from per-experiment files)


generate_snapshot_fixtures.py
-----------------------------

What it does:
  Creates synthetic .npz data files for prognostics, diagnostics, anomaly
  detection, and forecasting. Uses a fixed seed (42) for reproducibility.
  Output: test/fixtures/snapshot/data/*.npz

How to use:
  uv run python test/scripts/snapshot/generate_snapshot_fixtures.py

When to use:
  - Once when setting up the snapshot test suite
  - If you change the expected data format (keys, shapes) for snapshot tests
  - If fixtures are missing or corrupted


generate_snapshot_ragged_fixtures.py
------------------------------------

What it does:
  Creates synthetic ragged .pkl data for prognostics_ragged snapshot tests.
  Uses a fixed seed (42) for reproducibility across machines.
  Output: test/fixtures/snapshot/data/ragged_prognostics.pkl

How to use:
  uv run python test/scripts/snapshot/generate_snapshot_ragged_fixtures.py

When to use:
  - Once when setting up the ragged snapshot tests
  - If ragged fixtures are missing or corrupted


generate_pipeline_snapshots.py
------------------------------

What it does:
  Generates real-data pipeline snapshots for phme20 (loaded + transformed slices).
  Used by test_pipeline_phme20_snapshots.py. Requires datasets and --data-dir.

How to use:
  uv run python test/scripts/snapshot/generate_pipeline_snapshots.py
  uv run python test/scripts/snapshot/generate_pipeline_snapshots.py --data-dir "$(pwd)/datasets"

When to use:
  - When setting up real-data pipeline tests (requires_snapshots)
  - After changing phme20 or transforms that affect output


generate_snapshot_reference.py
------------------------------

What it does:
  Runs each snapshot experiment (prognostics, diagnostics, prognostics_ragged,
  anomaly_detection) and captures test metrics.
  Writes per-experiment files to reference/<task>/<model>.json (source of truth),
  then merges to reference/reference.json. The test suite compares against this.

How to use:
  uv run python test/scripts/snapshot/generate_snapshot_reference.py

Incremental regeneration (merge into existing reference):
  uv run python test/scripts/snapshot/generate_snapshot_reference.py --task diagnostics
  uv run python test/scripts/snapshot/generate_snapshot_reference.py --model lstm
  uv run python test/scripts/snapshot/generate_snapshot_reference.py --experiment diagnostics/linear_classifier

When to use:
  - After pipeline changes you intend to keep (transforms, evaluators, models)
  - After adding or removing snapshot experiments
  - After changing trainer.max_epochs or other snapshot-affecting config
  - When reference.json is missing (tests will skip with a hint)

When to use (incremental):
  - After changing diagnostics configs → --task diagnostics
  - After adding a new model to a task → --experiment <task>/<new_model>
  - After updating one model → --experiment <task>/<model>

Note: Regenerating the reference updates the baseline. Commit the new
reference/ files only when the new behavior is correct.


Order of operations
-------------------

1. Run generate_snapshot_fixtures.py (creates data/*.npz)
2. Run generate_snapshot_ragged_fixtures.py (creates data/ragged_prognostics.pkl)
3. Run generate_snapshot_reference.py (creates reference/<task>/<model>.json + reference.json)
4. Run tests:
   uv run pytest test/pipeline/test_pipeline_synthetic_snapshots*.py -v
   or: uv run nox -s pipeline_snapshot

For real-data pipeline tests (phme20):
  - Run generate_pipeline_snapshots.py --data-dir "$(pwd)/datasets"
  - Then: uv run pytest test/pipeline/test_pipeline_phme20_snapshots.py -v
