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
2. Run generate_snapshot_reference.py (creates reference.json)
3. Run tests: uv run pytest test/pipeline/test_snapshot.py -v
