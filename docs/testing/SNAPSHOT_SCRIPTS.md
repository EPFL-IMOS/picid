# Snapshot Test Scripts

Scripts in `test/scripts/snapshot/` generate fixtures and reference metrics for the pipeline snapshot test suite. They are colocated with tests because they are used solely for testing.

## Fixture layout

- **data/** — Input fixtures (`.npz`, `.pkl`) from `generate_snapshot_fixtures.py` and `generate_snapshot_ragged_fixtures.py`
- **reference/** — Per-experiment metrics (`reference/<task>/<model>.json`), merged to `reference/reference.json`

## Scripts

### generate_snapshot_fixtures.py

**What it does:** Creates synthetic `.npz` data for prognostics, diagnostics, anomaly detection, and forecasting. Uses seed 42. Output: `test/fixtures/snapshot/data/*.npz`.

**How to use:** `uv run python test/scripts/snapshot/generate_snapshot_fixtures.py`

**When to use:** Initial snapshot setup, or when changing expected data format.

**Tests that depend on it:**
- `test/pipeline/test_pipeline_synthetic_snapshots_prognostics.py`
- `test/pipeline/test_pipeline_synthetic_snapshots_diagnostics.py`
- `test/pipeline/test_pipeline_synthetic_snapshots_anomaly.py`

---

### generate_snapshot_ragged_fixtures.py

**What it does:** Creates synthetic ragged `.pkl` data for prognostics_ragged. Output: `test/fixtures/snapshot/data/ragged_prognostics.pkl`.

**How to use:** `uv run python test/scripts/snapshot/generate_snapshot_ragged_fixtures.py`

**When to use:** Initial ragged snapshot setup, or when fixtures are missing.

**Tests that depend on it:**
- `test/data/datasources/test_synthetic_ragged_loader.py`
- `test/pipeline/test_pipeline_synthetic_snapshots_prognostics_ragged.py`

---

### generate_pipeline_snapshots.py

**What it does:** Generates real-data pipeline snapshots for phme20 (loaded + transformed slices). Requires datasets and `--data-dir`.

**How to use:**
```bash
uv run python test/scripts/snapshot/generate_pipeline_snapshots.py
uv run python test/scripts/snapshot/generate_pipeline_snapshots.py --data-dir "$(pwd)/datasets"
```

**When to use:** Setting up real-data pipeline tests, after changing phme20 or transforms.

**Tests that depend on it:**
- `test/pipeline/test_pipeline_phme20_snapshots.py`

---

### generate_snapshot_reference.py

**What it does:** Runs each snapshot experiment and captures test metrics. Writes per-experiment files to `reference/<task>/<model>.json`, then merges to `reference/reference.json`.

**How to use:** `uv run python test/scripts/snapshot/generate_snapshot_reference.py`

**Incremental:** `--task diagnostics`, `--model lstm`, `--experiment diagnostics/linear_classifier`

**When to use:** After pipeline changes you intend to keep. Commit new `reference/` files only when behavior is correct.

**Tests that depend on it:**
- `test/pipeline/test_pipeline_synthetic_snapshots_prognostics.py`
- `test/pipeline/test_pipeline_synthetic_snapshots_diagnostics.py`
- `test/pipeline/test_pipeline_synthetic_snapshots_prognostics_ragged.py`
- `test/pipeline/test_pipeline_synthetic_snapshots_anomaly.py`

---

## Order of operations

1. `generate_snapshot_fixtures.py` → data/*.npz
2. `generate_snapshot_ragged_fixtures.py` → data/ragged_prognostics.pkl
3. `generate_snapshot_reference.py` → reference/
4. Run: `nox -f test/pipeline/noxfile.py -s pipeline_snapshot`

For phme20 real-data tests: run `generate_pipeline_snapshots.py --data-dir <datasets>`, then `pytest test/pipeline/test_pipeline_phme20_snapshots.py -v`.

---

## Re-establishing snapshots and references

When pipeline/data format changes have been made (transforms, evaluators, models, datasource output, expected fixture shapes), you must regenerate snapshots and references so tests compare against the new baseline. Run the scripts in this order:

### Synthetic data flow (prognostics, diagnostics, anomaly, prognostics_ragged)

```bash
# 1. Regenerate synthetic data fixtures (if data format changed)
uv run python test/scripts/snapshot/generate_snapshot_fixtures.py
uv run python test/scripts/snapshot/generate_snapshot_ragged_fixtures.py

# 2. Regenerate reference metrics (run pipeline experiments, capture output)
uv run python test/scripts/snapshot/generate_snapshot_reference.py
```

Then run `nox -f test/pipeline/noxfile.py -s pipeline_snapshot` (see Order of operations above).

**Commit:** `test/fixtures/snapshot/data/*` and `test/fixtures/snapshot/reference/*` only when the new behavior is correct.

### Real-data flow (phme20)

```bash
# Regenerate phme20 pipeline snapshots (loaded + transformed slices)
uv run python test/scripts/snapshot/generate_pipeline_snapshots.py --data-dir "$(pwd)/datasets"
```

Then run `pytest test/pipeline/test_pipeline_phme20_snapshots.py -v` (see Order of operations above).

**Commit:** `test/data/fixtures/pipeline_snapshots/phme20/*` only when the new behavior is correct.

### When to run which flows

| Change type | Synthetic flow | Real-data flow |
|-------------|----------------|----------------|
| Data format (keys, shapes) | Yes (step 1) | No |
| Transforms, evaluators, models | Yes (step 2) | Yes |
| Datasource config (phme20) | No | Yes |
| Initial setup | Yes (all) | Yes (if using phme20 tests) |
