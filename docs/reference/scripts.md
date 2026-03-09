# Scripts reference

Overview of scripts in the `scripts/` directory. Each subfolder has its own documentation — see the links below for full details.

**Root overview:** [scripts/README.txt](https://github.com/epfl-imos/TabPHM/blob/main/scripts/README.txt)

---

## Environment and setup

### env_setup.sh

**What it does:** Installs project dependencies with `uv sync` and `uv pip install -e ".[dev]"`.

**How to use:** `./scripts/env_setup.sh` or source it in your shell.

**When to use:** Initial project setup or after pulling dependency changes.

---

### datasets_setup.sh

**What it does:** Downloads datasets required for experiments. Pre-downloads PHME20 (via phmd) for pipeline tests and phme20 experiments. Contains commented examples for other datasets (railway, N-CMAPSS, etc.).

**How to use:** `./scripts/datasets_setup.sh`

**When to use:** Before running experiments that need PHME20 or other datasets. CI runs this before tests.

---

## Reproducibility

Scripts in `scripts/reproducibility/` support reproducing runs and debugging config differences. See [scripts/reproducibility/README.md](https://github.com/epfl-imos/TabPHM/blob/main/scripts/reproducibility/README.md) for full documentation.

### reproducibility/compare_configs.py

**What it does:** Compares two resolved config YAML files and prints key-level differences. Use when metrics differ between runs and you need to find what changed.

**How to use:**
```bash
uv run python scripts/reproducibility/compare_configs.py <config_a.yaml> <config_b.yaml>
```

**Output:** `~` different values, `-` only in first, `+` only in second.

**When to use:** Debugging metric differences, auditing config changes, verifying two runs used the same config.

---

### reproducibility/reproduce_from_run.py

**What it does:** Loads the resolved config from a previous run (`run_dir/config_resolved.yaml`) and re-runs the full pipeline with output to `run_dir/reproduce_<timestamp>`. Use for exact reproduction from a run's saved config (e.g. version comparison).

**How to use:**
```bash
export PROJECT_ROOT=/path/to/TabPHM
uv run python scripts/reproducibility/reproduce_from_run.py <run_output_dir>
```

**When to use:** Reproducing a run on another machine, comparing library versions, debugging with the exact config that produced a given output.

**See:** [Reproducibility guide](../guides/reproducibility.md).

---

## Snapshot tests

Scripts in `scripts/snapshot/` support the pipeline snapshot test suite. See [scripts/snapshot/README.txt](https://github.com/epfl-imos/TabPHM/blob/main/scripts/snapshot/README.txt) for details.

### snapshot/generate_snapshot_fixtures.py

**What it does:** Creates synthetic `.npz` data files for prognostics, diagnostics, anomaly detection, and forecasting. Uses seed 42. Output: `test/fixtures/snapshot/*.npz`.

**How to use:** `uv run python scripts/snapshot/generate_snapshot_fixtures.py`

**When to use:** Initial snapshot setup, or when changing expected data format.

---

### snapshot/generate_snapshot_ragged_fixtures.py

**What it does:** Creates synthetic ragged `.pkl` data for `prognostics_ragged` snapshot tests. Output: `test/fixtures/snapshot/ragged_prognostics.pkl`.

**How to use:** `uv run python scripts/snapshot/generate_snapshot_ragged_fixtures.py`

**When to use:** Initial ragged snapshot setup, or when fixtures are missing.

---

### snapshot/generate_pipeline_snapshots.py

**What it does:** Generates real-data pipeline snapshots for phme20 (loaded + transformed slices). Used by `test_pipeline_real_data.py`. Requires datasets.

**How to use:**
```bash
uv run python scripts/snapshot/generate_pipeline_snapshots.py
uv run python scripts/snapshot/generate_pipeline_snapshots.py --data-dir "$(pwd)/datasets"
```

**When to use:** Setting up real-data pipeline tests, after changing phme20 or transforms.

---

### snapshot/generate_snapshot_reference.py

**What it does:** Runs each snapshot experiment (linear_regression, lstm, patchtst, crossformer, etc.) and captures test metrics. Writes `test/fixtures/snapshot/reference.json`. The test suite compares current runs against this reference.

**How to use:** `uv run python scripts/snapshot/generate_snapshot_reference.py`

**When to use:** After pipeline changes you intend to keep. Commit the new `reference.json` only when the new behavior is correct.

---

## Publish (leaderboard submission)

Scripts in `scripts/publish/` package run results and upload them for leaderboard submission. See [scripts/publish/README.md](https://github.com/epfl-imos/TabPHM/blob/main/scripts/publish/README.md) for full documentation.

**Key scripts:** `push_run_to_gdrive.py`, `build_submission_message.py`

---

## Experiment Generator

Scripts in `scripts/experiment_generator/` generate the config map for the [Experiment Generator](../guides/experiment_generator.md) UI. See [scripts/experiment_generator/README.md](https://github.com/epfl-imos/TabPHM/blob/main/scripts/experiment_generator/README.md) for full documentation.

**Key script:** `generate_config_map.py` — crawls `configs/` and outputs `config_map.json`.

---

## Other scripts

### base/

Logging utilities sourced by `reproduce/` scripts.

### reproduce/

Generated by picid_report. Contains per-experiment reproduce scripts.

### paper_run_all.sh

Orchestrates paper experiments.

### paper_diagnostics/ and paper_prognostics/

Paper experiment scripts for diagnostics and prognostics.

---

[← Back to reference index](index.md)
