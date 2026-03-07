# Reproducibility scripts

Scripts for reproducing experiments and comparing runs.

---

## reproduce_from_run.py

**What it does:** Loads the resolved config from a previous run (`run_dir/config_resolved.yaml`) and re-runs the full pipeline with output to `run_dir/reproduce_<timestamp>`. Use when you want to reproduce exactly from a run's saved config (e.g. for version comparison) rather than from repo overrides.

**How to use:**
```bash
export PROJECT_ROOT=/path/to/TabPHM
uv run python scripts/reproducibility/reproduce_from_run.py <run_output_dir>
```

Example:
```bash
uv run python scripts/reproducibility/reproduce_from_run.py $PROJECT_ROOT/artifacts/debug/runs/.../2026-03-06_22-42-27
```

**When to use:**
- Reproducing a run on another machine (copy run dir, then run this)
- Comparing library versions (run with version A, upgrade, run this to compare metrics)
- Debugging: re-run with the exact config that produced a given output

**Requirements:** `PROJECT_ROOT` env var, `config_resolved.yaml` in the run dir.

**Output:** `run_dir/reproduce_<timestamp>/` with checkpoints, logs, and a new `config_resolved.yaml`.
