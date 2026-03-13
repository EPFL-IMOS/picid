#!/usr/bin/env python3
"""Tutorial: Run and reproduce experiments.

This script demonstrates:
1. Running an experiment (Option A: from repo configs)
2. Reproducing from a run's saved config (Option B: reproduce_from_run)

Run from project root:
    uv run python tutorials/workflow/reproduce_experiment.py

Requires: PROJECT_ROOT env var, datasets (or use snapshot experiment).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent)
)

print("=" * 60)
print("Tutorial: Run and Reproduce Experiments")
print("=" * 60)

# Step 1: Run an experiment (Option A)
print("\n1. Running experiment (Option A - from repo configs)...")
env = os.environ.copy()
env["PROJECT_ROOT"] = str(PROJECT_ROOT)
env["HYDRA_FULL_ERROR"] = "1"
root = str(PROJECT_ROOT).replace("\\", "/")
searchpath = f"file://{root}/test/configs,file://{root}/configs"

cmd = [
    sys.executable,
    "picid/run.py",
    "experiment=snapshot/prognostics/linear_regression",
    "paths=default",
    "debug=nox_testing_fit_predict",
    "trainer.max_epochs=2",
    "num_threads=1",
    "seed=42",
    "hydra.run.dir=artifacts/tutorial_reproduce",
    f"hydra.searchpath=[{searchpath}]",
]
r = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
if r.returncode != 0:
    print("Experiment failed. Ensure PROJECT_ROOT is set and datasets exist.")
    sys.exit(1)

run_dir = PROJECT_ROOT / "artifacts" / "tutorial_reproduce"
print(f"   Output: {run_dir}")

# Step 2: Reproduce from run config (Option B)
print("\n2. Reproducing from run config (Option B - reproduce_from_run)...")
r = subprocess.run(
    [sys.executable, "scripts/reproducibility/reproduce_from_run.py", str(run_dir)],
    env=env,
    cwd=PROJECT_ROOT,
)
if r.returncode != 0:
    print("Reproduce failed.")
    sys.exit(1)

repro_dir = next(run_dir.glob("reproduce_*"), None)
if repro_dir:
    print(f"   Reproduce output: {repro_dir}")

print("\n3. Check REPRODUCE.md in the run dir for manual instructions.")
print("   - Option A: uv run python picid/run.py <overrides>")
print(
    "   - Option B: uv run python scripts/reproducibility/reproduce_from_run.py <run_dir>"
)
print("\nDone.")
