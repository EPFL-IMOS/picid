#!/usr/bin/env python3
"""Generate reference metrics for pipeline snapshot tests.

Run once after pipeline changes you intend to keep. Commit the output.
Metrics are rounded to 4 decimals for precise comparison.
Usage: uv run python scripts/snapshot/generate_snapshot_reference.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REF_DIR = PROJECT_ROOT / "test" / "fixtures" / "snapshot"

# (task_type, experiment_path, model_key)
# Note: forecasting/anomaly_detection need dataset format adjustments for single-source
EXPERIMENTS = [
    ("prognostics", "snapshot/prognostics", "linear_regression"),
    ("prognostics", "snapshot/prognostics", "exponential_regression"),
    ("prognostics", "snapshot/prognostics", "lstm"),
    ("prognostics", "snapshot/prognostics", "patchtst"),
    ("prognostics", "snapshot/prognostics", "crossformer"),
    ("diagnostics", "snapshot/diagnostics", "linear_classifier"),
]

PRECISION = 4  # Round to 4 decimals


def round_metrics(obj):
    """Recursively round float values to PRECISION decimals."""
    if isinstance(obj, dict):
        return {k: round_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_metrics(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, PRECISION)
    return obj


def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["HYDRA_FULL_ERROR"] = "1"

    root = str(PROJECT_ROOT).replace("\\", "/")
    searchpath = f"file://{root}/test/configs,file://{root}/configs"

    ref = {}
    for task_type, exp_base, model in EXPERIMENTS:
        key = f"{task_type}/{model}"
        out = REF_DIR / f"{task_type}_{model}.json"
        env["PIPELINE_REGRESSION_OUTPUT"] = str(out)

        cmd = [
            sys.executable,
            "picid/run.py",
            f"experiment={exp_base}/{model}",
            "paths=default",
            "debug=nox_testing_fit_predict",
            "trainer.max_epochs=10",
            "num_threads=1",
            "seed=42",
            f"hydra.searchpath=[{searchpath}]",
        ]
        print(f"Running {key}...", flush=True)
        r = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
        if r.returncode != 0:
            print(f"Failed: {key}", file=sys.stderr)
            sys.exit(r.returncode)
        raw = json.loads(out.read_text())
        ref[key] = round_metrics(raw)

    (REF_DIR / "reference.json").write_text(json.dumps(ref, indent=2))
    print("Reference written to", REF_DIR / "reference.json")


if __name__ == "__main__":
    main()
