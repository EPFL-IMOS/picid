#!/usr/bin/env python3
"""Generate reference metrics for synthetic pipeline snapshot tests.

Output: test/fixtures/snapshot/reference/reference.json (merged from reference/<task>/<model>.json)

Supports:
  - test/pipeline/test_pipeline_synthetic_snapshots.py (prognostics + test_reproduce)
  - test/pipeline/test_pipeline_synthetic_snapshots_diagnostics.py
  - test/pipeline/test_pipeline_synthetic_snapshots_prognostics_ragged.py
  - test/pipeline/test_pipeline_synthetic_snapshots_anomaly.py

Uses only synthetic data (no downloads). Run once after pipeline changes you
intend to keep. Commit the output. Metrics rounded to 4 decimals.

Usage: uv run python test/scripts/snapshot/generate_snapshot_reference.py
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REF_DIR = PROJECT_ROOT / "test" / "fixtures" / "snapshot" / "reference"

# (task_type, experiment_path, model_key)
# Note: forecasting/anomaly_detection need dataset format adjustments for single-source
EXPERIMENTS = [
    ("prognostics", "snapshot/prognostics", "linear_regression"),
    ("prognostics", "snapshot/prognostics", "exponential_regression"),
    ("prognostics", "snapshot/prognostics", "lstm"),
    ("prognostics", "snapshot/prognostics", "patchtst"),
    ("prognostics", "snapshot/prognostics", "crossformer"),
    ("prognostics", "snapshot/prognostics", "mlp"),
    ("prognostics", "snapshot/prognostics", "cnn_1d"),
    ("prognostics", "snapshot/prognostics", "tide"),
    ("prognostics", "snapshot/prognostics", "timeseries_transformer"),
    ("prognostics", "snapshot/prognostics", "stf"),
    ("prognostics", "snapshot/prognostics", "xgboost_fit_predict"),
    ("diagnostics", "snapshot/diagnostics", "linear_classifier"),
    ("diagnostics", "snapshot/diagnostics", "mlp"),
    ("diagnostics", "snapshot/diagnostics", "lstm"),
    ("diagnostics", "snapshot/diagnostics", "patchtst"),
    ("diagnostics", "snapshot/diagnostics", "cnn_1d"),
    ("diagnostics", "snapshot/diagnostics", "crossformer"),
    ("diagnostics", "snapshot/diagnostics", "tide"),
    ("diagnostics", "snapshot/diagnostics", "timeseries_transformer"),
    ("diagnostics", "snapshot/diagnostics", "stf"),
    ("diagnostics", "snapshot/diagnostics", "xgboost_fit_predict"),
    ("prognostics_ragged", "snapshot/ragged_prognostics", "linear_regression"),
    ("prognostics_ragged", "snapshot/ragged_prognostics", "exponential_regression"),
    ("prognostics_ragged", "snapshot/ragged_prognostics", "lstm"),
    ("prognostics_ragged", "snapshot/ragged_prognostics", "patchtst"),
    ("prognostics_ragged", "snapshot/ragged_prognostics", "crossformer"),
    ("anomaly_detection", "snapshot/anomaly", "isolation_forest"),
]

PRECISION = 4  # Round to 4 decimals


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate reference metrics for synthetic pipeline snapshot tests."
    )
    ap.add_argument(
        "--task",
        choices=[
            "prognostics",
            "diagnostics",
            "prognostics_ragged",
            "anomaly_detection",
        ],
        default=None,
        help="Run only experiments for this task type",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Run only experiments with this model (e.g. lstm, linear_classifier)",
    )
    ap.add_argument(
        "--experiment",
        default=None,
        help="Run exactly one experiment: task/model (e.g. diagnostics/linear_classifier)",
    )
    return ap.parse_args()


def filter_experiments(
    experiments: list, task: str | None, model: str | None, experiment: str | None
) -> list:
    """Filter EXPERIMENTS by task, model, or single experiment."""
    if experiment:
        parts = experiment.split("/")
        if len(parts) != 2:
            raise ValueError(
                "--experiment must be task/model (e.g. diagnostics/linear_classifier)"
            )
        t, m = parts
        matched = [
            (tt, exp, mod) for (tt, exp, mod) in experiments if tt == t and mod == m
        ]
        if not matched:
            valid = [f"{tt}/{mod}" for (tt, _, mod) in experiments]
            raise ValueError(f"Unknown experiment '{experiment}'. Valid: {valid}")
        return matched
    out = experiments
    if task:
        out = [(t, e, m) for (t, e, m) in out if t == task]
    if model:
        out = [(t, e, m) for (t, e, m) in out if m == model]
    return out


def round_metrics(obj):
    """Recursively round float values to PRECISION decimals."""
    if isinstance(obj, dict):
        return {k: round_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_metrics(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, PRECISION)
    return obj


def prefix_test_metrics(obj: dict) -> dict:
    """Ensure top-level metric keys carry a 'test/' prefix.

    interface.train() returns keys like 'test/loss'; eval_details/best_epoch/test/metrics.json
    may omit the prefix. Normalise so reference files always match interface.train() output.
    Idempotent: keys that already start with 'test/' are left unchanged.
    """
    return {(k if k.startswith("test/") else f"test/{k}"): v for k, v in obj.items()}


def merge_reference_from_per_experiment_files(ref_dir: Path) -> dict:
    """Build ref dict from all per-experiment JSON files. Source of truth: reference/<task>/<model>.json"""
    ref = {}
    for task_type, _, model in EXPERIMENTS:
        path = ref_dir / task_type / f"{model}.json"
        if path.exists():
            raw = json.loads(path.read_text())
            ref[f"{task_type}/{model}"] = round_metrics(raw)
    return ref


def _metrics_path_from_run_dir(run_dir: Path) -> Path:
    """Return the best-epoch test metrics path for a subprocess pipeline run."""
    return run_dir / "eval_details" / "best_epoch" / "test" / "metrics.json"


def main():
    args = parse_args()
    to_run = filter_experiments(EXPERIMENTS, args.task, args.model, args.experiment)
    if not to_run:
        print("No experiments match the given filters.", file=sys.stderr)
        sys.exit(1)

    REF_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["PYTHONPATH"] = str(
        PROJECT_ROOT
    )  # Ensure test.data importable (not stdlib test)
    env["HYDRA_FULL_ERROR"] = "1"

    root = str(PROJECT_ROOT).replace("\\", "/")
    searchpath = f"file://{root}/test/configs,file://{root}/configs"

    print(f"Running {len(to_run)} experiments: {[f'{t}/{m}' for (t, _, m) in to_run]}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for task_type, exp_base, model in to_run:
            key = f"{task_type}/{model}"
            (REF_DIR / task_type).mkdir(parents=True, exist_ok=True)
            out = REF_DIR / task_type / f"{model}.json"
            run_dir = tmp_root / task_type / model
            run_dir.mkdir(parents=True, exist_ok=True)

            # Fit-predict models use max_epochs=1: no gradient loop, each epoch just
            # calls fit()/predict() once; running more epochs is wasteful and may
            # interact poorly with the one_epoch trainer config.
            _fit_predict_models = {"isolation_forest", "xgboost_fit_predict"}
            max_epochs = "1" if model in _fit_predict_models else "10"
            cmd = [
                sys.executable,
                "picid/run.py",
                f"experiment={exp_base}/{model}",
                "paths=default",
                "debug=nox_testing_fit_predict",
                f"trainer.max_epochs={max_epochs}",
                "num_threads=1",
                "seed=42",
                f"hydra.run.dir={run_dir}",
                f"hydra.searchpath=[{searchpath}]",
            ]
            print(f"Running {key}...", flush=True)
            r = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
            if r.returncode != 0:
                print(f"Failed: {key}", file=sys.stderr)
                sys.exit(r.returncode)

            metrics_path = _metrics_path_from_run_dir(run_dir)
            if not metrics_path.exists():
                print(f"Metrics not captured for {key}: {metrics_path}", file=sys.stderr)
                sys.exit(1)
            raw = json.loads(metrics_path.read_text())
            out.write_text(json.dumps(prefix_test_metrics(raw), indent=2))

    ref = merge_reference_from_per_experiment_files(REF_DIR)
    (REF_DIR / "reference.json").write_text(json.dumps(ref, indent=2))
    print("Reference written to", REF_DIR / "reference.json")


if __name__ == "__main__":
    main()
