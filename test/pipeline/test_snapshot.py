"""Pipeline snapshot: compare current run to reference metrics.

Fails if pipeline behavior changes unintentionally (e.g. transform, evaluator).
Metrics compared to 4 decimal places. Configs live in test/configs/.

When the pipeline has been modified (transforms, evaluators, models, metrics):
  uv run python scripts/snapshot/generate_snapshot_reference.py

Then commit test/fixtures/snapshot/reference.json.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REF_PATH = PROJECT_ROOT / "test" / "fixtures" / "snapshot" / "reference.json"

# Must match EXPERIMENTS in scripts/snapshot/generate_snapshot_reference.py
EXPERIMENTS = [
    ("prognostics", "snapshot/prognostics", "linear_regression"),
    ("prognostics", "snapshot/prognostics", "exponential_regression"),
    ("prognostics", "snapshot/prognostics", "lstm"),
    ("prognostics", "snapshot/prognostics", "patchtst"),
    ("prognostics", "snapshot/prognostics", "crossformer"),
    ("diagnostics", "snapshot/diagnostics", "linear_classifier"),
    ("prognostics_ragged", "snapshot/ragged_prognostics", "linear_regression"),
]

PRECISION = 4

# Allow small cross-platform variance for neural nets (patchtst, lstm, crossformer)
# e.g. 0.2362 vs 0.2364 on Linux vs macOS due to BLAS/PyTorch differences
FLOAT_ABS_TOL = 0.001
FLOAT_REL_TOL = 0.005  # 0.5%


def round_metrics(obj):
    if isinstance(obj, dict):
        return {k: round_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_metrics(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, PRECISION)
    return obj


def _run_experiment_to_dir(
    exp: str,
    out_dir: Path,
    metrics_path: Path | None,
    epochs: int = 5,
) -> int:
    """Run experiment to out_dir, optionally capture metrics. Returns exit code."""
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["PYTHONPATH"] = str(
        PROJECT_ROOT
    )  # Ensure test.data importable (not stdlib test)
    env["HYDRA_FULL_ERROR"] = "1"
    if metrics_path:
        env["PIPELINE_REGRESSION_OUTPUT"] = str(metrics_path)
    root = str(PROJECT_ROOT).replace("\\", "/")
    searchpath = f"file://{root}/test/configs,file://{root}/configs"
    cmd = [
        sys.executable,
        "picid/run.py",
        f"experiment={exp}",
        "paths=default",
        "debug=nox_testing_fit_predict",
        f"trainer.max_epochs={epochs}",
        "num_threads=1",
        "seed=42",
        f"hydra.run.dir={out_dir}",
        f"hydra.searchpath=[{searchpath}]",
    ]
    return subprocess.run(cmd, env=env, cwd=PROJECT_ROOT).returncode


@pytest.mark.skipif(
    not REF_PATH.exists(),
    reason="Run scripts/snapshot/generate_snapshot_reference.py first",
)
@pytest.mark.parametrize("task_type,exp_base,model", EXPERIMENTS)
def test_pipeline_snapshot_matches_reference(task_type, exp_base, model):
    """Current pipeline output matches committed reference (4 decimals)."""
    import tempfile

    key = f"{task_type}/{model}"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    try:
        env = os.environ.copy()
        env["PIPELINE_REGRESSION_OUTPUT"] = out_path
        env["PROJECT_ROOT"] = str(PROJECT_ROOT)
        env["PYTHONPATH"] = str(
            PROJECT_ROOT
        )  # Ensure test.data importable (not stdlib test)
        env["HYDRA_FULL_ERROR"] = "1"
        root = str(PROJECT_ROOT).replace("\\", "/")
        searchpath = f"file://{root}/test/configs,file://{root}/configs"
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
        r = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
        assert r.returncode == 0, f"Pipeline failed for {key}"
        current = round_metrics(json.loads(Path(out_path).read_text()))
    finally:
        Path(out_path).unlink(missing_ok=True)

    ref_all = json.loads(REF_PATH.read_text())
    ref = ref_all[key]

    for k in ref:
        assert k in current, f"Missing {k} in current"
        c, r = current[k], ref[k]
        if isinstance(c, float) and isinstance(r, float):
            assert math.isclose(
                c, r, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL
            ), f"{k}: current={c} ref={r} (must match within tolerance)"
        else:
            assert c == r, f"{k}: current={c} ref={r} (must match)"


def test_reproduce_from_run_matches_overrides():
    """Run via overrides and via reproduce_from_run produce matching metrics (5 epochs)."""
    import tempfile

    exp = "snapshot/prognostics/linear_regression"
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run_a"
        run_dir.mkdir()
        metrics_a = Path(tmp) / "metrics_a.json"
        metrics_b = Path(tmp) / "metrics_b.json"

        rc = _run_experiment_to_dir(exp, run_dir, metrics_a, epochs=5)
        assert rc == 0, "Initial run failed"
        assert metrics_a.exists(), "Metrics not captured"

        env = os.environ.copy()
        env["PROJECT_ROOT"] = str(PROJECT_ROOT)
        env["PYTHONPATH"] = str(
            PROJECT_ROOT
        )  # Ensure test.data importable (not stdlib test)
        env["HYDRA_FULL_ERROR"] = "1"
        env["PIPELINE_REGRESSION_OUTPUT"] = str(metrics_b)
        rc = subprocess.run(
            [
                sys.executable,
                "scripts/reproducibility/reproduce_from_run.py",
                str(run_dir),
            ],
            env=env,
            cwd=PROJECT_ROOT,
        )
        assert rc.returncode == 0, "reproduce_from_run failed"
        assert metrics_b.exists(), "Reproduce metrics not captured"

        a = round_metrics(json.loads(metrics_a.read_text()))
        b = round_metrics(json.loads(metrics_b.read_text()))
        for k in a:
            assert k in b, f"Missing {k} in reproduce output"
            assert (
                a[k] == b[k]
            ), f"{k}: overrides={a[k]} reproduce={b[k]} (must match to {PRECISION} decimals)"
