"""Pipeline snapshot tests using synthetic data only.

Compares current run to reference metrics. Fails if pipeline behavior changes
unintentionally (e.g. transform, evaluator). Uses only pre-generated synthetic
fixtures (prognostics.npz, diagnostics.npz, ragged_prognostics.pkl)—no downloads,
no real data. Metrics compared to 4 decimal places. Configs in test/configs/.

Fixture generators (run when needed, then commit output):
  - test/scripts/snapshot/generate_snapshot_fixtures.py  → prognostics.npz, diagnostics.npz, etc.
  - test/scripts/snapshot/generate_snapshot_ragged_fixtures.py → ragged_prognostics.pkl
  - test/scripts/snapshot/generate_snapshot_reference.py → reference.json

When the pipeline has been modified (transforms, evaluators, models, metrics):
  uv run python test/scripts/snapshot/generate_snapshot_reference.py

Then commit test/fixtures/snapshot/reference/reference.json.

Related snapshot test files:
  - test_pipeline_synthetic_snapshots_diagnostics.py
  - test_pipeline_synthetic_snapshots_prognostics_ragged.py
  - test_pipeline_synthetic_snapshots_anomaly.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from picid.interface import EntryInterface
from picid.config import project_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REF_PATH = (
    PROJECT_ROOT / "test" / "fixtures" / "snapshot" / "reference" / "reference.json"
)

# Must match EXPERIMENTS in test/scripts/snapshot/generate_snapshot_reference.py
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
]

PRECISION = 4

# Fit-predict models (XGBoost etc.) use a one-epoch trainer with no ModelCheckpoint;
# interface.train() must use max_epochs=1 to avoid the ckpt_path="best" branch.
_FIT_PREDICT_MODELS: frozenset[str] = frozenset({"xgboost_fit_predict"})

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
    epochs: int = 5,
) -> int:
    """Run experiment to out_dir. Returns exit code."""
    env = os.environ.copy()
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


def _metrics_path_from_run_dir(run_dir: Path) -> Path:
    """Return the best-epoch test metrics path for a subprocess pipeline run."""
    return run_dir / "eval_details" / "best_epoch" / "test" / "metrics.json"


@pytest.mark.requires_snapshots
@pytest.mark.skipif(
    not REF_PATH.exists(),
    reason="Run test/scripts/snapshot/generate_snapshot_reference.py first",
)
@pytest.mark.parametrize("task_type,exp_base,model", EXPERIMENTS)
def test_pipeline_snapshot_matches_reference(task_type, exp_base, model):
    """Current pipeline output matches committed reference (4 decimals)."""

    key = f"{task_type}/{model}"

    root = project_config.root_dir
    searchpath = f"{root}/test/configs,{project_config.config_path}"

    interface = EntryInterface()

    # Fit-predict models use a one-epoch trainer with no ModelCheckpoint;
    # max_epochs=1 keeps interface.train() out of the ckpt_path="best" branch.
    max_epochs = "1" if model in _FIT_PREDICT_MODELS else "10"

    # IMPLEMENTERE CARICAMENTO DA CFG
    results = interface.train(
        run_name=f"test_{key}",
        model=None,
        task_definition=None,
        datasource=None,
        overrides=[
            f"trainer.max_epochs={max_epochs}",
            "num_threads=1",
            "debug=nox_testing_fit_predict",
            "trainer.accelerator=cpu",
            "seed=42",
            f"experiment={exp_base}/{model}",
            f"hydra.searchpath=[{searchpath}]",
        ],
    )

    results = results[0]

    ref_all = json.loads(REF_PATH.read_text())
    ref = ref_all[key]

    for k in ref:
        assert k in results, f"Missing {k} in current"
        c, r = results[k], ref[k]
        if isinstance(c, float) and isinstance(r, float):
            assert math.isclose(
                c, r, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL
            ), f"{k}: current={c} ref={r} (must match within tolerance)"
        else:
            assert c == r, f"{k}: current={c} ref={r} (must match)"


@pytest.mark.requires_snapshots
def test_reproduce_from_run_matches_overrides():
    """Run via overrides and via reproduce_from_run produce matching metrics (5 epochs)."""
    import tempfile

    exp = "snapshot/prognostics/linear_regression"
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run_a"
        run_dir.mkdir()

        rc = _run_experiment_to_dir(exp, run_dir, epochs=5)
        assert rc == 0, "Initial run failed"
        metrics_a_path = _metrics_path_from_run_dir(run_dir)
        assert metrics_a_path.exists(), "Metrics not captured"

        env = os.environ.copy()
        env["PROJECT_ROOT"] = str(PROJECT_ROOT)
        env["PYTHONPATH"] = str(
            PROJECT_ROOT
        )  # Ensure test.data importable (not stdlib test)
        env["HYDRA_FULL_ERROR"] = "1"
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
        reproduce_dirs = list(run_dir.glob("reproduce_*"))
        assert reproduce_dirs, "Reproduce dir not created"
        reproduce_dir = max(reproduce_dirs, key=lambda p: p.stat().st_mtime)
        metrics_b_path = _metrics_path_from_run_dir(reproduce_dir)
        assert metrics_b_path.exists(), "Reproduce metrics not captured"

        a = round_metrics(json.loads(metrics_a_path.read_text()))
        b = round_metrics(json.loads(metrics_b_path.read_text()))
        for k in a:
            assert k in b, f"Missing {k} in reproduce output"
            assert (
                a[k] == b[k]
            ), f"{k}: overrides={a[k]} reproduce={b[k]} (must match to {PRECISION} decimals)"
