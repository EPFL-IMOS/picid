"""Pipeline snapshot tests for prognostics_ragged using synthetic data only.

Compares current run to reference metrics. Uses ragged_prognostics.pkl fixture.
Fixture generator: test/scripts/snapshot/generate_snapshot_ragged_fixtures.py
Reference generator: test/scripts/snapshot/generate_snapshot_reference.py
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
REF_PATH = (
    PROJECT_ROOT / "test" / "fixtures" / "snapshot" / "reference" / "reference.json"
)

# Must match EXPERIMENTS in test/scripts/snapshot/generate_snapshot_reference.py
EXPERIMENTS = [
    ("prognostics_ragged", "snapshot/ragged_prognostics", "linear_regression"),
    ("prognostics_ragged", "snapshot/ragged_prognostics", "exponential_regression"),
    ("prognostics_ragged", "snapshot/ragged_prognostics", "lstm"),
    ("prognostics_ragged", "snapshot/ragged_prognostics", "patchtst"),
    ("prognostics_ragged", "snapshot/ragged_prognostics", "crossformer"),
]

PRECISION = 4
FLOAT_ABS_TOL = 0.001
FLOAT_REL_TOL = 0.005


def _metrics_from_eval_details(run_dir: Path) -> dict:
    """Read metrics from eval_details/best_epoch/test/metrics.json, add test/ prefix to match ref."""
    raw = json.loads(
        (run_dir / "eval_details" / "best_epoch" / "test" / "metrics.json").read_text()
    )
    return {"test/" + k: v for k, v in raw.items()}


def round_metrics(obj):
    if isinstance(obj, dict):
        return {k: round_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_metrics(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, PRECISION)
    return obj


@pytest.mark.requires_snapshots
@pytest.mark.skipif(
    not REF_PATH.exists(),
    reason="Run test/scripts/snapshot/generate_snapshot_reference.py first",
)
@pytest.mark.parametrize("task_type,exp_base,model", EXPERIMENTS)
def test_pipeline_snapshot_matches_reference(task_type, exp_base, model):
    """Current pipeline output matches committed reference (4 decimals)."""
    import tempfile

    key = f"{task_type}/{model}"
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        env = os.environ.copy()
        env["PROJECT_ROOT"] = str(PROJECT_ROOT)
        env["PYTHONPATH"] = str(PROJECT_ROOT)
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
            f"hydra.run.dir={run_dir}",
            f"hydra.searchpath=[{searchpath}]",
        ]
        r = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
        assert r.returncode == 0, f"Pipeline failed for {key}"
        current = round_metrics(_metrics_from_eval_details(run_dir))

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
