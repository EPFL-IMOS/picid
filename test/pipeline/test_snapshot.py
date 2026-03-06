"""Pipeline snapshot: compare current run to reference metrics.

Fails if pipeline behavior changes unintentionally (e.g. transform, evaluator).
Metrics compared to 4 decimal places. Configs live in test/configs/.
"""
from __future__ import annotations

import json
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


def round_metrics(obj):
    if isinstance(obj, dict):
        return {k: round_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_metrics(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, PRECISION)
    return obj


@pytest.mark.skipif(not REF_PATH.exists(), reason="Run scripts/snapshot/generate_snapshot_reference.py first")
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
        assert current[k] == ref[k], (
            f"{k}: current={current[k]} ref={ref[k]} (must match to 4 decimals)"
        )
