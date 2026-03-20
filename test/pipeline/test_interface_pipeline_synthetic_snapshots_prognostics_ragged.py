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

from picid.config import project_config
from picid.interface import EntryInterface

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

    root = project_config.root_dir
    searchpath = f"{root}/test/configs,{project_config.config_path}"

    interface = EntryInterface()

    # IMPLEMENTERE CARICAMENTO DA CFG
    results = interface.train(run_name=f'test_{key}',
                              model=None,
                              task_definition=None,
                              datasource=None,
                              overrides=['trainer.max_epochs=10',
                                         "num_threads=1",
                                         "debug=nox_testing_fit_predict",
                                         "trainer.accelerator=cpu",
                                         "seed=42",
                                         f"experiment={exp_base}/{model}",
                                         f"hydra.searchpath=[{searchpath}]"])

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
