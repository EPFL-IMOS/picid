"""Pipeline snapshot tests for diagnostics using synthetic data only.

Compares current run to reference metrics. Uses diagnostics.npz fixture.
Fixture generator: test/scripts/snapshot/generate_snapshot_fixtures.py
Reference generator: test/scripts/snapshot/generate_snapshot_reference.py
"""

from __future__ import annotations

import json
import math
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
]

PRECISION = 4
FLOAT_ABS_TOL = 0.001
FLOAT_REL_TOL = 0.005

# Fit-predict models (XGBoost etc.) use a one-epoch trainer with no ModelCheckpoint;
# interface.train() must use max_epochs=1 to avoid the ckpt_path="best" branch.
_FIT_PREDICT_MODELS: frozenset[str] = frozenset({"xgboost_fit_predict"})


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

    key = f"{task_type}/{model}"

    root = project_config.root_dir
    searchpath = f"{root}/test/configs,{project_config.config_path}"

    interface = EntryInterface()

    # Fit-predict models use a one-epoch trainer with no ModelCheckpoint;
    # max_epochs=1 keeps interface.train() out of the ckpt_path="best" branch.
    max_epochs = "1" if model in _FIT_PREDICT_MODELS else "10"

    # IMPLEMENTERE CARICAMENTO DA CFG
    results = interface.train(
        run_name="test",
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
