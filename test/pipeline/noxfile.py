"""Nox sessions for pipeline snapshot tests."""

from pathlib import Path

import nox

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@nox.session(venv_backend="none")
def pipeline_snapshot(session):
    """Run pipeline snapshot tests: synthetic data, quick models, compare to reference."""
    session.chdir(_PROJECT_ROOT)
    session.run(
        "pytest",
        "test/pipeline/test_pipeline_synthetic_snapshots_prognostics.py",
        "test/pipeline/test_pipeline_synthetic_snapshots_diagnostics.py",
        "test/pipeline/test_pipeline_synthetic_snapshots_prognostics_ragged.py",
        "test/pipeline/test_pipeline_synthetic_snapshots_anomaly.py",
        "-v",
        *session.posargs,
    )
