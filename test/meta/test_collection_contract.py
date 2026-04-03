"""Meta-tests: registered pytest markers and documented collection smoke commands.

Collection smoke (contract; use ``uv run pytest``):

- ``uv run pytest test/ --collect-only -q`` — full tree collects without executing tests
- ``uv run pytest .../test_collection_contract.py -q`` — marker contract self-check

``requires_snapshots`` tests are verification-only against committed fixtures; do not
regenerate, rewrite, or bless snapshots from test runs.
"""

from pathlib import Path

import pytest


@pytest.fixture
def registered_marker_names(pytestconfig) -> set[str]:
    """Names of markers registered in pyproject.toml (ini ``markers``)."""
    lines = pytestconfig.getini("markers")
    return {line.split(":", 1)[0].strip() for line in lines}


def test_registered_test_markers_include_expected_families(registered_marker_names):
    expected = {
        "optional_dep",
        "integration",
        "tutorial",
        "benchmark",
        "slow",
        "requires_snapshots",
    }
    missing = expected - registered_marker_names
    assert not missing, f"markers missing from pyproject.toml: {sorted(missing)}"


def test_collection_smoke_commands_documented_in_this_module():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "uv run pytest" in src
    assert "--collect-only" in src
    assert "marker contract self-check" in src


def test_project_test_docs_reference_uv_entrypoint():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/testing/test-entrypoints.md").read_text(encoding="utf-8")
    assert "uv run pytest" in text


def test_pytest_basetemp_not_under_source_test_tree(pytestconfig):
    """Basetemp must stay outside test/ so pytest-generated dirs are not in the test source tree."""
    root = Path(__file__).resolve().parents[2]
    test_dir = (root / "test").resolve()
    basetemp = pytestconfig.option.basetemp
    assert basetemp is not None, "expected --basetemp in addopts"
    basetemp_resolved = Path(basetemp).resolve()
    assert not basetemp_resolved.is_relative_to(
        test_dir
    ), f"basetemp {basetemp_resolved} must not lie under the test tree {test_dir}"
