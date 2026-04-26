from importlib import import_module
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]

ACTIVE_PATHS = [
    REPO_ROOT / "picid",
    REPO_ROOT / "configs",
]

ALLOWED_PICID_FILES = {
    REPO_ROOT / "picid" / "baselines" / "__init__.py",
}


def _iter_active_files():
    for root in ACTIVE_PATHS:
        if root.name == "picid":
            for path in root.rglob("*.py"):
                if "baselines" in path.parts:
                    continue
                yield path
        elif root.name == "configs":
            yield from root.rglob("*.yaml")


def test_active_code_and_configs_do_not_reference_legacy_baselines_namespace():
    offenders = []
    for path in _iter_active_files():
        if path in ALLOWED_PICID_FILES:
            continue

        text = path.read_text()
        if "picid.baselines" in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_legacy_baselines_namespace_is_removed():
    sys.modules.pop("picid.baselines", None)
    with pytest.raises(ModuleNotFoundError):
        import_module("picid.baselines")
