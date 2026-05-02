"""Guard rails for canonical internal model imports."""

from __future__ import annotations

from pathlib import Path


def test_runtime_modules_do_not_import_legacy_methods_or_wrappers():
    checked = [
        Path("picid/model/__init__.py"),
        Path("picid/run.py"),
        Path("picid/pipeline/base.py"),
        Path("picid/interface/model/custom_model.py"),
        Path("picid/interface/utils.py"),
    ]
    forbidden = ("picid.model.methods", "picid.model.wrappers")

    for path in checked:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
