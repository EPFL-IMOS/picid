"""Guard rails that keep the family-first namespace canonical."""

from __future__ import annotations

from pathlib import Path


_FORBIDDEN = ("picid.model.methods", "picid.model.wrappers")


def _assert_no_forbidden_tokens(paths: list[Path]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in _FORBIDDEN), path


def test_runtime_and_user_facing_surfaces_do_not_point_to_legacy_namespaces():
    runtime_paths = [
        path
        for path in Path("picid").rglob("*.py")
        if "picid/model/methods/" not in str(path)
        and "picid/model/wrappers/" not in str(path)
    ]
    tutorial_paths = list(Path("tutorials/models").glob("*.py"))
    config_paths = list(Path("configs/model").glob("*.yaml"))

    _assert_no_forbidden_tokens(runtime_paths + tutorial_paths + config_paths)


def test_canonical_test_tree_does_not_depend_on_legacy_namespaces():
    canonical_tests = [
        path
        for path in Path("test/model/estimators").rglob("*.py")
        if path.name != "test_import_parity.py"
    ]
    canonical_tests.extend(
        path
        for path in Path("test/model/adapters").rglob("*.py")
        if path.name != "test_import_parity.py"
    )

    _assert_no_forbidden_tokens(canonical_tests)
