"""Smoke tests for HSF15 datasource tutorials (per-component loaders)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from picid.data.data_objects import SplitDatasetContainer

import tutorials.datasources.hsf15_component_loader as hsf15_tutorial


class _FakeHSF15LoaderForTutorial:
    """Minimal stand-in so tutorials run without PHMD / real HSF15 files."""

    def __init__(self, **kwargs: Any) -> None:
        self.fold = int(kwargs.get("fold", 0))
        self.cache_dir = str(kwargs.get("cache_dir", ""))
        self.data_name = str(kwargs.get("data_name", "HSF15"))
        self.task_mode = str(kwargs.get("task_mode", "accumulator"))
        self.auxiliary_tasks = list(kwargs.get("auxiliary_tasks", []))

    def load_data(self) -> None:
        return None

    def get_data(self) -> SplitDatasetContainer:
        feat = np.zeros((2, 4, 3), dtype=np.float32)
        tgt = np.zeros((2, 4, 1), dtype=np.float32)
        meta_unit = {"unit_name": "HSF15_Unit_1", "class_label": 0}
        return SplitDatasetContainer(
            features={
                "train": [feat.copy()],
                "val": [feat.copy()],
                "test": [feat.copy()],
            },
            target={
                "train": [tgt.copy()],
                "val": [tgt.copy()],
                "test": [tgt.copy()],
            },
            unit_metadata={
                "train": [dict(meta_unit)],
                "val": [dict(meta_unit)],
                "test": [dict(meta_unit)],
            },
        )

    def get_meta_data(self) -> dict[str, Any]:
        return {"note": "fake HSF15 meta for tutorial"}


@pytest.mark.parametrize(
    "component,expected_num_classes",
    [
        ("accumulator", 4),
        ("cooler", 3),
        ("pump", 3),
        ("valve", 4),
    ],
)
def test_hsf15_component_tutorial_reports_task_mode_and_num_classes(
    component: str,
    expected_num_classes: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each component tutorial documents task_mode and configured class count."""
    monkeypatch.setattr(
        hsf15_tutorial,
        "get_hsf15_loader_class",
        lambda: _FakeHSF15LoaderForTutorial,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hsf15_component_loader.py",
            "--component",
            component,
            "--data-dir",
            str(tmp_path),
        ],
    )
    hsf15_tutorial.main()
    out = capsys.readouterr().out
    assert "HSF15 TUTORIAL" in out
    assert "HSF15 COMPONENT NOTE" in out
    assert f"task_mode: {component}" in out
    assert f"num_classes: {expected_num_classes}" in out
    assert "\nLOADER CONFIG\n" in out
    assert "\nDATASET-SPECIFIC NOTES\n" in out


@pytest.mark.parametrize(
    "module_name,component,expected_num_classes",
    [
        ("tutorials.datasources.hsf15_accumulator_loader", "accumulator", 4),
        ("tutorials.datasources.hsf15_cooler_loader", "cooler", 3),
        ("tutorials.datasources.hsf15_pump_loader", "pump", 3),
        ("tutorials.datasources.hsf15_valve_loader", "valve", 4),
    ],
)
def test_hsf15_wrapper_scripts_match_component(
    module_name: str,
    component: str,
    expected_num_classes: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Thin wrapper entrypoints print the same component-specific contract."""
    monkeypatch.setattr(
        hsf15_tutorial,
        "get_hsf15_loader_class",
        lambda: _FakeHSF15LoaderForTutorial,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [module_name.rsplit(".", maxsplit=1)[-1] + ".py", "--data-dir", str(tmp_path)],
    )
    mod = importlib.import_module(module_name)
    mod.main()
    out = capsys.readouterr().out
    assert f"task_mode: {component}" in out
    assert f"num_classes: {expected_num_classes}" in out
