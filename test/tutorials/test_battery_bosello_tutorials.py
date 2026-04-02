"""Tests for NB14 and UNIBO21 Bosello datasource tutorials (patched loaders)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from picid.data.data_objects import SplitDatasetContainer

import tutorials.datasources.nb14_bosello_loader as nb14_tutorial
import tutorials.datasources.unibo21_bosello_loader as unibo21_tutorial


def _tiny_split_container() -> SplitDatasetContainer:
    """2D feature/target blocks so :func:`compute_feature_stats` can aggregate."""
    feat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    tgt = np.array([[0.5], [0.25]], dtype=np.float32)
    meta = {"unit_name": "fake_unit", "n_cycles": 2}
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
            "train": [dict(meta)],
            "val": [dict(meta)],
            "test": [dict(meta)],
        },
    )


class _FakeNB14LoaderForTutorial:
    """Stand-in for NB14Loader; avoids NASA battery files."""

    def __init__(self, **kwargs: Any) -> None:
        self.data_dir = str(kwargs.get("data_dir", ""))
        self.data_name = str(kwargs.get("data_name", "NB14"))
        self.task_mode = str(kwargs.get("task_mode", "ahRul"))
        self.data_path = self.data_dir

    def load_data(self) -> None:
        return None

    def get_data(self) -> SplitDatasetContainer:
        return _tiny_split_container()

    def get_meta_data(self) -> dict[str, Any]:
        return {"note": "fake NB14 Bosello tutorial meta"}


class _FakeUNIBO21LoaderForTutorial:
    """Stand-in for UNIBO21Loader; avoids Powertools files."""

    def __init__(self, **kwargs: Any) -> None:
        self.data_dir = str(kwargs.get("data_dir", ""))
        self.data_name = str(kwargs.get("data_name", "UNIBO21"))
        self.task_mode = str(kwargs.get("task_mode", "SOC"))
        self.data_path = self.data_dir

    def load_data(self) -> None:
        return None

    def get_data(self) -> SplitDatasetContainer:
        return _tiny_split_container()

    def get_meta_data(self) -> dict[str, Any]:
        return {"note": "fake UNIBO21 Bosello tutorial meta"}


def test_nb14_bosello_tutorial_includes_filtering_and_ah_rul_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        nb14_tutorial,
        "get_nb14_bosello_loader_class",
        lambda: _FakeNB14LoaderForTutorial,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["nb14_bosello_loader.py", "--data-dir", str(tmp_path)],
    )
    nb14_tutorial.main()
    out = capsys.readouterr().out
    assert "NB14 BOSSELLO TUTORIAL" in out
    assert "\nDATASET-SPECIFIC NOTES\n" in out
    assert "filtering" in out.lower()
    assert "ah-rul" in out.lower()
    assert "\nLOADER CONFIG\n" in out


def test_unibo21_bosello_tutorial_includes_capacity_threshold_and_group_split_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        unibo21_tutorial,
        "get_unibo21_bosello_loader_class",
        lambda: _FakeUNIBO21LoaderForTutorial,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["unibo21_bosello_loader.py", "--data-dir", str(tmp_path)],
    )
    unibo21_tutorial.main()
    out = capsys.readouterr().out
    assert "UNIBO21 BOSSELLO TUTORIAL" in out
    assert "\nDATASET-SPECIFIC NOTES\n" in out
    assert "capacity-threshold" in out.lower()
    assert "group-split" in out.lower()
    assert "\nLOADER CONFIG\n" in out
