"""Smoke tests for ``tutorials/datasources/airbus_helicopter_loader.py``."""

from __future__ import annotations

import sys
from collections import UserDict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tutorials.datasources import airbus_helicopter_loader as airbus_tutorial


class _FakeAirbusHelicopterLoaderForTutorial:
    """Minimal stand-in so ``main()`` runs without real Airbus HDF5 files."""

    def __init__(
        self,
        *,
        data_dir: str = "",
        data_name: str = "",
        task_mode: str = "anomaly_detection",
        download: bool = True,
        **_: Any,
    ) -> None:
        self.data_path = data_dir
        self.data_name = data_name
        self.task_mode = task_mode

    def load_data(self) -> None:
        return None

    def get_data(self) -> dict[str, Any]:
        feat = np.zeros((60, 1024, 1), dtype=np.float32)
        tgt_train = np.zeros((60, 1, 1), dtype=np.float32)
        tgt_anom = np.ones((60, 1, 1), dtype=np.float32)
        return {
            "features": {
                "train": [feat.copy()],
                "val": [feat.copy()],
                "test": [feat.copy()],
            },
            "target": {
                "train": [tgt_train],
                "val": [tgt_anom],
                "test": [tgt_anom],
            },
            "metadata": {
                "train": [
                    {
                        "unit_name": "train_seq_0",
                        "split": "train",
                    }
                ],
                "val": [{"unit_name": "test_seq_0", "split": "test"}],
                "test": [{"unit_name": "test_seq_0", "split": "test"}],
            },
        }

    def get_meta_data(self) -> dict[str, Any]:
        return {
            "unit_ids": {"train": [0], "val": [0], "test": [0]},
            "unit_names": {
                "train": ["train_seq_0"],
                "val": ["test_seq_0"],
                "test": ["test_seq_0"],
            },
            "dims_explanation": "Ragged: (60, 1024, 1) features; (60, 1, 1) targets.",
        }


def test_airbus_helicopter_tutorial_smoke_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tutorial stdout must include the banner, split overview, and split contract note."""
    monkeypatch.setattr(
        airbus_tutorial,
        "AirbusHelicopterLoader",
        _FakeAirbusHelicopterLoaderForTutorial,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "airbus_helicopter_loader.py",
            "--data-dir",
            str(tmp_path),
            "--no-download",
        ],
    )
    airbus_tutorial.main()
    out = capsys.readouterr().out
    assert "AIRBUS HELICOPTER TUTORIAL" in out
    assert "SPLIT OVERVIEW" in out
    assert "PREDEFINED SPLIT NOTE" in out


def test_split_branch_accepts_mapping_subclasses() -> None:
    raw = UserDict({"train": [np.zeros((2, 2), dtype=np.float32)]})
    out = airbus_tutorial._split_branch({"features": raw}, "features")
    assert "train" in out
