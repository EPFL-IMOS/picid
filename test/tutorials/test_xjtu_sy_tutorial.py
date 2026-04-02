"""Smoke tests for ``tutorials/datasources/xjtu_sy_loader.py``."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from picid.data.data_objects import SplitDatasetContainer

from tutorials.datasources import xjtu_sy_loader as xjtu_tutorial


class _FakeXJtuSyLoaderForTutorial:
    """Stand-in so ``main()`` runs without PHMD files or network."""

    split_mode = "in_domain"
    fold_id = 1
    data_name = "XJTU-SY"
    task_mode = "rul"

    def load_data(self) -> None:
        return None

    def get_data(self) -> SplitDatasetContainer:
        z = np.zeros((1, 10, 2), dtype=np.float32)
        t = np.zeros((1, 10, 1), dtype=np.float32)
        return SplitDatasetContainer(
            features={
                "train": [z, z.copy(), z.copy()],
                "val": [z.copy()],
                "test": [z.copy()],
            },
            target={
                "train": [t, t.copy(), t.copy()],
                "val": [t.copy()],
                "test": [t.copy()],
            },
            unit_metadata={
                "train": [
                    {"unit_name": "Bearing 1_3"},
                    {"unit_name": "Bearing 2_3"},
                    {"unit_name": "Bearing 3_3"},
                ],
                "val": [{"unit_name": "Bearing 1_1"}],
                "test": [{"unit_name": "Bearing 1_2"}],
            },
        )

    def get_meta_data(self) -> dict[str, Any]:
        return {"features": ["x", "y"]}


def test_xjtu_sy_tutorial_output_includes_split_protocol_and_oc_coverage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stdout must document split mode, fold id, and operating-condition coverage."""
    monkeypatch.setattr(
        xjtu_tutorial,
        "_build_loader",
        lambda _cfg: _FakeXJtuSyLoaderForTutorial(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["xjtu_sy_loader.py"],
    )
    xjtu_tutorial.main()
    out = capsys.readouterr().out
    assert "SPLIT MODE: in_domain" in out
    assert "FOLD ID" in out
    assert "OPERATING CONDITION COVERAGE" in out


def test_xjtu_compose_resolves_without_project_root_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydra compose for tutorial should not depend on PROJECT_ROOT env being set."""
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    cfg = xjtu_tutorial._compose_xjtu_sy_config()
    OmegaConf.resolve(cfg.datasource)
