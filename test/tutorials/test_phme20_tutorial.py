"""Smoke tests for ``tutorials/datasources/phme20_loader.py``."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from picid.data.data_objects import SplitDatasetContainer

from tutorials.datasources._standard_report import STANDARD_SECTIONS
from tutorials.datasources import phme20_loader as ph20_tutorial


class _FakePHME20LoaderForTutorial:
    """Stand-in so ``main()`` runs without PHMD files or network."""

    fold = 0
    data_name = "PHME20"
    task_mode = "rul"

    def load_data(self) -> None:
        return None

    def get_data(self) -> SplitDatasetContainer:
        z = np.zeros((5, 3), dtype=np.float32)
        t = np.linspace(4.0, 0.0, 5, dtype=np.float32).reshape(-1, 1)
        meta_train = {
            "unit_id": np.array(12),
            "unit_name": "Unit_12",
            "target_col": "rul",
            "target_in_the_featurs": True,
        }
        return SplitDatasetContainer(
            features={
                "train": [z],
                "val": [z.copy()],
                "test": [z.copy()],
            },
            target={
                "train": [t],
                "val": [t.copy()],
                "test": [t.copy()],
            },
            unit_metadata={
                "train": [meta_train],
                "val": [{"unit_id": np.array(3), "unit_name": "Unit_3"}],
                "test": [{"unit_id": np.array(99), "unit_name": "Unit_99"}],
            },
        )

    def get_meta_data(self) -> dict[str, Any]:
        return {"features": ["s1", "s2", "s3"], "identifier": "unit"}


def test_phme20_tutorial_includes_standard_report_and_rul_context(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stdout must include all standard section headers and PHME20 RUL/split notes."""
    monkeypatch.setattr(
        ph20_tutorial,
        "_build_loader",
        lambda _cfg: _FakePHME20LoaderForTutorial(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["phme20_loader.py"],
    )
    ph20_tutorial.main()
    out = capsys.readouterr().out
    for title in STANDARD_SECTIONS:
        assert f"\n{title}\n" in out, f"missing standard section: {title!r}"
    assert "TASK MODE: rul" in out
    assert "PHMD FOLD" in out
    assert "RUL target context" in out
    assert "PHMD filter" in out
    assert "Unit identifiers" in out


def test_phme20_compose_resolves_without_project_root_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydra compose for tutorial should not depend on PROJECT_ROOT env being set."""
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    cfg = ph20_tutorial._compose_phme20_config()
    OmegaConf.resolve(cfg.datasource)
