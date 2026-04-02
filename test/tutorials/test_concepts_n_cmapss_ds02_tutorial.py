"""Smoke tests for ``tutorials/datasources/concepts_n_cmapss_ds02_loader.py``."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from picid.data.data_objects import SplitDatasetContainer
from picid.data.split_strategies.by_source_splitter import BySourceSplitter

from tutorials.datasources import concepts_n_cmapss_ds02_loader as ds02_tutorial


class _FakeMultiSourceLoaderForTutorial:
    """Stand-in so ``main()`` runs without N-CMAPSS HDF5 files or child loads."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self.data_name = str(kwargs.get("data_name", ""))
        self.task_mode = str(kwargs.get("task_mode", ""))

    def load_data(self) -> None:
        return None

    def split_data(self) -> None:
        return None

    def get_data(self) -> SplitDatasetContainer:
        return SplitDatasetContainer(
            features={
                "train": [np.zeros((5, 3), dtype=np.float32)],
                "val": [np.zeros((4, 3), dtype=np.float32)],
                "test": [np.zeros((6, 3), dtype=np.float32)],
            },
            target={
                "train": [np.zeros((5, 1), dtype=np.float32)],
                "val": [np.zeros((4, 1), dtype=np.float32)],
                "test": [np.zeros((6, 1), dtype=np.float32)],
            },
        )

    def get_meta_data(self) -> dict[str, Any]:
        return {}

    def get_source_names(self) -> tuple[str, ...]:
        return ("train1", "test1", "val1")

    def get_multisource_data_splitter(self) -> BySourceSplitter:
        return BySourceSplitter(
            sources_train=["train1"],
            sources_test=["test1"],
            sources_val=["val1"],
        )


def test_concepts_n_cmapss_ds02_tutorial_output_includes_required_sections(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tutorial stdout must document multisource layout, units, and concepts."""
    monkeypatch.setattr(
        ds02_tutorial,
        "_build_multisource_loader",
        lambda _cfg: _FakeMultiSourceLoaderForTutorial(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["concepts_n_cmapss_ds02_loader.py"],
    )
    ds02_tutorial.main()
    out = capsys.readouterr().out
    assert "MULTISOURCE OVERVIEW" in out
    assert "SELECTED UNITS" in out
    assert "REQUESTED CONCEPTS" in out
