"""Tests for the shared PHMD repository and assembly helpers."""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.modules["phmd"] = MagicMock()
sys.modules["phmd.datasets"] = MagicMock()

from picid.data.datasources.base.phmd_assembly import build_split_payloads  # noqa: E402
from picid.data.datasources.base.phmd_repository import PHMDRepository  # noqa: E402


@dataclass
class _FakeTask:
    """Lightweight stand-in for a PHMD task."""

    fold_payloads: dict[int, object]
    meta: dict[str, object]

    def __getitem__(self, fold: int):
        return self.fold_payloads[fold]


class _FakeDataset:
    """Dataset stub that exposes ``get_task`` like the PHMD package."""

    def __init__(self, tasks: dict[str, _FakeTask]):
        self._tasks = tasks

    def get_task(self, task_name: str):
        return self._tasks[task_name]


def test_phmd_repository_load_task_bundle_and_metadata(monkeypatch, tmp_path):
    """Repository resolves task folds and captures metadata from the main task."""
    tasks = {
        "rul": _FakeTask(
            fold_payloads={0: {"train": pd.DataFrame({"unit": [1], "x": [1.0]})}},
            meta={"features": ["x"], "identifier": "unit"},
        ),
        "aux": _FakeTask(
            fold_payloads={0: {"train": pd.DataFrame({"unit": [2], "y": [2.0]})}},
            meta={"features": ["y"], "identifier": "unit"},
        ),
    }
    monkeypatch.setattr(
        "picid.data.datasources.base.phmd_repository.datasets.Dataset",
        lambda data_name, cache_dir: _FakeDataset(tasks),
    )

    repository = PHMDRepository(
        data_name="DemoPHMD",
        cache_dir=str(tmp_path),
        fold=0,
        task_mode="rul",
        auxiliary_tasks=["aux"],
    )
    bundle = repository.load_task_bundle(["aux", "rul"])

    assert set(bundle.task_folds) == {"aux", "rul"}
    assert bundle.meta_data == tasks["rul"].meta


def test_phmd_repository_cache_helpers(tmp_path):
    """Cache helper methods remain stable and round-trip cached payloads."""
    repository = PHMDRepository(
        data_name="DemoPHMD",
        cache_dir=str(tmp_path),
        fold=3,
        task_mode="rul",
        auxiliary_tasks=["aux1", "aux2"],
    )

    cache_name = repository.get_cache_file_name("train")
    assert cache_name.endswith(".pkl")
    assert "taskmode_rul" in cache_name
    assert "fold_3" in cache_name
    assert "aux1_aux2" in cache_name
    assert "dataset_DemoPHMD" in cache_name

    cache_path = tmp_path / cache_name
    payload = {"x": [1, 2, 3]}
    with open(cache_path, "wb") as file_obj:
        pickle.dump(payload, file_obj)

    assert repository.retrieve_cached_task_data(cache_path) == payload

    with pytest.raises(FileNotFoundError, match="Cache file not found"):
        repository.retrieve_cached_task_data(tmp_path / "missing.pkl")


def test_build_split_payloads_returns_split_container_and_unit_metadata():
    """Assembly keeps the split container shape and collects unit metadata."""
    main_fold = {
        "train": pd.DataFrame(
            {
                "unit_id": [1, 1],
                "feat": [1.0, 2.0],
                "target": [0.1, 0.2],
            }
        ),
        "val": pd.DataFrame(
            {
                "unit_id": [2],
                "feat": [3.0],
                "target": [0.3],
            }
        ),
        "test": pd.DataFrame(
            {
                "unit_id": [3],
                "feat": [4.0],
                "target": [0.4],
            }
        ),
    }

    def _process_unit(df_unit, unit_name, features_col, target_col):
        return {
            "features": df_unit[features_col].to_numpy(),
            "target": df_unit[[target_col]].to_numpy(),
            "metadata": {"unit_name": f"unit-{unit_name}", "unit_id": int(unit_name)},
        }

    payload, extra_meta = build_split_payloads(
        main_fold,
        unit_column="unit_id",
        feature_columns=["feat"],
        target_column="target",
        process_unit=_process_unit,
    )

    assert set(payload) == {"features", "target", "metadata"}
    assert len(payload["features"]["train"]) == 1
    assert len(payload["target"]["test"]) == 1
    assert len(payload["metadata"]["train"]) == 1
    assert extra_meta["unit_names"]["train"] == ["unit-1"]
    assert extra_meta["unit_ids"]["val"] == [2]
