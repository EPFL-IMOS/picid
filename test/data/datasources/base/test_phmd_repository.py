"""Tests for the shared PHMD repository and assembly helpers."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import pandas as pd
import pytest

from picid.data.datasources.base.phmd_assembly import build_split_payloads
from picid.data.datasources.base.phmd_repository import PHMDRepository


@dataclass
class _FakeTask:
    """Lightweight stand-in for a PHMD task."""

    fold_payloads: dict[int, object]
    meta: dict[str, object]

    def __getitem__(self, fold: int):
        """
        Return the fold payload for the requested fold index.

        Parameters
        ----------
        fold : int
            Fold index to retrieve.

        Returns
        -------
        object
            Stored payload for the requested fold.
        """
        return self.fold_payloads[fold]


class _FakeDataset:
    """
    Dataset stub that exposes ``get_task`` like the PHMD package.

    Parameters
    ----------
    tasks : dict[str, _FakeTask]
        Task mapping returned by the fake PHMD dataset.
    """

    def __init__(self, tasks: dict[str, _FakeTask]):
        """
        Store the task mapping used by the fake dataset.

        Parameters
        ----------
        tasks : dict[str, _FakeTask]
            Task mapping returned by the fake PHMD dataset.
        """
        self._tasks = tasks

    def get_task(self, task_name: str):
        """
        Return the fake task for the requested task name.

        Parameters
        ----------
        task_name : str
            Requested task name.

        Returns
        -------
        _FakeTask
            Fake task associated with ``task_name``.
        """
        return self._tasks[task_name]


def test_phmd_repository_load_task_bundle_and_metadata(monkeypatch, tmp_path):
    """
    Repository resolves task folds and captures metadata from the main task.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the PHMD dataset factory.
    tmp_path : pathlib.Path
        Temporary cache directory for the repository instance.
    """
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
    """
    Cache helper methods remain stable and round-trip cached payloads.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used for repository cache files.
    """
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
        """
        Convert one fake unit into the assembly payload shape.

        Parameters
        ----------
        df_unit : pandas.DataFrame
            Unit dataframe for the fake PHMD split.
        unit_name : object
            Unit identifier.
        features_col : list[str]
            Feature columns to extract.
        target_col : str
            Target column to extract.

        Returns
        -------
        dict[str, object]
            Split payload for one unit.
        """
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

    assert set(payload) == {"features", "target", "unit_metadata"}
    assert len(payload["features"]["train"]) == 1
    assert len(payload["target"]["test"]) == 1
    assert len(payload["unit_metadata"]["train"]) == 1
    assert extra_meta["unit_names"]["train"] == ["unit-1"]
    assert extra_meta["unit_ids"]["val"] == [2]


def test_build_split_payloads_preserves_keys_when_train_split_is_empty():
    """Assembly should preserve payload keys even when the train split is empty."""
    main_fold = {
        "train": pd.DataFrame({"unit_id": [], "feat": [], "target": []}),
        "val": pd.DataFrame({"unit_id": [2], "feat": [3.0], "target": [0.3]}),
        "test": pd.DataFrame({"unit_id": [3], "feat": [4.0], "target": [0.4]}),
    }

    def _process_unit(df_unit, unit_name, features_col, target_col):
        """
        Convert one fake unit into the assembly payload shape.

        Parameters
        ----------
        df_unit : pandas.DataFrame
            Unit dataframe for the fake PHMD split.
        unit_name : object
            Unit identifier.
        features_col : list[str]
            Feature columns to extract.
        target_col : str
            Target column to extract.

        Returns
        -------
        dict[str, object]
            Split payload for one unit.
        """
        return {
            "features": df_unit[features_col].to_numpy(),
            "target": df_unit[[target_col]].to_numpy(),
        }

    payload, extra_meta = build_split_payloads(
        main_fold,
        unit_column="unit_id",
        feature_columns=["feat"],
        target_column="target",
        process_unit=_process_unit,
    )

    assert set(payload) == {"features", "target"}
    assert payload["features"]["train"] == []
    assert len(payload["features"]["val"]) == 1
    assert len(payload["target"]["test"]) == 1
    assert extra_meta == {}
