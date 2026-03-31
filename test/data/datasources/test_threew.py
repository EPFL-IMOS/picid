"""Characterization tests for ThreeWLoader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from picid.data.data_objects import SplitDatasetContainer
from picid.data.datasources.base.exceptions import DatasourceConfigurationError
from picid.data.datasources.threew import ThreeWLoader


@pytest.fixture
def threew_kwargs(tmp_path: Path):
    """
    Return baseline loader kwargs for synthetic 3W tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-test temporary directory.

    Returns
    -------
    dict[str, object]
        Loader kwargs used by the synthetic 3W tests.
    """
    return {
        "data_dir": str(tmp_path),
        "data_name": "threew",
        "task_mode": "anomaly_detection",
        "download": False,
        "validation_fold": 0,
        "test_fold": 1,
        "include_ova": False,
        "export_event_class": True,
    }


def _touch_instance(tmp_path: Path, class_id: int, instance_name: str) -> Path:
    """
    Create an empty parquet placeholder at the expected 3W path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root directory.
    class_id : int
        3W class folder id.
    instance_name : str
        Instance stem without extension.

    Returns
    -------
    pathlib.Path
        Created parquet placeholder path.
    """
    class_dir = tmp_path / "dataset" / str(class_id)
    class_dir.mkdir(parents=True, exist_ok=True)
    instance_path = class_dir / f"{instance_name}.parquet"
    instance_path.touch()
    return instance_path


def _write_folds(tmp_path: Path, rows: list[tuple[str, int, bool]]) -> None:
    """
    Write a minimal folds CSV understood by the 3W loader.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root directory.
    rows : list[tuple[str, int, bool]]
        Fold rows as ``(instancia, fold, is_ova)`` tuples.
    """
    folds_dir = tmp_path / "dataset" / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["instancia", "fold", "is_ova"])
    df.to_csv(folds_dir / "folds_clf_02.csv", index=False)


def _fake_frame(class_id: int, n_rows: int = 6) -> pd.DataFrame:
    """
    Create a deterministic in-memory frame matching 3W columns.

    Parameters
    ----------
    class_id : int
        Event class assigned to the synthetic frame.
    n_rows : int, optional
        Number of timesteps in the frame.

    Returns
    -------
    pandas.DataFrame
        Synthetic frame compatible with the loader.
    """
    values = np.arange(n_rows, dtype=np.float32)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="min"),
            "P-PDG": values,
            "P-TPT": values + 1.0,
            "T-TPT": values + 2.0,
            "P-MON-CKP": values + 3.0,
            "T-JUS-CKP": values + 4.0,
            "P-JUS-CKGL": values + 5.0,
            "T-JUS-CKGL": values + 6.0,
            "QGL": values + 7.0,
            "class": np.full(n_rows, class_id, dtype=np.int64),
        }
    )


def test_threew_loader_builds_predefined_splits_and_labels(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Verify split construction and binary target mapping.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace parquet reads with synthetic frames.
    """
    # Build a tiny synthetic 3W-like layout with fold assignments.
    _touch_instance(Path(threew_kwargs["data_dir"]), 0, "WELL-NORMAL-0001")
    _touch_instance(Path(threew_kwargs["data_dir"]), 3, "WELL-FAULT-0002")
    _touch_instance(Path(threew_kwargs["data_dir"]), 7, "WELL-FAULT-0003")
    _touch_instance(Path(threew_kwargs["data_dir"]), 5, "SIMULATED_0001")
    _write_folds(
        Path(threew_kwargs["data_dir"]),
        [
            ("0/WELL-NORMAL-0001.csv", 0, False),  # val
            ("3/WELL-FAULT-0002.csv", 1, False),  # test
            ("7/WELL-FAULT-0003.csv", 3, False),  # train
            ("5/SIMULATED_0001.csv", -1, False),  # train (simulated)
        ],
    )

    instance_by_name = {
        "WELL-NORMAL-0001": _fake_frame(0),
        "WELL-FAULT-0002": _fake_frame(3),
        "WELL-FAULT-0003": _fake_frame(7),
        "SIMULATED_0001": _fake_frame(5),
    }

    def _fake_reader(self, path: Path) -> pd.DataFrame:
        """
        Return a synthetic frame by instance stem.

        Parameters
        ----------
        path : pathlib.Path
            Source parquet path used to select a synthetic frame.

        Returns
        -------
        pandas.DataFrame
            Synthetic frame copy for the requested instance.
        """
        return instance_by_name[path.stem].copy()

    monkeypatch.setattr(ThreeWLoader, "_read_instance_frame", _fake_reader)

    loader = ThreeWLoader(**threew_kwargs)
    loader.load_data()

    data = loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert set(data.keys()) == {"features", "target", "unit_id", "event_class"}

    assert len(data["features"]["train"]) == 2
    assert len(data["features"]["val"]) == 1
    assert len(data["features"]["test"]) == 1

    # Binary anomaly target: class 0 -> 0, non-zero -> 1.
    val_target = np.asarray(data["target"]["val"][0]).reshape(-1)
    test_target = np.asarray(data["target"]["test"][0]).reshape(-1)
    assert np.all(val_target == 0.0)
    assert np.all(test_target == 1.0)

    meta = loader.get_meta_data()
    assert set(meta.keys()) >= {"unit_ids", "unit_names", "class_labels"}
    assert meta["class_labels"]["val"] == [0]
    assert meta["class_labels"]["test"] == [3]


def test_threew_loader_filters_ova_instances_when_disabled(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Filter rows marked as OVA when include_ova is false.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace parquet reads with synthetic frames.
    """
    _touch_instance(Path(threew_kwargs["data_dir"]), 0, "WELL-NORMAL-0001")
    _touch_instance(Path(threew_kwargs["data_dir"]), 2, "WELL-DHSV-0002")
    _write_folds(
        Path(threew_kwargs["data_dir"]),
        [
            ("0/WELL-NORMAL-0001.csv", 2, True),  # OVA -> excluded
            ("2/WELL-DHSV-0002.csv", 2, False),  # kept
        ],
    )

    def _fake_reader(self, path: Path) -> pd.DataFrame:
        """
        Return normal or fault synthetic frame from the filename.

        Parameters
        ----------
        path : pathlib.Path
            Source parquet path used to infer class for synthetic data.

        Returns
        -------
        pandas.DataFrame
            Synthetic frame with the inferred class label.
        """
        class_id = 0 if "NORMAL" in path.stem else 2
        return _fake_frame(class_id)

    monkeypatch.setattr(ThreeWLoader, "_read_instance_frame", _fake_reader)

    loader = ThreeWLoader(**threew_kwargs)
    loader.load_data()

    data = loader.get_data()
    assert len(data["features"]["train"]) == 1
    event_class = int(np.asarray(data["event_class"]["train"][0]).reshape(-1)[0])
    assert event_class == 2


def test_threew_loader_rejects_multisource_splitter(threew_kwargs: dict):
    """
    Reject incompatible multisource splitter configuration.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    """
    with pytest.raises(
        DatasourceConfigurationError,
        match="does not accept multisource_data_splitter",
    ):
        ThreeWLoader(**threew_kwargs, multisource_data_splitter=object())
