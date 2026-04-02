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
    class_dir = tmp_path / "threew_dataset" / str(class_id)
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
    folds_dir = tmp_path / "threew_dataset" / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["instancia", "fold", "is_ova"])
    df.to_csv(folds_dir / "folds_clf_02.csv", index=False)


def _write_folds_dataset_layout(
    tmp_path: Path, rows: list[tuple[str, int, bool]]
) -> None:
    """
    Write folds under ``dataset/folds/`` (upstream 3W repo layout, no ``threew_dataset``).

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


def _touch_instance_dataset_layout(
    tmp_path: Path, class_id: int, instance_name: str
) -> Path:
    """
    Create a parquet placeholder under ``dataset/<class_id>/`` (not ``threew_dataset``).

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


def _fake_frame_with_class_series(class_ids: list[int] | np.ndarray) -> pd.DataFrame:
    """
    Build a synthetic frame with a per-row ``class`` column (varying labels).

    Parameters
    ----------
    class_ids : list[int] or numpy.ndarray
        Event class id per timestep.

    Returns
    -------
    pandas.DataFrame
        Frame matching 3W columns with non-constant ``class``.
    """
    series = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    n_rows = int(series.shape[0])
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
            "class": series,
        }
    )


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


def _flatten_records_for_test(
    container: dict[str, dict[str, list[np.ndarray]]],
) -> list[dict[str, object]]:
    """
    Flatten split payloads for local assertions in tests.

    Parameters
    ----------
    container : dict[str, dict[str, list[numpy.ndarray]]]
        Split container-like payload with features, target, and optional metadata.

    Returns
    -------
    list[dict[str, object]]
        Flat records with split name, unit index, features, target, and metadata.
    """
    records: list[dict[str, object]] = []
    split_metadata = getattr(container, "unit_metadata", None)
    if not isinstance(split_metadata, dict):
        split_metadata = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        features_list = container.get("features", {}).get(split, [])
        metadata_list = split_metadata.get(split, [])
        target_list = container.get("target", {}).get(split, [])
        for idx, (features, metadata, target) in enumerate(
            zip(features_list, metadata_list, target_list)
        ):
            records.append(
                {
                    "split": split,
                    "unit_idx": idx,
                    "features": np.asarray(features),
                    "target": np.asarray(target),
                    "metadata": metadata,
                }
            )
    return records


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


def test_threew_loader_rejects_unsupported_task_mode(threew_kwargs: dict) -> None:
    """
    Reject unsupported ``task_mode`` with an explicit supported-modes hint.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    """
    with pytest.raises(ValueError, match="does not support task_mode=") as exc_info:
        ThreeWLoader(**{**threew_kwargs, "task_mode": "classification"})
    assert "anomaly_detection" in str(exc_info.value)


def test_threew_loader_uses_row_level_labels_for_anomaly_target(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Use per-timestep ``class`` values to build binary anomaly targets.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace parquet reads with synthetic frames.
    """
    root = Path(threew_kwargs["data_dir"])
    _touch_instance(root, 5, "SIMULATED_ROWLABELS")
    _write_folds(
        root,
        [("5/SIMULATED_ROWLABELS.csv", 2, False)],
    )

    frame = _fake_frame_with_class_series([0, 0, 2, 2, 0, 2])

    monkeypatch.setattr(
        ThreeWLoader,
        "_read_instance_frame",
        lambda self, path: frame.copy(),
    )

    loader = ThreeWLoader(**threew_kwargs)
    loader.load_data()
    data = loader.get_data()
    target = np.asarray(data["target"]["train"][0]).reshape(-1)
    assert target.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_threew_loader_supports_fault_classification_task_mode(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Export per-timestep multiclass labels in ``fault_classification`` mode.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace parquet reads with synthetic frames.
    """
    root = Path(threew_kwargs["data_dir"])
    _touch_instance(root, 5, "SIMULATED_FC")
    _write_folds(
        root,
        [("5/SIMULATED_FC.csv", 2, False)],
    )
    row_classes = [0, 2, 5, 0, 2, 5]
    frame = _fake_frame_with_class_series(row_classes)

    monkeypatch.setattr(
        ThreeWLoader,
        "_read_instance_frame",
        lambda self, path: frame.copy(),
    )

    kwargs = {**threew_kwargs, "task_mode": "fault_classification"}
    assert kwargs.get("export_event_class") is True
    loader = ThreeWLoader(**kwargs)
    loader.load_data()
    data = loader.get_data()
    assert "fault_classification" in data
    fc = np.asarray(data["fault_classification"]["train"][0]).reshape(-1)
    assert np.array_equal(fc, np.array(row_classes, dtype=np.int64))


def test_threew_loader_exposes_dual_task_modes(threew_kwargs: dict) -> None:
    """
    Expose anomaly and fault-classification task modes.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    """
    loader = ThreeWLoader(**threew_kwargs)
    expected_modes = {"anomaly_detection", "fault_classification"}
    assert set(loader.get_available_task_modes()) == expected_modes
    assert set(ThreeWLoader.SUPPORTED_TASK_MODES) == expected_modes


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


def test_threew_loader_resolves_instance_root_from_folds_layout(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Resolve instances from ``dataset/<class>`` when folds live in ``dataset/folds``.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace parquet reads with synthetic frames.
    """
    root = Path(threew_kwargs["data_dir"])
    _touch_instance_dataset_layout(root, 0, "WELL-NORMAL-0001")
    _write_folds_dataset_layout(
        root,
        [("0/WELL-NORMAL-0001.csv", 2, False)],
    )
    kwargs = {
        **threew_kwargs,
        "folds_file": "dataset/folds/folds_clf_02.csv",
    }
    monkeypatch.setattr(
        ThreeWLoader,
        "_read_instance_frame",
        lambda self, path: _fake_frame(0),
    )
    loader = ThreeWLoader(**kwargs)
    loader.load_data()
    data = loader.get_data()
    assert len(data["features"]["train"]) == 1


def test_threew_loader_raises_when_all_rows_missing(threew_kwargs: dict):
    """
    Raise an actionable error when folds exist but all listed rows are missing.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    """
    _write_folds(
        Path(threew_kwargs["data_dir"]),
        [("0/WELL-NORMAL-0001.csv", 2, False)],
    )
    loader = ThreeWLoader(**threew_kwargs)
    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load_data()
    msg = str(exc_info.value)
    assert "dataset/<class>/*.parquet" in msg
    assert "threew_dataset/<class>/*.parquet" in msg


def test_threew_loader_raises_when_folds_reference_only_missing_instances(
    threew_kwargs: dict,
):
    """
    Raise when folds reference only missing instances after row filtering.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    """
    root = Path(threew_kwargs["data_dir"])
    _touch_instance(root, 0, "WELL-OTHER-9999")
    _write_folds(
        root,
        [("0/WELL-NORMAL-0001.csv", 2, False)],
    )
    loader = ThreeWLoader(**threew_kwargs, skip_missing_instances=True)
    with pytest.raises(FileNotFoundError, match="no instance parquet"):
        loader.load_data()


def test_threew_loader_raises_clear_error_when_data_missing_and_download_disabled(
    threew_kwargs: dict,
):
    """
    Fail with an actionable message when folds are missing and download is off.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    """
    loader = ThreeWLoader(**threew_kwargs)
    with pytest.raises(FileNotFoundError, match="Set download=true"):
        loader.load_data()


def test_threew_loader_downloads_dataset_when_missing(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Trigger auto-download once when folds are missing and download is enabled.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace download and parquet reads.
    """
    kwargs = {**threew_kwargs, "download": True}
    download_calls = {"count": 0}

    def _fake_download(self) -> None:
        """Create a minimal dataset layout during mocked download."""
        download_calls["count"] += 1
        _touch_instance(Path(self.data_path), 0, "WELL-NORMAL-0001")
        _write_folds(
            Path(self.data_path),
            [("0/WELL-NORMAL-0001.csv", 2, False)],
        )

    monkeypatch.setattr(ThreeWLoader, "_download_missing_dataset", _fake_download)
    monkeypatch.setattr(
        ThreeWLoader,
        "_read_instance_frame",
        lambda self, path: _fake_frame(0),
    )

    loader = ThreeWLoader(**kwargs)
    loader.load_data()

    assert download_calls["count"] == 1
    data = loader.get_data()
    assert len(data["features"]["train"]) == 1


def test_threew_loader_resolves_shifted_well_timestamps(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Resolve missing WELL timestamp stems to nearest local timestamp variant.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace parquet reads with synthetic frames.
    """
    _touch_instance(Path(threew_kwargs["data_dir"]), 2, "WELL-00012_20170320123144")
    _write_folds(
        Path(threew_kwargs["data_dir"]),
        [("2/WELL-00012_20170320143144.csv", 2, False)],
    )
    monkeypatch.setattr(
        ThreeWLoader,
        "_read_instance_frame",
        lambda self, path: _fake_frame(2),
    )

    loader = ThreeWLoader(**threew_kwargs)
    loader.load_data()
    data = loader.get_data()
    assert len(data["features"]["train"]) == 1
    meta = loader.get_meta_data()
    assert meta["unit_names"]["train"][0] == "2/WELL-00012_20170320123144"


def test_threew_loader_skips_unresolvable_rows_when_enabled(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Skip rows with missing files when no exact/fallback parquet exists.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace parquet reads with synthetic frames.
    """
    _touch_instance(Path(threew_kwargs["data_dir"]), 2, "WELL-00012_20170320123144")
    _write_folds(
        Path(threew_kwargs["data_dir"]),
        [
            ("2/WELL-00012_20170320143144.csv", 2, False),
            ("5/WELL-00017_20140318160220.csv", 2, False),
        ],
    )
    monkeypatch.setattr(
        ThreeWLoader,
        "_read_instance_frame",
        lambda self, path: _fake_frame(2),
    )

    loader = ThreeWLoader(**threew_kwargs, skip_missing_instances=True)
    loader.load_data()
    data = loader.get_data()
    assert len(data["features"]["train"]) == 1


def test_threew_loader_imputes_feature_nans(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Replace NaNs in numeric feature payload before transform checks run.

    Parameters
    ----------
    threew_kwargs : dict
        Baseline loader kwargs fixture.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace parquet reads with synthetic frames.
    """
    _touch_instance(Path(threew_kwargs["data_dir"]), 2, "WELL-00012_20170320123144")
    _write_folds(
        Path(threew_kwargs["data_dir"]),
        [("2/WELL-00012_20170320123144.csv", 2, False)],
    )

    def _nan_frame(self, path: Path) -> pd.DataFrame:
        """
        Return synthetic frame containing NaNs in feature columns.

        Parameters
        ----------
        path : pathlib.Path
            Unused parquet path argument required by loader API.

        Returns
        -------
        pandas.DataFrame
            Synthetic frame with NaNs for imputation coverage.
        """
        frame = _fake_frame(2)
        frame.loc[1, "P-PDG"] = np.nan
        frame.loc[2, "QGL"] = np.nan
        return frame

    monkeypatch.setattr(ThreeWLoader, "_read_instance_frame", _nan_frame)
    loader = ThreeWLoader(**threew_kwargs)
    loader.load_data()
    features = np.asarray(loader.get_data()["features"]["train"][0])
    assert not np.isnan(features).any()


def test_flatten_records_uses_unit_metadata_from_split_container() -> None:
    """
    Keep tutorial flattening compatible with SplitDatasetContainer metadata layout.
    """
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target = np.zeros((2, 1, 1), dtype=np.float32)
    container = SplitDatasetContainer(
        features={"train": [features], "val": [], "test": []},
        target={"train": [target], "val": [], "test": []},
        unit_id={"train": [np.array([0], dtype=np.int64)], "val": [], "test": []},
        unit_metadata={
            "train": [
                {
                    "unit_name": "2/WELL-00012_20170320123144",
                    "class_label": 2,
                    "feature_columns": ["P-PDG", "P-TPT"],
                }
            ],
            "val": [],
            "test": [],
        },
        metadata={"column_map": {}},
    )

    records = _flatten_records_for_test(container)

    assert len(records) == 1
    assert records[0]["split"] == "train"
    assert records[0]["metadata"]["class_label"] == 2
    assert records[0]["metadata"]["unit_name"] == "2/WELL-00012_20170320123144"
