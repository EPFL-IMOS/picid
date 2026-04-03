"""Characterization tests for ThreeWLoader."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from picid.data.data_objects import SplitDatasetContainer
from picid.data.datasources.base.exceptions import DatasourceConfigurationError
from picid.data.datasources.threew import ThreeWLoader
from test.fixtures.datasource_layouts import (
    make_threew_frame,
    make_threew_frame_with_class_series,
    touch_threew_instance,
    touch_threew_instance_dataset_layout,
    write_threew_folds,
    write_threew_folds_dataset_layout,
)


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
    touch_threew_instance(Path(threew_kwargs["data_dir"]), 0, "WELL-NORMAL-0001")
    touch_threew_instance(Path(threew_kwargs["data_dir"]), 3, "WELL-FAULT-0002")
    touch_threew_instance(Path(threew_kwargs["data_dir"]), 7, "WELL-FAULT-0003")
    touch_threew_instance(Path(threew_kwargs["data_dir"]), 5, "SIMULATED_0001")
    write_threew_folds(
        Path(threew_kwargs["data_dir"]),
        [
            ("0/WELL-NORMAL-0001.csv", 0, False),  # val
            ("3/WELL-FAULT-0002.csv", 1, False),  # test
            ("7/WELL-FAULT-0003.csv", 3, False),  # train
            ("5/SIMULATED_0001.csv", -1, False),  # train (simulated)
        ],
    )

    instance_by_name = {
        "WELL-NORMAL-0001": make_threew_frame(0),
        "WELL-FAULT-0002": make_threew_frame(3),
        "WELL-FAULT-0003": make_threew_frame(7),
        "SIMULATED_0001": make_threew_frame(5),
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
    touch_threew_instance(Path(threew_kwargs["data_dir"]), 0, "WELL-NORMAL-0001")
    touch_threew_instance(Path(threew_kwargs["data_dir"]), 2, "WELL-DHSV-0002")
    write_threew_folds(
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
        return make_threew_frame(class_id)

    monkeypatch.setattr(ThreeWLoader, "_read_instance_frame", _fake_reader)

    loader = ThreeWLoader(**threew_kwargs)
    loader.load_data()

    data = loader.get_data()
    assert len(data["features"]["train"]) == 1
    event_class = int(np.asarray(data["event_class"]["train"][0]).reshape(-1)[0])
    assert event_class == 2


def test_threew_loader_accepts_task_mode_without_altering_fold_derived_targets(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``task_mode`` is not validated by ThreeWLoader; binary targets still follow
    the class id parsed from the folds path.
    """
    root = Path(threew_kwargs["data_dir"])
    touch_threew_instance(root, 3, "WELL-X")
    write_threew_folds(root, [("3/WELL-X.csv", 2, False)])
    monkeypatch.setattr(
        ThreeWLoader,
        "_read_instance_frame",
        lambda self, path: make_threew_frame(3),
    )
    loader = ThreeWLoader(**{**threew_kwargs, "task_mode": "classification"})
    loader.load_data()
    assert loader.task_mode == "classification"
    target = np.asarray(loader.get_data()["target"]["train"][0]).reshape(-1)
    assert np.all(target == 1.0)


def test_threew_loader_anomaly_target_uses_fold_class_not_frame_class_series(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Binary targets are constant from the folds directory class id, not from
    per-row ``class`` values in the frame.
    """
    root = Path(threew_kwargs["data_dir"])
    touch_threew_instance(root, 5, "SIMULATED_ROWLABELS")
    write_threew_folds(
        root,
        [("5/SIMULATED_ROWLABELS.csv", 2, False)],
    )

    frame = make_threew_frame_with_class_series([0, 0, 2, 2, 0, 2])

    monkeypatch.setattr(
        ThreeWLoader,
        "_read_instance_frame",
        lambda self, path: frame.copy(),
    )

    loader = ThreeWLoader(**threew_kwargs)
    loader.load_data()
    data = loader.get_data()
    target = np.asarray(data["target"]["train"][0]).reshape(-1)
    assert np.all(target == 1.0)
    event = np.asarray(data["event_class"]["train"][0]).reshape(-1)
    assert np.all(event == 5)


def test_threew_loader_event_class_is_fold_constant_not_multiclass_payload(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    There is no ``fault_classification`` tensor; optional ``event_class`` is
    the fold class id broadcast to every timestep (frame ``class`` ignored).
    """
    root = Path(threew_kwargs["data_dir"])
    touch_threew_instance(root, 5, "SIMULATED_FC")
    write_threew_folds(
        root,
        [("5/SIMULATED_FC.csv", 2, False)],
    )
    row_classes = [0, 2, 5, 0, 2, 5]
    frame = make_threew_frame_with_class_series(row_classes)

    monkeypatch.setattr(
        ThreeWLoader,
        "_read_instance_frame",
        lambda self, path: frame.copy(),
    )

    kwargs = {**threew_kwargs, "task_mode": "fault_classification"}
    loader = ThreeWLoader(**kwargs)
    loader.load_data()
    data = loader.get_data()
    assert "fault_classification" not in data
    ec = np.asarray(data["event_class"]["train"][0]).reshape(-1)
    assert np.all(ec == 5)


def test_threew_loader_get_data_omits_event_class_when_export_disabled(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``export_event_class`` is false, the container has no ``event_class`` key."""
    root = Path(threew_kwargs["data_dir"])
    touch_threew_instance(root, 0, "WELL-NORMAL-0001")
    write_threew_folds(
        root,
        [("0/WELL-NORMAL-0001.csv", 2, False)],
    )
    monkeypatch.setattr(
        ThreeWLoader,
        "_read_instance_frame",
        lambda self, path: make_threew_frame(0),
    )
    loader = ThreeWLoader(**{**threew_kwargs, "export_event_class": False})
    loader.load_data()
    assert "event_class" not in loader.get_data()


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
    touch_threew_instance_dataset_layout(root, 0, "WELL-NORMAL-0001")
    write_threew_folds_dataset_layout(
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
        lambda self, path: make_threew_frame(0),
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
    write_threew_folds(
        Path(threew_kwargs["data_dir"]),
        [("0/WELL-NORMAL-0001.csv", 2, False)],
    )
    loader = ThreeWLoader(**threew_kwargs)
    with pytest.raises(
        FileNotFoundError, match="3W instance file not found"
    ) as exc_info:
        loader.load_data()
    msg = str(exc_info.value)
    assert "dataset/0/WELL-NORMAL-0001.parquet" in msg


def test_threew_loader_raises_when_folds_stem_has_no_matching_parquet(
    threew_kwargs: dict,
):
    """
    Exact stem from folds must exist under ``dataset/<class>/``; a different
    file in the same class directory does not satisfy the reference.
    """
    root = Path(threew_kwargs["data_dir"])
    touch_threew_instance(root, 0, "WELL-OTHER-9999")
    write_threew_folds(
        root,
        [("0/WELL-NORMAL-0001.csv", 2, False)],
    )
    loader = ThreeWLoader(**threew_kwargs)
    with pytest.raises(FileNotFoundError, match="WELL-NORMAL-0001.parquet"):
        loader.load_data()


def test_threew_loader_raises_when_folds_file_missing(threew_kwargs: dict):
    """Missing default folds CSV raises ``FileNotFoundError`` (no download hint)."""
    loader = ThreeWLoader(**threew_kwargs)
    with pytest.raises(FileNotFoundError, match="3W folds file not found"):
        loader.load_data()


def test_threew_loader_warns_on_download_true_without_auto_download(
    threew_kwargs: dict,
    caplog: pytest.LogCaptureFixture,
):
    """
    ``download=True`` only logs that auto-download is unimplemented; loading
    still requires a local folds file.
    """
    kwargs = {**threew_kwargs, "download": True}
    with caplog.at_level(logging.WARNING, logger="picid.data.datasources.threew"):
        loader = ThreeWLoader(**kwargs)
        with pytest.raises(FileNotFoundError, match="3W folds file not found"):
            loader.load_data()
    assert any(
        "Automatic download is not implemented" in r.message for r in caplog.records
    )


def test_threew_loader_unit_name_matches_folds_instance_stem(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """``unit_name`` in metadata uses the class id and stem from the folds CSV."""
    touch_threew_instance(
        Path(threew_kwargs["data_dir"]), 2, "WELL-00012_20170320123144"
    )
    write_threew_folds(
        Path(threew_kwargs["data_dir"]),
        [("2/WELL-00012_20170320123144.csv", 2, False)],
    )
    monkeypatch.setattr(
        ThreeWLoader,
        "_read_instance_frame",
        lambda self, path: make_threew_frame(2),
    )

    loader = ThreeWLoader(**threew_kwargs)
    loader.load_data()
    data = loader.get_data()
    assert len(data["features"]["train"]) == 1
    meta = loader.get_meta_data()
    assert meta["unit_names"]["train"][0] == "2/WELL-00012_20170320123144"


def test_threew_loader_requires_exact_parquet_stem_no_fuzzy_match(
    threew_kwargs: dict,
):
    """
    Loader resolves ``dataset/<class>/<stem>.parquet`` from the folds entry only;
    there is no fallback to a different timestamp stem on disk.
    """
    touch_threew_instance(
        Path(threew_kwargs["data_dir"]), 2, "WELL-00012_20170320123144"
    )
    write_threew_folds(
        Path(threew_kwargs["data_dir"]),
        [("2/WELL-00012_20170320143144.csv", 2, False)],
    )
    loader = ThreeWLoader(**threew_kwargs)
    with pytest.raises(FileNotFoundError, match="20170320143144"):
        loader.load_data()


def test_threew_loader_skip_missing_instances_kwarg_does_not_skip_rows(
    threew_kwargs: dict,
):
    """
    Compatibility kwargs such as ``skip_missing_instances`` are not consumed by
    ThreeWLoader; missing files still fail load.
    """
    write_threew_folds(
        Path(threew_kwargs["data_dir"]),
        [("0/WELL-NORMAL-0001.csv", 2, False)],
    )
    loader = ThreeWLoader(**threew_kwargs, skip_missing_instances=True)
    with pytest.raises(FileNotFoundError, match="3W instance file not found"):
        loader.load_data()


def test_threew_loader_loads_multiple_train_instances_when_paths_exist(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """Each folds row with a resolvable parquet becomes one train unit (no skipping)."""
    root = Path(threew_kwargs["data_dir"])
    touch_threew_instance(root, 2, "WELL-00012_20170320123144")
    touch_threew_instance(root, 5, "WELL-00017_20140318160220")
    write_threew_folds(
        root,
        [
            ("2/WELL-00012_20170320123144.csv", 2, False),
            ("5/WELL-00017_20140318160220.csv", 2, False),
        ],
    )
    monkeypatch.setattr(
        ThreeWLoader,
        "_read_instance_frame",
        lambda self, path: make_threew_frame(2 if "00012" in path.stem else 5),
    )

    loader = ThreeWLoader(**threew_kwargs)
    loader.load_data()
    data = loader.get_data()
    assert len(data["features"]["train"]) == 2


def test_threew_loader_preserves_feature_nans_in_raw_payload(
    threew_kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """Numeric features are taken from the frame with ``to_numpy``; NaNs are kept."""
    touch_threew_instance(
        Path(threew_kwargs["data_dir"]), 2, "WELL-00012_20170320123144"
    )
    write_threew_folds(
        Path(threew_kwargs["data_dir"]),
        [("2/WELL-00012_20170320123144.csv", 2, False)],
    )

    def _nan_frame(self, path: Path) -> pd.DataFrame:
        frame = make_threew_frame(2)
        frame.loc[1, "P-PDG"] = np.nan
        frame.loc[2, "QGL"] = np.nan
        return frame

    monkeypatch.setattr(ThreeWLoader, "_read_instance_frame", _nan_frame)
    loader = ThreeWLoader(**threew_kwargs)
    loader.load_data()
    features = np.asarray(loader.get_data()["features"]["train"][0])
    assert np.isnan(features[1, 0])
    assert np.isnan(features[2, 7])


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
