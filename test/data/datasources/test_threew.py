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


# ----- Scenario builders: layout on disk + optional reader contract -----


def _layout_predefined_splits_and_labels(root: Path) -> dict[str, pd.DataFrame]:
    touch_threew_instance(root, 0, "WELL-NORMAL-0001")
    touch_threew_instance(root, 3, "WELL-FAULT-0002")
    touch_threew_instance(root, 7, "WELL-FAULT-0003")
    touch_threew_instance(root, 5, "SIMULATED_0001")
    write_threew_folds(
        root,
        [
            ("0/WELL-NORMAL-0001.csv", 0, False),
            ("3/WELL-FAULT-0002.csv", 1, False),
            ("7/WELL-FAULT-0003.csv", 3, False),
            ("5/SIMULATED_0001.csv", -1, False),
        ],
    )
    return {
        "WELL-NORMAL-0001": make_threew_frame(0),
        "WELL-FAULT-0002": make_threew_frame(3),
        "WELL-FAULT-0003": make_threew_frame(7),
        "SIMULATED_0001": make_threew_frame(5),
    }


def _layout_ova_filter_two_train_candidates(root: Path) -> None:
    touch_threew_instance(root, 0, "WELL-NORMAL-0001")
    touch_threew_instance(root, 2, "WELL-DHSV-0002")
    write_threew_folds(
        root,
        [
            ("0/WELL-NORMAL-0001.csv", 2, True),
            ("2/WELL-DHSV-0002.csv", 2, False),
        ],
    )


def _reader_ova_normal_vs_fault(self: ThreeWLoader, path: Path) -> pd.DataFrame:
    class_id = 0 if "NORMAL" in path.stem else 2
    return make_threew_frame(class_id)


def _layout_single_train_fold(root: Path, class_id: int, stem: str, fold: int) -> None:
    touch_threew_instance(root, class_id, stem)
    write_threew_folds(root, [(f"{class_id}/{stem}.csv", fold, False)])


def _layout_dataset_folds_alternate_root(root: Path) -> None:
    touch_threew_instance_dataset_layout(root, 0, "WELL-NORMAL-0001")
    write_threew_folds_dataset_layout(
        root,
        [("0/WELL-NORMAL-0001.csv", 2, False)],
    )


def _layout_multi_train_timestamps(root: Path) -> None:
    touch_threew_instance(root, 2, "WELL-00012_20170320123144")
    touch_threew_instance(root, 5, "WELL-00017_20140318160220")
    write_threew_folds(
        root,
        [
            ("2/WELL-00012_20170320123144.csv", 2, False),
            ("5/WELL-00017_20140318160220.csv", 2, False),
        ],
    )


def _reader_multi_train_by_stem(self: ThreeWLoader, path: Path) -> pd.DataFrame:
    return make_threew_frame(2 if "00012" in path.stem else 5)


def _layout_nan_features_instance(root: Path) -> None:
    touch_threew_instance(root, 2, "WELL-00012_20170320123144")
    write_threew_folds(
        root,
        [("2/WELL-00012_20170320123144.csv", 2, False)],
    )


def _reader_with_nans_in_sensor_columns(self: ThreeWLoader, path: Path) -> pd.DataFrame:
    frame = make_threew_frame(2)
    frame.loc[1, "P-PDG"] = np.nan
    frame.loc[2, "QGL"] = np.nan
    return frame


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


class TestSplitAssignmentAndContainerContract:
    """Predefined fold assignment and ``SplitDatasetContainer`` keys/shapes."""

    def test_builds_predefined_splits_and_labels(
        self,
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
        root = Path(threew_kwargs["data_dir"])
        instance_by_name = _layout_predefined_splits_and_labels(root)

        def _fake_reader(self, path: Path) -> pd.DataFrame:
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

        val_target = np.asarray(data["target"]["val"][0]).reshape(-1)
        test_target = np.asarray(data["target"]["test"][0]).reshape(-1)
        assert np.all(val_target == 0.0)
        assert np.all(test_target == 1.0)

        meta = loader.get_meta_data()
        assert set(meta.keys()) >= {"unit_ids", "unit_names", "class_labels"}
        assert meta["class_labels"]["val"] == [0]
        assert meta["class_labels"]["test"] == [3]

    def test_fold_minus_one_simulated_instance_stays_in_train(
        self,
        threew_kwargs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fold ``-1`` must map to train (default ``include_simulated_train``), not be dropped."""
        root = Path(threew_kwargs["data_dir"])
        instance_by_name = _layout_predefined_splits_and_labels(root)

        def _fake_reader(self, path: Path) -> pd.DataFrame:
            return instance_by_name[path.stem].copy()

        monkeypatch.setattr(ThreeWLoader, "_read_instance_frame", _fake_reader)

        loader = ThreeWLoader(**threew_kwargs)
        loader.load_data()

        meta = loader.get_meta_data()
        train_names = meta["unit_names"]["train"]
        assert any(
            "SIMULATED_0001" in name for name in train_names
        ), "fold -1 simulated row must remain in train split"
        assert not any("SIMULATED_0001" in name for name in meta["unit_names"]["val"])
        assert not any("SIMULATED_0001" in name for name in meta["unit_names"]["test"])


class TestOvaFiltering:
    """Rows marked OVA in folds are excluded when ``include_ova`` is false."""

    def test_filters_ova_instances_when_disabled(
        self,
        threew_kwargs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        root = Path(threew_kwargs["data_dir"])
        _layout_ova_filter_two_train_candidates(root)
        monkeypatch.setattr(
            ThreeWLoader, "_read_instance_frame", _reader_ova_normal_vs_fault
        )

        loader = ThreeWLoader(**threew_kwargs)
        loader.load_data()

        data = loader.get_data()
        assert len(data["features"]["train"]) == 1
        event_class = int(np.asarray(data["event_class"]["train"][0]).reshape(-1)[0])
        assert event_class == 2


class TestTargetAndEventClassSemantics:
    """Binary targets and ``event_class`` follow folds, not frame row labels."""

    def test_task_mode_does_not_alter_fold_derived_targets(
        self,
        threew_kwargs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``task_mode`` is not validated; binary targets still follow fold class id."""
        root = Path(threew_kwargs["data_dir"])
        _layout_single_train_fold(root, 3, "WELL-X", 2)
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

    def test_anomaly_target_uses_fold_class_not_frame_class_series(
        self,
        threew_kwargs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = Path(threew_kwargs["data_dir"])
        _layout_single_train_fold(root, 5, "SIMULATED_ROWLABELS", 2)
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

    def test_event_class_is_fold_constant_not_multiclass_payload(
        self,
        threew_kwargs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = Path(threew_kwargs["data_dir"])
        _layout_single_train_fold(root, 5, "SIMULATED_FC", 2)
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


class TestFlagsAndOptionalExport:
    """Optional exports and incompatible configuration."""

    def test_get_data_omits_event_class_when_export_disabled(
        self,
        threew_kwargs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = Path(threew_kwargs["data_dir"])
        _layout_single_train_fold(root, 0, "WELL-NORMAL-0001", 2)
        monkeypatch.setattr(
            ThreeWLoader,
            "_read_instance_frame",
            lambda self, path: make_threew_frame(0),
        )
        loader = ThreeWLoader(**{**threew_kwargs, "export_event_class": False})
        loader.load_data()
        assert "event_class" not in loader.get_data()

    def test_rejects_multisource_splitter(self, threew_kwargs: dict):
        with pytest.raises(
            DatasourceConfigurationError,
            match="does not accept multisource_data_splitter",
        ):
            ThreeWLoader(**threew_kwargs, multisource_data_splitter=object())

    def test_skip_missing_instances_kwarg_does_not_skip_rows(
        self,
        threew_kwargs: dict,
    ):
        write_threew_folds(
            Path(threew_kwargs["data_dir"]),
            [("0/WELL-NORMAL-0001.csv", 2, False)],
        )
        loader = ThreeWLoader(**threew_kwargs, skip_missing_instances=True)
        with pytest.raises(FileNotFoundError, match="3W instance file not found"):
            loader.load_data()

    def test_warns_on_download_true_without_auto_download(
        self,
        threew_kwargs: dict,
        caplog: pytest.LogCaptureFixture,
    ):
        kwargs = {**threew_kwargs, "download": True}
        with caplog.at_level(logging.WARNING, logger="picid.data.datasources.threew"):
            loader = ThreeWLoader(**kwargs)
            with pytest.raises(FileNotFoundError, match="3W folds file not found"):
                loader.load_data()
        assert any(
            "Automatic download is not implemented" in r.message for r in caplog.records
        )


class TestFilesystemAndLayoutFailures:
    """Missing folds, missing instances, and stem mismatches."""

    def test_resolves_instance_root_from_folds_layout(
        self,
        threew_kwargs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        root = Path(threew_kwargs["data_dir"])
        _layout_dataset_folds_alternate_root(root)
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

    def test_raises_when_all_rows_missing(self, threew_kwargs: dict):
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

    def test_raises_when_folds_stem_has_no_matching_parquet(
        self,
        threew_kwargs: dict,
    ):
        root = Path(threew_kwargs["data_dir"])
        touch_threew_instance(root, 0, "WELL-OTHER-9999")
        write_threew_folds(
            root,
            [("0/WELL-NORMAL-0001.csv", 2, False)],
        )
        loader = ThreeWLoader(**threew_kwargs)
        with pytest.raises(FileNotFoundError, match="WELL-NORMAL-0001.parquet"):
            loader.load_data()

    def test_raises_when_folds_file_missing(self, threew_kwargs: dict):
        loader = ThreeWLoader(**threew_kwargs)
        with pytest.raises(FileNotFoundError, match="3W folds file not found"):
            loader.load_data()


class TestMetadataAndPathIdentity:
    """Metadata export and exact path resolution from folds."""

    def test_unit_name_matches_folds_instance_stem(
        self,
        threew_kwargs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        root = Path(threew_kwargs["data_dir"])
        touch_threew_instance(root, 2, "WELL-00012_20170320123144")
        write_threew_folds(
            root,
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

    def test_requires_exact_parquet_stem_no_fuzzy_match(
        self,
        threew_kwargs: dict,
    ):
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

    def test_loads_multiple_train_instances_when_paths_exist(
        self,
        threew_kwargs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        root = Path(threew_kwargs["data_dir"])
        _layout_multi_train_timestamps(root)
        monkeypatch.setattr(
            ThreeWLoader, "_read_instance_frame", _reader_multi_train_by_stem
        )
        loader = ThreeWLoader(**threew_kwargs)
        loader.load_data()
        data = loader.get_data()
        assert len(data["features"]["train"]) == 2


class TestRawFeaturePayload:
    """Raw numeric features preserve NaNs."""

    def test_preserves_feature_nans_in_raw_payload(
        self,
        threew_kwargs: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        root = Path(threew_kwargs["data_dir"])
        _layout_nan_features_instance(root)
        monkeypatch.setattr(
            ThreeWLoader, "_read_instance_frame", _reader_with_nans_in_sensor_columns
        )
        loader = ThreeWLoader(**threew_kwargs)
        loader.load_data()
        features = np.asarray(loader.get_data()["features"]["train"][0])
        assert np.isnan(features[1, 0])
        assert np.isnan(features[2, 7])


class TestSplitContainerFlattenHelper:
    """Local helper stays aligned with ``SplitDatasetContainer`` metadata layout."""

    def test_flatten_records_uses_unit_metadata_from_split_container(self) -> None:
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
