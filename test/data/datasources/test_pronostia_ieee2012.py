"""Tests for the raw IEEE 2012 PRONOSTIA datasource."""

from __future__ import annotations

import hydra
from pathlib import Path
import subprocess
import tempfile

import awkward as ak
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import pytest

import picid.data.datasources.pronostia_ieee2012 as pronostia_ieee2012_module
from picid.data.datasources.pronostia_ieee2012 import (
    PRONOSTIA_TOTAL_LIFE_SECONDS,
    TEST_RULS,
    UNIT_NAMES_TO_ID,
)


SAMPLE_RANGE_SIZE = 4
CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
LEARNING_UNITS = ("1_1", "1_2", "2_1", "2_2", "3_1", "3_2")
FULL_TEST_UNITS = (
    "1_3",
    "1_4",
    "1_5",
    "1_6",
    "1_7",
    "2_3",
    "2_4",
    "2_5",
    "2_6",
    "2_7",
    "3_3",
)
PRONOSTIA_IN_DOMAIN_FOLDS = {
    "fold_1": {
        "train": ["1_4", "1_5", "1_6", "1_7", "2_4", "2_5", "2_6", "2_7", "3_1"],
        "val": ["1_1", "2_1", "3_2"],
        "test": ["1_2", "1_3", "2_2", "2_3", "3_3"],
    },
    "fold_2": {
        "train": ["1_1", "1_5", "1_6", "1_7", "2_1", "2_5", "2_6", "2_7", "3_2"],
        "val": ["1_2", "2_2", "3_3"],
        "test": ["1_3", "1_4", "2_3", "2_4", "3_1"],
    },
    "fold_3": {
        "train": ["1_1", "1_2", "1_6", "1_7", "2_1", "2_2", "2_6", "2_7", "3_3"],
        "val": ["1_3", "2_3", "3_1"],
        "test": ["1_4", "1_5", "2_4", "2_5", "3_2"],
    },
    "fold_4": {
        "train": ["1_1", "1_2", "1_3", "1_7", "2_1", "2_2", "2_3", "2_7", "3_1"],
        "val": ["1_4", "2_4", "3_2"],
        "test": ["1_5", "1_6", "2_5", "2_6", "3_3"],
    },
    "fold_5": {
        "train": ["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "2_4", "3_2"],
        "val": ["1_5", "2_5", "3_3"],
        "test": ["1_6", "1_7", "2_6", "2_7", "3_1"],
    },
}
PRONOSTIA_DOMAIN_SHIFT_FOLDS = {
    "fold_1": {
        "train": ["1_3", "1_4", "1_5", "1_6", "1_7", "2_3", "2_4", "2_5", "2_6", "2_7"],
        "val": ["1_1", "1_2", "2_1", "2_2"],
        "test": ["3_1", "3_2", "3_3"],
    },
    "fold_2": {
        "train": ["1_1", "1_4", "1_5", "1_6", "1_7", "2_1", "2_4", "2_5", "2_6", "2_7"],
        "val": ["1_2", "1_3", "2_2", "2_3"],
        "test": ["3_1", "3_2", "3_3"],
    },
    "fold_3": {
        "train": ["1_1", "1_2", "1_5", "1_6", "1_7", "2_1", "2_2", "2_5", "2_6", "2_7"],
        "val": ["1_3", "1_4", "2_3", "2_4"],
        "test": ["3_1", "3_2", "3_3"],
    },
    "fold_4": {
        "train": ["1_1", "1_2", "1_3", "1_6", "1_7", "2_1", "2_2", "2_3", "2_6", "2_7"],
        "val": ["1_4", "1_5", "2_4", "2_5"],
        "test": ["3_1", "3_2", "3_3"],
    },
    "fold_5": {
        "train": ["1_1", "1_2", "1_3", "1_4", "1_7", "2_1", "2_2", "2_3", "2_4", "2_7"],
        "val": ["1_5", "1_6", "2_5", "2_6"],
        "test": ["3_1", "3_2", "3_3"],
    },
}


def _write_acc_file(path: Path, value: float, sample_range_size: int) -> None:
    signal = np.column_stack(
        [
            np.full(sample_range_size, value, dtype=np.float32),
            np.full(sample_range_size, value + 0.5, dtype=np.float32),
        ]
    )
    np.savetxt(path, signal, delimiter=",", fmt="%.3f")


def _write_six_column_acc_file(
    path: Path,
    *,
    marker_a: float,
    marker_b: float,
    value: float,
    sample_range_size: int,
    delimiter: str = ",",
) -> None:
    signal = np.column_stack(
        [
            np.full(sample_range_size, marker_a, dtype=np.float32),
            np.full(sample_range_size, marker_b, dtype=np.float32),
            np.full(sample_range_size, marker_b + 10.0, dtype=np.float32),
            np.arange(sample_range_size, dtype=np.float32),
            np.full(sample_range_size, value, dtype=np.float32),
            np.full(sample_range_size, value + 0.5, dtype=np.float32),
        ]
    )
    np.savetxt(path, signal, delimiter=delimiter, fmt="%.3f")


def _unit_subset(unit_key: str) -> str:
    return "Learning_set" if unit_key in LEARNING_UNITS else "Full_Test_Set"


def _create_unit_tree(
    root: Path,
    unit_key: str,
    *,
    file_count: int,
    sample_range_size: int = SAMPLE_RANGE_SIZE,
) -> None:
    (root / "Learning_set").mkdir(parents=True, exist_ok=True)
    (root / "Full_Test_Set").mkdir(parents=True, exist_ok=True)
    unit_dir = root / _unit_subset(unit_key) / f"Bearing{unit_key}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(file_count):
        _write_acc_file(unit_dir / f"acc_{idx:05d}.csv", float(idx), sample_range_size)


def _create_complete_dataset_tree(
    root: Path, *, sample_range_size: int = SAMPLE_RANGE_SIZE
) -> None:
    for unit_key in LEARNING_UNITS + FULL_TEST_UNITS:
        _create_unit_tree(
            root,
            unit_key,
            file_count=1,
            sample_range_size=sample_range_size,
        )


def _make_loader(data_dir: Path, **overrides):
    from picid.data.datasources.pronostia_ieee2012 import PronostiaIEEE2012Loader

    kwargs = {
        "data_name": "pronostia_ieee2012_test",
        "task_mode": "rul",
        "data_dir": str(data_dir),
        "split_assignments": {"train": [], "val": [], "test": []},
        "sample_range_size": SAMPLE_RANGE_SIZE,
    }
    kwargs.update(overrides)
    return PronostiaIEEE2012Loader(**kwargs)


@pytest.fixture(autouse=True)
def hydra_initialized():
    """Re-initialize Hydra for each test so compose() is order-independent."""

    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS_DIR)):
        yield
    GlobalHydra.instance().clear()


def _assert_disjoint_and_cover_all(
    split_table: dict[str, list[str]], expected_units: set[str]
) -> None:
    seen: set[str] = set()
    for split_name in ("train", "val", "test"):
        split_units = set(split_table[split_name])
        assert split_units.isdisjoint(seen), f"{split_name} overlaps prior splits"
        seen |= split_units
    assert seen == expected_units


def test_pronostia_ieee2012_loader_reads_sorted_two_channel_windows_and_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "Full_Test_Set").mkdir(parents=True, exist_ok=True)
    unit_dir = tmp_path / "Learning_set" / "Bearing1_1"
    unit_dir.mkdir(parents=True)
    _write_acc_file(unit_dir / "acc_00002.csv", 2.0, SAMPLE_RANGE_SIZE)
    _write_acc_file(unit_dir / "acc_00000.csv", 0.0, SAMPLE_RANGE_SIZE)
    _write_acc_file(unit_dir / "acc_00001.csv", 1.0, SAMPLE_RANGE_SIZE)

    loader = _make_loader(
        tmp_path,
        split_assignments={"train": ["1_1"], "val": [], "test": []},
    )

    loader.load_data()
    data = loader.get_data()

    features = ak.to_numpy(data["features"]["train"][0])
    target = ak.to_numpy(data["target"]["train"][0])
    metadata = data.unit_metadata["train"][0]

    assert features.shape == (2, SAMPLE_RANGE_SIZE, 2)
    assert np.allclose(features[0, :, 0], 1.5)
    assert np.allclose(features[0, :, 1], 1.0)
    assert np.allclose(features[1, :, 0], 2.5)
    assert np.allclose(features[1, :, 1], 2.0)

    assert target.shape == (2, SAMPLE_RANGE_SIZE, 1)
    assert np.allclose(target[:, 0, 0], [28010.0, 28000.0])
    assert np.allclose(target[:, -1, 0], [28010.0, 28000.0])

    assert metadata["unit_name"] == "Bearing 1_1"
    assert metadata["unit_id"] == UNIT_NAMES_TO_ID["1_1"]
    assert metadata["unit_length"] == 2
    assert metadata["dataset_subset"] == "Learning_set"


def test_pronostia_ieee2012_loader_supports_split_mode_without_false_choice_flags(
    tmp_path: Path,
) -> None:
    (tmp_path / "Full_Test_Set").mkdir(parents=True, exist_ok=True)
    unit_dir = tmp_path / "Learning_set" / "Bearing1_1"
    unit_dir.mkdir(parents=True)
    _write_acc_file(unit_dir / "acc_00001.csv", 0.0, SAMPLE_RANGE_SIZE)
    _write_acc_file(unit_dir / "acc_00002.csv", 1.0, SAMPLE_RANGE_SIZE)

    from picid.data.datasources.pronostia_ieee2012 import PronostiaIEEE2012Loader

    loader = PronostiaIEEE2012Loader(
        data_name="pronostia_ieee2012_split_mode_test",
        task_mode="rul",
        data_dir=str(tmp_path),
        split_mode="in_domain",
        fold_id=3,
        sample_range_size=SAMPLE_RANGE_SIZE,
        split_assignments_by_mode={
            "in_domain": {
                "fold_1": {"train": [], "val": ["1_1"], "test": []},
                "fold_3": {"train": ["1_1"], "val": [], "test": []},
            },
            "domain_shift": {
                "fold_1": {"train": [], "val": ["1_1"], "test": []},
            },
        },
    )

    loader.load_data()
    data = loader.get_data()

    assert len(data["features"]["train"]) == 1
    assert loader.split_assignments == {"train": ["1_1"], "val": [], "test": []}
    assert not hasattr(loader, "use_ragged")
    assert not hasattr(loader, "include_temperature")
    assert not hasattr(loader, "derive_full_test_rul")
    assert not hasattr(loader, "split_assignments_by_mode")


def test_pronostia_ieee2012_loader_resolves_exact_fold_tables() -> None:
    from picid.data.datasources.pronostia_ieee2012 import PronostiaIEEE2012Loader

    loader = PronostiaIEEE2012Loader(
        data_name="pronostia_ieee2012_split_resolution_test",
        task_mode="rul",
        data_dir="/tmp/unused_pronostia",
        split_mode="domain_shift",
        fold_id=4,
        split_assignments_by_mode={
            "in_domain": PRONOSTIA_IN_DOMAIN_FOLDS,
            "domain_shift": PRONOSTIA_DOMAIN_SHIFT_FOLDS,
        },
        download_if_missing=False,
        sample_range_size=SAMPLE_RANGE_SIZE,
    )

    assert loader.split_assignments == PRONOSTIA_DOMAIN_SHIFT_FOLDS["fold_4"]


def test_pronostia_cv_tables_match_expected_structure() -> None:
    all_units = set(UNIT_NAMES_TO_ID)
    source_units = {unit for unit in UNIT_NAMES_TO_ID if not unit.startswith("3_")}
    target_units = {"3_1", "3_2", "3_3"}

    for split_table in PRONOSTIA_IN_DOMAIN_FOLDS.values():
        _assert_disjoint_and_cover_all(split_table, all_units)
        assert len(split_table["train"]) == 9
        assert len(split_table["val"]) == 3
        assert len(split_table["test"]) == 5
        assert sum(unit.startswith("3_") for unit in split_table["train"]) == 1
        assert sum(unit.startswith("3_") for unit in split_table["val"]) == 1
        assert sum(unit.startswith("3_") for unit in split_table["test"]) == 1

    for split_table in PRONOSTIA_DOMAIN_SHIFT_FOLDS.values():
        _assert_disjoint_and_cover_all(split_table, source_units | target_units)
        assert set(split_table["test"]) == target_units
        assert set(split_table["train"]).issubset(source_units)
        assert set(split_table["val"]).issubset(source_units)
        assert len(split_table["train"]) == 10
        assert len(split_table["val"]) == 4
        assert len(split_table["test"]) == 3


def test_pronostia_ieee2012_loader_uses_last_two_columns_from_raw_acc_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "Full_Test_Set").mkdir(parents=True, exist_ok=True)
    unit_dir = tmp_path / "Learning_set" / "Bearing1_2"
    unit_dir.mkdir(parents=True)
    _write_six_column_acc_file(
        unit_dir / "acc_00001.csv",
        marker_a=8.0,
        marker_b=47.0,
        value=1.25,
        sample_range_size=SAMPLE_RANGE_SIZE,
    )
    _write_six_column_acc_file(
        unit_dir / "acc_00002.csv",
        marker_a=11.0,
        marker_b=12.0,
        value=-2.5,
        sample_range_size=SAMPLE_RANGE_SIZE,
    )

    loader = _make_loader(
        tmp_path,
        split_assignments={"train": ["1_2"], "val": [], "test": []},
    )

    loader.load_data()
    data = loader.get_data()
    features = ak.to_numpy(data["features"]["train"][0])
    target = ak.to_numpy(data["target"]["train"][0])

    assert features.shape == (1, SAMPLE_RANGE_SIZE, 2)
    assert np.allclose(features[0, :, 0], -2.0)
    assert np.allclose(features[0, :, 1], -2.5)
    assert target.shape == (1, SAMPLE_RANGE_SIZE, 1)
    assert np.allclose(target[:, 0, 0], [8690.0])


def test_pronostia_ieee2012_loader_accepts_semicolon_delimited_acc_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "Full_Test_Set").mkdir(parents=True, exist_ok=True)
    unit_dir = tmp_path / "Learning_set" / "Bearing2_1"
    unit_dir.mkdir(parents=True)
    _write_six_column_acc_file(
        unit_dir / "acc_00001.csv",
        marker_a=1.0,
        marker_b=2.0,
        value=3.5,
        sample_range_size=SAMPLE_RANGE_SIZE,
        delimiter=";",
    )
    _write_six_column_acc_file(
        unit_dir / "acc_00002.csv",
        marker_a=4.0,
        marker_b=5.0,
        value=-7.0,
        sample_range_size=SAMPLE_RANGE_SIZE,
        delimiter=";",
    )

    loader = _make_loader(
        tmp_path,
        split_assignments={"train": ["2_1"], "val": [], "test": []},
    )

    loader.load_data()
    data = loader.get_data()
    features = ak.to_numpy(data["features"]["train"][0])
    target = ak.to_numpy(data["target"]["train"][0])

    assert features.shape == (1, SAMPLE_RANGE_SIZE, 2)
    assert np.allclose(features[0, :, 0], -6.5)
    assert np.allclose(features[0, :, 1], -7.0)
    assert np.allclose(target[:, 0, 0], [9090.0])


def test_pronostia_ieee2012_loader_applies_bearing_1_4_feature_alignment_fix(
    tmp_path: Path,
) -> None:
    unit_dir = tmp_path / "Full_Test_Set" / "Bearing1_4"
    (tmp_path / "Learning_set").mkdir(parents=True, exist_ok=True)
    unit_dir.mkdir(parents=True)
    for idx in range(13):
        _write_acc_file(unit_dir / f"acc_{idx:05d}.csv", float(idx), SAMPLE_RANGE_SIZE)

    loader = _make_loader(
        tmp_path,
        split_assignments={"train": ["1_4"], "val": [], "test": []},
    )

    loader.load_data()
    data = loader.get_data()
    features = ak.to_numpy(data["features"]["train"][0])
    target = ak.to_numpy(data["target"]["train"][0])
    metadata = data.unit_metadata["train"][0]

    # Bearing 1_4 needs the normal leading-window drop plus an extra 10-window
    # feature shift to align with the PHMD PRONOSTIA payload.
    assert features.shape == (2, SAMPLE_RANGE_SIZE, 2)
    assert np.allclose(features[0, :, 0], 11.5)
    assert np.allclose(features[0, :, 1], 11.0)
    assert np.allclose(features[1, :, 0], 12.5)
    assert np.allclose(features[1, :, 1], 12.0)
    # Targets are regenerated from the aligned observation sequence, so they stay
    # consistent with the framework's PHMD-facing PRONOSTIA semantics.
    assert np.allclose(target[:, 0, 0], [14170.0, 14160.0])
    assert metadata["unit_length"] == 2


def test_pronostia_ieee2012_loader_rejects_incomplete_split_assignments(
    tmp_path: Path,
) -> None:
    _create_complete_dataset_tree(tmp_path)
    loader = _make_loader(
        tmp_path,
        split_assignments={
            "train": ["1_1"],
            "val": ["1_2"],
            "test": ["2_1"],
        },
    )

    with pytest.raises(ValueError, match="Unassigned units"):
        loader.load_data()


def test_pronostia_ieee2012_loader_rejects_duplicate_split_membership(
    tmp_path: Path,
) -> None:
    _create_complete_dataset_tree(tmp_path)
    loader = _make_loader(
        tmp_path,
        split_assignments={
            "train": ["1_1", "1_2"],
            "val": ["1_2"],
            "test": list(LEARNING_UNITS[2:] + FULL_TEST_UNITS),
        },
    )

    with pytest.raises(ValueError, match="assigned to multiple splits"):
        loader.load_data()


def test_pronostia_ieee2012_full_test_unit_uses_known_pronostia_life_lookup(
    tmp_path: Path,
) -> None:
    _create_unit_tree(tmp_path, "1_3", file_count=1802)
    loader = _make_loader(
        tmp_path,
        split_assignments={"train": ["1_3"], "val": [], "test": []},
    )

    loader.load_data()
    data = loader.get_data()

    target = ak.to_numpy(data["target"]["train"][0])
    metadata = data.unit_metadata["train"][0]
    unit_id = UNIT_NAMES_TO_ID["1_3"]

    assert target.shape == (1801, SAMPLE_RANGE_SIZE, 1)
    assert float(target[0, 0, 0]) == 23730.0
    assert float(target[-1, 0, 0]) == 5730.0
    assert metadata["unit_length"] == 1801
    assert metadata["hidden_rul_seconds"] == TEST_RULS["1_3"]
    assert (
        metadata["expected_total_life_seconds"] == PRONOSTIA_TOTAL_LIFE_SECONDS[unit_id]
    )


def test_pronostia_ieee2012_loader_prepares_missing_dataset_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_root = tmp_path / "missing_pronostia_root"
    clone_calls: list[list[str]] = []

    def fake_run(cmd, check, **kwargs):
        clone_calls.append(list(cmd))
        (missing_root / "Full_Test_Set").mkdir(parents=True, exist_ok=True)
        prepared_unit_dir = missing_root / "Learning_set" / "Bearing1_1"
        prepared_unit_dir.mkdir(parents=True, exist_ok=True)
        _write_acc_file(prepared_unit_dir / "acc_00001.csv", 0.0, SAMPLE_RANGE_SIZE)
        _write_acc_file(prepared_unit_dir / "acc_00002.csv", 1.0, SAMPLE_RANGE_SIZE)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(pronostia_ieee2012_module.subprocess, "run", fake_run)

    loader = _make_loader(
        missing_root,
        split_assignments={"train": ["1_1"], "val": [], "test": []},
        download_if_missing=True,
        download_url="https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset",
    )

    loader.load_data()

    assert clone_calls, "Expected load_data() to prepare the missing dataset root."
    assert clone_calls[0][:6] == [
        "git",
        "clone",
        "--depth",
        "1",
        "--filter=blob:none",
        "--sparse",
    ]
    assert clone_calls[0][-1] == str(missing_root)
    assert clone_calls[1] == [
        "git",
        "-C",
        str(missing_root),
        "sparse-checkout",
        "set",
        "Learning_set",
        "Full_Test_Set",
    ]
    assert missing_root.joinpath("Learning_set", "Bearing1_1", "acc_00001.csv").exists()


def test_pronostia_ieee2012_loader_fails_clearly_when_download_is_disabled(
    tmp_path: Path,
) -> None:
    loader = _make_loader(
        tmp_path / "missing_pronostia_root",
        split_assignments={"train": ["1_1"], "val": [], "test": []},
        download_if_missing=False,
    )

    with pytest.raises(FileNotFoundError, match="download_if_missing=False"):
        loader.load_data()


def test_pronostia_ieee2012_pipeline_smoke_with_existing_raw_transforms(
    tmp_path: Path,
) -> None:
    from picid.data.data_objects import SplitViewPolicy
    from picid.data.preprocessing.preprocessor import PreProcessor
    from picid.transforms.base.transform_manager import ConfigTransformManager

    for unit_key in ("1_1", "1_5", "1_6"):
        _create_unit_tree(tmp_path, unit_key, file_count=2)

    loader = _make_loader(
        tmp_path,
        split_assignments={"train": ["1_1"], "val": ["1_5"], "test": ["1_6"]},
    )

    test_output_dir = Path(tempfile.gettempdir()) / "pronostia_ieee2012_smoke"
    cfg = compose(
        config_name="run",
        overrides=[
            "transforms=bearings/pronostia/raw",
            "task_definition=prognostics/rul",
            f"paths.output_dir={test_output_dir}",
            f"paths.work_dir={test_output_dir}",
        ],
    )

    manager = ConfigTransformManager(transforms_config=cfg.transforms)
    preprocessor = PreProcessor(datasource=loader, transforms=manager)
    preprocessor.pipeline(cache_preprocessed=False)
    processed = preprocessor.get_processed_split_dict(
        view_policy=SplitViewPolicy.KEEP_UNIT_LISTS
    )

    assert "train" in processed
    assert "features" in processed["train"]
    assert "rul" in processed["train"]
    assert len(processed["train"]["features"]) == 1
    assert len(processed["val"]["features"]) == 1
    assert len(processed["test"]["features"]) == 1


def test_pronostia_ieee2012_configs_and_experiment_bases_compose() -> None:
    cfg = compose(config_name="run", overrides=["datasource=pronostia"])
    assert (
        cfg.datasource._target_
        == "picid.data.datasources.pronostia_ieee2012.PronostiaIEEE2012Loader"
    )
    assert cfg.datasource.split_mode == "in_domain"
    assert cfg.datasource.fold_id == 1
    assert (
        dict(cfg.datasource.split_assignments_by_mode.in_domain.fold_1)
        == PRONOSTIA_IN_DOMAIN_FOLDS["fold_1"]
    )
    assert (
        dict(cfg.datasource.split_assignments_by_mode.domain_shift.fold_5)
        == PRONOSTIA_DOMAIN_SHIFT_FOLDS["fold_5"]
    )

    cfg = compose(
        config_name="run",
        overrides=[
            "datasource=pronostia",
            "datasource.split_mode=domain_shift",
            "datasource.fold_id=4",
        ],
    )
    assert cfg.datasource.split_mode == "domain_shift"
    assert cfg.datasource.fold_id == 4

    cfg = compose(config_name="run", overrides=["datasource=pronostia_phmd"])
    assert (
        cfg.datasource._target_
        == "picid.data.datasources.phmd_pronostia.PronostiaLoader"
    )

    for experiment_name, split_mode, expected_transform_key in (
        (
            "pronostia/prognostics/in_domain/combined/base",
            "in_domain",
            "concatenate_features",
        ),
        ("pronostia/prognostics/in_domain/raw/base", "in_domain", "reshape_x"),
        (
            "pronostia/prognostics/domain_shift/combined/base",
            "domain_shift",
            "concatenate_features",
        ),
        ("pronostia/prognostics/domain_shift/raw/base", "domain_shift", "reshape_x"),
    ):
        cfg = compose(config_name="run", overrides=[f"experiment={experiment_name}"])
        assert (
            cfg.datasource._target_
            == "picid.data.datasources.pronostia_ieee2012.PronostiaIEEE2012Loader"
        )
        assert cfg.datasource.split_mode == split_mode
        assert cfg.datasource.fold_id == 1
        assert cfg.datasource.task_mode == "rul"
        assert expected_transform_key in cfg.transforms


PRONOSTIA_PHMD_SPLIT = {
    "train": ["1_2", "2_1", "3_1", "3_2"],
    "val": ["1_1", "2_2"],
    "test": list(FULL_TEST_UNITS),
}


def test_pronostia_ieee2012_loader_resolves_phmd_split() -> None:
    from picid.data.datasources.pronostia_ieee2012 import PronostiaIEEE2012Loader

    loader = PronostiaIEEE2012Loader(
        data_name="pronostia_phmd_split_test",
        task_mode="rul",
        data_dir="/tmp/unused_pronostia",
        split_mode="phmd_split",
        split_assignments_by_mode={
            "in_domain": PRONOSTIA_IN_DOMAIN_FOLDS,
            "domain_shift": PRONOSTIA_DOMAIN_SHIFT_FOLDS,
            "phmd_split": PRONOSTIA_PHMD_SPLIT,
        },
        download_if_missing=False,
        sample_range_size=SAMPLE_RANGE_SIZE,
    )

    assert loader.split_assignments == PRONOSTIA_PHMD_SPLIT


def test_pronostia_ieee2012_configs_expose_phmd_split() -> None:
    cfg = compose(
        config_name="run",
        overrides=["datasource=pronostia", "datasource.split_mode=phmd_split"],
    )
    assert cfg.datasource.split_mode == "phmd_split"
    phmd_split = cfg.datasource.split_assignments_by_mode.phmd_split
    assert list(phmd_split.train) == PRONOSTIA_PHMD_SPLIT["train"]
    assert list(phmd_split.val) == PRONOSTIA_PHMD_SPLIT["val"]
    assert list(phmd_split.test) == PRONOSTIA_PHMD_SPLIT["test"]
