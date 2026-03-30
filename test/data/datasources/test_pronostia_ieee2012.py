"""Tests for the raw IEEE 2012 PRONOSTIA datasource."""

from __future__ import annotations

from pathlib import Path
import tempfile

import awkward as ak
from hydra import compose
import numpy as np
import pytest

from picid.data.datasources.pronostia_ieee2012 import (
    PRONOSTIA_TOTAL_LIFE_SECONDS,
    TEST_RULS,
    UNIT_NAMES_TO_ID,
)


SAMPLE_RANGE_SIZE = 4
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


def _write_acc_file(path: Path, value: float, sample_range_size: int) -> None:
    signal = np.column_stack(
        [
            np.full(sample_range_size, value, dtype=np.float32),
            np.full(sample_range_size, value + 0.5, dtype=np.float32),
        ]
    )
    np.savetxt(path, signal, delimiter=",", fmt="%.3f")


def _unit_subset(unit_key: str) -> str:
    return "Learning_set" if unit_key in LEARNING_UNITS else "Full_Test_Set"


def _create_unit_tree(
    root: Path,
    unit_key: str,
    *,
    file_count: int,
    sample_range_size: int = SAMPLE_RANGE_SIZE,
) -> None:
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
        "use_ragged": True,
        "sample_range_size": SAMPLE_RANGE_SIZE,
        "derive_full_test_rul": True,
        "include_temperature": False,
    }
    kwargs.update(overrides)
    return PronostiaIEEE2012Loader(**kwargs)


def test_pronostia_ieee2012_loader_reads_sorted_two_channel_windows_and_metadata(
    tmp_path: Path,
) -> None:
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

    assert features.shape == (3, SAMPLE_RANGE_SIZE, 2)
    assert np.allclose(features[0, :, 0], 0.0)
    assert np.allclose(features[1, :, 0], 1.0)
    assert np.allclose(features[2, :, 0], 2.0)

    assert target.shape == (3, SAMPLE_RANGE_SIZE, 1)
    assert np.allclose(target[:, 0, 0], [0.0, 10.0, 20.0])
    assert np.allclose(target[:, -1, 0], [0.0, 10.0, 20.0])

    assert metadata["unit_name"] == "Bearing 1_1"
    assert metadata["unit_id"] == UNIT_NAMES_TO_ID["1_1"]
    assert metadata["unit_length"] == 3
    assert metadata["dataset_subset"] == "Learning_set"


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

    assert target.shape == (1802, SAMPLE_RANGE_SIZE, 1)
    assert float(target[0, 0, 0]) == 0.0
    assert float(target[-1, 0, 0]) == 18010.0
    assert metadata["hidden_rul_seconds"] == TEST_RULS["1_3"]
    assert metadata["expected_total_life_seconds"] == PRONOSTIA_TOTAL_LIFE_SECONDS[unit_id]


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
    for datasource_name in (
        "pronostia_ieee2012_in_domain",
        "pronostia_ieee2012_domain_shift",
    ):
        cfg = compose(config_name="run", overrides=[f"datasource={datasource_name}"])
        assert (
            cfg.datasource._target_
            == "picid.data.datasources.pronostia_ieee2012.PronostiaIEEE2012Loader"
        )

    for experiment_name, datasource_name in (
        (
            "pronostia_ieee2012_in_domain/prognostics/base",
            "pronostia_ieee2012_in_domain",
        ),
        (
            "pronostia_ieee2012_domain_shift/prognostics/base",
            "pronostia_ieee2012_domain_shift",
        ),
    ):
        cfg = compose(config_name="run", overrides=[f"experiment={experiment_name}"])
        assert cfg.datasource.data_name == datasource_name
        assert cfg.datasource.task_mode == "rul"


def test_existing_phmd_pronostia_datasource_config_still_instantiates() -> None:
    cfg = compose(config_name="run", overrides=["datasource=pronostia"])
    assert (
        cfg.datasource._target_ == "picid.data.datasources.phmd_pronostia.PronostiaLoader"
    )
