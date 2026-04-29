"""Tests for custom XJTU-SY split protocols."""

from __future__ import annotations

import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path
import pytest

from picid.data.datasources.phmd_xjtu_sy import XJTU_SYLoader

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"

XJTU_IN_DOMAIN_FOLDS = {
    "fold_1": {
        "train": ["1_3", "1_4", "1_5", "2_3", "2_4", "2_5", "3_3", "3_4", "3_5"],
        "val": ["1_1", "2_1", "3_1"],
        "test": ["1_2", "2_2", "3_2"],
    },
    "fold_2": {
        "train": ["1_1", "1_4", "1_5", "2_1", "2_4", "2_5", "3_1", "3_4", "3_5"],
        "val": ["1_2", "2_2", "3_2"],
        "test": ["1_3", "2_3", "3_3"],
    },
    "fold_3": {
        "train": ["1_1", "1_2", "1_5", "2_1", "2_2", "2_5", "3_1", "3_2", "3_5"],
        "val": ["1_3", "2_3", "3_3"],
        "test": ["1_4", "2_4", "3_4"],
    },
    "fold_4": {
        "train": ["1_1", "1_2", "1_3", "2_1", "2_2", "2_3", "3_1", "3_2", "3_3"],
        "val": ["1_4", "2_4", "3_4"],
        "test": ["1_5", "2_5", "3_5"],
    },
    "fold_5": {
        "train": ["1_2", "1_3", "1_4", "2_2", "2_3", "2_4", "3_2", "3_3", "3_4"],
        "val": ["1_5", "2_5", "3_5"],
        "test": ["1_1", "2_1", "3_1"],
    },
}
XJTU_DOMAIN_SHIFT_FOLDS = {
    "fold_1": {
        "train": ["1_2", "1_3", "1_4", "1_5", "2_2", "2_3", "2_4", "2_5"],
        "val": ["1_1", "2_1"],
        "test": ["3_1", "3_2", "3_3", "3_4", "3_5"],
    },
    "fold_2": {
        "train": ["1_1", "1_3", "1_4", "1_5", "2_1", "2_3", "2_4", "2_5"],
        "val": ["1_2", "2_2"],
        "test": ["3_1", "3_2", "3_3", "3_4", "3_5"],
    },
    "fold_3": {
        "train": ["1_1", "1_2", "1_4", "1_5", "2_1", "2_2", "2_4", "2_5"],
        "val": ["1_3", "2_3"],
        "test": ["3_1", "3_2", "3_3", "3_4", "3_5"],
    },
    "fold_4": {
        "train": ["1_1", "1_2", "1_3", "1_5", "2_1", "2_2", "2_3", "2_5"],
        "val": ["1_4", "2_4"],
        "test": ["3_1", "3_2", "3_3", "3_4", "3_5"],
    },
    "fold_5": {
        "train": ["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "2_4"],
        "val": ["1_5", "2_5"],
        "test": ["3_1", "3_2", "3_3", "3_4", "3_5"],
    },
}


@pytest.fixture(autouse=True)
def hydra_initialized():
    """Re-initialize Hydra for each test so compose() is order-independent."""

    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS_DIR)):
        yield
    GlobalHydra.instance().clear()


def _loader_kwargs() -> dict:
    return {
        "fold": 0,
        "data_name": "XJTU-SY",
        "task_mode": "rul",
        "cache_dir": "/tmp/phmd_xjtu_sy",
    }


def _mock_payload() -> dict:
    split_dict = {"train": [], "val": [], "test": []}
    payload = {
        "features": {split: [] for split in split_dict},
        "target": {split: [] for split in split_dict},
        "unit_id": {split: [] for split in split_dict},
        "unit_metadata": {split: [] for split in split_dict},
    }

    all_units = [
        "1_1",
        "1_2",
        "1_3",
        "1_4",
        "1_5",
        "2_1",
        "2_2",
        "2_3",
        "2_4",
        "2_5",
        "3_1",
        "3_2",
        "3_3",
        "3_4",
        "3_5",
    ]
    for unit_key in all_units:
        payload["features"]["train"].append(f"features-{unit_key}")
        payload["target"]["train"].append(f"target-{unit_key}")
        payload["unit_id"]["train"].append(
            tuple(int(part) for part in unit_key.split("_"))
        )
        payload["unit_metadata"]["train"].append(
            {
                "unit_name": f"Bearing {unit_key}",
                "unit_id": tuple(int(part) for part in unit_key.split("_")),
            }
        )
    return payload


def _assert_disjoint_and_cover_all(
    split_table: dict[str, list[str]], expected_units: set[str]
) -> None:
    seen: set[str] = set()
    for split_name in ("train", "val", "test"):
        split_units = set(split_table[split_name])
        assert split_units.isdisjoint(seen), f"{split_name} overlaps prior splits"
        seen |= split_units
    assert seen == expected_units


def test_xjtu_sy_loader_domain_shift_reassigns_units_to_paper_protocol() -> None:
    loader = XJTU_SYLoader(**_loader_kwargs(), split_mode="domain_shift", fold_id=5)

    assert loader.split_assignments == {
        "train": ["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "2_4"],
        "val": ["1_5", "2_5"],
        "test": ["3_1", "3_2", "3_3", "3_4", "3_5"],
    }

    payload = _mock_payload()
    loader.meta_data = {"unit_names": {}, "unit_ids": {}, "features": ["f1", "f2"]}

    remapped = loader._apply_custom_split_assignments(payload)
    loader._refresh_split_metadata(remapped)

    assert [meta["unit_name"] for meta in remapped["unit_metadata"]["train"]] == [
        "Bearing 1_1",
        "Bearing 1_2",
        "Bearing 1_3",
        "Bearing 1_4",
        "Bearing 2_1",
        "Bearing 2_2",
        "Bearing 2_3",
        "Bearing 2_4",
    ]
    assert [meta["unit_name"] for meta in remapped["unit_metadata"]["val"]] == [
        "Bearing 1_5",
        "Bearing 2_5",
    ]
    assert [meta["unit_name"] for meta in remapped["unit_metadata"]["test"]] == [
        "Bearing 3_1",
        "Bearing 3_2",
        "Bearing 3_3",
        "Bearing 3_4",
        "Bearing 3_5",
    ]
    assert loader.meta_data["unit_names"]["test"] == [
        "Bearing 3_1",
        "Bearing 3_2",
        "Bearing 3_3",
        "Bearing 3_4",
        "Bearing 3_5",
    ]


def test_xjtu_sy_loader_resolves_exact_fold_tables() -> None:
    loader = XJTU_SYLoader(
        **_loader_kwargs(),
        split_mode="in_domain",
        fold_id=2,
        split_assignments_by_mode={
            "in_domain": XJTU_IN_DOMAIN_FOLDS,
            "domain_shift": XJTU_DOMAIN_SHIFT_FOLDS,
        },
    )

    assert loader.split_assignments == XJTU_IN_DOMAIN_FOLDS["fold_2"]


def test_xjtu_sy_cv_tables_match_expected_structure() -> None:
    all_units = {
        "1_1",
        "1_2",
        "1_3",
        "1_4",
        "1_5",
        "2_1",
        "2_2",
        "2_3",
        "2_4",
        "2_5",
        "3_1",
        "3_2",
        "3_3",
        "3_4",
        "3_5",
    }
    source_units = {unit for unit in all_units if not unit.startswith("3_")}
    target_units = {unit for unit in all_units if unit.startswith("3_")}

    for split_table in XJTU_IN_DOMAIN_FOLDS.values():
        _assert_disjoint_and_cover_all(split_table, all_units)
        assert len(split_table["train"]) == 9
        assert len(split_table["val"]) == 3
        assert len(split_table["test"]) == 3
        assert sum(unit.startswith("3_") for unit in split_table["train"]) == 3
        assert sum(unit.startswith("3_") for unit in split_table["val"]) == 1
        assert sum(unit.startswith("3_") for unit in split_table["test"]) == 1

    for split_table in XJTU_DOMAIN_SHIFT_FOLDS.values():
        _assert_disjoint_and_cover_all(split_table, source_units | target_units)
        assert set(split_table["test"]) == target_units
        assert set(split_table["train"]).issubset(source_units)
        assert set(split_table["val"]).issubset(source_units)
        assert len(split_table["train"]) == 8
        assert len(split_table["val"]) == 2
        assert len(split_table["test"]) == 5


def test_xjtu_sy_configs_and_experiment_bases_compose() -> None:
    cfg = compose(config_name="run", overrides=["datasource=xjtu_sy"])
    assert (
        cfg.datasource._target_ == "picid.data.datasources.phmd_xjtu_sy.XJTU_SYLoader"
    )
    assert cfg.datasource.split_mode == "in_domain"
    assert cfg.datasource.fold_id == 1
    assert (
        dict(cfg.datasource.split_assignments_by_mode.in_domain.fold_1)
        == XJTU_IN_DOMAIN_FOLDS["fold_1"]
    )
    assert (
        dict(cfg.datasource.split_assignments_by_mode.domain_shift.fold_5)
        == XJTU_DOMAIN_SHIFT_FOLDS["fold_5"]
    )

    cfg = compose(
        config_name="run",
        overrides=[
            "datasource=xjtu_sy",
            "datasource.split_mode=domain_shift",
            "datasource.fold_id=4",
        ],
    )
    assert cfg.datasource.split_mode == "domain_shift"
    assert cfg.datasource.fold_id == 4

    for experiment_name, split_mode, expected_transform_key in (
        (
            "xjtu_sy/prognostics/in_domain/combined/base",
            "in_domain",
            "concatenate_features",
        ),
        ("xjtu_sy/prognostics/in_domain/raw/base", "in_domain", "scaler_features"),
        (
            "xjtu_sy/prognostics/domain_shift/combined/base",
            "domain_shift",
            "concatenate_features",
        ),
        (
            "xjtu_sy/prognostics/domain_shift/raw/base",
            "domain_shift",
            "scaler_features",
        ),
    ):
        cfg = compose(config_name="run", overrides=[f"experiment={experiment_name}"])
        assert (
            cfg.datasource._target_
            == "picid.data.datasources.phmd_xjtu_sy.XJTU_SYLoader"
        )
        assert cfg.datasource.split_mode == split_mode
        assert cfg.datasource.fold_id == 1
        assert expected_transform_key in cfg.transforms


XJTU_PHMD_SPLIT = {
    "train": ["1_3", "1_4", "2_1", "2_4", "2_5", "3_1", "3_2", "3_3"],
    "val": ["1_1", "1_2", "3_5"],
    "test": ["1_5", "2_2", "2_3", "3_4"],
}


def test_xjtu_sy_loader_resolves_phmd_split() -> None:
    loader = XJTU_SYLoader(**_loader_kwargs(), split_mode="phmd_split")

    assert loader.split_assignments == XJTU_PHMD_SPLIT


def test_xjtu_sy_configs_expose_phmd_split() -> None:
    cfg = compose(
        config_name="run",
        overrides=["datasource=xjtu_sy", "datasource.split_mode=phmd_split"],
    )
    assert cfg.datasource.split_mode == "phmd_split"
    assert dict(cfg.datasource.split_assignments_by_mode.phmd_split) == XJTU_PHMD_SPLIT
