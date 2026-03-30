"""Tests for custom XJTU-SY split protocols."""

from __future__ import annotations

from hydra import compose

from picid.data.datasources.phmd_xjtu_sy import XJTU_SYLoader


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
        payload["unit_id"]["train"].append(tuple(int(part) for part in unit_key.split("_")))
        payload["unit_metadata"]["train"].append(
            {
                "unit_name": f"Bearing {unit_key}",
                "unit_id": tuple(int(part) for part in unit_key.split("_")),
            }
        )
    return payload


def test_xjtu_sy_loader_domain_shift_reassigns_units_to_paper_protocol() -> None:
    loader = XJTU_SYLoader(**_loader_kwargs(), split_mode="domain_shift")

    assert loader.split_assignments == {
        "train": ["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "2_4"],
        "val": ["1_5", "2_5"],
        "test": ["3_1", "3_2", "3_3", "3_4", "3_5"],
    }

    payload = _mock_payload()
    loader.meta_data = {"unit_names": {}, "unit_ids": {}, "features": ["f1", "f2"]}

    remapped = loader._apply_custom_split_assignments(payload)
    loader._refresh_split_metadata(remapped)

    assert [
        meta["unit_name"] for meta in remapped["unit_metadata"]["train"]
    ] == [
        "Bearing 1_1",
        "Bearing 1_2",
        "Bearing 1_3",
        "Bearing 1_4",
        "Bearing 2_1",
        "Bearing 2_2",
        "Bearing 2_3",
        "Bearing 2_4",
    ]
    assert [
        meta["unit_name"] for meta in remapped["unit_metadata"]["val"]
    ] == ["Bearing 1_5", "Bearing 2_5"]
    assert [
        meta["unit_name"] for meta in remapped["unit_metadata"]["test"]
    ] == ["Bearing 3_1", "Bearing 3_2", "Bearing 3_3", "Bearing 3_4", "Bearing 3_5"]
    assert loader.meta_data["unit_names"]["test"] == [
        "Bearing 3_1",
        "Bearing 3_2",
        "Bearing 3_3",
        "Bearing 3_4",
        "Bearing 3_5",
    ]


def test_xjtu_sy_configs_and_experiment_bases_compose() -> None:
    cfg = compose(config_name="run", overrides=["datasource=xjtu_sy"])
    assert cfg.datasource._target_ == "picid.data.datasources.phmd_xjtu_sy.XJTU_SYLoader"
    assert cfg.datasource.split_mode == "in_domain"

    cfg = compose(
        config_name="run",
        overrides=["datasource=xjtu_sy", "datasource.split_mode=domain_shift"],
    )
    assert cfg.datasource.split_mode == "domain_shift"

    for experiment_name, split_mode, expected_transform_key in (
        ("xjtu_sy/prognostics/in_domain/combined/base", "in_domain", "concatenate_features"),
        ("xjtu_sy/prognostics/in_domain/raw/base", "in_domain", "scaler_features"),
        ("xjtu_sy/prognostics/domain_shift/combined/base", "domain_shift", "concatenate_features"),
        ("xjtu_sy/prognostics/domain_shift/raw/base", "domain_shift", "scaler_features"),
    ):
        cfg = compose(config_name="run", overrides=[f"experiment={experiment_name}"])
        assert cfg.datasource._target_ == "picid.data.datasources.phmd_xjtu_sy.XJTU_SYLoader"
        assert cfg.datasource.split_mode == split_mode
        assert expected_transform_key in cfg.transforms
