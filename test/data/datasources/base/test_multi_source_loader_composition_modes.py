"""Regression matrix for MultiSourceLoader source/splitter permutations."""

import pytest
from omegaconf import OmegaConf

from picid.data.datasources.base.multi_source_loader import MultiSourceLoader
from picid.data.split_strategies import BySourceSplitter
from test.data.datasources.base.conftest import InMemorySingleSourceLoader


def _source_cfg(name: str) -> dict:
    return {
        "_target_": "test.data.datasources.base.conftest.InMemorySingleSourceLoader",
        "n_samples": 120,
        "n_features": 4,
        "data_name": name,
        "task_mode": "regression",
    }


@pytest.mark.integration
def test_config_sources_with_config_splitter_work():
    unit_a = _source_cfg("unit_a")
    unit_b = _source_cfg("unit_b")
    unit_c = _source_cfg("unit_c")

    loader = MultiSourceLoader(
        data_name="multi_cfg_cfg",
        task_mode="regression",
        source_list={"unit_a": unit_a, "unit_b": unit_b, "unit_c": unit_c},
        multisource_data_splitter={
            "_target_": "picid.data.split_strategies.BySourceSplitter",
            "sources_train": ["unit_a"],
            "sources_val": ["unit_b"],
            "sources_test": ["unit_c"],
        },
        unit_a=OmegaConf.create(unit_a),
        unit_b=OmegaConf.create(unit_b),
        unit_c=OmegaConf.create(unit_c),
    )

    loader.load_data()
    loader.split_data()
    data = loader.get_data()

    assert set(loader.get_source_names()) == {"unit_a", "unit_b", "unit_c"}
    assert "features" in data and "target" in data
    assert all(split in data["features"] for split in ("train", "val", "test"))


@pytest.mark.integration
def test_instance_sources_with_instance_splitter_work():
    loader = MultiSourceLoader(
        data_name="multi_inst_inst",
        task_mode="regression",
        source_list={
            "unit_a": InMemorySingleSourceLoader(data_name="unit_a"),
            "unit_b": InMemorySingleSourceLoader(data_name="unit_b"),
            "unit_c": InMemorySingleSourceLoader(data_name="unit_c"),
        },
        multisource_data_splitter=BySourceSplitter(
            sources_train=["unit_a"], sources_val=["unit_b"], sources_test=["unit_c"]
        ),
    )

    loader.load_data()
    loader.split_data()
    data = loader.get_data()

    assert set(loader.get_source_names()) == {"unit_a", "unit_b", "unit_c"}
    assert "features" in data and "target" in data
    assert all(split in data["features"] for split in ("train", "val", "test"))
