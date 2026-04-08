"""Validation tests for MultiSourceLoader constructor contracts."""

import pytest

from picid.data.datasources.base.exceptions import DatasourceConfigurationError
from picid.data.datasources.base.multi_source_loader import MultiSourceLoader
from picid.data.split_strategies import BySourceSplitter
from test.data.datasources.base.conftest import InMemorySingleSourceLoader


@pytest.mark.unit
def test_rejects_empty_source_list():
    with pytest.raises(
        DatasourceConfigurationError, match="source_list cannot be empty"
    ):
        MultiSourceLoader(data_name="m", task_mode="regression", source_list={})


@pytest.mark.unit
def test_rejects_unsupported_multisource_splitter_type():
    with pytest.raises(DatasourceConfigurationError, match="multisource_data_splitter"):
        MultiSourceLoader(
            data_name="m",
            task_mode="regression",
            source_list={"a": InMemorySingleSourceLoader(data_name="a")},
            multisource_data_splitter=object(),
        )


@pytest.mark.integration
def test_accepts_instantiated_sources_without_hydra_source_kwargs():
    source_a = InMemorySingleSourceLoader(data_name="a")
    source_b = InMemorySingleSourceLoader(data_name="b")
    source_c = InMemorySingleSourceLoader(data_name="c")

    loader = MultiSourceLoader(
        data_name="multi_instances",
        task_mode="regression",
        source_list={"a": source_a, "b": source_b, "c": source_c},
        multisource_data_splitter=BySourceSplitter(
            sources_train=["a"], sources_val=["b"], sources_test=["c"]
        ),
    )

    loader.load_data()
    loader.split_data()
    data = loader.get_data()

    assert "features" in data
    assert "train" in data["features"]
    assert "val" in data["features"]
    assert "test" in data["features"]
