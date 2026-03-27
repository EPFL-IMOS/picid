"""
Tests for base datasource interfaces (AbstractDataSourceLoader contract).
"""

from __future__ import annotations

import pytest

import picid.data.datasources.base as base_package
from picid.data.data_objects import SplitDatasetContainer
from picid.data.datasources.base.contracts import DatasourceProtocol, LoaderState
from picid.data.datasources.base.interfaces import AbstractDataSourceLoader


class ConcreteLoader(AbstractDataSourceLoader):
    """Minimal concrete loader for interface tests."""

    def __init__(self, **kwargs):
        kwargs.setdefault("data_name", "concrete")
        kwargs.setdefault("task_mode", "regression")
        super().__init__(**kwargs)
        self._loaded = False

    def load_data(self) -> None:
        self._loaded = True

    def get_data(self):
        return SplitDatasetContainer()

    def get_data_name(self) -> str:
        return self.data_name

    def get_meta_data(self) -> dict:
        return {"data_name": self.data_name}

    def split_data(self) -> None:
        pass


def test_interface_get_split_mode_within_units_when_no_multisource_splitter():
    """get_split_mode returns the canonical within-units mode when no multisource splitter is set."""
    loader = ConcreteLoader()
    assert loader.get_split_mode() == "within_units"


def test_interface_get_split_mode_between_units_when_multisource_splitter_present():
    """get_split_mode returns the canonical between-units mode when a multisource splitter is provided."""
    loader = ConcreteLoader(multisource_data_splitter=object())
    assert loader.get_split_mode() == "between_units"


def test_interface_get_split_mode_treats_falsy_splitter_values_as_multisource():
    """Split mode should key off None checks, not truthiness, for splitter presence."""
    loader = ConcreteLoader(multisource_data_splitter=[])
    assert loader.get_split_mode() == "between_units"


def test_interface_repr_includes_class_name():
    """__repr__ includes class name and init args."""
    loader = ConcreteLoader(data_name="test", task_mode="rul")
    r = repr(loader)
    assert "ConcreteLoader" in r
    assert "test" in r
    assert "rul" in r


def test_interface_data_name_and_task_mode_from_kwargs():
    """data_name and task_mode are set from kwargs."""
    loader = ConcreteLoader(data_name="mydata", task_mode="forecasting")
    assert loader.data_name == "mydata"
    assert loader.task_mode == "forecasting"


def test_interface_get_data_names_returns_tuple():
    """New contract: get_data_names returns a tuple, even for a single source."""
    loader = ConcreteLoader(data_name="mydata")
    assert loader.get_data_names() == ("mydata",)


def test_interface_get_data_name_remains_compatibility_shim():
    """Legacy callers can still resolve a single datasource name through the shim."""
    loader = ConcreteLoader(data_name="mydata")
    assert loader.get_data_name() == "mydata"


def test_interface_get_meta_data_returns_dict():
    """Base datasource contract includes get_meta_data."""
    loader = ConcreteLoader(data_name="mydata")
    assert loader.get_meta_data() == {"data_name": "mydata"}


def test_interface_get_cache_fingerprint_contains_constructor_config():
    """Cache fingerprint should be derived from the datasource constructor config."""
    loader = ConcreteLoader(data_name="mydata", task_mode="forecasting")
    assert loader.get_cache_fingerprint() == {
        "data_name": "mydata",
        "task_mode": "forecasting",
    }


def test_interface_cache_fingerprint_is_not_affected_by_repr_config_mutation():
    """Cache fingerprints should remain stable after repr snapshots are mutated."""
    loader = ConcreteLoader(data_name="mydata", task_mode="forecasting")
    loader._repr_config["data_name"] = "mutated"
    loader._repr_config["task_mode"] = "fault"

    assert loader.get_cache_fingerprint() == {
        "data_name": "mydata",
        "task_mode": "forecasting",
    }


def test_interface_cache_fingerprint_isolated_from_original_nested_kwargs():
    """Cache fingerprints should snapshot nested constructor inputs by value."""
    nested = {"tags": ["train", "test"]}
    loader = ConcreteLoader(
        data_name="mydata",
        task_mode="forecasting",
        nested=nested,
    )

    nested["tags"].append("mutated")

    assert loader.get_cache_fingerprint() == {
        "data_name": "mydata",
        "task_mode": "forecasting",
        "nested": {"tags": ["train", "test"]},
    }


def test_interface_split_composition_helper_uses_public_split_container():
    loader = ConcreteLoader()

    data = loader.get_split_data_for_composition()

    assert isinstance(data, SplitDatasetContainer)


def test_interface_loaded_composition_helper_rejects_split_only_loader():
    loader = ConcreteLoader()

    with pytest.raises(TypeError, match="does not expose an unsplit DatasetContainer"):
        loader.get_loaded_data_for_composition()


def test_interface_loader_state_helpers_follow_shared_state_machine():
    loader = ConcreteLoader()

    assert loader.get_loader_state() == LoaderState.INITIALIZED
    assert loader.is_loaded() is False
    assert loader.is_split_ready() is False

    loader._state = LoaderState.LOADED
    assert loader.is_loaded() is True
    assert loader.is_split_ready() is False

    loader._state = LoaderState.SPLIT
    assert loader.is_loaded() is True
    assert loader.is_split_ready() is True


def test_interface_protocol_keeps_composition_helpers_off_public_contract():
    """The public datasource protocol should not expose composition-only methods."""
    assert not hasattr(DatasourceProtocol, "get_loaded_data_for_composition")
    assert not hasattr(DatasourceProtocol, "get_split_data_for_composition")


def test_base_package_does_not_export_legacy_split_mode_normalizer():
    """Legacy split-mode normalization helpers are no longer part of the public base API."""
    assert not hasattr(base_package, "normalize_split_mode")
    assert "normalize_split_mode" not in getattr(base_package, "__all__", [])
