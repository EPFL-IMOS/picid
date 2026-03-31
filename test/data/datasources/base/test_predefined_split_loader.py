"""Tests for the shared predefined-split loader base."""

from __future__ import annotations

import numpy as np
import pytest

from picid.data.data_objects.data import SplitDatasetContainer
from picid.data.datasources.base.exceptions import (
    DatasourceConfigurationError,
    DatasourceStateError,
)
from picid.data.datasources.base.predefined_split_loader import (
    PredefinedSplitLoaderBase,
)


class InMemoryPredefinedSplitLoader(PredefinedSplitLoaderBase):
    def __init__(self, **kwargs):
        kwargs.setdefault("data_name", "predefined")
        kwargs.setdefault("task_mode", "regression")
        super().__init__(**kwargs)

    def _load_data(self) -> dict:
        return {
            "features": {
                "train": [np.ones((3, 2))],
                "val": [np.ones((2, 2))],
                "test": [np.ones((1, 2))],
            },
            "target": {
                "train": [np.ones((3, 1))],
                "val": [np.ones((2, 1))],
                "test": [np.ones((1, 1))],
            },
            "metadata": {
                "train": [{"unit_name": "u1"}],
                "val": [{"unit_name": "u2"}],
                "test": [{"unit_name": "u3"}],
            },
        }


class FailingReloadPredefinedSplitLoader(InMemoryPredefinedSplitLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fail_next_load = False

    def _load_data(self) -> dict:
        if self._fail_next_load:
            raise RuntimeError("reload failed")
        return super()._load_data()


class SingleUnitPredefinedSplitLoader(InMemoryPredefinedSplitLoader):
    def _load_data(self) -> dict:
        return {
            "features": {
                "train": [np.ones((3, 2))],
                "val": [np.ones((2, 2))],
                "test": [np.ones((1, 2))],
            },
            "target": {
                "train": [np.ones((3, 1))],
                "val": [np.ones((2, 1))],
                "test": [np.ones((1, 1))],
            },
            "metadata": {
                "train": [{"unit_name": "u1"}],
                "val": [{"unit_name": "u1"}],
                "test": [{"unit_name": "u1"}],
            },
        }


def test_predefined_split_loader_load_get_and_noop_split():
    loader = InMemoryPredefinedSplitLoader()
    loader.load_data()
    loader.split_data()
    data = loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert loader.get_data_names() == ("predefined",)
    assert loader.get_data_name() == "predefined"
    assert loader.get_split_mode() == "within_units"


def test_predefined_split_loader_get_data_is_stable_across_repeated_calls():
    """Repeated get_data() calls must preserve split payload list storage."""
    loader = InMemoryPredefinedSplitLoader()
    loader.load_data()

    first = loader.get_data()
    second = loader.get_data()

    assert len(first.to_split_dict()["train"]["features"]) == 1
    assert len(second.to_split_dict()["train"]["features"]) == 1


def test_predefined_split_loader_one_unit_per_split_is_stable_across_repeated_calls():
    """Repeated get_data() calls must preserve one-unit-per-split list storage."""
    loader = SingleUnitPredefinedSplitLoader()
    loader.load_data()

    first = loader.get_data()
    second = loader.get_data()

    assert len(first.to_split_dict()["train"]["features"]) == 1
    split_dict = second.to_split_dict()
    assert len(split_dict["train"]["features"]) == 1
    assert split_dict["train"]["features"][0].shape == (3, 2)


def test_predefined_split_loader_rejects_multisource_splitter():
    """Predefined-split loaders fail fast on incompatible multisource config."""
    with pytest.raises(
        DatasourceConfigurationError,
        match="does not accept multisource_data_splitter",
    ):
        InMemoryPredefinedSplitLoader(multisource_data_splitter=object())


def test_predefined_split_loader_get_data_before_load_raises():
    """Predefined-split loaders still guard against access before load_data()."""
    loader = InMemoryPredefinedSplitLoader()

    with pytest.raises(DatasourceStateError, match="must be loaded"):
        loader.get_data()


def test_predefined_split_loader_failed_reload_clears_previous_live_state():
    """A failed reload must not leave the previous split payload marked as live."""
    loader = FailingReloadPredefinedSplitLoader()
    loader.load_data()
    assert loader.is_loaded() is True
    assert loader.is_split_ready() is True

    loader._fail_next_load = True
    with pytest.raises(RuntimeError, match="reload failed"):
        loader.load_data()

    assert loader.is_loaded() is False
    assert loader.is_split_ready() is False
    assert loader.get_loader_state().value == "initialized"
    with pytest.raises(DatasourceStateError, match="must be loaded"):
        loader.get_data()


def test_predefined_split_loader_cache_fingerprint_uses_constructor_config():
    loader = InMemoryPredefinedSplitLoader(data_name="custom", task_mode="forecasting")
    assert loader.get_cache_fingerprint() == {
        "data_name": "custom",
        "task_mode": "forecasting",
    }


def test_predefined_split_loader_cache_fingerprint_ignores_late_repr_config_mutation():
    loader = InMemoryPredefinedSplitLoader(data_name="custom", task_mode="forecasting")
    original = loader.get_cache_fingerprint()

    loader._repr_config["data_name"] = "mutated"
    loader._repr_config["task_mode"] = "fault"
    loader._repr_config["multisource_data_splitter"] = {"_target_": "changed"}

    assert loader.get_cache_fingerprint() == original


def test_predefined_split_loader_cache_fingerprint_isolated_from_nested_kwargs():
    nested = {"channels": ["x", "y"]}
    loader = InMemoryPredefinedSplitLoader(
        data_name="custom",
        task_mode="forecasting",
        nested=nested,
    )

    nested["channels"].append("z")

    assert loader.get_cache_fingerprint() == {
        "data_name": "custom",
        "task_mode": "forecasting",
        "nested": {"channels": ["x", "y"]},
    }
