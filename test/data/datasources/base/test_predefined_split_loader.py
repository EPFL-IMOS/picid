"""Tests for the shared predefined-split loader base."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from picid.data.data_objects import SplitDatasetContainer
from picid.data.datasources.base.exceptions import (
    DatasourceConfigurationError,
    DatasourceStateError,
)
from picid.data.datasources.base.predefined_split_loader import (
    PredefinedSplitLoaderBase,
)

_PREDEFINED_LOG = "picid.data.datasources.base.predefined_split_loader"


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


class TestLoadSplitGetDataContract:
    """Happy path: load, noop split, container identity."""

    def test_load_get_and_noop_split(self):
        loader = InMemoryPredefinedSplitLoader()
        loader.load_data()
        loader.split_data()
        data = loader.get_data()
        assert isinstance(data, SplitDatasetContainer)
        assert loader.get_data_names() == ("predefined",)
        assert loader.get_data_name() == "predefined"
        assert loader.get_split_mode() == "within_units"


class TestRepeatedGetDataStability:
    """``get_data()`` returns fresh deep copies with stable list structure."""

    @pytest.mark.parametrize(
        "loader_factory,expect_train_feature_shape",
        [
            (InMemoryPredefinedSplitLoader, None),
            (SingleUnitPredefinedSplitLoader, (3, 2)),
        ],
    )
    def test_stable_across_repeated_calls(
        self, loader_factory, expect_train_feature_shape
    ):
        """Repeated get_data() calls must preserve split payload list storage."""
        loader = loader_factory()
        loader.load_data()

        first = loader.get_data()
        second = loader.get_data()

        assert len(first.to_split_dict()["train"]["features"]) == 1
        assert len(second.to_split_dict()["train"]["features"]) == 1
        if expect_train_feature_shape is not None:
            assert (
                second.to_split_dict()["train"]["features"][0].shape
                == expect_train_feature_shape
            )


class TestConfigurationRejects:
    """Incompatible multisource splitter is rejected at construction."""

    def test_rejects_multisource_splitter(self):
        with pytest.raises(
            DatasourceConfigurationError,
            match="does not accept multisource_data_splitter",
        ):
            InMemoryPredefinedSplitLoader(multisource_data_splitter=object())


class TestStateGuards:
    """Lifecycle ordering and split_data no-op before load."""

    def test_get_data_before_load_raises(self):
        loader = InMemoryPredefinedSplitLoader()
        with pytest.raises(DatasourceStateError, match="must be loaded"):
            loader.get_data()

    def test_split_data_before_load_logs_and_get_data_still_raises(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        """split_data() is a no-op warning before load; data remains inaccessible."""
        loader = InMemoryPredefinedSplitLoader()
        with caplog.at_level(logging.WARNING, logger=_PREDEFINED_LOG):
            loader.split_data()
        assert any("relies on predefined splits" in r.message for r in caplog.records)
        with pytest.raises(DatasourceStateError, match="must be loaded"):
            loader.get_data()


class TestFailedReload:
    """Failed reload clears live state."""

    def test_failed_reload_clears_previous_live_state(self):
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


class TestCacheFingerprint:
    """Constructor snapshot drives fingerprint; repr and nested kwargs isolated."""

    def test_uses_constructor_config(self):
        loader = InMemoryPredefinedSplitLoader(
            data_name="custom", task_mode="forecasting"
        )
        assert loader.get_cache_fingerprint() == {
            "data_name": "custom",
            "task_mode": "forecasting",
        }

    def test_ignores_late_repr_config_mutation(self):
        loader = InMemoryPredefinedSplitLoader(
            data_name="custom", task_mode="forecasting"
        )
        original = loader.get_cache_fingerprint()

        loader._repr_config["data_name"] = "mutated"
        loader._repr_config["task_mode"] = "fault"
        loader._repr_config["multisource_data_splitter"] = {"_target_": "changed"}

        assert loader.get_cache_fingerprint() == original

    def test_isolated_from_nested_kwargs_mutation(self):
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
