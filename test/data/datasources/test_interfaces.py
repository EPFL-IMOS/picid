"""
Tests for base datasource interfaces (AbstractDataSourceLoader contract).
"""
from __future__ import annotations

import pytest

from picid.data.datasources.base.interfaces import AbstractDataSourceLoader
from picid.data.data_objects.data import SplitDatasetContainer


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

    def split_data(self) -> None:
        pass


def test_interface_get_split_mode_per_unit_when_no_multisource_splitter():
    """get_split_mode returns 'per_unit' when multisource_data_splitter is absent."""
    loader = ConcreteLoader()
    assert loader.get_split_mode() == "per_unit"


def test_interface_get_split_mode_cross_unit_when_multisource_splitter_present():
    """get_split_mode returns 'cross_unit' when multisource_data_splitter is provided."""
    loader = ConcreteLoader(multisource_data_splitter=object())
    assert loader.get_split_mode() == "cross_unit"


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
