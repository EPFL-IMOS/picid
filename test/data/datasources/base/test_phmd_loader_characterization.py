"""
Characterization tests for phmd_loader: convert_outer_list_to_inner and PHMDMultiSourceLoader interface.

Uses mock for phmd package so tests run without PHMD data. Covers convert_outer_list_to_inner
and a minimal concrete PHMD loader subclass (load_data, get_data, get_meta_data, get_data_name, split_data, __repr__).
"""
from __future__ import annotations

import numpy as np
import pytest

# Mock phmd before importing phmd_loader so tests run without PHMD data
import sys
from unittest.mock import MagicMock

sys.modules["phmd"] = MagicMock()
sys.modules["phmd.datasets"] = MagicMock()

from lmetk.data.datasources.base.phmd_loader import (
    convert_outer_list_to_inner,
    PHMDMultiSourceLoader,
)
from lmetk.data.data_objects.data import SplitDatasetContainer


# ----- convert_outer_list_to_inner -----


def test_convert_outer_list_to_inner_empty():
    assert convert_outer_list_to_inner([]) == {}


def test_convert_outer_list_to_inner_single():
    data_list = [{"a": [1, 2], "b": [3, 4]}]
    out = convert_outer_list_to_inner(data_list)
    assert out == {"a": [[1, 2]], "b": [[3, 4]]}


def test_convert_outer_list_to_inner_multiple():
    data_list = [
        {"features": np.zeros((2, 3)), "target": np.zeros(2)},
        {"features": np.ones((2, 3)), "target": np.ones(2)},
    ]
    out = convert_outer_list_to_inner(data_list)
    assert list(out.keys()) == ["features", "target"]
    assert len(out["features"]) == 2
    assert len(out["target"]) == 2


def test_convert_outer_list_to_inner_mismatch_keys_raises():
    data_list = [
        {"a": [1], "b": [2]},
        {"a": [3], "c": [4]},
    ]
    with pytest.raises(ValueError, match="same keys"):
        convert_outer_list_to_inner(data_list)


def test_convert_outer_list_to_inner_same_length_per_key():
    # stacked[key] has length len(data_list); all keys have same length so no raise
    data_list = [
        {"a": [1, 2], "b": [3, 4]},
        {"a": [5, 6], "b": [7, 8]},
    ]
    out = convert_outer_list_to_inner(data_list)
    assert len(out["a"]) == 2 and len(out["b"]) == 2


# ----- PHMDMultiSourceLoader: minimal concrete subclass (no phmd dependency) -----


class MockPHMDLoader(PHMDMultiSourceLoader):
    """Minimal PHMD loader that returns fixed data without calling phmd."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_source_dict = {}  # repr uses len(self.data_source_dict) when loaded

    def _load_data(self):
        # Return same structure as real _load_data: out_dict with features, target, metadata, etc.
        self.meta_data = {"unit_ids": [0], "source": "mock"}
        train_f = [np.zeros((10, 2))]
        val_f = [np.zeros((5, 2))]
        test_f = [np.zeros((5, 2))]
        train_t = [np.zeros((10, 1))]
        val_t = [np.zeros((5, 1))]
        test_t = [np.zeros((5, 1))]
        meta = [{"unit_id": 0}]
        return {
            "features": {"train": train_f, "val": val_f, "test": test_f},
            "target": {"train": train_t, "val": val_t, "test": test_t},
            "metadata": {"train": meta, "val": meta, "test": meta},
        }

    def _preprocess_fold(self, main_fold, auxiliary_tasks_fold_dict):
        return main_fold

    def _get_features_columns(self):
        return ["feat0", "feat1"]

    def _get_target_column(self):
        return "target"

    def _get_unit_column(self):
        return "unit_id"

    def _process_unit(self, df_unit, unit_name, features_col, target_col):
        raise NotImplementedError("Not used in mock _load_data")


@pytest.fixture
def mock_phmd_loader(tmp_path):
    return MockPHMDLoader(
        data_name="MockPHMD",
        task_mode="rul",
        fold=0,
        cache_dir=str(tmp_path),
    )


def test_phmd_loader_load_data_get_data(mock_phmd_loader):
    mock_phmd_loader.load_data()
    data = mock_phmd_loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert "features" in data
    assert "target" in data
    assert "train" in data["features"]


def test_phmd_loader_get_data_before_load_raises(mock_phmd_loader):
    with pytest.raises(AssertionError, match="must be loaded"):
        mock_phmd_loader.get_data()


def test_phmd_loader_get_meta_data(mock_phmd_loader):
    mock_phmd_loader.load_data()
    meta = mock_phmd_loader.get_meta_data()
    assert meta is not None


def test_phmd_loader_get_data_name(mock_phmd_loader):
    mock_phmd_loader.load_data()
    # get_data_name returns self.data_dict["metadata"] (structure differs from multi_source)
    name = mock_phmd_loader.get_data_name()
    assert name is not None


def test_phmd_loader_split_data_warns(mock_phmd_loader, caplog):
    mock_phmd_loader.split_data()
    assert "PHMD" in caplog.text or "split" in caplog.text.lower()


def test_phmd_loader_repr_unloaded(mock_phmd_loader):
    r = repr(mock_phmd_loader)
    assert "MultiSource" in r or "unloaded" in r.lower()


def test_phmd_loader_repr_loaded(mock_phmd_loader):
    mock_phmd_loader.load_data()
    r = repr(mock_phmd_loader)
    assert "MultiSource" in r or "units" in r
