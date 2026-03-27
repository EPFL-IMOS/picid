"""
Characterization tests for phmd_loader: convert_outer_list_to_inner and PHMDMultiSourceLoader interface.

Uses mock for phmd package so tests run without PHMD data. Covers convert_outer_list_to_inner
and a minimal concrete PHMD loader subclass (load_data, get_data, get_meta_data, get_data_name, split_data, __repr__).
"""

from __future__ import annotations

import numpy as np
import pytest

from picid.data.datasources.base.phmd_loader import (
    convert_outer_list_to_inner,
    PHMDMultiSourceLoader,
)
from picid.data.datasources.base.exceptions import (
    DatasourceConfigurationError,
    DatasourceStateError,
)
from picid.data.data_objects import SplitDatasetContainer


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
    with pytest.raises(DatasourceStateError, match="must be loaded"):
        mock_phmd_loader.get_data()


def test_phmd_loader_get_meta_data(mock_phmd_loader):
    mock_phmd_loader.load_data()
    meta = mock_phmd_loader.get_meta_data()
    assert meta is not None


def test_phmd_loader_get_data_name(mock_phmd_loader):
    mock_phmd_loader.load_data()
    assert mock_phmd_loader.get_data_name() == "MockPHMD"
    assert mock_phmd_loader.get_data_names() == ("MockPHMD",)


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


def test_phmd_loader_rejects_multisource_splitter(tmp_path):
    """PHMDMultiSourceLoader should fail fast on incompatible splitter config."""
    with pytest.raises(
        DatasourceConfigurationError,
        match="does not accept multisource_data_splitter",
    ):
        MockPHMDLoader(
            data_name="X",
            task_mode="rul",
            fold=0,
            cache_dir=str(tmp_path),
            multisource_data_splitter={"_target_": "some.Splitter"},
        )


def test_phmd_loader_get_cache_file_name(mock_phmd_loader):
    """_get_cache_file_name returns expected pattern with task_mode, fold, dataset, type."""
    name = mock_phmd_loader._get_cache_file_name("train")
    assert "rul" in name
    assert "fold_0" in name
    assert "MockPHMD" in name
    assert "train" in name
    assert name.endswith(".pkl")


def test_phmd_loader_get_cache_file_name_with_auxiliary(mock_phmd_loader):
    """_get_cache_file_name with auxiliary_tasks includes them."""
    mock_phmd_loader.auxiliary_tasks = ["aux1", "aux2"]
    name = mock_phmd_loader._get_cache_file_name("val")
    assert "aux1_aux2" in name or "aux_" in name


def test_phmd_loader_retrieve_cached_task_data_missing_raises(
    mock_phmd_loader, tmp_path
):
    """_retrieve_cached_task_data raises FileNotFoundError when path does not exist."""
    missing = tmp_path / "missing.pkl"
    with pytest.raises(FileNotFoundError, match="Cache file not found"):
        mock_phmd_loader._retrieve_cached_task_data(missing)


def test_phmd_loader_retrieve_cached_task_data_exists(mock_phmd_loader, tmp_path):
    """_retrieve_cached_task_data returns unpickled data when file exists."""
    import pickle

    cache_file = tmp_path / "cached.pkl"
    data = {"x": [1, 2, 3]}
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    out = mock_phmd_loader._retrieve_cached_task_data(cache_file)
    assert out == data


def test_phmd_loader_init_creates_cache_dir(tmp_path):
    """When cache_dir path does not exist, __init__ creates it."""
    cache_dir = tmp_path / "new_cache_subdir"
    assert not cache_dir.exists()
    MockPHMDLoader(
        data_name="X",
        task_mode="rul",
        fold=0,
        cache_dir=str(cache_dir),
    )
    assert cache_dir.exists()
    assert cache_dir.is_dir()


class LoaderWithMockedInit(PHMDMultiSourceLoader):
    """Runs real _load_data loop by mocking _init_phmd_dataset; no phmd I/O."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_source_dict = {}

    def _init_phmd_dataset(self, task_names):
        """Return minimal main_fold so the for-split/unit loop runs."""
        import pandas as pd

        self.meta_data = {}
        unit_col = "unit_id"
        train_df = pd.DataFrame(
            {
                unit_col: [1],
                "feat0": [1.0],
                "feat1": [2.0],
                "rul": [10.0],
            }
        )
        val_df = train_df.copy()
        test_df = train_df.copy()
        main_fold = {"train": train_df, "val": val_df, "test": test_df}
        return {self.task_mode: main_fold}

    def _preprocess_fold(self, main_fold, auxiliary_tasks_fold_dict):
        return main_fold

    def _get_features_columns(self):
        return ["feat0", "feat1"]

    def _get_target_column(self):
        return "rul"

    def _get_unit_column(self):
        return "unit_id"

    def _process_unit(self, df_unit, unit_name, features_col, target_col):
        return {
            "features": df_unit[features_col].values,
            "target": df_unit[[target_col]].values,
            "metadata": {"unit_name": f"unit_{unit_name}", "unit_id": unit_name},
        }


def test_phmd_loader_load_data_full_path_mocked_init(tmp_path):
    """Full _load_data path with _init_phmd_dataset mocked (no real phmd)."""
    loader = LoaderWithMockedInit(
        data_name="TestDS",
        task_mode="rul",
        fold=0,
        cache_dir=str(tmp_path),
    )
    loader.load_data()
    data = loader.get_data()
    assert "features" in data
    assert "train" in data["features"]
    assert len(data["features"]["train"]) == 1
    assert loader.get_meta_data() is not None
    assert "unit_names" in loader.meta_data or "unit_ids" in loader.meta_data
