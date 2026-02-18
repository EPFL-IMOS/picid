"""Fixtures for loader characterization tests: in-memory SingleSourceLoader and configs."""
import numpy as np
import pytest
from omegaconf import OmegaConf

from lmetk.data.datasources.base.single_source_loader import SingleSourceLoader
from lmetk.data.preprocessing.splits import TimeSplitter


class InMemorySingleSourceLoader(SingleSourceLoader):
    """Concrete SingleSourceLoader that returns fixed in-memory arrays (for tests only)."""

    def __init__(self, n_samples=200, n_features=5, data_name="in_memory", **kwargs):
        if "data_name" not in kwargs:
            kwargs["data_name"] = data_name
        if "task_mode" not in kwargs:
            kwargs["task_mode"] = "regression"
        data_splitter = kwargs.pop("data_splitter", None)
        if data_splitter is None:
            # Use ratio-based splits with test=None so TimeSplitter uses n - train_len - val_len for test
            data_splitter = TimeSplitter(
                train=0.5,
                val=0.25,
                test=None,
                seq_len=10,
                pred_len=5,
                create_splits_for=["features", "timestamps", "target"],
            )
        super().__init__(data_splitter=data_splitter, **kwargs)
        self._n_samples = n_samples
        self._n_features = n_features

    def _load_data(self):
        rng = np.random.default_rng(42)
        return {
            "features": rng.standard_normal((self._n_samples, self._n_features)),
            "timestamps": np.arange(self._n_samples, dtype=float),
            "target": rng.standard_normal((self._n_samples, 1)),
        }


@pytest.fixture
def in_memory_loader():
    """Single in-memory loader (standalone: load → split → get_data returns SplitDatasetContainer)."""
    return InMemorySingleSourceLoader(n_samples=200, n_features=5, data_name="unit_a")


@pytest.fixture
def in_memory_loader_part_of_multi():
    """In-memory loader as part of multi (no splitter; get_data returns DatasetContainer)."""
    return InMemorySingleSourceLoader(
        n_samples=100,
        n_features=3,
        data_name="unit_b",
        is_part_of_multisource=True,
    )


def _two_source_config():
    """Config for two sources without multisource splitter (each source splits itself)."""
    train, val, test = 0.5, 0.25, 0.25
    splitter_cfg = {
        "train": train,
        "val": val,
        "test": None,  # ratio-based: test = n - train - val
        "seq_len": 10,
        "pred_len": 5,
        "create_splits_for": ["features", "timestamps", "target"],
    }
    source_a = {
        "_target_": "test.data.datasources.base.conftest.InMemorySingleSourceLoader",
        "n_samples": 150,
        "n_features": 4,
        "data_name": "unit_a",
        "task_mode": "regression",
        "data_splitter": {"_target_": "lmetk.data.preprocessing.splits.TimeSplitter", **splitter_cfg},
    }
    source_b = {
        "_target_": "test.data.datasources.base.conftest.InMemorySingleSourceLoader",
        "n_samples": 120,
        "n_features": 4,
        "data_name": "unit_b",
        "task_mode": "regression",
        "data_splitter": {"_target_": "lmetk.data.preprocessing.splits.TimeSplitter", **splitter_cfg},
    }
    return {
        "data_name": "multi_test",
        "task_mode": "regression",
        "source_list": {"unit_a": source_a, "unit_b": source_b},
        "multisource_data_splitter": None,
        "unit_a": OmegaConf.create(source_a),
        "unit_b": OmegaConf.create(source_b),
    }


@pytest.fixture
def two_sources_config_no_multisource_splitter():
    return _two_source_config()


def _two_sources_with_by_source_splitter_config():
    """Config for three sources with BySourceSplitter (train=unit_a, val=unit_b, test=unit_c)."""
    source_a = {
        "_target_": "test.data.datasources.base.conftest.InMemorySingleSourceLoader",
        "n_samples": 100,
        "n_features": 3,
        "data_name": "unit_a",
        "task_mode": "regression",
        "is_part_of_multisource": True,
    }
    source_b = {
        "_target_": "test.data.datasources.base.conftest.InMemorySingleSourceLoader",
        "n_samples": 80,
        "n_features": 3,
        "data_name": "unit_b",
        "task_mode": "regression",
        "is_part_of_multisource": True,
    }
    source_c = {
        "_target_": "test.data.datasources.base.conftest.InMemorySingleSourceLoader",
        "n_samples": 60,
        "n_features": 3,
        "data_name": "unit_c",
        "task_mode": "regression",
        "is_part_of_multisource": True,
    }
    return {
        "data_name": "multi_by_source",
        "task_mode": "regression",
        "source_list": {"unit_a": source_a, "unit_b": source_b, "unit_c": source_c},
        "multisource_data_splitter": {
            "_target_": "lmetk.data.preprocessing.splits.BySourceSplitter",
            "sources_train": ["unit_a"],
            "sources_val": ["unit_b"],
            "sources_test": ["unit_c"],
        },
        "unit_a": OmegaConf.create(source_a),
        "unit_b": OmegaConf.create(source_b),
        "unit_c": OmegaConf.create(source_c),
    }


@pytest.fixture
def two_sources_config_with_by_source_splitter():
    return _two_sources_with_by_source_splitter_config()
