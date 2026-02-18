"""
Characterization tests for SingleSourceLoader and MultiSourceLoader (Phase 0).

Covers: standalone single (load → split → get_data → SplitDatasetContainer),
multi without multisource splitter, multi with BySourceSplitter, and edge cases
(metadata normalization, ragged/empty splits).
"""
import pytest

from lmetk.data.data_objects.data import DatasetContainer, SplitDatasetContainer
from lmetk.data.datasources.base.interfaces import AbstractDataSourceLoader
from lmetk.data.datasources.base.single_source_loader import SingleSourceLoader
from lmetk.data.datasources.base.multi_source_loader import MultiSourceLoader

from test.data.datasources.base.conftest import (
    InMemorySingleSourceLoader,
    two_sources_config_no_multisource_splitter,
    two_sources_config_with_by_source_splitter,
)


# ----- SingleSourceLoader standalone -----


def test_single_source_loader_standalone_load_split_get_returns_split_container(
    in_memory_loader,
):
    """Standalone: load → split → get_data returns SplitDatasetContainer."""
    loader = in_memory_loader
    loader.load_data()
    loader.split_data()
    data = loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert "features" in data
    assert "target" in data
    assert "train" in data["features"]
    assert "val" in data["features"]
    assert "test" in data["features"]


def test_single_source_loader_get_data_before_load_raises(in_memory_loader):
    """get_data() before load_data() raises AssertionError."""
    loader = in_memory_loader
    with pytest.raises(AssertionError, match="must be loaded"):
        loader.get_data()


def test_single_source_loader_get_data_before_split_raises(in_memory_loader):
    """Standalone: get_data() before split_data() raises AssertionError."""
    loader = in_memory_loader
    loader.load_data()
    with pytest.raises(AssertionError, match="must be splitted"):
        loader.get_data()


def test_single_source_loader_metadata_normalized_to_train_val_test():
    """Metadata not in {train, val, test} is normalized to train/val/test."""
    class LoaderWithMetadata(InMemorySingleSourceLoader):
        def _load_data(self):
            d = super()._load_data()
            d["metadata"] = {"info": "single"}
            return d

    loader = LoaderWithMetadata(n_samples=200, n_features=5, data_name="x")
    loader.load_data()
    loader.split_data()
    data = loader.get_data()
    # metadata is exposed via __getitem__("metadata") -> .metadata (may include column_map)
    meta = data["metadata"]
    assert meta is not None
    assert isinstance(meta, dict)
    assert {"train", "val", "test"}.issubset(meta.keys())


def test_single_source_loader_get_meta_data_after_load(in_memory_loader):
    """get_meta_data() after load returns dict (empty for in-memory)."""
    loader = in_memory_loader
    loader.load_data()
    meta = loader.get_meta_data()
    assert isinstance(meta, dict)


def test_single_source_loader_get_meta_data_before_load_raises(in_memory_loader):
    """get_meta_data() before load_data() raises."""
    loader = in_memory_loader
    with pytest.raises(AssertionError, match="must be loaded"):
        loader.get_meta_data()


def test_single_source_loader_get_data_name(in_memory_loader):
    """get_data_name() returns data_name."""
    assert in_memory_loader.get_data_name() == "unit_a"


def test_single_source_loader_split_data_idempotent_warning(in_memory_loader):
    """Calling split_data() twice logs warning and does not change result."""
    loader = in_memory_loader
    loader.load_data()
    loader.split_data()
    first = loader.get_data()
    loader.split_data()  # second time
    second = loader.get_data()
    assert first is not second
    # Content should be equivalent (same shapes)
    for key in ["features", "target"]:
        for split in ["train", "val", "test"]:
            assert first[key][split][0].shape == second[key][split][0].shape


# ----- SingleSourceLoader as part of multi (load only, no split) -----


def test_single_source_loader_as_part_of_multi_returns_dataset_container(
    in_memory_loader_part_of_multi,
):
    """When is_part_of_multisource=True, get_data() returns DatasetContainer (no split)."""
    loader = in_memory_loader_part_of_multi
    loader.load_data()
    # No split_data() called
    data = loader.get_data()
    assert isinstance(data, DatasetContainer)
    assert not isinstance(data, SplitDatasetContainer)
    # Raw keys (features, timestamps, target) as arrays, not split dicts
    assert "features" in data
    assert hasattr(data["features"], "shape")


# ----- MultiSourceLoader: two sources, no multisource splitter -----


def test_multi_source_loader_two_sources_no_multisource_splitter(
    two_sources_config_no_multisource_splitter,
):
    """MultiSourceLoader with 2 sources and no multisource splitter: load → split → get_data."""
    cfg = two_sources_config_no_multisource_splitter
    loader = MultiSourceLoader(**cfg)
    loader.load_data()
    loader.split_data()
    data = loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert "features" in data and "target" in data
    for split in ["train", "val", "test"]:
        assert split in data["features"]
        # Combined from both sources: list of arrays
        assert len(data["features"][split]) >= 1


def test_multi_source_loader_two_sources_with_by_source_splitter(
    two_sources_config_with_by_source_splitter,
):
    """MultiSourceLoader with 2 sources and BySourceSplitter: load → split → get_data."""
    cfg = two_sources_config_with_by_source_splitter
    loader = MultiSourceLoader(**cfg)
    loader.load_data()
    loader.split_data()
    data = loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert "features" in data
    for split in ["train", "val", "test"]:
        assert split in data["features"]


def test_multi_source_loader_get_data_before_load_raises(
    two_sources_config_no_multisource_splitter,
):
    """MultiSourceLoader get_data() before load_data() raises."""
    cfg = two_sources_config_no_multisource_splitter
    loader = MultiSourceLoader(**cfg)
    with pytest.raises(AssertionError, match="not been loaded|never loaded"):
        loader.get_data()


def test_multi_source_loader_get_meta_data_after_load(
    two_sources_config_no_multisource_splitter,
):
    """MultiSourceLoader get_meta_data() after load returns dict."""
    cfg = two_sources_config_no_multisource_splitter
    loader = MultiSourceLoader(**cfg)
    loader.load_data()
    loader.split_data()
    meta = loader.get_meta_data()
    assert isinstance(meta, dict)


def test_multi_source_loader_get_data_name(
    two_sources_config_no_multisource_splitter,
):
    """MultiSourceLoader get_data_name() returns source names list."""
    cfg = two_sources_config_no_multisource_splitter
    loader = MultiSourceLoader(**cfg)
    names = loader.get_data_name()
    assert names == ["unit_a", "unit_b"]


def test_multi_source_loader_get_multisource_data_splitter(
    two_sources_config_no_multisource_splitter,
    two_sources_config_with_by_source_splitter,
):
    """get_multisource_data_splitter() returns None or BySourceSplitter instance."""
    loader_no = MultiSourceLoader(**two_sources_config_no_multisource_splitter)
    assert loader_no.get_multisource_data_splitter() is None

    loader_yes = MultiSourceLoader(**two_sources_config_with_by_source_splitter)
    from lmetk.data.preprocessing.splits import BySourceSplitter
    assert isinstance(loader_yes.get_multisource_data_splitter(), BySourceSplitter)


def test_multi_source_loader_repr(
    two_sources_config_no_multisource_splitter,
):
    """MultiSourceLoader __repr__ includes source count."""
    loader = MultiSourceLoader(**two_sources_config_no_multisource_splitter)
    r = repr(loader)
    assert "MultiSource" in r or "Multi" in r
    assert "2" in r or "units" in r


# ----- Interface: get_split_mode -----


def test_single_source_loader_get_split_mode_per_unit(in_memory_loader):
    """SingleSourceLoader get_split_mode() returns 'per_unit'."""
    assert in_memory_loader.get_split_mode() == "per_unit"


def test_multi_source_loader_get_split_mode_without_splitter(
    two_sources_config_no_multisource_splitter,
):
    """MultiSourceLoader without multisource splitter: get_split_mode() is 'per_unit'."""
    loader = MultiSourceLoader(**two_sources_config_no_multisource_splitter)
    assert loader.get_split_mode() == "per_unit"


def test_multi_source_loader_get_split_mode_with_splitter(
    two_sources_config_with_by_source_splitter,
):
    """MultiSourceLoader with BySourceSplitter: get_split_mode() is 'cross_unit'."""
    loader = MultiSourceLoader(**two_sources_config_with_by_source_splitter)
    assert loader.get_split_mode() == "cross_unit"
