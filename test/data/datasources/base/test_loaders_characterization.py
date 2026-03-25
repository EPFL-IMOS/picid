"""
Characterization tests for SingleSourceLoader and MultiSourceLoader (Phase 0).

Covers: standalone single (load → split → get_data → SplitDatasetContainer),
multi without multisource splitter, multi with BySourceSplitter, and edge cases
(metadata normalization, ragged/empty splits).
"""

import pytest
from omegaconf import OmegaConf

from picid.data.data_objects.data import DatasetContainer, SplitDatasetContainer
from picid.data.datasources.base.contracts import LoaderState
from picid.data.datasources.base.exceptions import DatasourceStateError
from picid.data.datasources.base.multi_source_loader import MultiSourceLoader
from picid.data.datasources.base.predefined_split_loader import (
    PredefinedSplitLoaderBase,
)

from test.data.datasources.base.conftest import (
    InMemorySingleSourceLoader,
)


class InMemoryPredefinedChildLoader(PredefinedSplitLoaderBase):
    """Minimal predefined-split child loader for multisource regression tests."""

    def __init__(self, **kwargs):
        kwargs.setdefault("data_name", "predefined_child")
        kwargs.setdefault("task_mode", "regression")
        super().__init__(**kwargs)

    def _load_data(self) -> dict:
        return {
            "features": {
                "train": [["train_child"]],
                "val": [["val_child"]],
                "test": [["test_child"]],
            },
            "target": {
                "train": [[1]],
                "val": [[2]],
                "test": [[3]],
            },
        }


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
    """get_data() before load_data() raises a datasource state error."""
    loader = in_memory_loader
    with pytest.raises(DatasourceStateError, match="must be loaded"):
        loader.get_data()


def test_single_source_loader_get_data_before_split_raises(in_memory_loader):
    """Standalone: get_data() before split_data() raises a datasource state error."""
    loader = in_memory_loader
    loader.load_data()
    with pytest.raises(DatasourceStateError, match="must be splitted"):
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
    """get_meta_data() before load_data() raises a datasource state error."""
    loader = in_memory_loader
    with pytest.raises(DatasourceStateError, match="must be loaded"):
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


def test_single_source_loader_internal_raw_helper_returns_dataset_container(
    in_memory_loader,
):
    """Internal raw helper returns the unsplit unit payload before public split access."""
    loader = in_memory_loader
    loader.load_data()
    data = loader._get_loaded_data_container()
    assert isinstance(data, DatasetContainer)
    assert not isinstance(data, SplitDatasetContainer)
    assert hasattr(data["features"], "shape")


def test_single_source_loader_internal_split_helper_returns_split_container(
    in_memory_loader,
):
    """Internal split helper returns the normalized split payload for orchestration."""
    loader = in_memory_loader
    loader.load_data()
    loader.split_data()
    data = loader._get_split_data_container()
    assert isinstance(data, SplitDatasetContainer)
    assert isinstance(data["features"]["train"], list)
    assert len(data["features"]["train"]) == 1


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


def test_multi_source_loader_accepts_predefined_split_children():
    """MultiSourceLoader should concatenate predefined-split child loaders."""
    loader = MultiSourceLoader(
        data_name="multi_predefined",
        task_mode="regression",
        source_list={"child_a": {}, "child_b": {}},
        child_a=OmegaConf.create(
            {
                "_target_": (
                    "test.data.datasources.base.test_loaders_characterization."
                    "InMemoryPredefinedChildLoader"
                ),
                "data_name": "child_a",
                "task_mode": "regression",
            }
        ),
        child_b=OmegaConf.create(
            {
                "_target_": (
                    "test.data.datasources.base.test_loaders_characterization."
                    "InMemoryPredefinedChildLoader"
                ),
                "data_name": "child_b",
                "task_mode": "regression",
            }
        ),
    )

    loader.load_data()
    loader.split_data()
    data = loader.get_data()

    assert isinstance(data, SplitDatasetContainer)
    assert data["features"]["train"] == [["train_child"], ["train_child"]]
    assert data["target"]["test"] == [[3], [3]]


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


def test_multi_source_loader_without_multisource_splitter_uses_internal_split_helper(
    two_sources_config_no_multisource_splitter_private_contract,
):
    """MultiSourceLoader should not depend on child public get_data() for per-source splits."""
    loader = MultiSourceLoader(
        **two_sources_config_no_multisource_splitter_private_contract
    )
    loader.load_data()
    loader.split_data()
    data = loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert len(data["features"]["train"]) >= 1


def test_multi_source_loader_with_by_source_splitter_uses_internal_raw_helper(
    two_sources_config_with_by_source_splitter_private_contract,
):
    """MultiSourceLoader should consume raw child payloads through the internal helper path."""
    loader = MultiSourceLoader(
        **two_sources_config_with_by_source_splitter_private_contract
    )
    loader.load_data()
    loader.split_data()
    data = loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert set(data["features"]) == {"train", "val", "test"}


def test_multi_source_loader_get_data_before_load_raises(
    two_sources_config_no_multisource_splitter,
):
    """MultiSourceLoader get_data() before load_data() raises a datasource state error."""
    cfg = two_sources_config_no_multisource_splitter
    loader = MultiSourceLoader(**cfg)
    with pytest.raises(DatasourceStateError, match="must be loaded"):
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
    """MultiSourceLoader get_data_name() returns the outer datasource identity."""
    cfg = two_sources_config_no_multisource_splitter
    loader = MultiSourceLoader(**cfg)
    names = loader.get_data_name()
    assert names == "multi_test"


def test_multi_source_loader_get_source_names(
    two_sources_config_no_multisource_splitter,
):
    """MultiSourceLoader exposes child source identities separately from the outer datasource."""
    cfg = two_sources_config_no_multisource_splitter
    loader = MultiSourceLoader(**cfg)
    assert loader.get_source_names() == ("unit_a", "unit_b")


def test_multi_source_loader_get_data_names_returns_outer_datasource_name(
    two_sources_config_no_multisource_splitter,
):
    """Canonical datasource names should identify the outer datasource, not child units."""
    cfg = two_sources_config_no_multisource_splitter
    loader = MultiSourceLoader(**cfg)
    assert loader.get_data_names() == ("multi_test",)


def test_multi_source_loader_get_multisource_data_splitter(
    two_sources_config_no_multisource_splitter,
    two_sources_config_with_by_source_splitter,
):
    """get_multisource_data_splitter() returns None or BySourceSplitter instance."""
    loader_no = MultiSourceLoader(**two_sources_config_no_multisource_splitter)
    assert loader_no.get_multisource_data_splitter() is None

    loader_yes = MultiSourceLoader(**two_sources_config_with_by_source_splitter)
    from picid.data.split_strategies import BySourceSplitter

    assert isinstance(loader_yes.get_multisource_data_splitter(), BySourceSplitter)


def test_multi_source_loader_split_data_warns_when_already_splitted(
    two_sources_config_no_multisource_splitter,
):
    """split_data when loader._is_splitted already True logs warning and returns (coverage)."""
    loader = MultiSourceLoader(**two_sources_config_no_multisource_splitter)
    loader.load_data()
    loader.split_data()
    loader._is_splitted = True
    loader.split_data()  # early return with warning


def test_multi_source_loader_split_data_raises_when_source_not_loaded(
    two_sources_config_no_multisource_splitter,
):
    """split_data raises when a source has not been loaded (coverage)."""
    loader = MultiSourceLoader(**two_sources_config_no_multisource_splitter)
    loader.load_data()
    loader._is_loaded = True
    loader.data_source_dict["unit_b"]._is_loaded = False
    loader.data_source_dict["unit_b"]._state = LoaderState.INITIALIZED
    with pytest.raises(DatasourceStateError, match="has not been loaded"):
        loader.split_data()


def test_multi_source_loader_warns_when_source_fails_to_load():
    """load_data logs warning when a source stays outside the loaded lifecycle state."""
    from omegaconf import OmegaConf

    ok_source = {
        "_target_": "test.data.datasources.base.conftest.InMemorySingleSourceLoader",
        "n_samples": 50,
        "n_features": 2,
        "data_name": "ok",
        "task_mode": "regression",
    }
    fail_source = {
        "_target_": "test.data.datasources.base.conftest.FailingSingleSourceLoader",
        "data_name": "failing",
    }
    config = {
        "data_name": "mixed",
        "task_mode": "regression",
        "source_list": {"ok": ok_source, "failing": fail_source},
        "multisource_data_splitter": None,
        "ok": OmegaConf.create(ok_source),
        "failing": OmegaConf.create(fail_source),
    }
    loader = MultiSourceLoader(**config)
    loader.load_data()
    assert loader.data_source_dict["ok"].is_loaded()
    assert not loader.data_source_dict["failing"].is_loaded()


def test_single_source_loader_reload_clears_previous_split_state(in_memory_loader):
    """Reloading a single-source loader should require an explicit split again."""
    loader = in_memory_loader
    loader.load_data()
    loader.split_data()
    loader.load_data()
    assert loader._is_loaded is True
    assert loader._is_splitted is False
    with pytest.raises(DatasourceStateError, match="must be splitted"):
        loader.get_data()


def test_multi_source_loader_reload_clears_previous_container_state(
    two_sources_config_no_multisource_splitter,
):
    """Reloading a multisource loader should drop stale split containers until split_data() runs again."""
    loader = MultiSourceLoader(**two_sources_config_no_multisource_splitter)
    loader.load_data()
    loader.split_data()
    loader.load_data()
    assert loader._is_loaded is True
    assert loader._is_splitted is False
    assert loader.container is None
    with pytest.raises(DatasourceStateError, match="must be splitted"):
        loader.get_data()


def test_multi_source_loader_cache_fingerprint_deduplicates_child_configs(
    two_sources_config_no_multisource_splitter,
):
    """Cache fingerprints should avoid hashing the same child config twice."""
    loader = MultiSourceLoader(**two_sources_config_no_multisource_splitter)
    fingerprint = loader.get_cache_fingerprint()
    assert fingerprint["data_name"] == "multi_test"
    assert fingerprint["source_order"] == ["unit_a", "unit_b"]
    assert set(fingerprint["source_list"]) == {"unit_a", "unit_b"}
    assert "unit_a" not in fingerprint
    assert "unit_b" not in fingerprint


def test_multi_source_loader_cache_fingerprint_ignores_late_repr_config_mutation(
    two_sources_config_no_multisource_splitter,
):
    """Cache identity should be driven by explicit state, not mutable repr snapshots."""
    loader = MultiSourceLoader(**two_sources_config_no_multisource_splitter)
    original = loader.get_cache_fingerprint()

    loader._repr_config["data_name"] = "mutated"
    loader._repr_config["multisource_data_splitter"] = {"_target_": "changed"}
    loader._repr_config["unit_a"] = {"_target_": "mutated.child"}

    assert loader.get_cache_fingerprint() == original


def test_multi_source_loader_repr(
    two_sources_config_no_multisource_splitter,
):
    """MultiSourceLoader __repr__ includes source count."""
    loader = MultiSourceLoader(**two_sources_config_no_multisource_splitter)
    r = repr(loader)
    assert "MultiSource" in r or "Multi" in r
    assert "2" in r or "units" in r


# ----- Interface: get_split_mode -----


def test_single_source_loader_get_split_mode_within_units(in_memory_loader):
    """SingleSourceLoader get_split_mode() returns the canonical within-units mode."""
    assert in_memory_loader.get_split_mode() == "within_units"


def test_multi_source_loader_get_split_mode_without_splitter(
    two_sources_config_no_multisource_splitter,
):
    """MultiSourceLoader without multisource splitter uses the canonical within-units mode."""
    loader = MultiSourceLoader(**two_sources_config_no_multisource_splitter)
    assert loader.get_split_mode() == "within_units"


def test_multi_source_loader_get_split_mode_with_splitter(
    two_sources_config_with_by_source_splitter,
):
    """MultiSourceLoader with BySourceSplitter uses the canonical between-units mode."""
    loader = MultiSourceLoader(**two_sources_config_with_by_source_splitter)
    assert loader.get_split_mode() == "between_units"
