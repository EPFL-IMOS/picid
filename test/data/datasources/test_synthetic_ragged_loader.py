"""Tests for SyntheticRaggedLoader: ragged data mimicking unibo/pronostia structure."""
from __future__ import annotations

import pytest

try:
    import awkward as ak
except ImportError:
    ak = None

from test.data.datasources.synthetic_ragged_loader import SyntheticRaggedLoader


@pytest.mark.skipif(ak is None, reason="awkward not installed")
class TestSyntheticRaggedLoader:
    """Test SyntheticRaggedLoader produces valid ragged output."""

    def test_load_split_get_returns_split_dataset_container(self):
        """Instantiate loader, call load_data, split_data, get_data; assert SplitDatasetContainer."""
        loader = SyntheticRaggedLoader(seed=42, n_units=3)
        loader.load_data()
        loader.split_data()
        container = loader.get_data()

        from picid.data.data_objects.data import SplitDatasetContainer

        assert isinstance(container, SplitDatasetContainer)
        assert "features" in container
        assert "rul" in container
        assert "unit_id" in container

    def test_features_are_list_of_ak_array(self):
        """Assert features per split are list of ak.Array."""
        loader = SyntheticRaggedLoader(seed=42, n_units=3)
        loader.load_data()
        loader.split_data()
        container = loader.get_data()

        for split in ["train", "val", "test"]:
            feats = container["features"][split]
            assert isinstance(feats, list), f"{split} features should be list"
            assert len(feats) >= 1, f"{split} should have at least one unit"
            for arr in feats:
                assert isinstance(arr, ak.Array), f"{split} unit should be ak.Array, got {type(arr)}"

    def test_variable_lengths_across_units(self):
        """Assert variable lengths across units (ragged structure)."""
        loader = SyntheticRaggedLoader(seed=123, n_units=5)
        loader.load_data()

        # Before split: raw lists
        features_list = loader.data_dict["features"]
        assert isinstance(features_list, list)
        lengths = [len(ak.to_numpy(arr)) for arr in features_list]
        # With seed 123, n_units=5, lengths should vary (50-80 range)
        assert len(set(lengths)) > 1, "Units should have variable lengths (ragged)"

    def test_get_data_structure(self):
        """Assert get_data returns split -> key -> list of arrays structure."""
        loader = SyntheticRaggedLoader(seed=42, n_units=3)
        loader.load_data()
        loader.split_data()
        container = loader.get_data()

        split_dict = container.to_split_dict()
        for split in ["train", "val", "test"]:
            assert split in split_dict
            assert "features" in split_dict[split]
            assert "rul" in split_dict[split]
            assert "unit_id" in split_dict[split]
            # Each split has list of units
            feats = split_dict[split]["features"]
            assert isinstance(feats, list)
            assert len(feats) == 1, "RaggedByUnitSplitter assigns 1 unit per split"
