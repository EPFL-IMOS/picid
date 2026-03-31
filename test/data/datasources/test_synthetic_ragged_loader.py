"""Tests for SyntheticRaggedFromFileLoader: ragged data from pre-generated fixtures.

Fixture generator: test/scripts/snapshot/generate_snapshot_ragged_fixtures.py
  Output: test/fixtures/snapshot/data/ragged_prognostics.pkl

When the ragged fixture format has been modified:
  uv run python test/scripts/snapshot/generate_snapshot_ragged_fixtures.py

Then commit test/fixtures/snapshot/ragged_prognostics.pkl.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import awkward as ak
except ImportError:
    ak = None

from test.data.datasources.synthetic_ragged_loader import SyntheticRaggedFromFileLoader

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "snapshot"
    / "data"
    / "ragged_prognostics.pkl"
)


@pytest.mark.skipif(ak is None, reason="awkward not installed")
@pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="Run test/scripts/snapshot/generate_snapshot_ragged_fixtures.py first",
)
class TestSyntheticRaggedFromFileLoader:
    """Test SyntheticRaggedFromFileLoader loads valid ragged output from fixtures."""

    def test_load_split_get_returns_split_dataset_container(self):
        """Instantiate loader, call load_data, split_data, get_data; assert SplitDatasetContainer."""
        loader = SyntheticRaggedFromFileLoader(data_path=str(FIXTURE_PATH))
        loader.load_data()
        loader.split_data()
        container = loader.get_data()

        from picid.data.data_objects import SplitDatasetContainer

        assert isinstance(container, SplitDatasetContainer)
        assert "features" in container
        assert "rul" in container
        assert "unit_id" in container

    def test_features_are_list_of_ak_array(self):
        """Assert features per split are list of ak.Array."""
        loader = SyntheticRaggedFromFileLoader(data_path=str(FIXTURE_PATH))
        loader.load_data()
        loader.split_data()
        container = loader.get_data()

        for split in ["train", "val", "test"]:
            feats = container["features"][split]
            assert isinstance(feats, list), f"{split} features should be list"
            assert len(feats) >= 1, f"{split} should have at least one unit"
            for arr in feats:
                assert isinstance(
                    arr, ak.Array
                ), f"{split} unit should be ak.Array, got {type(arr)}"

    def test_variable_lengths_across_units(self):
        """Assert variable lengths across units (ragged structure)."""
        loader = SyntheticRaggedFromFileLoader(data_path=str(FIXTURE_PATH))
        loader.load_data()

        # Before split: raw lists
        features_list = loader.data_dict["features"]
        assert isinstance(features_list, list)
        lengths = [len(ak.to_numpy(arr)) for arr in features_list]
        assert len(set(lengths)) > 1, "Units should have variable lengths (ragged)"

    def test_get_data_structure(self):
        """Assert get_data returns split -> key -> list of arrays structure."""
        loader = SyntheticRaggedFromFileLoader(data_path=str(FIXTURE_PATH))
        loader.load_data()
        loader.split_data()
        container = loader.get_data()

        split_dict = container.to_split_dict()
        for split in ["train", "val", "test"]:
            assert split in split_dict
            assert "features" in split_dict[split]
            assert "rul" in split_dict[split]
            assert "unit_id" in split_dict[split]
            feats = split_dict[split]["features"]
            assert isinstance(feats, list)
            assert len(feats) == 1, "One unit per split"
