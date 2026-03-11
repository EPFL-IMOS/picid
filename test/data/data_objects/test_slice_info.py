"""Tests for picid.data.data_objects.slice_info (SliceInfo) to reach full coverage."""

from picid.data.data_objects.slice_info import SliceInfo


class TestSliceInfoCoverage:
    """Target: cover copy(deep=True/False) branches including optional fields."""

    def test_copy_deep_with_all_fields_set(self):
        """copy(deep=True) with unit_ids, cycle_ids, bounds, index_map all set."""
        s = SliceInfo(
            split="train",
            unit_ids=[1, 2],
            cycle_ids=[10, 20],
            bounds={"t_min": 0, "t_max": 100},
            index_map={"orig": [0, 1], "sliced": [0, 1]},
        )
        s2 = s.copy(deep=True)
        assert s2.split == "train"
        assert s2.unit_ids == [1, 2]
        assert s2.unit_ids is not s.unit_ids
        assert s2.cycle_ids == [10, 20]
        assert s2.bounds == {"t_min": 0, "t_max": 100}
        assert s2.bounds is not s.bounds
        assert s2.index_map == {"orig": [0, 1], "sliced": [0, 1]}
        assert s2.index_map is not s.index_map
        s2.index_map["orig"].append(2)
        assert s.index_map["orig"] == [0, 1]

    def test_copy_deep_with_none_optionals(self):
        """copy(deep=True) when unit_ids/cycle_ids/bounds/index_map are None."""
        s = SliceInfo(split="val")
        s2 = s.copy(deep=True)
        assert s2.split == "val"
        assert s2.unit_ids is None
        assert s2.cycle_ids is None
        assert s2.bounds is None
        assert s2.index_map is None

    def test_copy_shallow_with_all_fields_set(self):
        """copy(deep=False) with all optional fields set (shallow list/dict copy)."""
        s = SliceInfo(
            split="test",
            unit_ids=[3, 4],
            cycle_ids=[30],
            bounds={"a": 1},
            index_map={"k": "v"},
        )
        s2 = s.copy(deep=False)
        assert s2.split == "test"
        assert s2.unit_ids == [3, 4]
        assert s2.cycle_ids == [30]
        assert s2.bounds == {"a": 1}
        assert s2.index_map == {"k": "v"}
        # Shallow: new list/dict but same content (implementation uses list()/dict())

    def test_copy_shallow_with_none_optionals(self):
        """copy(deep=False) when optionals are None."""
        s = SliceInfo()
        s2 = s.copy(deep=False)
        assert s2.split is None
        assert s2.unit_ids is None
        assert s2.cycle_ids is None
        assert s2.bounds is None
        assert s2.index_map is None
