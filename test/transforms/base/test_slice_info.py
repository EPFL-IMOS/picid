"""Tests for Phase 4.2: slice-awareness (SliceInfo, container, context)."""

from picid.data.data_objects.slice_info import SliceInfo
from picid.data.data_objects.manifest import MetadataManifest
from picid.transforms.base.pipeline import TransformContext

from test.transforms.base.conftest import (
    create_dummy_single_unit_container,
    DummyStatelessTransform,
)


class TestSliceInfo:
    def test_defaults(self):
        s = SliceInfo()
        assert s.split is None
        assert s.unit_ids is None
        assert s.bounds is None

    def test_with_split_and_units(self):
        s = SliceInfo(split="train", unit_ids=[0, 1, 2])
        assert s.split == "train"
        assert s.unit_ids == [0, 1, 2]

    def test_copy_deep(self):
        s = SliceInfo(split="val", unit_ids=[1, 2], bounds={"t_min": 0, "t_max": 10})
        s2 = s.copy(deep=True)
        s2.unit_ids.append(3)
        assert s.unit_ids == [1, 2]
        assert s2.unit_ids == [1, 2, 3]

    def test_copy_shallow(self):
        s = SliceInfo(unit_ids=[1, 2])
        s2 = s.copy(deep=False)
        assert s2.unit_ids == [1, 2]


class TestContainerSliceInfo:
    def test_container_accepts_slice_info(self):
        container = create_dummy_single_unit_container()
        si = SliceInfo(split="train", unit_ids=[0])
        container.slice_info = si
        assert container.slice_info is si

    def test_container_copy_copies_slice_info(self):
        si = SliceInfo(split="train", unit_ids=[0, 1])
        container = create_dummy_single_unit_container()
        container.slice_info = si
        copied = container.copy(deep=False)
        assert copied.slice_info is not si
        assert copied.slice_info.split == "train"
        assert copied.slice_info.unit_ids == [0, 1]

    def test_container_copy_without_slice_info(self):
        container = create_dummy_single_unit_container()
        assert getattr(container, "slice_info", None) is None
        copied = container.copy(deep=True)
        assert getattr(copied, "slice_info", None) is None


class TestContextSliceInfo:
    def test_context_slice_info_from_data(self):
        container = create_dummy_single_unit_container()
        container.slice_info = SliceInfo(split="train", unit_ids=[0, 1])
        from picid.transforms.base.strategy import TransformStrategy

        strategy = TransformStrategy()
        # Build context via apply; we only need to check that with manifest we get slice_info in payload
        container.manifest = MetadataManifest()
        result, _ = strategy.apply(
            transform_instance=DummyStatelessTransform(),
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
        )
        entry = result.manifest.query(category="transform")[0]
        assert "slice_info" in entry.payload
        assert entry.payload["slice_info"]["split"] == "train"
        assert entry.payload["slice_info"]["n_units"] == 2

    def test_context_slice_info_none_when_data_has_none(self):
        container = create_dummy_single_unit_container()
        ctx = TransformContext(
            data=container,
            transform_instance=DummyStatelessTransform(),
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
            slice_info=getattr(container, "slice_info", None),
        )
        assert ctx.slice_info is None
