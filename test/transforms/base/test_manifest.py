"""Tests for Phase 4.1: metadata manifest (ManifestEntry, MetadataManifest, pipeline recording)."""

import pytest

from picid.data.data_objects.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestEntry,
    MetadataManifest,
)
from picid.transforms.base.pipeline import RecordManifestStep, TransformContext

from test.transforms.base.conftest import (
    create_dummy_single_unit_container,
    DummyStatelessTransform,
)


class TestManifestEntry:
    def test_valid_category(self):
        e = ManifestEntry(
            schema_version="1.0",
            producer_version="0.1.0",
            category="transform",
            payload={"k": 1},
        )
        assert e.category == "transform"
        assert e.step_id is None

    def test_datasource_category(self):
        e = ManifestEntry(
            schema_version="1.0",
            producer_version="0.1.0",
            category="datasource",
            payload={},
            key="features",
        )
        assert e.key == "features"

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="category must be"):
            ManifestEntry(
                schema_version="1.0",
                producer_version="0.1.0",
                category="invalid",
                payload={},
            )


class TestMetadataManifest:
    def test_add_and_query(self):
        m = MetadataManifest()
        m.add(
            ManifestEntry(
                schema_version="1.0",
                producer_version="0.1.0",
                category="transform",
                payload={"a": 1},
                step_id="Scale",
                key="features",
            )
        )
        m.add(
            ManifestEntry(
                schema_version="1.0",
                producer_version="0.1.0",
                category="transform",
                payload={"b": 2},
                step_id="Other",
                split="train",
            )
        )
        assert len(m) == 2
        one = m.query(step_id="Scale")
        assert len(one) == 1
        assert one[0].payload["a"] == 1
        two = m.query(split="train")
        assert len(two) == 1
        assert two[0].step_id == "Other"
        all_t = m.query(category="transform")
        assert len(all_t) == 2

    def test_copy_deep(self):
        m = MetadataManifest()
        m.add(
            ManifestEntry(
                schema_version="1.0",
                producer_version="0.1.0",
                category="transform",
                payload={"nested": [1, 2]},
                step_id="X",
            )
        )
        m2 = m.copy(deep=True)
        m2._entries[0].payload["nested"].append(3)
        assert m._entries[0].payload["nested"] == [1, 2]

    def test_copy_shallow(self):
        m = MetadataManifest()
        m.add(
            ManifestEntry(
                schema_version="1.0",
                producer_version="0.1.0",
                category="transform",
                payload={"x": 1},
                step_id="X",
            )
        )
        m2 = m.copy(deep=False)
        assert m2.query(step_id="X")[0].payload["x"] == 1


class TestContainerManifest:
    def test_container_accepts_optional_manifest(self):
        man = MetadataManifest()
        container = create_dummy_single_unit_container()
        container.manifest = man
        assert container.manifest is man
        assert len(container.manifest) == 0

    def test_container_copy_copies_manifest(self):
        man = MetadataManifest()
        man.add(
            ManifestEntry(
                schema_version="1.0",
                producer_version="0.1.0",
                category="transform",
                payload={},
                step_id="A",
            )
        )
        container = create_dummy_single_unit_container()
        container.manifest = man
        copied = container.copy(deep=True)
        assert copied.manifest is not man
        assert len(copied.manifest) == 1
        assert copied.manifest.query(step_id="A")

    def test_container_init_manifest_kwarg(self):
        man = MetadataManifest()
        container = create_dummy_single_unit_container()
        container.manifest = man
        assert getattr(container, "manifest", None) is man

    def test_container_copy_without_manifest(self):
        container = create_dummy_single_unit_container()
        container.manifest = None  # explicit no-manifest
        assert getattr(container, "manifest", None) is None
        copied = container.copy(deep=True)
        assert getattr(copied, "manifest", None) is None


class TestRecordManifestStep:
    def test_no_manifest_no_op(self):
        container = create_dummy_single_unit_container()
        container.manifest = (
            None  # explicit no-manifest: RecordManifestStep should no-op
        )
        assert getattr(container, "manifest", None) is None
        ctx = TransformContext(
            data=container,
            transform_instance=DummyStatelessTransform(),
            apply_to_keys=["features"],
            assign_to_keys=["features"],
            assign_to_keys_map=["features"],
            strategy=None,
        )
        ctx.transformed_data = container.copy(deep=False)
        RecordManifestStep().run(ctx)
        assert getattr(ctx.transformed_data, "manifest", None) is None

    def test_with_manifest_appends_entry(self):
        container = create_dummy_single_unit_container()
        container.manifest = MetadataManifest()
        from picid.transforms.base.strategy import TransformStrategy

        strategy = TransformStrategy()
        result, log = strategy.apply(
            transform_instance=DummyStatelessTransform(),
            data=container,
            apply_to_keys="features",
            assign_to_keys="features",
            assign_to_keys_map=["features"],
        )
        # Result is a copy of the container; its manifest is a copy, and the entry was appended to that copy.
        assert result.manifest is not None
        assert len(result.manifest) == 1
        entry = result.manifest.query(category="transform")[0]
        assert entry.schema_version == MANIFEST_SCHEMA_VERSION
        assert entry.step_id == "DummyStatelessTransform"
        assert entry.payload["apply_to_keys"] == ["features"]
        assert "log_splits" in entry.payload
