"""
Characterization tests for DatasetContainer and SplitDatasetContainer (Phase 0).

Covers: split payload normalization, cardinality reporting, validate(),
to_split_dict(), group_by_split(), get_instance_cls().
"""

import logging

import numpy as np
import pandas as pd
import pytest
import awkward as ak

from picid.data.data_objects import (
    DatasetContainer,
    SplitDatasetContainer,
    SplitUnitCardinality,
    SplitViewPolicy,
)
from picid.data.data_objects import BaseDataObjectWithMetadata
from picid.data.data_objects.validation import describe_unit_payload


# ----- Fixtures -----


@pytest.fixture
def single_unit_split_container():
    """SplitDatasetContainer with one unit per split (single-unit, wrapped in lists)."""
    return SplitDatasetContainer(
        features={
            "train": [np.random.randn(100, 5)],
            "val": [np.random.randn(30, 5)],
            "test": [np.random.randn(30, 5)],
        },
        target={
            "train": [np.random.randn(100, 1)],
            "val": [np.random.randn(30, 1)],
            "test": [np.random.randn(30, 1)],
        },
    )


@pytest.fixture
def multi_unit_split_container():
    """SplitDatasetContainer with three units per split."""
    n_units = 3
    return SplitDatasetContainer(
        features={
            "train": [np.random.randn(100, 5) for _ in range(n_units)],
            "val": [np.random.randn(30, 5) for _ in range(n_units)],
            "test": [np.random.randn(30, 5) for _ in range(n_units)],
        },
        target={
            "train": [np.random.randn(100, 1) for _ in range(n_units)],
            "val": [np.random.randn(30, 1) for _ in range(n_units)],
            "test": [np.random.randn(30, 1) for _ in range(n_units)],
        },
    )


# ----- DatasetContainer: split payload normalization -----


def test_dataset_container_single_unit_normalization():
    """Single arrays per split are wrapped in canonical one-unit lists."""
    c = DatasetContainer(
        features={"train": np.zeros((10, 2)), "val": np.zeros((5, 2))},
        target={"train": np.zeros((10, 1)), "val": np.zeros((5, 1))},
    )
    assert len(c["features"]["train"]) == 1
    assert c["features"]["train"][0].shape == (10, 2)
    assert c.unit_cardinality == SplitUnitCardinality.SINGLE_UNIT_PER_SPLIT


def test_dataset_container_multi_unit():
    """Explicit lists per split stay as-is and report multi-unit cardinality."""
    c = DatasetContainer(
        features={
            "train": [np.zeros((10, 2)), np.zeros((10, 2))],
            "val": [np.zeros((5, 2))],
        },
        target={
            "train": [np.zeros((10, 1)), np.zeros((10, 1))],
            "val": [np.zeros((5, 1))],
        },
    )
    assert len(c["features"]["train"]) == 2
    assert c.unit_cardinality == SplitUnitCardinality.MIXED


def test_dataset_container_multi_unit_per_split_cardinality():
    """Two-or-more units in every populated split report MULTI_UNIT_PER_SPLIT."""
    c = DatasetContainer(
        features={
            "train": [np.zeros((10, 2)), np.zeros((10, 2))],
            "val": [np.zeros((5, 2)), np.zeros((5, 2))],
        },
        target={
            "train": [np.zeros((10, 1)), np.zeros((10, 1))],
            "val": [np.zeros((5, 1)), np.zeros((5, 1))],
        },
    )

    assert c.unit_cardinality == SplitUnitCardinality.MULTI_UNIT_PER_SPLIT


def test_dataset_container_empty_cardinality_for_empty_lists():
    """All-empty splits report EMPTY cardinality without special flags."""
    c = DatasetContainer(
        features={"train": [], "val": [], "test": []},
        target={"train": [], "val": [], "test": []},
    )

    assert c.unit_cardinality == SplitUnitCardinality.EMPTY


def test_dataset_container_ignores_metadata_for_cardinality():
    """Metadata payloads do not affect split cardinality calculation."""
    c = DatasetContainer(
        features={"train": [np.zeros((10, 2))], "val": [np.zeros((5, 2))]},
        target={"train": [np.zeros((10, 1))], "val": [np.zeros((5, 1))]},
        metadata={
            "train": [{"unit_name": "u1"}, {"unit_name": "u2"}],
            "val": [{"unit_name": "u3"}],
        },
    )

    assert c.unit_cardinality == SplitUnitCardinality.SINGLE_UNIT_PER_SPLIT


def test_dataset_container_constructor_derives_cardinality_from_payload():
    """Constructor derives cardinality from normalized split payloads alone."""
    c = DatasetContainer(
        features={"train": np.zeros((10, 2)), "val": np.zeros((5, 2))},
        target={"train": np.zeros((10, 1)), "val": np.zeros((5, 1))},
    )

    assert c.unit_cardinality == SplitUnitCardinality.SINGLE_UNIT_PER_SPLIT


# ----- SplitDatasetContainer.validate() -----


def test_split_container_validate_success(single_unit_split_container):
    """validate() does not raise when structure is consistent."""
    report = single_unit_split_container.validate()
    assert report["is_consistent"] is True


def test_split_container_validate_success_prints_rich_report_without_ascii_logs(
    single_unit_split_container,
    monkeypatch,
    caplog,
):
    """validate() prints the Rich report and logs only the short status line."""
    printed_reports = []
    monkeypatch.setattr(
        SplitDatasetContainer,
        "_print_split_alignment_report",
        lambda self, report: printed_reports.append(report),
    )

    with caplog.at_level(
        logging.INFO,
        logger="picid.data.data_objects.containers.split_dataset_container",
    ):
        report = single_unit_split_container.validate()

    assert printed_reports == [report]
    assert report["is_consistent"] is True
    assert "SplitDatasetContainer validation successful." in caplog.text
    assert "| key" not in caplog.text
    assert "+-" not in caplog.text


def test_split_container_validate_inconsistent_units_reports_heterogeneity():
    """validate() reports heterogeneous unit counts instead of raising by default."""
    c = SplitDatasetContainer(
        features={
            "train": [np.zeros((100, 5)), np.zeros((100, 5))],
            "val": [np.zeros((30, 5))],
            "test": [np.zeros((30, 5))],
        },
        target={
            "train": [np.zeros((100, 1))],
            "val": [np.zeros((30, 1))],
            "test": [np.zeros((30, 1))],
        },
    )
    report = c.validate()
    assert report["is_consistent"] is False
    assert report["unit_counts_match"] is False


def test_split_container_validate_strict_inconsistent_units_raises():
    """validate(strict=True) still raises on heterogeneous split cardinality."""
    c = SplitDatasetContainer(
        features={
            "train": [np.zeros((100, 5)), np.zeros((100, 5))],
            "val": [np.zeros((30, 5))],
            "test": [np.zeros((30, 5))],
        },
        target={
            "train": [np.zeros((100, 1))],
            "val": [np.zeros((30, 1))],
            "test": [np.zeros((30, 1))],
        },
    )
    with pytest.raises(ValueError, match="heterogeneous split payloads") as exc_info:
        c.validate(strict=True)
    message = str(exc_info.value)
    assert "| key" in message
    assert "unit_counts" in message
    assert "features" in message


def test_split_container_validate_reports_unit_schema_mismatch():
    """validate() should report mismatched per-unit payload schemas within a split."""
    c = SplitDatasetContainer(
        features={
            "train": [
                {"sensor_a": np.zeros((2,)), "sensor_b": np.ones((2,))},
                {"sensor_a": np.zeros((2,)), "sensor_c": np.ones((2,))},
            ]
        },
        target={"train": [np.zeros((1,)), np.zeros((1,))]},
    )

    report = c.validate()

    assert report["is_consistent"] is False
    assert report["unit_schema_match"] is False
    assert report["schema_mismatches"]
    features_row = next(row for row in report["rows"] if row["key"] == "features")
    assert features_row["schema_status"]["train"] == "heterogeneous"


def test_split_container_validate_strict_schema_mismatch_raises():
    """validate(strict=True) should raise on per-unit payload schema mismatch."""
    c = SplitDatasetContainer(
        features={
            "train": [
                {"sensor_a": np.zeros((2,)), "sensor_b": np.ones((2,))},
                {"sensor_a": np.zeros((2,)), "sensor_c": np.ones((2,))},
            ]
        },
        target={"train": [np.zeros((1,)), np.zeros((1,))]},
    )

    with pytest.raises(ValueError, match="schema"):
        c.validate(strict=True)


def test_split_container_validate_reports_homogeneous_unit_schema_details():
    """validate() should report homogeneous unit schemas explicitly for mapping payloads."""
    c = SplitDatasetContainer(
        features={
            "train": [
                {"sensor_a": np.zeros((2,)), "sensor_b": np.ones((2,))},
                {"sensor_a": np.zeros((2,)), "sensor_b": np.ones((2,))},
            ]
        },
        target={"train": [np.zeros((1,)), np.zeros((1,))]},
    )

    report = c.validate()
    features_row = next(row for row in report["rows"] if row["key"] == "features")

    assert features_row["schema_status"]["train"] == "homogeneous"
    assert "sensor_a" in features_row["schemas"]["train"]
    assert "sensor_b" in features_row["schemas"]["train"]


def test_dataset_container_split_name_mismatch_is_reported():
    """validate() reports split-name mismatches without making cardinality invalid."""
    c = SplitDatasetContainer(
        features={"train": [np.zeros((10, 2))], "val": [np.zeros((5, 2))]},
        target={"train": [np.zeros((10, 1))], "test": [np.zeros((5, 1))]},
    )

    report = c.validate()
    assert report["split_names_match"] is False
    assert c.unit_cardinality == SplitUnitCardinality.SINGLE_UNIT_PER_SPLIT


# ----- to_split_dict() -----


def test_split_container_to_split_dict_structure(single_unit_split_container):
    """Default to_split_dict() keeps the canonical list-per-unit view."""
    d = single_unit_split_container.to_split_dict()
    assert set(d.keys()) == {"train", "val", "test"}
    assert "features" in d["train"]
    assert "target" in d["train"]
    assert isinstance(d["train"]["features"], list)
    assert len(d["train"]["features"]) == 1


def test_split_container_to_split_dict_explicit_singleton_unwrap():
    """UNWRAP_SINGLETONS turns one-unit split lists into direct payload values."""
    c = SplitDatasetContainer(
        features={"train": np.zeros((10, 2)), "val": np.zeros((5, 2))},
        target={"train": np.zeros((10, 1)), "val": np.zeros((5, 1))},
    )

    d = c.to_split_dict(SplitViewPolicy.UNWRAP_SINGLETONS)

    assert isinstance(d["train"]["features"], np.ndarray)
    assert d["train"]["features"].shape == (10, 2)


def test_split_container_to_split_dict_unwrap_raises_for_multi_unit():
    """UNWRAP_SINGLETONS raises when any populated split has multiple units."""
    with pytest.raises(ValueError, match="requires exactly one unit"):
        multi_unit_split_container = SplitDatasetContainer(
            features={
                "train": [np.random.randn(100, 5), np.random.randn(100, 5)],
                "val": [np.random.randn(30, 5), np.random.randn(30, 5)],
            },
            target={
                "train": [np.random.randn(100, 1), np.random.randn(100, 1)],
                "val": [np.random.randn(30, 1), np.random.randn(30, 1)],
            },
        )
        multi_unit_split_container.to_split_dict(SplitViewPolicy.UNWRAP_SINGLETONS)


def test_split_container_to_split_dict_unwrap_raises_for_mixed_cardinality():
    """UNWRAP_SINGLETONS raises when split cardinality mixes one-unit and multi-unit splits."""
    c = SplitDatasetContainer(
        features={
            "train": [np.random.randn(100, 5), np.random.randn(100, 5)],
            "val": [np.random.randn(30, 5)],
        },
        target={
            "train": [np.random.randn(100, 1), np.random.randn(100, 1)],
            "val": [np.random.randn(30, 1)],
        },
    )

    with pytest.raises(ValueError, match="requires exactly one unit"):
        c.to_split_dict(SplitViewPolicy.UNWRAP_SINGLETONS)


def test_split_container_to_split_dict_empty():
    """Empty container returns {}."""
    c = SplitDatasetContainer()
    assert c.to_split_dict() == {}


def test_dataset_container_repr_reports_relaxed_cardinality():
    """__repr__ should stay safe when split alignment is heterogeneous."""
    c = SplitDatasetContainer(
        features={"train": [np.zeros((10, 2)), np.zeros((10, 2))]},
        target={"train": [np.zeros((10, 1))]},
    )

    assert "unit_cardinality=mixed" in repr(c)


def test_split_container_to_split_dict_allows_heterogeneous_counts():
    """Default split export keeps heterogeneous payloads without raising."""
    c = SplitDatasetContainer(
        features={"train": [np.zeros((10, 2))]},
        target={"train": [np.zeros((10, 1)), np.zeros((10, 1))]},
    )

    split_dict = c.to_split_dict()

    assert len(split_dict["train"]["features"]) == 1
    assert len(split_dict["train"]["target"]) == 2


def test_split_container_validate_handles_awkward_payload_descriptions():
    """validate() should describe awkward payloads without assuming ``.shape`` exists."""
    c = SplitDatasetContainer(
        features={"train": [ak.Array([{"a": 1, "b": [1, 2]}])]},
        target={"train": [np.zeros((1, 1))]},
    )

    report = c.validate()

    assert report["rows"][0]["shapes"]["train"].startswith("ak[")


def test_describe_unit_payload_empty():
    """_describe_unit_payload should label empty split payloads explicitly."""
    assert describe_unit_payload([]) == "empty"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (np.zeros((2, 3)), "(2, 3)"),
        (pd.Series([1, 2, 3]), "(3,)"),
        (pd.DataFrame({"a": [1, 2], "b": [3, 4]}), "(2, 2)"),
    ],
)
def test_describe_unit_payload_uses_tabular_shapes(payload, expected):
    """_describe_unit_payload should report NumPy/pandas payload shapes directly."""
    assert describe_unit_payload([payload]) == expected


def test_describe_unit_payload_reports_mapping():
    """_describe_unit_payload should label mapping-like payloads explicitly."""
    payload = BaseDataObjectWithMetadata(
        features=np.zeros((2, 2)), target=np.zeros((2, 1))
    )

    assert describe_unit_payload([payload]) == "mapping"


class _BrokenShapePayload:
    @property
    def shape(self):
        raise AttributeError("shape unavailable")

    def __len__(self):
        return 7


def test_describe_unit_payload_falls_back_to_len_when_shape_is_broken():
    """_describe_unit_payload should fall back to length when ``shape`` is unusable."""
    assert describe_unit_payload([_BrokenShapePayload()]) == "len=7"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ([1, 2, 3], "len=3"),
        ("abc", "str"),
        (b"abc", "bytes"),
        (42, "int"),
    ],
)
def test_describe_unit_payload_fallback_branches(payload, expected):
    """_describe_unit_payload should cover generic length and scalar fallbacks."""
    assert describe_unit_payload([payload]) == expected


# ----- group_by_split() -----


def test_split_container_group_by_split(single_unit_split_container):
    """group_by_split() returns {split: list of BaseDataObjectWithMetadata}."""
    grouped = single_unit_split_container.group_by_split()
    assert set(grouped.keys()) == {"train", "val", "test"}
    for split in ["train", "val", "test"]:
        assert isinstance(grouped[split], list)
        assert len(grouped[split]) >= 1
        for unit in grouped[split]:
            assert isinstance(unit, BaseDataObjectWithMetadata)
            assert "features" in unit
            assert "target" in unit


def test_split_container_group_by_split_multi_unit(multi_unit_split_container):
    """group_by_split() returns one DataChunk per unit per split."""
    grouped = multi_unit_split_container.group_by_split()
    assert len(grouped["train"]) == 3
    assert len(grouped["val"]) == 3
    assert len(grouped["test"]) == 3


def test_split_container_group_by_split_empty():
    """Empty container returns {}."""
    c = SplitDatasetContainer()
    assert c.group_by_split() == {}


# ----- get_instance_cls() -----


def test_split_container_get_instance_cls(single_unit_split_container):
    """get_instance_cls() returns dict mapping data key to stored value type."""
    cls_map = single_unit_split_container.get_instance_cls()
    assert isinstance(cls_map, dict)
    assert "features" in cls_map
    assert "target" in cls_map
    # Stored value type (nested container or ndarray)
    assert isinstance(cls_map["features"], type)


# ----- Optional manifest / slice_info -----


def test_dataset_container_optional_manifest_slice_info():
    """DatasetContainer accepts optional manifest= and slice_info= (no error)."""
    from picid.data.data_objects.manifest import MetadataManifest
    from picid.data.data_objects.slice_info import SliceInfo

    manifest = MetadataManifest()
    slice_info = SliceInfo(split="train", unit_ids=[0])
    c = DatasetContainer(
        features={"train": [np.zeros((10, 2))]},
        target={"train": [np.zeros((10, 1))]},
        manifest=manifest,
        slice_info=slice_info,
    )
    assert c.manifest is manifest
    assert c.slice_info is slice_info


def test_split_container_copy_preserves_manifest_slice_info(
    single_unit_split_container,
):
    """copy(deep=True) preserves manifest and slice_info."""
    from picid.data.data_objects.manifest import MetadataManifest
    from picid.data.data_objects.slice_info import SliceInfo

    single_unit_split_container.manifest = MetadataManifest()
    single_unit_split_container.slice_info = SliceInfo(split="train", unit_ids=[0])
    c2 = single_unit_split_container.copy(deep=True)
    assert c2.manifest is not None
    assert c2.slice_info is not None
    assert c2.slice_info.split == "train"


# ----- data_objects.utils (coverage) -----


def test_utils_get_length():
    """get_length returns shape[0] for arrays and len() for list/tuple."""
    from picid.data.data_objects.utils import get_length

    assert get_length(np.zeros((10, 2))) == 10
    assert get_length([1, 2, 3]) == 3
    assert get_length((1, 2)) == 2
    assert get_length(np.array(3)) is None  # 0-dim


def test_utils_check_length_consistency_ok():
    """check_length_consistency does not raise when lengths match."""
    from picid.data.data_objects.utils import check_length_consistency

    check_length_consistency(
        [np.zeros((5, 2)), np.zeros(5)],
        ["a", "b"],
    )


def test_utils_check_length_consistency_raises():
    """check_length_consistency raises when lengths differ."""
    from picid.data.data_objects.utils import check_length_consistency

    with pytest.raises(ValueError, match="Length mismatch"):
        check_length_consistency(
            [np.zeros((5, 2)), np.zeros(3)],
            ["a", "b"],
        )


def test_utils_check_for_nans_raises():
    """check_for_nans raises when NaN present in numeric array."""
    from picid.data.data_objects.utils import check_for_nans

    arr = np.array([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="NaN"):
        check_for_nans([arr], ["x"])


def test_utils_check_for_nans_ok():
    """check_for_nans does not raise when no NaN."""
    from picid.data.data_objects.utils import check_for_nans

    check_for_nans([np.array([1.0, 2.0])], ["x"])


# ----- data_objects.manifest (coverage) -----


def test_manifest_add_and_query():
    """MetadataManifest add and query by step_id/key/split/category."""
    from picid.data.data_objects.manifest import MetadataManifest, ManifestEntry

    m = MetadataManifest()
    m.add(
        ManifestEntry(
            schema_version="1.0",
            producer_version="0.1",
            category="datasource",
            payload={"k": 1},
            step_id="load",
            key="features",
            split="train",
        )
    )
    m.add(
        ManifestEntry(
            schema_version="1.0",
            producer_version="0.1",
            category="transform",
            payload={},
            step_id="scaler",
            key="features",
            split="train",
        )
    )
    assert len(m) == 2
    q = m.query(step_id="load")
    assert len(q) == 1
    assert q[0].payload["k"] == 1
    q2 = m.query(category="transform")
    assert len(q2) == 1
    assert q2[0].step_id == "scaler"


def test_manifest_copy():
    """MetadataManifest copy(deep=True) returns independent copy."""
    from picid.data.data_objects.manifest import MetadataManifest, ManifestEntry

    m = MetadataManifest()
    m.add(
        ManifestEntry(
            schema_version="1.0",
            producer_version="0.1",
            category="datasource",
            payload={"x": 1},
            step_id="s",
            key="k",
            split="train",
        )
    )
    m2 = m.copy(deep=True)
    assert len(m2) == 1
    assert m2._entries[0].payload["x"] == 1
