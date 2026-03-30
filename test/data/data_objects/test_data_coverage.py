"""
Coverage-focused tests for picid.data.data_objects.data (DatasetContainer, get_sanitized_data, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest

from picid.data.data_objects import (
    DatasetContainer,
    NamedTransformInput,
    SplitDatasetContainer,
    SplitUnitCardinality,
    SplitViewPolicy,
)


def test_get_sanitized_data_size_exceeds_threshold_warns():
    """get_sanitized_data with total size > size_threshold triggers warning and skips length check."""
    # One array of size 2e6 > default 1e6
    big = np.zeros((2_000_000, 1), dtype=np.float32)
    inp = NamedTransformInput(features=big)
    with pytest.warns(UserWarning, match="exceeds threshold"):
        inp.get_sanitized_data(check_lengths=True, size_threshold=1_000_000)


def test_dataset_container_repr_empty():
    """__repr__ on empty DatasetContainer returns class name and 'empty'."""
    c = DatasetContainer()
    r = repr(c)
    assert "DatasetContainer" in r
    assert "empty" in r


def test_dataset_container_repr_with_data():
    """__repr__ on DatasetContainer includes keys, splits, unit counts, and cardinality."""
    c = DatasetContainer(
        features={"train": [np.zeros((10, 2))], "val": [np.zeros((5, 2))]},
        target={"train": [np.zeros((10, 1))], "val": [np.zeros((5, 1))]},
    )
    r = repr(c)
    assert "DatasetContainer" in r or "SplitDatasetContainer" in r
    assert "features" in r
    assert "target" in r
    assert "train" in r
    assert "val" in r
    assert "unit_cardinality" in r


def test_split_dataset_container_repr_includes_unit_counts():
    """__repr__ on SplitDatasetContainer includes unit counts per split."""
    c = SplitDatasetContainer(
        features={
            "train": [np.zeros((10, 2)), np.zeros((10, 2))],
            "val": [np.zeros((5, 2))],
        },
        target={
            "train": [np.zeros((10, 1)), np.zeros((10, 1))],
            "val": [np.zeros((5, 1))],
        },
    )
    r = repr(c)
    assert "SplitDatasetContainer" in r
    assert "features" in r
    assert "target" in r
    # Unit counts e.g. {'train': 2, 'val': 1}
    assert "'train': 2" in r or '"train": 2' in r
    assert "'val': 1" in r or '"val": 1' in r
    assert "unit_cardinality=mixed" in r


def test_dataset_container_repr_no_recursion():
    """__repr__ returns a string without recursing into nested data (no stack overflow)."""
    c = SplitDatasetContainer(
        features={"train": [np.zeros((100, 10))], "val": [np.zeros((50, 10))]},
        target={"train": [np.zeros((100, 1))], "val": [np.zeros((50, 1))]},
    )
    # Should return quickly; nested arrays are never repr'd
    r = repr(c)
    assert isinstance(r, str)
    assert len(r) < 500  # Summary only, not full array content


def test_dataset_container_validate_empty():
    """validate() on empty container logs and returns without error."""
    c = DatasetContainer()
    c.validate()


def test_split_container_validate_empty():
    """validate() on empty SplitDatasetContainer logs and returns."""
    c = SplitDatasetContainer()
    c.validate()


def test_split_container_to_split_dict_single_unit_keeps_lists_by_default():
    """Default to_split_dict() preserves list-of-one payloads for one-unit splits."""
    c = SplitDatasetContainer(
        features={"train": np.zeros((10, 2)), "val": np.zeros((5, 2))},
        target={"train": np.zeros((10, 1)), "val": np.zeros((5, 1))},
    )
    d = c.to_split_dict()
    assert "train" in d
    assert "features" in d["train"]
    assert isinstance(d["train"]["features"], list)
    assert d["train"]["features"][0].shape == (10, 2)
    assert c.unit_cardinality == SplitUnitCardinality.SINGLE_UNIT_PER_SPLIT


def test_split_container_to_split_dict_single_unit_unwraps_explicitly():
    """Explicit unwrapping keeps legacy behavior available without being implicit."""
    c = SplitDatasetContainer(
        features={"train": np.zeros((10, 2)), "val": np.zeros((5, 2))},
        target={"train": np.zeros((10, 1)), "val": np.zeros((5, 1))},
    )
    d = c.to_split_dict(SplitViewPolicy.UNWRAP_SINGLETONS)
    assert isinstance(d["train"]["features"], np.ndarray)
    assert d["train"]["features"].shape == (10, 2)


def test_group_by_split_empty():
    """group_by_split() on empty container returns {}."""
    c = SplitDatasetContainer()
    assert c.group_by_split() == {}


def test_split_container_validate_success_with_consistent_multi_unit():
    """validate() passes and logs success when all splits have consistent units (lines 219, 223-257)."""
    c = SplitDatasetContainer(
        features={
            "train": [np.zeros((10, 2)), np.zeros((10, 2))],
            "val": [np.zeros((5, 2))],
        },
        target={
            "train": [np.zeros((10, 1)), np.zeros((10, 1))],
            "val": [np.zeros((5, 1))],
        },
    )
    # Inject raw dicts so validate() sees dict (not BaseDataObject) and runs the loop (lines 219-257)
    c._data["features"] = {
        "train": [np.zeros((10, 2)), np.zeros((10, 2))],
        "val": [np.zeros((5, 2))],
    }
    c._data["target"] = {
        "train": [np.zeros((10, 1)), np.zeros((10, 1))],
        "val": [np.zeros((5, 1))],
    }
    report = c.validate()
    assert report["is_consistent"] is True


def test_split_container_validate_inconsistent_unit_counts_reports_heterogeneity():
    """validate() reports heterogeneous unit counts instead of raising by default."""
    c = SplitDatasetContainer(
        features={"train": [np.zeros((5, 2))], "val": [np.zeros((3, 2))]},
        target={"train": [np.zeros((5, 1))], "val": [np.zeros((3, 1))]},
    )
    c._data["features"] = {
        "train": [np.zeros((10, 2)), np.zeros((10, 2))],
        "val": [np.zeros((5, 2))],
    }
    c._data["target"] = {
        "train": [np.zeros((10, 1))],  # 1 unit vs 2 for features/train
        "val": [np.zeros((5, 1))],
    }
    report = c.validate()
    assert report["is_consistent"] is False
    assert report["unit_counts_match"] is False


def test_dataset_container_copy_manifest_none():
    """copy() when manifest is None sets manifest to None (line 198)."""
    c = DatasetContainer(
        features={"train": [np.zeros((5, 2))]},
    )
    object.__setattr__(c, "manifest", None)
    c2 = c.copy(deep=True)
    assert getattr(c2, "manifest") is None


def test_group_by_split_produces_units():
    """group_by_split() with multi-unit data appends BaseDataObjectWithMetadata (line 339)."""
    from picid.data.data_objects import BaseDataObjectWithMetadata

    c = SplitDatasetContainer(
        features={
            "train": [np.zeros((10, 2)), np.zeros((10, 2))],
            "val": [np.zeros((5, 2))],
        },
        target={
            "train": [np.zeros((10, 1)), np.zeros((10, 1))],
            "val": [np.zeros((5, 1))],
        },
    )
    grouped = c.group_by_split()
    assert "train" in grouped
    assert "val" in grouped
    assert len(grouped["train"]) == 2
    assert all(isinstance(u, BaseDataObjectWithMetadata) for u in grouped["train"])
