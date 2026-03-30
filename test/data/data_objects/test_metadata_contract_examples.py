"""Example-style regression tests for metadata-aware dataset containers."""

from __future__ import annotations

import numpy as np
import pytest

from picid.data.data_objects import SplitDatasetContainer


def _build_container(**kwargs) -> SplitDatasetContainer:
    payload = {
        "features": {
            "train": [np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]])],
            "val": [np.array([[5.0]])],
            "test": [],
        },
        "target": {
            "train": [np.array([[0.1], [0.2]]), np.array([[0.3], [0.4]])],
            "val": [np.array([[0.5]])],
            "test": [],
        },
        "container_metadata": {
            "source": "demo",
            "column_map": {"features": ["sensor_a"]},
        },
        "unit_metadata": {
            "train": [
                {"unit_name": "train-1", "split": "train"},
                {"unit_name": "train-2", "split": "train"},
            ],
            "val": [{"unit_name": "val-1", "split": "val"}],
            "test": [],
        },
    }
    payload.update(kwargs)
    return SplitDatasetContainer(
        **payload,
    )


def test_split_container_separates_container_and_unit_metadata():
    """Container metadata and per-unit metadata should remain distinct."""
    container = _build_container()

    assert container.container_metadata["source"] == "demo"
    assert container.metadata["source"] == "demo"
    assert container["metadata"]["source"] == "demo"
    assert "metadata" not in container.keys()
    assert "unit_metadata" not in container.keys()
    assert container.unit_metadata["train"][0]["unit_name"] == "train-1"

    split_dict = container.to_split_dict()
    assert split_dict["train"]["unit_metadata"][0]["unit_name"] == "train-1"
    assert split_dict["test"]["features"] == []
    assert split_dict["test"]["unit_metadata"] == []


def test_split_container_group_by_split_attaches_per_unit_metadata():
    """Unit metadata should be attached to each grouped unit object."""
    container = _build_container()

    grouped = container.group_by_split()

    assert len(grouped["train"]) == 2
    assert grouped["train"][0].metadata["unit_name"] == "train-1"
    assert grouped["train"][1].metadata["unit_name"] == "train-2"
    assert grouped["val"][0].metadata["split"] == "val"
    assert grouped["test"] == []


def test_split_container_preserves_empty_splits_in_copy_and_export():
    """Known empty splits should remain visible after copy/export round-trips."""
    container = _build_container()

    copied = container.copy(deep=True)
    split_dict = copied.to_split_dict()

    assert "test" in copied["features"]
    assert copied["features"]["test"] == []
    assert copied["target"]["test"] == []
    assert copied.unit_metadata["test"] == []
    assert "test" in split_dict
    assert split_dict["test"]["features"] == []
    assert split_dict["test"]["target"] == []
    assert split_dict["test"]["unit_metadata"] == []


def test_split_container_reports_misaligned_unit_metadata():
    """Validation should flag unit metadata that does not align with split units."""
    container = _build_container(
        unit_metadata={
            "train": [{"unit_name": "only-one"}],
            "val": [{"unit_name": "val-1"}],
            "test": [],
        }
    )

    report = container.validate()

    assert report["is_consistent"] is False
    assert report["unit_metadata_match"] is False
    assert report["unit_metadata_mismatches"]["train"] == {
        "expected": 2,
        "actual": 1,
    }

    with pytest.raises(ValueError, match="unit_metadata"):
        container.validate(strict=True)


def test_split_container_normalizes_legacy_split_metadata_into_unit_metadata():
    """Legacy top-level split metadata should be adapted into canonical unit metadata."""
    container = SplitDatasetContainer(
        features={"train": [np.array([[1.0]])], "val": [], "test": []},
        target={"train": [np.array([[0.1]])], "val": [], "test": []},
        metadata={
            "train": [{"unit_name": "legacy-train"}],
            "val": [],
            "test": [],
        },
    )

    assert container.container_metadata == {"column_map": {}}
    assert container.unit_metadata["train"][0]["unit_name"] == "legacy-train"
    assert container.to_split_dict()["train"]["unit_metadata"][0]["unit_name"] == (
        "legacy-train"
    )
