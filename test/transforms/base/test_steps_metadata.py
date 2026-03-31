"""Tests that pipeline metadata is passed to transforms with clear scope."""

import copy
import numpy as np
import pytest

from picid.data.data_objects import SplitDatasetContainer
from picid.data.data_objects.slice_info import SliceInfo
from picid.exceptions import TransformError
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin
from picid.data.data_objects import NamedTransformInput

from test.transforms.base.conftest import create_dummy_split_container


class MetadataRecordingTransform(NoFitPerSegmentMixin, DenseTransform):
    """Transform that records the metadata it receives in transform_data."""

    def __init__(self):
        super().__init__()
        self.last_metadata = None

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> np.ndarray:
        self.last_metadata = metadata
        key = list(data.keys())[0]
        return data[key] * 2


class ScopedMetadataRecordingTransform(NoFitPerSegmentMixin, DenseTransform):
    """Record both per-unit metadata and pipeline/container metadata."""

    def __init__(self):
        super().__init__()
        self.seen_unit_metadata = []
        self.seen_pipeline_metadata = []

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> np.ndarray:
        self.seen_unit_metadata.append(copy.deepcopy(data.metadata))
        self.seen_pipeline_metadata.append(copy.deepcopy(metadata))
        return data["features"][:, :1] * 2


def _container_with_slice_info():
    """SplitDatasetContainer with slice_info set."""
    base = create_dummy_split_container(n_units=2, n_samples_per_unit=5, n_features=2)
    # Rebuild container with slice_info; SplitDatasetContainer passes kwargs to DatasetContainer
    return SplitDatasetContainer(
        features=base["features"],
        target=base["target"],
        slice_info=SliceInfo(split="train", unit_ids=[1, 2]),
    )


def _container_with_metadata() -> SplitDatasetContainer:
    """SplitDatasetContainer with both container-level and per-unit metadata."""
    return SplitDatasetContainer(
        features={
            "train": [
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[5.0, 6.0], [7.0, 8.0]]),
            ],
            "val": [np.array([[9.0, 10.0]])],
            "test": [],
        },
        target={
            "train": [np.array([[0.0], [1.0]]), np.array([[2.0], [3.0]])],
            "val": [np.array([[4.0]])],
            "test": [],
        },
        container_metadata={"dataset_name": "demo", "column_map": {}},
        unit_metadata={
            "train": [
                {"unit_name": "unit-1", "machine_id": 1},
                {"unit_name": "unit-2", "machine_id": 2},
            ],
            "val": [{"unit_name": "unit-3", "machine_id": 3}],
            "test": [],
        },
    )


def test_slice_info_not_in_metadata_when_opt_in_false():
    """With include_slice_info_in_metadata False or omitted, metadata has no slice_info."""
    container = _container_with_slice_info()
    transform = MetadataRecordingTransform()
    metadata_config = {
        "apply_to": "features",
        "assign_to": "features",
    }
    dt = DataTransform("test", transform, metadata_config)
    dt.forward(container)
    # TransformStep runs per split; we only care that slice_info was not added
    assert transform.last_metadata is not None
    assert "slice_info" not in transform.last_metadata


def test_slice_info_in_metadata_when_opt_in_true():
    """With include_slice_info_in_metadata True, metadata contains slice_info."""
    container = _container_with_slice_info()
    transform = MetadataRecordingTransform()
    metadata_config = {
        "apply_to": "features",
        "assign_to": "features",
        "include_slice_info_in_metadata": True,
    }
    dt = DataTransform("test", transform, metadata_config)
    dt.forward(container)
    assert transform.last_metadata is not None
    assert "slice_info" in transform.last_metadata
    si = transform.last_metadata["slice_info"]
    assert si["split"] == "train"
    assert si["unit_ids"] == [1, 2]


def test_transform_receives_unit_metadata_on_named_transform_input():
    """Per-unit metadata should be attached to each chunk as data.metadata."""
    container = _container_with_metadata()
    transform = ScopedMetadataRecordingTransform()
    dt = DataTransform(
        "scoped-metadata",
        transform,
        {"apply_to": "features", "assign_to": "features"},
    )

    out, _ = dt.forward(container)

    assert transform.seen_unit_metadata[0]["unit_name"] == "unit-1"
    assert transform.seen_unit_metadata[1]["machine_id"] == 2
    assert transform.seen_unit_metadata[2]["unit_name"] == "unit-3"
    assert out.unit_metadata == container.unit_metadata


def test_transform_receives_container_metadata_in_pipeline_metadata():
    """Container metadata should be available through the transform metadata dict."""
    container = _container_with_metadata()
    transform = ScopedMetadataRecordingTransform()
    dt = DataTransform(
        "container-metadata",
        transform,
        {"apply_to": "features", "assign_to": "features"},
    )

    dt.forward(container)

    pipeline_metadata = transform.seen_pipeline_metadata[0]
    assert pipeline_metadata["container_metadata"]["dataset_name"] == "demo"
    assert "slice_info" not in pipeline_metadata


def test_transform_preserves_unit_metadata_when_unit_count_is_preserved():
    """Unit metadata should survive unit-preserving transforms unchanged."""
    container = _container_with_metadata()
    transform = ScopedMetadataRecordingTransform()
    dt = DataTransform(
        "preserve-unit-metadata",
        transform,
        {"apply_to": "features", "assign_to": "features"},
    )

    out, _ = dt.forward(container)

    assert out.unit_metadata == container.unit_metadata
    assert out["features"]["train"][0].shape == (2, 1)
    assert out["features"]["train"][1].shape == (2, 1)


def test_transform_raises_on_misaligned_unit_metadata_before_chunk_building():
    """Chunk preparation should fail early when per-unit metadata is misaligned."""
    container = SplitDatasetContainer(
        features={
            "train": [np.array([[1.0]]), np.array([[2.0]])],
            "val": [],
            "test": [],
        },
        target={
            "train": [np.array([[0.0]]), np.array([[1.0]])],
            "val": [],
            "test": [],
        },
        unit_metadata={
            "train": [{"unit_name": "only-one"}],
            "val": [],
            "test": [],
        },
    )
    transform = MetadataRecordingTransform()
    dt = DataTransform(
        "misaligned-unit-metadata",
        transform,
        {"apply_to": "features", "assign_to": "features"},
    )

    with pytest.raises(TransformError, match="unit_metadata"):
        dt.forward(container)


def test_transform_raises_when_output_cannot_preserve_unit_metadata_alignment():
    """Unit metadata should block transforms that collapse multiple units into one."""
    from test.transforms.base.conftest import ConcatenateUnitsTransform

    container = _container_with_metadata()
    dt = DataTransform(
        "concat-units",
        ConcatenateUnitsTransform(axis=0),
        {"apply_to": "features", "assign_to": "features"},
    )

    with pytest.raises(TransformError, match="unit metadata|unit_metadata"):
        dt.forward(container)


def test_transform_raises_actionable_error_for_unit_metadata_hook():
    """The alignment error should point users to transform-level propagation hooks."""
    from test.transforms.base.conftest import ConcatenateUnitsTransform

    container = _container_with_metadata()
    dt = DataTransform(
        "concat-units",
        ConcatenateUnitsTransform(axis=0),
        {"apply_to": "features", "assign_to": "features"},
    )

    with pytest.raises(TransformError) as exc_info:
        dt.forward(container)

    msg = str(exc_info.value)
    assert "propagate_unit_metadata" in msg
    assert "aggregate_unit_metadata" in msg
    assert "drop_unit_metadata" in msg
