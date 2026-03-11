"""Tests that slice_info is passed in metadata to transforms only when opted in."""
import numpy as np

from picid.data.data_objects.data import SplitDatasetContainer
from picid.data.data_objects.slice_info import SliceInfo
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin
from picid.data.data_objects.data import NamedTransformInput

from test.transforms.base.conftest import create_dummy_split_container


class MetadataRecordingTransform(NoFitPerSegmentMixin, DenseTransform):
    """Transform that records the metadata it receives in transform_data."""

    def __init__(self):
        super().__init__()
        self.last_metadata = None

    def transform_data(
        self, data: NamedTransformInput, metadata: dict
    ) -> np.ndarray:
        self.last_metadata = metadata
        key = list(data.keys())[0]
        return data[key] * 2


def _container_with_slice_info():
    """SplitDatasetContainer with slice_info set."""
    base = create_dummy_split_container(n_units=2, n_samples_per_unit=5, n_features=2)
    # Rebuild container with slice_info; SplitDatasetContainer passes kwargs to DatasetContainer
    return SplitDatasetContainer(
        features=base["features"],
        target=base["target"],
        slice_info=SliceInfo(split="train", unit_ids=[1, 2]),
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
