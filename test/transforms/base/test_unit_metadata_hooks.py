"""Tests for transform-level unit_metadata propagation hooks."""


import numpy as np
import awkward as ak
import pytest

from picid.data.data_objects import SplitDatasetContainer
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base.multisource import NoFitConcatAlongAxisMixin
from picid.exceptions import TransformError
from picid.transforms.base.pipeline.unit_metadata import (
    aggregate_unit_metadata,
    drop_unit_metadata,
    preserve_unit_metadata,
)
from picid.transforms.base_transforms.concatenate import (
    MultiDatasetRuggedToDenseTransform,
)


class _CollapsingDenseTransform(NoFitConcatAlongAxisMixin, DenseTransform):
    """Simple collapsing transform used to exercise the no-hook failure path."""

    def __init__(self) -> None:
        DenseTransform.__init__(self)
        NoFitConcatAlongAxisMixin.__init__(self, axis=0)

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> np.ndarray:
        key = list(data.keys())[0]
        return data[key] * 2


def _container_with_unit_metadata() -> SplitDatasetContainer:
    return SplitDatasetContainer(
        features={
            "train": [
                ak.Array([[1.0, 2.0], [3.0, 4.0]]),
                ak.Array([[5.0, 6.0], [7.0, 8.0]]),
            ],
            "val": [ak.Array([[9.0, 10.0], [11.0, 12.0]])],
            "test": [],
        },
        unit_metadata={
            "train": [
                {"unit_name": "train-1", "unit_id": 1},
                {"unit_name": "train-2", "unit_id": 2},
            ],
            "val": [{"unit_name": "val-1", "unit_id": 3}],
            "test": [],
        },
    )


def test_base_transform_unit_metadata_hook_defaults_to_noop() -> None:
    from test.transforms.base.conftest import DummyStatelessTransform

    transform = DummyStatelessTransform()

    propagated = transform.propagate_unit_metadata(
        unit_metadata_by_split={"train": [{"unit_name": "u1"}]},
        transformed_results_for_new_key={},
        metadata={},
    )

    assert propagated is None


def test_preserve_and_drop_unit_metadata_helpers() -> None:
    unit_metadata = {
        "train": [{"unit_name": "u1", "unit_id": 1}],
        "val": [{"unit_name": "u2", "unit_id": 2}],
    }

    preserved = preserve_unit_metadata(unit_metadata_by_split=unit_metadata)
    dropped = drop_unit_metadata(unit_metadata_by_split=unit_metadata)

    assert preserved == unit_metadata
    assert preserved is not unit_metadata
    assert preserved["train"] is not unit_metadata["train"]
    assert dropped == {}


def test_aggregate_unit_metadata_builds_split_summary() -> None:
    unit_metadata = {
        "train": [
            {"unit_name": "train-1", "unit_id": 10},
            {"unit_name": "train-2", "unit_id": 11},
        ],
        "val": [{"unit_name": "val-1", "unit_id": 12}],
    }
    transformed_results = {
        "features": {
            "train": [np.array([[1.0, 2.0]])],
            "val": [np.array([[3.0, 4.0]])],
        }
    }

    aggregated = aggregate_unit_metadata(
        unit_metadata_by_split=unit_metadata,
        transformed_results_for_new_key=transformed_results,
        metadata={"transform_name": "collapse"},
    )

    assert len(aggregated["train"]) == 1
    assert aggregated["train"][0]["unit_name"] == "aggregated::train"
    assert aggregated["train"][0]["source_unit_count"] == 2
    assert aggregated["train"][0]["source_unit_names"] == ["train-1", "train-2"]
    assert aggregated["train"][0]["source_unit_ids"] == [10, 11]
    assert aggregated["val"][0]["unit_name"] == "aggregated::val"
    assert aggregated["val"][0]["source_unit_count"] == 1


def test_multi_dataset_rugged_to_dense_transform_aggregates_unit_metadata() -> None:
    container = _container_with_unit_metadata()
    transform = DataTransform(
        "collapse-units",
        MultiDatasetRuggedToDenseTransform(axis=0),
        {"apply_to": "features", "assign_to": "features"},
    )

    out, _ = transform.forward(container)

    assert len(out.features["train"]) == 1
    assert len(out.unit_metadata["train"]) == 1
    assert out.unit_metadata["train"][0]["unit_name"] == "aggregated::train"
    assert out.unit_metadata["train"][0]["source_unit_count"] == 2


def test_collapsing_transform_without_hook_reports_actionable_metadata_error() -> None:
    container = _container_with_unit_metadata()
    transform = DataTransform(
        "collapse-without-hook",
        _CollapsingDenseTransform(),
        {"apply_to": "features", "assign_to": "features"},
    )

    with pytest.raises(TransformError) as exc_info:
        transform.forward(container)

    message = str(exc_info.value)
    assert "unit_metadata" in message
    assert "propagate_unit_metadata" in message
    assert "aggregate_unit_metadata" in message
    assert "drop_unit_metadata" in message
