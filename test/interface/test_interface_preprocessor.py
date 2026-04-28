from __future__ import annotations

import numpy as np

from picid.data.data_objects import SplitDatasetContainer, SplitViewPolicy
from picid.interface.utils import InterfacePreProcessor
from picid.transforms.base import DataTransform
from test.transforms.base.conftest import DummyStatelessTransform


def _small_split_container() -> SplitDatasetContainer:
    return SplitDatasetContainer(
        features={
            "train": [np.array([[1.0], [2.0]])],
            "val": [np.array([[3.0]])],
            "test": [np.array([[4.0]])],
        },
        target={
            "train": [np.array([[10.0], [20.0]])],
            "val": [np.array([[30.0]])],
            "test": [np.array([[40.0]])],
        },
    )


def test_interface_apply_transforms_prints_run_style_summary(monkeypatch):
    captured_summaries = []
    monkeypatch.setattr(
        "picid.interface.utils.print_transforms_summary",
        captured_summaries.append,
    )
    data = _small_split_container()
    transform = DataTransform(
        "double_features",
        DummyStatelessTransform(),
        {"apply_to": "features", "assign_to": "features"},
    )

    result = InterfacePreProcessor.apply_transforms(data, [transform], None)

    assert len(captured_summaries) == 1
    summary = captured_summaries[0]
    assert len(summary) == 1
    assert summary[0]["transform_name"] == "double_features"
    assert summary[0]["status"] == "Success"
    assert "dense" in summary[0]["details"]
    assert "features" in summary[0]["inputs"]

    split_result = result.to_split_dict(SplitViewPolicy.KEEP_UNIT_LISTS)
    np.testing.assert_array_equal(
        split_result["train"]["features"][0],
        np.array([[2.0], [4.0]]),
    )


def test_interface_apply_transforms_empty_list_prints_empty_summary(monkeypatch):
    captured_summaries = []
    monkeypatch.setattr(
        "picid.interface.utils.print_transforms_summary",
        captured_summaries.append,
    )
    data = _small_split_container()

    result = InterfacePreProcessor.apply_transforms(data, [], None)

    assert result is data
    assert captured_summaries == [[]]
