"""
Coverage-focused tests for picid.data.data_objects.data (DatasetContainer, get_sanitized_data, etc.).
"""
from __future__ import annotations

import numpy as np
import pytest

from picid.data.data_objects.data import (
    DatasetContainer,
    NamedTransformInput,
    SplitDatasetContainer,
)


def test_get_sanitized_data_size_exceeds_threshold_warns():
    """get_sanitized_data with total size > size_threshold triggers warning and skips length check."""
    # One array of size 2e6 > default 1e6
    big = np.zeros((2_000_000, 1), dtype=np.float32)
    inp = NamedTransformInput(features=big)
    with pytest.warns(UserWarning, match="exceeds threshold"):
        inp.get_sanitized_data(check_lengths=True, size_threshold=1_000_000)


def test_dataset_container_validate_empty():
    """validate() on empty container logs and returns without error."""
    c = DatasetContainer()
    c.validate()


def test_split_container_validate_empty():
    """validate() on empty SplitDatasetContainer logs and returns."""
    c = SplitDatasetContainer()
    c.validate()


def test_split_container_to_split_dict_single_unit_unwraps():
    """to_split_dict() when is_multi_unit is False stores unit_list[0] per key."""
    # Pass non-list per split so container wraps and sets is_multi_unit=False
    c = SplitDatasetContainer(
        features={"train": np.zeros((10, 2)), "val": np.zeros((5, 2))},
        target={"train": np.zeros((10, 1)), "val": np.zeros((5, 1))},
    )
    assert c.is_multi_unit is False
    d = c.to_split_dict()
    assert "train" in d
    assert "features" in d["train"]
    assert isinstance(d["train"]["features"], np.ndarray)
    assert d["train"]["features"].shape == (10, 2)


def test_group_by_split_empty():
    """group_by_split() on empty container returns {}."""
    c = SplitDatasetContainer()
    assert c.group_by_split() == {}
