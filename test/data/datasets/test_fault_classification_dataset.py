"""Tests for FaultClassificationBatchDataset (picid.data.datasets.fault_classification_dataset)."""

from __future__ import annotations

import numpy as np
import pytest

from picid.data.datasets.fault_classification_dataset import (
    FaultClassificationBatchDataset,
)


@pytest.fixture
def fault_class_data():
    """Features and fault_class for one unit (same shape as RUL-style)."""
    T, F = 60, 4
    np.random.seed(42)
    return {
        "features": np.random.randn(T, F).astype(np.float32),
        "fault_classification": np.random.randint(0, 5, size=(T, 1)).astype(np.float32),
    }


@pytest.fixture
def meta_with_unit_ids():
    return {
        "unit_ids": {"train": [10], "val": [10], "test": [10]},
        "unit_names": {"train": ["u0"], "val": ["u0"], "test": ["u0"]},
        "current_data_split": "train",
        "concat_dataset_index": 0,
    }


def test_fault_classification_init_and_len(fault_class_data):
    """FaultClassificationBatchDataset initializes with pred_len=1 and has length > 0."""
    ds = FaultClassificationBatchDataset(
        data_dict=fault_class_data,
        task_type="fault_classification",
        seq_len=10,
        label_len=0,
        pred_len=0,
        stride=1,
        meta_data_dict={},
    )
    assert len(ds) > 0


def test_fault_classification_getitem_structure(fault_class_data):
    """__getitem__(list) returns batch_idx, fault_classification (last step), features."""
    ds = FaultClassificationBatchDataset(
        data_dict=fault_class_data,
        task_type="fault_classification",
        seq_len=5,
        label_len=0,
        pred_len=0,
        stride=1,
        meta_data_dict={},
    )
    batch = ds[[0, 1]]
    assert "batch_idx" in batch
    assert "fault_classification" in batch
    assert "features" in batch
    assert batch["features"].dim() == 3
    assert batch["features"].shape[0] == 2
    assert batch["fault_classification"].shape[0] == 2


def test_fault_classification_get_unit_id(fault_class_data, meta_with_unit_ids):
    """When get_unit_id=True, __getitem__ includes unit_id."""
    ds = FaultClassificationBatchDataset(
        data_dict=fault_class_data,
        task_type="fault_classification",
        seq_len=5,
        label_len=0,
        pred_len=0,
        stride=1,
        get_unit_id=True,
        meta_data_dict=meta_with_unit_ids,
    )
    batch = ds[[0]]
    assert "unit_id" in batch
    assert batch["unit_id"].shape[0] == 1
    assert batch["unit_id"][0].item() == 10


def test_fault_classification_get_collate_fn(fault_class_data):
    """get_collate_fn returns a callable."""
    ds = FaultClassificationBatchDataset(
        data_dict=fault_class_data,
        task_type="fault_classification",
        seq_len=5,
        label_len=0,
        pred_len=0,
        stride=1,
        meta_data_dict={},
    )
    fn = ds.get_collate_fn()
    assert callable(fn)
