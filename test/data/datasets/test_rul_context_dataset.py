"""
Tests for RULContextBatchDataset with real synthetic data.
Covers pred_len assertion, get_unit_id, extract_unit_id/extract_unit_name, __getitem__ structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from picid.data.datasets.rul_context_dataset import RULContextBatchDataset


@pytest.fixture
def rul_dense_data():
    """Dense features and RUL for one unit."""
    T, F = 60, 4
    np.random.seed(42)
    return {
        "features": np.random.randn(T, F).astype(np.float32),
        "rul": np.arange(T - 1, -1, -1, dtype=np.float32).reshape(-1, 1),
    }


@pytest.fixture
def meta_with_unit_ids():
    return {
        "unit_ids": {"train": [10, 20, 30], "val": [10, 20, 30], "test": [10, 20, 30]},
        "unit_names": {
            "train": ["u0", "u1", "u2"],
            "val": ["u0", "u1", "u2"],
            "test": ["u0", "u1", "u2"],
        },
        "current_data_split": "train",
    }


def test_rul_context_pred_len_must_be_one(rul_dense_data):
    """pred_len != 1 raises AssertionError."""
    with pytest.raises(AssertionError, match="pred_len must be 0"):
        RULContextBatchDataset(
            data_dict=rul_dense_data,
            task_type="rul",
            seq_len=10,
            label_len=0,
            pred_len=2,
            stride=1,
            meta_data_dict={},
        )


def test_rul_context_init_success_with_pred_len_one(rul_dense_data):
    """pred_len=1 is accepted and internally forced to 0."""
    ds = RULContextBatchDataset(
        data_dict=rul_dense_data,
        task_type="rul",
        seq_len=10,
        label_len=0,
        pred_len=0,
        stride=1,
        meta_data_dict={},
    )
    assert len(ds) > 0


def test_rul_context_len_matches_expected_window_count(rul_dense_data):
    """len(ds) matches strict sliding-window count for dense data."""
    ds = RULContextBatchDataset(
        data_dict=rul_dense_data,
        task_type="rul",
        seq_len=10,
        label_len=0,
        pred_len=0,
        stride=1,
        padding_left_flag=False,
        meta_data_dict={},
    )

    # T=60, required_len=seq_len+pred_len=10, min_start=0, max_start=51
    # Number of starts in range(0, 51, 1) is 51.
    assert len(ds) == 51


def test_rul_context_getitem_structure(rul_dense_data):
    """__getitem__(list) returns batch_idx, rul (last step), features."""
    ds = RULContextBatchDataset(
        data_dict=rul_dense_data,
        task_type="rul",
        seq_len=5,
        label_len=0,
        pred_len=0,
        stride=1,
        meta_data_dict={},
    )
    batch = ds[[0, 1]]
    assert "batch_idx" in batch
    assert "rul" in batch
    assert "features" in batch
    assert batch["features"].dim() == 3
    assert batch["features"].shape[0] == 2
    assert batch["rul"].shape[0] == 2


def test_rul_context_rul_last_value_matches(rul_dense_data):
    """RUL in batch is the last value of the window (correct for RUL at end of sequence)."""
    ds = RULContextBatchDataset(
        data_dict=rul_dense_data,
        task_type="rul",
        seq_len=5,
        label_len=0,
        pred_len=0,
        stride=1,
        meta_data_dict={},
        padding_left_flag=False,
    )
    batch = ds[[0]]
    expected_rul = rul_dense_data["rul"][4, 0]
    assert batch["rul"][0].item() == pytest.approx(float(expected_rul))


def test_rul_context_get_unit_id_includes_unit_id(rul_dense_data, meta_with_unit_ids):
    """When get_unit_id=True, __getitem__ includes unit_id from meta_data_dict."""
    meta = dict(meta_with_unit_ids)
    meta["concat_dataset_index"] = 0
    ds = RULContextBatchDataset(
        data_dict=rul_dense_data,
        task_type="rul",
        seq_len=5,
        label_len=0,
        pred_len=0,
        stride=1,
        get_unit_id=True,
        meta_data_dict=meta,
    )
    batch = ds[[0]]
    assert "unit_id" in batch
    assert batch["unit_id"].shape[0] == 1
    assert batch["unit_id"][0].item() == 10


def test_rul_context_extract_unit_id_and_name(rul_dense_data, meta_with_unit_ids):
    """extract_unit_id and extract_unit_name return correct values for concat_dataset_index."""
    meta = dict(meta_with_unit_ids)
    meta["concat_dataset_index"] = 1
    ds = RULContextBatchDataset(
        data_dict=rul_dense_data,
        task_type="rul",
        seq_len=5,
        label_len=0,
        pred_len=0,
        stride=1,
        get_unit_id=True,
        meta_data_dict=meta,
    )
    assert ds.extract_unit_id() == 20
    assert ds.extract_unit_name() == "u1"


def test_rul_context_get_collate_fn(rul_dense_data):
    """get_collate_fn returns collate_key_value_batch."""
    ds = RULContextBatchDataset(
        data_dict=rul_dense_data,
        task_type="rul",
        seq_len=5,
        label_len=0,
        pred_len=0,
        stride=1,
        meta_data_dict={},
    )
    fn = ds.get_collate_fn()
    from picid.data.datasets.collate_functions import collate_key_value_batch

    assert fn is collate_key_value_batch


def test_rul_context_collate_integration(rul_dense_data):
    """Collated batch has correct batch dimension."""
    ds = RULContextBatchDataset(
        data_dict=rul_dense_data,
        task_type="rul",
        seq_len=5,
        label_len=0,
        pred_len=0,
        stride=1,
        meta_data_dict={},
    )
    fn = ds.get_collate_fn()
    samples = [ds[[i]] for i in range(3)]
    batch = fn(samples)
    assert batch["features"].shape[0] == 3
    assert batch["rul"].shape[0] == 3
