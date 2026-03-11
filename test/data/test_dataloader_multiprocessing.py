"""
Phase 2: Worker multiprocessing — num_workers > 0.

Runs DataLoader with multiple workers to surface pickling errors,
race conditions, or shared-memory issues. Uses a wrapper that
converts scalar indices to list so SlidingWindowBatchDataset
receives __getitem__(list) as in production.
"""

from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import Dataset, DataLoader

from picid.data.datasets.sliding_window_batch_dataset import SlidingWindowBatchDataset
from picid.data.datasets.collate_functions import collate_key_value_batch


class _ListIndexWrapper(Dataset):
    """Wraps a dataset that expects list indices; DataLoader passes scalar idx."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[[idx]]


@pytest.fixture
def small_sliding_dataset():
    """Small dataset that can be used with multiple workers."""
    data = {"f": np.random.randn(80, 4).astype(np.float32)}
    return SlidingWindowBatchDataset(
        data_dict=data,
        seq_len=5,
        label_len=0,
        pred_len=1,
        stride=2,
    )


@pytest.mark.parametrize("num_workers", [1, 2])
def test_dataloader_with_workers(small_sliding_dataset, num_workers):
    """DataLoader with num_workers > 0 completes without pickling/race errors (wrapper gives list indices)."""
    wrapped = _ListIndexWrapper(small_sliding_dataset)
    dl = DataLoader(
        wrapped,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_key_value_batch,
    )
    batches = list(dl)
    assert len(batches) >= 1
    b = batches[0]
    assert b["f_seq_x"].dim() == 3 and b["f_seq_x"].size(0) == 4


def test_dataloader_multiple_epochs_with_workers(small_sliding_dataset):
    """Multiple epochs with num_workers=2 do not crash (e.g. no stale handles)."""
    wrapped = _ListIndexWrapper(small_sliding_dataset)
    dl = DataLoader(
        wrapped,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_key_value_batch,
    )
    for _ in range(3):
        for batch in dl:
            assert batch["f_seq_x"].shape[0] <= 4
