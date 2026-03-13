"""
Phase 2: Transform integration & reproducibility.

Asserts that dataset output is deterministic given fixed seed and same index.
When random augmentations are applied in the pipeline (e.g. via transform Mixins),
reproducibility should hold for a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from picid.data.datasets.sliding_window_batch_dataset import SlidingWindowBatchDataset
from picid.data.datasets.context_dataset import ContextBatchDataset


@pytest.fixture
def seeded_data(fixed_seed):
    """Deterministic data for reproducibility checks."""
    np.random.seed(42)
    feats = np.random.randn(60, 4).astype(np.float32)
    target = np.random.randn(60, 1).astype(np.float32)
    return {"features": feats, "rul": target, "target": target}


def test_sliding_window_deterministic_same_index(seeded_data):
    """Same index returns identical tensors (no randomness in dataset)."""
    data = {"f": seeded_data["features"]}
    ds = SlidingWindowBatchDataset(
        data_dict=data,
        seq_len=5,
        label_len=0,
        pred_len=1,
        stride=1,
    )
    a = ds[[0]]
    b = ds[[0]]
    torch.testing.assert_close(a["f_seq_x"], b["f_seq_x"])
    torch.testing.assert_close(a["f_seq_y"], b["f_seq_y"])


def test_context_dataset_deterministic_same_index(seeded_data):
    """ContextBatchDataset returns same output for same index (deterministic)."""
    ds = ContextBatchDataset(
        data_dict=seeded_data,
        task_type="rul",
        seq_len=5,
        label_len=0,
        pred_len=1,
        stride=1,
    )
    a = ds[[0]]
    b = ds[[0]]
    torch.testing.assert_close(
        a.context.features_seq_x,
        b.context.features_seq_x,
    )
    torch.testing.assert_close(
        a.target.rul_seq_x,
        b.target.rul_seq_x,
    )


def test_reproducibility_after_reseed(seeded_data):
    """Resetting seed and re-creating dataset yields same __getitem__ result."""

    def make_and_get():
        np.random.seed(123)
        data = {"f": np.random.randn(40, 3).astype(np.float32)}
        ds = SlidingWindowBatchDataset(
            data_dict=data,
            seq_len=4,
            label_len=0,
            pred_len=1,
            stride=1,
        )
        return ds[[0]]["f_seq_x"].clone()

    t1 = make_and_get()
    t2 = make_and_get()
    torch.testing.assert_close(t1, t2)
