"""
Phase 1: BaseVectorizedConcatDataset — list indexing and collation.

Uses real sub-datasets (e.g. SlidingWindowBatchDataset) to assert
correct routing of list indices across concatenated datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

from picid.data.datasets.base import BaseVectorizedConcatDataset
from picid.data.datasets.sliding_window_batch_dataset import SlidingWindowBatchDataset
from picid.data.datasets.collate_functions import collate_key_value_batch


class _ConcatWithCollate(BaseVectorizedConcatDataset):
    """Concrete implementation that provides collate_fn."""

    def get_collate_fn(self):
        return collate_key_value_batch


@pytest.fixture
def two_small_sliding_datasets():
    """Two SlidingWindowBatchDataset with 5 windows each."""
    data_a = {"f": np.arange(20).reshape(20, 1).astype(np.float32)}
    data_b = {"f": np.arange(30, 50).reshape(20, 1).astype(np.float32)}
    ds_a = SlidingWindowBatchDataset(
        data_dict=data_a, seq_len=4, label_len=0, pred_len=1, stride=1
    )
    ds_b = SlidingWindowBatchDataset(
        data_dict=data_b, seq_len=4, label_len=0, pred_len=1, stride=1
    )
    return [ds_a, ds_b]


def test_base_vectorized_concat_list_indexing_single_dataset(
    two_small_sliding_datasets,
):
    """When len(datasets)==1, list index goes to the only dataset."""
    concat = _ConcatWithCollate(two_small_sliding_datasets[:1])
    out = concat[[0, 1]]
    assert "f_seq_x" in out
    assert out["f_seq_x"].shape[0] == 2


def test_base_vectorized_concat_list_indexing_two_datasets(two_small_sliding_datasets):
    """List indices are split across two datasets; batch has correct size."""
    concat = _ConcatWithCollate(two_small_sliding_datasets)
    # First dataset has 16 windows (20 - 4 - 1 + 1), second same
    total = len(concat)
    assert total == len(two_small_sliding_datasets[0]) + len(
        two_small_sliding_datasets[1]
    )
    indices = [0, total // 2, total - 1]
    out = concat[indices]
    assert out["f_seq_x"].shape[0] == 3


def test_base_vectorized_concat_scalar_index_not_supported(two_small_sliding_datasets):
    """Single int index raises NotImplementedError (list indexing only)."""
    concat = _ConcatWithCollate(two_small_sliding_datasets)
    with pytest.raises(NotImplementedError):
        concat[0]
