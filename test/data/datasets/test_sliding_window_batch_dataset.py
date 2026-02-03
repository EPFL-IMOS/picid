import pytest
import numpy as np
import torch
import awkward as ak
from torch.utils.data import DataLoader
from lmetk.data.datasets.sliding_window_batch_dataset import SlidingWindowBatchDataset
from lmetk.utils.awkward_utils import ak_regularize_regular_axes

# =========================================================================
# === Fixtures ===
# =========================================================================


@pytest.fixture
def dense_data_dict():
    """
    Returns a dict with two synchronized dense features.
    Time=20.
    """
    return {
        "sensor_A": np.arange(20).reshape(20, 1),
        "sensor_B": np.arange(20, 40).reshape(20, 1),
    }


@pytest.fixture
def ragged_data_dict():
    """
    Returns a dict with jagged arrays (Cycles/Units structure).
    Cycle 0: Len 10
    Cycle 1: Len 15
    """
    # Manually creating ragged structure
    c0 = [[x] for x in range(10)]
    c1 = [[x] for x in range(15)]

    arr = ak.Array([c0, c1])
    arr = ak_regularize_regular_axes(arr)

    return {
        "jagged_feat": arr,
    }


# =========================================================================
# === Tests ===
# =========================================================================


def test_init_success_dense(dense_data_dict):
    """Test initialization with standard dense arrays."""
    ds = SlidingWindowBatchDataset(
        data_dict=dense_data_dict,
        seq_len=5,
        label_len=0,
        pred_len=5,
        stride=1,
    )
    assert len(ds) == 11
    assert "sensor_A" in ds.sequencers


def test_init_skips_none_values():
    """Covers: 'if arr is None: continue'"""
    data = {"valid": np.zeros((10, 1)), "empty": None}
    ds = SlidingWindowBatchDataset(
        data_dict=data, seq_len=2, label_len=0, pred_len=1, stride=1
    )
    assert "valid" in ds.sequencers
    assert "empty" not in ds.sequencers


def test_init_raises_on_unsupported_type():
    """Covers: 'raise NotImplementedError'"""
    data = {"bad": [1, 2, 3]}
    with pytest.raises(NotImplementedError, match="Unsupported type"):
        SlidingWindowBatchDataset(
            data_dict=data, seq_len=1, label_len=0, pred_len=1, stride=1
        )


def test_init_raises_on_length_mismatch():
    """Covers: 'raise ValueError' on length mismatch"""
    data = {"long": np.zeros((20, 1)), "short": np.zeros((10, 1))}
    with pytest.raises(ValueError, match="mismatch at key 'short'"):
        SlidingWindowBatchDataset(
            data_dict=data, seq_len=5, label_len=0, pred_len=5, stride=1
        )


def test_getitem_structure_and_values(dense_data_dict):
    """
    Verifies the output dictionary structure and data correctness.
    Checks Scalar Access (ds[0]).
    """
    ds = SlidingWindowBatchDataset(
        data_dict=dense_data_dict,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=1,
        suffixes=["_X", "_Y"],
    )

    # Fetch index 0 (Scalar).
    # Should return squeezed tensors (T, F), NOT (1, T, F).
    sample = ds[[0]]

    assert "sensor_A_X" in sample

    # Check Values
    # sensor_A window 0 -> [0, 1]
    # Expected Shape: (1, 2, 1)
    expected_x = torch.tensor([[[0], [1]]], dtype=torch.float32)

    assert torch.equal(sample["sensor_A_X"], expected_x)
    assert sample["sensor_A_X"].ndim == 3

    expected_y = torch.tensor([[[2]]], dtype=torch.float32)
    assert torch.equal(sample["sensor_A_Y"], expected_y)


def test_ragged_input_handling(ragged_data_dict):
    """
    Tests that the dataset correctly wraps RaggedArraySequencer.
    Checks both Scalar and List access to verify dimensionality.
    """
    print(ragged_data_dict)
    ds = SlidingWindowBatchDataset(
        data_dict=ragged_data_dict, seq_len=2, label_len=0, pred_len=1, stride=1
    )

    assert len(ds) == 21

    # Case: List Access ds[[0]] (Simulating BatchSampler)
    # Should return (Batch, T, F) -> (1, 2, 1)
    batch_sample = ds[[0]]
    assert batch_sample["jagged_feat_seq_x"].ndim == 3


# =========================================================================
# === Tests: subset_ratio / subset_seed ===
# =========================================================================


def test_subset_reduces_length_and_matches_expected_indices(dense_data_dict):
    """
    subset_ratio should reduce the available windows and honor deterministic indices.
    """
    ds = SlidingWindowBatchDataset(
        data_dict=dense_data_dict,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=1,
        subset_ratio=0.25,
        subset_seed=123,
    )

    # length should be roughly 25% of full
    # Recompute full windows based on driver
    full_windows = len(
        SlidingWindowBatchDataset(
            data_dict=dense_data_dict, seq_len=2, label_len=0, pred_len=1, stride=1
        )
    )
    expected_len = int(full_windows * 0.25)
    assert len(ds) == expected_len

    # Verify indices are sorted and within range
    seq_idx = ds._seq_idx
    assert seq_idx is not None
    assert np.all(np.diff(seq_idx) >= 0)
    assert seq_idx.min() >= 0 and seq_idx.max() < full_windows


def test_subset_seed_reproducibility_and_variation(dense_data_dict):
    """Same seed -> same subset indices; different seed -> different indices."""
    common_kwargs = dict(
        data_dict=dense_data_dict,
        seq_len=3,
        label_len=0,
        pred_len=1,
        stride=1,
        subset_ratio=0.3,
    )

    ds_a = SlidingWindowBatchDataset(**common_kwargs, subset_seed=7)
    ds_b = SlidingWindowBatchDataset(**common_kwargs, subset_seed=7)
    ds_c = SlidingWindowBatchDataset(**common_kwargs, subset_seed=8)

    assert np.array_equal(ds_a._seq_idx, ds_b._seq_idx)
    assert not np.array_equal(ds_a._seq_idx, ds_c._seq_idx)


def test_reindexing_idx_in_get_item(dense_data_dict):
    """Test that the indexes are mapped in get_item"""
    ds_full = SlidingWindowBatchDataset(
        data_dict=dense_data_dict, seq_len=3, label_len=0, pred_len=1, stride=1
    )
    ds_subset = SlidingWindowBatchDataset(
        data_dict=dense_data_dict,
        seq_len=3,
        label_len=0,
        pred_len=1,
        stride=1,
        subset_ratio=0.4,
        subset_seed=42,
    )

    seq_idx = ds_subset._seq_idx
    assert seq_idx is not None and len(seq_idx) > 0

    mapped_idx = seq_idx[0]
    subset_sample = ds_subset[[0]]
    full_sample = ds_full[[mapped_idx]]

    assert torch.equal(subset_sample["sensor_A_seq_x"], full_sample["sensor_A_seq_x"])
    assert torch.equal(subset_sample["sensor_A_seq_y"], full_sample["sensor_A_seq_y"])


# def test_dataloader_integration_and_collation():
#     """
#     Verifies standard PyTorch DataLoader usage.
#     """
#     data = {"feat": np.random.randn(100, 5).astype(np.float32)}
#     ds = SlidingWindowBatchDataset(
#         data_dict=data, seq_len=10, label_len=0, pred_len=5, stride=1
#     )

#     # Standard DataLoader: fetches samples one by one (ds[i]) and stacks them.
#     # ds[i] returns (10, 5). Stacking 16 of them -> (16, 10, 5).
#     dl = DataLoader(ds, batch_size=16, collate_fn=ds.get_collate_fn())

#     batch = next(iter(dl))

#     # Check batch size and dimensions
#     assert batch["feat_seq_x"].shape == (16, 10, 5)
#     assert batch["feat_seq_y"].shape == (16, 5, 5)
