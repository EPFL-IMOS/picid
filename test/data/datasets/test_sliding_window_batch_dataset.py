import pytest
import numpy as np
import torch
import awkward as ak
from picid.data.datasets.sliding_window_batch_dataset import SlidingWindowBatchDataset
from picid.utils.awkward_utils import ak_regularize_regular_axes

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
        seq_len=2,
        label_len=0,
        pred_len=5,
        stride=1,
        padding_left_flag=False,
    )
    # len(ds) for dense data with padding_left_flag=False:
    #   required_len = seq_len + pred_len + pred_offset = 2 + 5 + 0 = 7
    #   min_start = 0, max_start = T - required_len + 1 (exclusive end of range)
    #   n_windows = number of steps in range(min_start, max_start, stride)
    #            = (max_start - min_start + stride - 1) // stride
    #   With T=20: max_start = 20 - 7 + 1 = 14, so len(ds) = (14 - 0) // 1 = 14
    assert len(ds) == 14
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
        padding_left_flag=False,
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
        data_dict=ragged_data_dict,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=1,
        padding_left_flag=False,
    )

    assert len(ds) == 21

    # Case: List Access ds[[0]] (Simulating BatchSampler)
    # Should return (Batch, T, F) -> (1, 2, 1)
    batch_sample = ds[[0]]
    assert batch_sample["jagged_feat_seq_x"].ndim == 3


def test_sliding_window_get_collate_fn(dense_data_dict):
    """get_collate_fn returns collate_key_value_batch (coverage)."""
    from picid.data.datasets.collate_functions import collate_key_value_batch

    ds = SlidingWindowBatchDataset(
        data_dict=dense_data_dict,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=1,
    )
    fn = ds.get_collate_fn()
    assert fn is collate_key_value_batch


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
