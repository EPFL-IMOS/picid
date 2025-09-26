from pprint import pprint
import pytest
import awkward as ak

from lmetk.data.optimization.ak_jagged_sequencer import (
    AkwardJaggedAutoregressiveSequencer,
)
from lmetk.utils.awkward_utils import ak_regularize_regular_axes


def test_fetch_one_item_per_sequence():
    features = ak_regularize_regular_axes(
        ak.Array(
            [
                [0, 2, 3],
                [1, 3, 4],
                [2, 4, 5],
                [3, 5, 6],
                [4, 6, 7],
                [5, 7, 8],
                [6, 8, 9],
            ]
        )
    )
    time_feats = None

    seq = AkwardJaggedAutoregressiveSequencer(
        features=features,
        time_features=time_feats,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=2,
    )
    for i in range(seq.__len__()):
        seq_x, seq_y = seq.sequences_batch([i])

        # Assertions for seq_x and seq_y
        assert seq_x is not None, "seq_x should not be None"
        assert seq_y is not None, "seq_y should not be None"
        assert len(seq_x) > 0, "seq_x should not be empty"
        assert len(seq_y) > 0, "seq_y should not be empty"
        assert (
            seq_x.shape[1] == seq.seq_len
        ), f"seq_x should have sequence length {seq.seq_len}"
        assert (
            seq_y.shape[1] == seq.pred_len
        ), f"seq_y should have prediction length {seq.pred_len}"

    # Assertions for total batch length
    total_batches = seq.__len__()
    total_items = sum(
        seq.sequences_batch([i])[0].shape[0] for i in range(total_batches)
    )
    expected_items = (len(features) - seq.seq_len - seq.pred_len) // seq.stride + 1
    assert (
        total_items == expected_items
    ), "Total items in batches should match the expected number based on stride"

    # Check if dtype of seq_x and seq_y is torch
    for i in range(seq.__len__()):
        seq_x, seq_y = seq.sequences_batch([i])
        assert hasattr(seq_x, "dtype"), "seq_x should have a dtype attribute"
        assert hasattr(seq_y, "dtype"), "seq_y should have a dtype attribute"
        assert str(seq_x.dtype).startswith("torch"), "seq_x dtype should be torch"
        assert str(seq_y.dtype).startswith("torch"), "seq_y dtype should be torch"


def test_fetch_array_of_indices():
    features = ak_regularize_regular_axes(
        ak.Array(
            [
                [0, 2, 3, 4],
                [1, 3, 4, 5],
                [2, 4, 5, 6],
                [3, 5, 6, 7],
                [4, 6, 7, 8],
                [5, 7, 8, 9],
                [6, 8, 9, 10],
            ]
        )
    )
    time_feats = None

    seq = AkwardJaggedAutoregressiveSequencer(
        features=features,
        time_features=time_feats,
        seq_len=3,
        label_len=1,
        pred_len=2,
        stride=1,
    )
    indices = [0, 2]
    seq_x, seq_y = seq.sequences_batch(indices)

    pprint(f"seq_x for indices {indices}: {seq_x.shape}")
    pprint(f"seq_y for indices {indices}: {seq_y.shape}")
