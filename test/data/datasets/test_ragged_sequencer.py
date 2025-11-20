import pytest
import awkward as ak
from unittest.mock import patch
import numpy as np
from lmetk.data.optimization.ak_jagged_sequencer import (
    AkwardJaggedAutoregressiveSequencer,
)
from lmetk.utils.awkward_utils import ak_regularize_regular_axes

# --- Fixtures ---


@pytest.fixture
def dense_features_ak():
    """
    Creates a regular (dense) Awkward Array.
    Shape: (7, 3) -> Interpreted as 1 Unit, Time=7, Feat=3
    Values: Row index + col index/10 (for easy tracking)
    """
    # Using a predictable pattern: value = row + col*0.1
    data = [
        [0.0, 0.1, 0.2],  # t=0
        [1.0, 1.1, 1.2],  # t=1
        [2.0, 2.1, 2.2],  # t=2
        [3.0, 3.1, 3.2],  # t=3
        [4.0, 4.1, 4.2],  # t=4
        [5.0, 5.1, 5.2],  # t=5
        [6.0, 6.1, 6.2],  # t=6
    ]
    return ak_regularize_regular_axes(ak.Array(data))


@pytest.fixture
def jagged_features_ak():
    """
    Creates a Jagged Awkward Array.
    Unit 0: Length 5
    Unit 1: Length 3 (Short)
    Unit 2: Length 6
    Feat dim: 2
    """
    data = [
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],  # Unit 0
        [[10, 10], [11, 11], [12, 12]],  # Unit 1
        [[20, 20], [21, 21], [22, 22], [23, 23], [24, 24], [25, 25]],  # Unit 2
    ]
    return ak_regularize_regular_axes(ak.Array(data))


# --- Tests ---


def test_fetch_one_item_per_sequence_dense_values(dense_features_ak):
    """
    Tests iterating through a dense array with a stride, verifying EXACT content.
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_features_ak,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=2,
    )

    # L=7. Req = 2+1 = 3.
    # Valid starts: 0, 2, 4.
    # Window 0: Input [t0, t1], Pred [t2]
    # Window 1: Input [t2, t3], Pred [t4]
    # Window 2: Input [t4, t5], Pred [t6]

    expected_items = 3
    assert len(seq) == expected_items

    # Check Window 1 (Index 1) -> Starts at t=2
    seq_x, seq_y = seq.sequences_batch([1])

    # Expected X: t2, t3
    expected_x = np.array([[[2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]], dtype=np.float32)
    # Expected Y: t4 (since pred_len=1, label_len=0)
    expected_y = np.array([[[4.0, 4.1, 4.2]]], dtype=np.float32)

    np.testing.assert_allclose(seq_x.numpy(), expected_x, rtol=1e-5)
    np.testing.assert_allclose(seq_y.numpy(), expected_y, rtol=1e-5)


def test_fetch_array_of_indices_dense_overlap(dense_features_ak):
    """
    Tests fetching a batch of indices with overlap (label_len > 0).
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_features_ak,
        seq_len=3,
        label_len=1,
        pred_len=2,
        stride=1,
    )

    # L=7. Req = 3 + 2 = 5 (offset=0).
    # Max start = 7 - 5 = 2. Indices: 0, 1, 2.
    assert len(seq) == 3

    # Fetch index 2 (Start t=2)
    # X: [t2, t3, t4]
    # Y: label_len=1 (overlap t4) + pred_len=2 (t5, t6) -> [t4, t5, t6]
    seq_x, seq_y = seq.sequences_batch([2])

    assert seq_x.shape == (1, 3, 3)
    assert seq_y.shape == (1, 3, 3)  # 1 label + 2 pred

    # Verify overlap
    # Last step of X (t4) matches first step of Y (t4)
    np.testing.assert_allclose(seq_x[0, -1].numpy(), seq_y[0, 0].numpy())

    # Verify last step of Y is t6 (last element of array)
    expected_last = np.array([6.0, 6.1, 6.2], dtype=np.float32)
    np.testing.assert_allclose(seq_y[0, -1].numpy(), expected_last, rtol=1e-5)


def test_jagged_structure_strict_windowing_and_mapping(jagged_features_ak):
    """
    Tests that:
    1. Short units are dropped (Unit 1 len 3 < req 5).
    2. Indices map correctly to the remaining units (Unit 0 and Unit 2).
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=jagged_features_ak,
        seq_len=3,
        label_len=1,
        pred_len=2,
        stride=1,
    )

    # Req = 3 + 2 = 5.
    # Unit 0 (Len 5): Start 0. (1 window)
    # Unit 1 (Len 3): Dropped.
    # Unit 2 (Len 6): Start 0, 1. (2 windows)
    # Total = 3 windows.
    # Indices map: 0->(U0, t0), 1->(U2, t0), 2->(U2, t1)

    assert len(seq) == 3
    indices = seq.get_index_array()

    # Check mapping
    # Index 0 should be Unit 0
    assert indices[0, 0] == 0
    # Index 1 should be Unit 2
    assert indices[1, 0] == 2

    # Test fetching Index 2 (Unit 2, Start 1)
    # Unit 2 Data: 20, 21, 22, 23, 24, 25
    # Start 1.
    # X: [21, 22, 23]
    # Y: overlap(23) + pred(24, 25) -> [23, 24, 25]

    seq_x, seq_y = seq.sequences_batch([2])

    expected_x = np.array([[[21, 21], [22, 22], [23, 23]]], dtype=np.float32)
    expected_y = np.array([[[23, 23], [24, 24], [25, 25]]], dtype=np.float32)

    np.testing.assert_array_equal(seq_x.numpy(), expected_x)
    np.testing.assert_array_equal(seq_y.numpy(), expected_y)


def test_pred_offset_logic(dense_features_ak):
    """
    Tests the pred_offset parameter.
    Useful for creating gaps between input and target (e.g. seq [0,1], offset 1, pred [3])
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_features_ak,
        seq_len=2,
        label_len=0,
        pred_len=1,
        pred_offset=1,  # Skip 1 step
        stride=1,
    )

    # L=7. Req = seq(2) + offset(1) + pred(1) = 4.
    # Valid starts: 0, 1, 2, 3. (Max 7-4=3).

    assert len(seq) == 4

    # Fetch index 0
    # X: [t0, t1]
    # Gap: t2
    # Y: [t3]

    seq_x, seq_y = seq.sequences_batch([0])

    expected_x = np.array([[[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]], dtype=np.float32)
    expected_y = np.array([[[3.0, 3.1, 3.2]]], dtype=np.float32)  # t3

    np.testing.assert_allclose(seq_x.numpy(), expected_x, rtol=1e-5)
    np.testing.assert_allclose(seq_y.numpy(), expected_y, rtol=1e-5)


def test_edge_case_single_step_jagged():
    """
    Tests edge case where seq_len=1, pred_len=1.
    We use ACTUAL jagged data to force the sequencer into 'Ragged' mode.
    """
    # Unit 0: Len 2 [[10], [11]]. (Valid: 2 >= 2)
    # Unit 1: Len 1 [[20]].       (Invalid: 1 < 2, dropped)
    # This structure forces is_ragged=True
    data = ak.Array([[[10], [11]], [[20]]])
    # Regularize the feature dimension (inner list)
    data = ak_regularize_regular_axes(data)

    seq = AkwardJaggedAutoregressiveSequencer(
        features=data, seq_len=1, label_len=0, pred_len=1, stride=1
    )

    # Unit 0 is valid. Len 2. Req 2.
    # Valid start: 0.
    # X: [10] (t=0)
    # Y: [11] (t=1)
    assert len(seq) == 1

    x, y = seq.sequences_batch([0])

    #

    # Now shape should be correct (Batch, Time, Feat) -> (1, 1, 1)
    assert x.shape == (1, 1, 1)
    assert y.shape == (1, 1, 1)

    assert x[0, 0, 0] == 10
    assert y[0, 0, 0] == 11


def test_edge_case_single_step_multi_feature():
    """
    Tests edge case (B, 1, F) where F > 1.
    Verifies that Feature dimension is preserved and not squeezed/flattened.
    """
    # Create Jagged Data with 3 Features (F=3)
    # Unit 0: Len 2 (Valid for seq_len=1, pred_len=1)
    #   t=0: [1, 2, 3]
    #   t=1: [4, 5, 6]
    # Unit 1: Len 1 (Invalid, will be dropped)
    #   t=0: [7, 8, 9]
    data = ak.Array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]])
    data = ak_regularize_regular_axes(data)  # Type: 2 * var * 3 * int64

    seq = AkwardJaggedAutoregressiveSequencer(
        features=data, seq_len=1, label_len=0, pred_len=1, stride=1
    )

    # Unit 0 is valid. Start 0.
    assert len(seq) == 1

    x, y = seq.sequences_batch([0])

    # Check Shapes: (Batch, Time, Feat)
    # Expected: (1, 1, 3)
    assert x.shape == (1, 1, 3)
    assert y.shape == (1, 1, 3)

    # Verify Content
    # x should be Unit 0, t=0: [1, 2, 3]
    np.testing.assert_array_equal(x[0, 0].numpy(), [1, 2, 3])

    # y should be Unit 0, t=1: [4, 5, 6]
    np.testing.assert_array_equal(y[0, 0].numpy(), [4, 5, 6])


def test_jagged_with_large_stride(jagged_features_ak):
    """
    Tests that stride resets correctly between jagged units.
    Ensures indices are calculated relative to the start of EACH unit.
    """
    # Unit 0: Len 5
    # Unit 1: Len 3 (Dropped)
    # Unit 2: Len 6

    seq = AkwardJaggedAutoregressiveSequencer(
        features=jagged_features_ak,
        seq_len=2,
        pred_len=1,
        label_len=0,
        stride=2,  # Stride > 1
    )

    # Required Length = 2 + 1 = 3.

    # --- Unit 0 (Len 5) ---
    # Valid length: 5 - 3 + 1 = 3 slots (indices 0, 1, 2).
    # Stride 2:
    #   Start 0 (Valid) -> Input [0,1], Pred [2]
    #   Start 2 (Valid) -> Input [2,3], Pred [4]
    #   Start 4 (Invalid, > 2)
    # Expected: 2 windows.

    # --- Unit 1 (Len 3) ---
    # Valid length: 3 - 3 + 1 = 1 slot (index 0).
    # Stride 2:
    #   Start 0 (Valid) -> Input [10,11], Pred [12]
    # Expected: 1 window.

    # --- Unit 2 (Len 6) ---
    # Valid length: 6 - 3 + 1 = 4 slots (indices 0, 1, 2, 3).
    # Stride 2:
    #   Start 0 (Valid)
    #   Start 2 (Valid)
    #   Start 4 (Invalid, > 3)
    # Expected: 2 windows.

    # Total Expected: 2 + 1 + 2 = 5 windows.

    indices = seq.get_index_array()
    assert len(indices) == 5, f"Expected 5 windows, got {len(indices)}"

    # Check mapping for Unit 2
    # It should strictly contain starts 0 and 2.
    u2_indices = indices[indices[:, 0] == 2]
    u2_starts = u2_indices[:, -1]

    np.testing.assert_array_equal(u2_starts, [0, 2])

    # Verify data for Unit 2, Start 2
    # Data: 20, 21, 22, 23, 24, 25
    # Start 2 -> Input [22, 23], Pred [24]

    # We need to find the batch index that corresponds to Unit 2, Start 2.
    # Since we can't guarantee order without sorting, we filter by logic or just search.
    # For this test, we can just loop or trust the get_index_array order if deterministic.
    # Let's use the sequencer to fetch based on the raw index known from the array.

    # Find the index in the master list where Unit=2 and Start=2
    target_row = np.where((indices[:, 0] == 2) & (indices[:, -1] == 2))[0][0]

    x, y = seq.sequences_batch([target_row])

    expected_x = np.array([[[22, 22], [23, 23]]], dtype=np.float32)
    np.testing.assert_array_equal(x.numpy(), expected_x)


def test_dense_array_too_short():
    """
    Covers Line 97: Dense array length < required length.
    Should return empty indices.
    """
    # Data length 5
    data = ak_regularize_regular_axes(ak.Array(np.zeros((5, 1))))

    # Required: seq(3) + pred(3) = 6
    seq = AkwardJaggedAutoregressiveSequencer(
        features=data,
        seq_len=3,
        label_len=0,
        pred_len=3,
        stride=1,
    )

    assert len(seq) == 0
    assert len(seq.get_index_array()) == 0


def test_ragged_dim_zero_coverage(dense_features_ak):
    """
    Forces execution of Line 115 (ragged_dim == 0 logic) by mocking internal helpers.
    We ensure 'ragged_index_tuples' returns a value so the loop runs.
    """
    # Initialize with dense data (will effectively treat as 1 unit)
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_features_ak, seq_len=2, label_len=0, pred_len=1, stride=1
    )

    # FORCE ragged_dim to 0.
    seq.ragged_dim = 0

    # We need to mock 'ragged_index_tuples' to return a list containing an empty tuple [()].
    # This simulates the root level iteration.
    # We patch it where it is IMPORTED in the source file, not where it is defined.
    with patch(
        "lmetk.data.optimization.ak_jagged_sequencer.ragged_index_tuples"
    ) as mock_tuples:
        mock_tuples.return_value = [()]  # List containing one empty tuple

        # We also need to ensure 'ak.num' returns a valid length (int) for axis 0
        # The dense_features_ak has length 7. ak.num(..., axis=0) -> 7.
        # ak.to_list(7) -> 7. So 'lengths' will be 7.

        # Now run _build_indices.
        # It calls prefixes = [()]
        # Loop runs for prefix = ()
        # Calls get_length(())
        # Inside get_length: ragged_dim==0 is True.
        # 'if prefix' is False (empty tuple).
        # Returns 'lengths' (7).

        # This successfully executes the 'else lengths' branch of line 115.
        indices = seq._build_indices()

        # Optional: Verify we got some indices back (Length 7. Req 3. Valid 0,1,2,3,4. 5 windows?)
        assert len(indices) > 0
