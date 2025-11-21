import pytest
import numpy as np
import awkward as ak
import torch

from lmetk.data.optimization.ak_jagged_sequencer import (
    AkwardJaggedAutoregressiveSequencer,
)
from lmetk.utils.awkward_utils import ak_regularize_regular_axes

# =========================================================================
# === Fixtures ===
# =========================================================================


@pytest.fixture
def dense_data():
    """
    Simple Dense Array: [10, 11, 12, 13, 14]
    Shape: (5, 1) -> (Batch=1, Time=5, Feat=1)
    """
    data = np.arange(10, 15).reshape(-1, 1).astype(np.float32)
    return ak_regularize_regular_axes(ak.Array(data))


@pytest.fixture
def jagged_data():
    """
    Jagged Array with mixed lengths.
    Unit 0: [10, 11, 12] (Len 3) - Short
    Unit 1: [20, 21, 22, 23, 24] (Len 5) - Long
    Feat dim: 1
    """
    data = [[[10], [11], [12]], [[20], [21], [22], [23], [24]]]
    return ak_regularize_regular_axes(ak.Array(data))


# =========================================================================
# === Tests ===
# =========================================================================


def test_padding_activates_for_short_sequences(dense_data):
    """
    Scenario: Data length (5) is shorter than required (Seq 6 + Pred 1 = 7).

    1. Strict Mode: Should return 0 windows.
    2. Padding Mode: Should return valid windows by zero-padding the left side.
    """
    # 1. Strict Mode (Default)
    seq_strict = AkwardJaggedAutoregressiveSequencer(
        features=dense_data, seq_len=6, pred_len=1, label_len=0, padding_left_flag=False
    )
    assert len(seq_strict) == 0

    # 2. Padding Mode
    # We want to predict starting at index 1 (2nd element, value 11).
    # History needed: 6. Available before index 1: 1 (value 10). Padding needed: 5.
    seq_pad = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=6,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        warmup_steps=1,  # Start predicting at index 1
    )

    # Valid Targets: Index 1, 2, 3, 4 (Values 11, 12, 13, 14). Total 4 windows.
    assert len(seq_pad) == 4

    # Check first window content
    # Window 0 -> Target Index 1 (Val 11)
    x, y = seq_pad.sequences_batch([0])

    # Expected X: 5 zeros + [10] -> Shape (1, 6, 1)
    expected_x = np.array([0, 0, 0, 0, 0, 10], dtype=np.float32).reshape(1, 6, 1)
    # Expected Y: [11] -> Shape (1, 1, 1)
    expected_y = np.array([11], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x.numpy(), expected_x)
    np.testing.assert_array_equal(y.numpy(), expected_y)


def test_padding_jagged_rescues_short_units(jagged_data):
    """
    Scenario: One unit is short, one is long.
    Config: Seq 4, Pred 1. Req 5.

    - Unit 0 (Len 3): Shorter than Req. Normally dropped.
    - Unit 1 (Len 5): Exactly Req. Valid.

    With Padding + Warmup=1, Unit 0 should now produce windows.
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=jagged_data,
        seq_len=4,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        warmup_steps=1,
    )

    indices = seq.get_index_array()

    # Verify Unit 0 is present
    # Unit 0 (Len 3). Targets at indices 1, 2. (2 windows).
    u0_indices = indices[indices[:, 0] == 0]
    assert len(u0_indices) == 2

    # Verify Unit 1 is present
    # Unit 1 (Len 5). Targets at indices 1, 2, 3, 4. (4 windows).
    u1_indices = indices[indices[:, 0] == 1]
    assert len(u1_indices) == 4

    # Check Data for Unit 0, first window (Target index 1 -> value 11)
    # Input needs 4 steps ending at index 0.
    # Data: [10, 11, 12]
    # Logic: Start = 1 - 4 = -3. Input Indices: [-3, -2, -1, 0] -> [0, 0, 0, 10]

    # We find the index corresponding to Unit 0.
    # Since u0_indices exist, the first row of the main index array (row 0) is Unit 0.
    x, y = seq.sequences_batch([0])

    expected_x = np.array([0, 0, 0, 10], dtype=np.float32).reshape(1, 4, 1)
    np.testing.assert_array_equal(x.numpy(), expected_x)


def test_warmup_steps_logic(dense_data):
    """
    Verifies how 'warmup_steps' controls the starting point.
    Data: [10, 11, 12, 13, 14] (Len 5)
    Seq: 2
    """
    # Case A: warmup_steps = 0 (Predict from very start, index 0)
    # Target: 0 (Value 10). Input: [-2, -1] -> [0, 0].
    seq_0 = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=2,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        warmup_steps=0,
    )
    # Targets: 0, 1, 2, 3, 4 (5 windows)
    assert len(seq_0) == 5
    x, y = seq_0.sequences_batch([0])

    # Input [0, 0] -> Target [10]
    # Reshape to (Batch=1, Seq=2, Feat=1)
    expected_x = np.array([0, 0], dtype=np.float32).reshape(1, 2, 1)
    expected_y = np.array([10], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x.numpy(), expected_x)
    np.testing.assert_array_equal(y.numpy(), expected_y)

    # Case B: warmup_steps = 2 (Predict starting index 2)
    # Target: 2 (Value 12). Input: [0, 1] -> [10, 11].
    # Since start index is 0 (non-negative), NO padding should be applied.
    seq_2 = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=2,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        warmup_steps=2,
    )
    # Targets: 2, 3, 4 (3 windows)
    assert len(seq_2) == 3
    x, y = seq_2.sequences_batch([0])

    # Input [10, 11] -> Target [12]
    expected_x_2 = np.array([10, 11], dtype=np.float32).reshape(1, 2, 1)
    expected_y_2 = np.array([12], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x.numpy(), expected_x_2)
    np.testing.assert_array_equal(y.numpy(), expected_y_2)


def test_padding_with_stride(dense_data):
    """
    Verifies that stride works correctly when starting indices are negative.
    Data: [10, 11, 12, 13, 14]
    """
    # Seq 3, Warmup 1, Stride 2.
    # First Target: 1 (Value 11).
    # First Input Start: 1 - 3 = -2.

    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=3,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        warmup_steps=1,
        stride=2,
    )

    # Window 1: Target 1 (11). Start -2. Input [-2, -1, 0] -> [0, 0, 10]
    # Window 2: Target 1+2=3 (13). Start 0. Input [0, 1, 2] -> [10, 11, 12]
    # Window 3: Target 3+2=5 (Out of bounds).

    assert len(seq) == 2

    # Check Window 1
    x1, y1 = seq.sequences_batch([0])
    np.testing.assert_array_equal(
        x1.numpy(), np.array([0, 0, 10], dtype=np.float32).reshape(1, 3, 1)
    )
    np.testing.assert_array_equal(
        y1.numpy(), np.array([11], dtype=np.float32).reshape(1, 1, 1)
    )

    # Check Window 2
    x2, y2 = seq.sequences_batch([1])
    np.testing.assert_array_equal(
        x2.numpy(), np.array([10, 11, 12], dtype=np.float32).reshape(1, 3, 1)
    )
    np.testing.assert_array_equal(
        y2.numpy(), np.array([13], dtype=np.float32).reshape(1, 1, 1)
    )


def test_padding_with_offset(dense_data):
    """
    Verifies pred_offset logic combined with padding.
    Data: [10, 11, 12, 13, 14]
    """
    # Seq 2. Offset 1 (Skip 1 step). Pred 1. Warmup 1.
    # Logic: First Target at Index 1 (Value 11).
    # Formula: min_start = warmup - seq - offset
    # min_start = 1 - 2 - 1 = -2.

    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=2,
        pred_len=1,
        label_len=0,
        pred_offset=1,
        padding_left_flag=True,
        warmup_steps=1,
    )

    # Window 1:
    # Start: -2. Input Indices: [-2, -1]. Data: [0, 0].
    # Offset Gap: Index 0 (Value 10).
    # Target: Index 1 (Value 11).

    x, y = seq.sequences_batch([0])

    np.testing.assert_array_equal(
        x.numpy(), np.array([0, 0], dtype=np.float32).reshape(1, 2, 1)
    )
    np.testing.assert_array_equal(
        y.numpy(), np.array([11], dtype=np.float32).reshape(1, 1, 1)
    )
