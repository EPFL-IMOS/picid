import pytest
import numpy as np
import awkward as ak

from picid.data.optimization.ak_jagged_sequencer import (
    AkwardJaggedAutoregressiveSequencer,
)
from picid.utils.awkward_utils import ak_regularize_regular_axes

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
# === PART 1: Original Tests (Zero Padding) ===
# Updated `warmup_steps` to represent padding magnitude
# =========================================================================


def test_padding_left_flag_invalid_raises(dense_data):
    """padding_left_flag not in [0, 1] raises ValueError (coverage)."""
    with pytest.raises(ValueError, match="padding_left_flag must be boolean or int"):
        AkwardJaggedAutoregressiveSequencer(
            features=dense_data,
            seq_len=2,
            pred_len=1,
            label_len=0,
            padding_left_flag=2,
        )


def test_padding_activates_for_short_sequences(dense_data):
    """
    Scenario: Data length (5) is shorter than required (Seq 6 + Pred 1 = 7).
    """
    # 1. Strict Mode (Default)
    seq_strict = AkwardJaggedAutoregressiveSequencer(
        features=dense_data, seq_len=6, pred_len=1, label_len=0, padding_left_flag=False
    )
    assert len(seq_strict) == 0

    # 2. Padding Mode (Zero)
    # Goal: Target Index 1 (Value 11).
    # Math: Start = Target(1) - Seq(6) - Offset(0) = -5
    # WarmupSteps = 5
    seq_pad = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=6,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        padding_mode="zero",
        warmup_steps=5,  # <--- CHANGED: 5 padded steps needed
    )

    # Valid Targets: Indices 1, 2, 3, 4. Total 4 windows.
    assert len(seq_pad) == 4

    # Check first window content
    # Window 0 -> Start -5. Input [-5..0]
    x, y = seq_pad.sequences_batch([0])

    # Expected X: 5 zeros + [10] -> Shape (1, 6, 1)
    expected_x = np.array([0, 0, 0, 0, 0, 10], dtype=np.float32).reshape(1, 6, 1)
    expected_y = np.array([11], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x.numpy(), expected_x)
    np.testing.assert_array_equal(y.numpy(), expected_y)


def test_padding_jagged_rescues_short_units(jagged_data):
    """
    Scenario: One unit is short, one is long.
    Config: Seq 4, Pred 1. Req 5.
    """
    # Goal: Target Index 1 (Value 11).
    # Math: Start = 1 - 4 = -3. WarmupSteps = 3.
    seq = AkwardJaggedAutoregressiveSequencer(
        features=jagged_data,
        seq_len=4,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        padding_mode="zero",
    )

    indices = seq.get_index_array()

    # Verify Unit 0 is present
    u0_indices = indices[indices[:, 0] == 0]
    assert len(u0_indices) == 2

    # Verify Unit 1 is present
    u1_indices = indices[indices[:, 0] == 1]
    assert len(u1_indices) == 4

    # Check Data for Unit 0, first window
    x, y = seq.sequences_batch([0])

    expected_x = np.array([0, 0, 0, 10], dtype=np.float32).reshape(1, 4, 1)
    np.testing.assert_array_equal(x.numpy(), expected_x)


def test_padding_jagged_long_seq_len(jagged_data):
    """
    Scenario: One unit is short, one is long.
    Config: Seq 4, Pred 1. Req 5.
    """
    # Goal: Target Index 1 (Value 11).
    # Math: Start = 1 - 4 = -3. WarmupSteps = 3.
    seq = AkwardJaggedAutoregressiveSequencer(
        features=jagged_data,
        seq_len=200,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        padding_mode="zero",
    )

    assert len(seq) == 6
    indices = seq.get_index_array()

    # Verify Unit 0 is present
    u0_indices = indices[indices[:, 0] == 0]
    assert len(u0_indices) == 2

    # Verify Unit 1 is present
    u1_indices = indices[indices[:, 0] == 1]
    assert len(u1_indices) == 4

    # Check Data for Unit 0, first window
    x, y = seq.sequences_batch([0])

    expected_x = np.array(199 * [0] + [10], dtype=np.float32).reshape(1, 200, 1)
    np.testing.assert_array_equal(x.numpy(), expected_x)


def test_padding_jagged_long_seq_len_pred_len_zero(jagged_data):
    """
    Scenario: One unit is short, one is long.
    Config: Seq 4, Pred 1. Req 5.
    """
    # Goal: Target Index 1 (Value 11).
    # Math: Start = 1 - 4 = -3. WarmupSteps = 3.
    seq = AkwardJaggedAutoregressiveSequencer(
        features=jagged_data,
        seq_len=200,
        pred_len=0,
        label_len=0,
        padding_left_flag=True,
        padding_mode="edge",
    )

    assert len(seq) == 8


def test_warmup_steps_logic(dense_data):
    """
    Verifies how 'warmup_steps' controls the starting point.
    Data: [10, 11, 12, 13, 14] (Len 5)
    Seq: 2
    """
    # Case A: Predict from very start, index 0.
    # Math: Start = Target(0) - Seq(2) = -2.
    # WarmupSteps = 2.
    seq_0 = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=2,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        padding_mode="zero",
        warmup_steps=2,  # <--- CHANGED: Need 2 padded zeros to target index 0
    )
    # Targets: 0, 1, 2, 3, 4 (5 windows)
    assert len(seq_0) == 5
    x, y = seq_0.sequences_batch([0])

    # Input [-2, -1] -> [0, 0] -> Target [10]
    expected_x = np.array([0, 0], dtype=np.float32).reshape(1, 2, 1)
    expected_y = np.array([10], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x.numpy(), expected_x)
    np.testing.assert_array_equal(y.numpy(), expected_y)

    # Case B: Predict starting index 2.
    # Math: Start = Target(2) - Seq(2) = 0.
    # WarmupSteps = 0 (No padding).
    seq_2 = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=2,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        warmup_steps=0,  # <--- CHANGED: 0 padding means start at index 0
    )
    # Targets: 2, 3, 4 (3 windows)
    assert len(seq_2) == 3
    x, y = seq_2.sequences_batch([0])

    # Input [0, 1] -> [10, 11] -> Target [12]
    expected_x_2 = np.array([10, 11], dtype=np.float32).reshape(1, 2, 1)
    expected_y_2 = np.array([12], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x.numpy(), expected_x_2)
    np.testing.assert_array_equal(y.numpy(), expected_y_2)


def test_padding_with_stride(dense_data):
    """
    Verifies that stride works correctly when starting indices are negative.
    Data: [10, 11, 12, 13, 14]
    """
    # Seq 3. Target 1.
    # Math: Start = 1 - 3 = -2. WarmupSteps = 2.
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=3,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        padding_mode="zero",
        warmup_steps=2,  # <--- CHANGED
        stride=2,
    )

    # Window 1: Start -2. Target 1 (11).
    # Window 2: Start 0. Target 3 (13).
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
    # Seq 2. Offset 1. Target 1.
    # Math: Start = 1 - 2 - 1 = -2. WarmupSteps = 2.
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=2,
        pred_len=1,
        label_len=0,
        pred_offset=1,
        padding_left_flag=True,
        padding_mode="zero",
        warmup_steps=2,  # <--- CHANGED
    )

    x, y = seq.sequences_batch([0])

    np.testing.assert_array_equal(
        x.numpy(), np.array([0, 0], dtype=np.float32).reshape(1, 2, 1)
    )
    np.testing.assert_array_equal(
        y.numpy(), np.array([11], dtype=np.float32).reshape(1, 1, 1)
    )


# =========================================================================
# === PART 2: New Tests (Edge Padding) ===
# Explicit tests for padding_mode="edge"
# =========================================================================


def test_edge_padding_repeats_first_value(dense_data):
    """
    Scenario: Validate that padding_mode="edge" repeats the first available value (index 0).
    """
    # Seq 4. Target 1.
    # Math: Start = 1 - 4 = -3. WarmupSteps = 3.
    seq_edge = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=4,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        padding_mode="edge",
        warmup_steps=3,  # <--- CHANGED
    )

    # Check Window 0
    x, y = seq_edge.sequences_batch([0])

    # Expected X: [10, 10, 10, 10]
    expected_x = np.array([10, 10, 10, 10], dtype=np.float32).reshape(1, 4, 1)
    expected_y = np.array([11], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x.numpy(), expected_x)
    np.testing.assert_array_equal(y.numpy(), expected_y)


def test_edge_padding_jagged_unit(jagged_data):
    """
    Scenario: Validate edge padding on a short unit in a jagged array.
    """
    # Seq 5. Target 1.
    # Math: Start = 1 - 5 = -4. WarmupSteps = 4.
    seq = AkwardJaggedAutoregressiveSequencer(
        features=jagged_data,
        seq_len=5,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        padding_mode="edge",
        warmup_steps=4,  # <--- CHANGED
    )

    # Unit 0 is the first unit, so batch_idx=[0] should fetch it.
    x, y = seq.sequences_batch([0])

    # Expected X: [10, 10, 10, 10, 10]
    expected_x = np.array([10, 10, 10, 10, 10], dtype=np.float32).reshape(1, 5, 1)

    np.testing.assert_array_equal(x.numpy(), expected_x)


def test_padding_mode_validation(dense_data):
    """
    Verifies that providing an invalid padding mode raises an error.
    """
    with pytest.raises(ValueError, match="padding_mode must be 'zero' or 'edge'"):
        AkwardJaggedAutoregressiveSequencer(
            features=dense_data,
            seq_len=2,
            label_len=0,
            pred_len=1,
            padding_left_flag=True,
            padding_mode="invalid_mode",
        )


# =========================================================================
# === PART 3: Offset Logic Tests ===
# =========================================================================


def test_offset_no_padding(dense_data):
    """
    Scenario: Strict Mode (No Padding).
    Data: [10, 11, 12, 13, 14] (Len 5)
    Seq: 2, Offset: 2, Pred: 1
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=2,
        pred_len=1,
        label_len=0,
        pred_offset=2,
        padding_left_flag=False,  # Strict
    )

    # Valid Windows:
    # 1. Input [10, 11] (Idx 0,1) -> Target [14] (Idx 4).
    # Next start would be index 1 -> Input [11, 12] -> Target 15 (Index 5 - Out of bounds).
    assert len(seq) == 1

    # Check Window 0
    x, y = seq.sequences_batch([0])

    expected_x = np.array([10, 11], dtype=np.float32).reshape(1, 2, 1)
    expected_y = np.array([14], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x.numpy(), expected_x)
    np.testing.assert_array_equal(y.numpy(), expected_y)


def test_offset_with_padding_zero(dense_data):
    """
    Scenario: Padding Mode (Zero) with Offset 2.
    Data: [10, 11, 12, 13, 14] (Len 5)
    Seq: 3, Offset: 2, Pred: 1
    """
    # Logic: Target 1.
    # Math: Start = 1 - 3 - 2 = -4. WarmupSteps = 4.
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=3,
        pred_len=1,
        label_len=0,
        pred_offset=2,
        padding_left_flag=True,
        padding_mode="zero",
        warmup_steps=4,  # <--- CHANGED
    )

    # Targets: 1, 2, 3, 4. Total 4.
    assert len(seq) == 4

    # Window 1 (Index 0): Target 11. Start -4.
    x, y = seq.sequences_batch([0])
    expected_x = np.array([0, 0, 0], dtype=np.float32).reshape(1, 3, 1)
    expected_y = np.array([11], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x.numpy(), expected_x)
    np.testing.assert_array_equal(y.numpy(), expected_y)

    # Window 4 (Index 3): Target 14. Start -1.
    # Logic: Start -1 -> Input [-1, 0, 1] -> [0, 10, 11].
    x_last, y_last = seq.sequences_batch([3])
    expected_x_last = np.array([0, 10, 11], dtype=np.float32).reshape(1, 3, 1)
    expected_y_last = np.array([14], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x_last.numpy(), expected_x_last)
    np.testing.assert_array_equal(y_last.numpy(), expected_y_last)


def test_offset_with_padding_edge(dense_data):
    """
    Scenario: Padding Mode (Edge) with Offset 2.
    """
    # Logic: Target 1.
    # Math: Start = 1 - 3 - 2 = -4. WarmupSteps = 4.
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=3,
        pred_len=1,
        label_len=0,
        pred_offset=2,
        padding_left_flag=True,
        padding_mode="edge",
        warmup_steps=4,  # <--- CHANGED
    )

    # Window 1 (Index 0): Target 11. Start -4.
    # Edge Logic: [-4, -3, -2] -> [10, 10, 10].
    x, y = seq.sequences_batch([0])
    expected_x = np.array([10, 10, 10], dtype=np.float32).reshape(1, 3, 1)
    expected_y = np.array([11], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x.numpy(), expected_x)
    np.testing.assert_array_equal(y.numpy(), expected_y)

    # Optional Check: Last Window (Index 3)
    # Target: 14. Start: -1. Input [-1, 0, 1] -> [10, 10, 11]
    x_last, y_last = seq.sequences_batch([3])
    expected_x_last = np.array([10, 10, 11], dtype=np.float32).reshape(1, 3, 1)
    expected_y_last = np.array([14], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x_last.numpy(), expected_x_last)
    np.testing.assert_array_equal(y_last.numpy(), expected_y_last)


# =========================================================================
# === PART 4: seq_len=0 Edge Cases ===
# =========================================================================


def test_seq_len_zero_strict_dense(dense_data):
    """
    seq_len=0 should return empty encoder windows and valid targets in strict mode.
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=0,
        pred_len=1,
        label_len=0,
        padding_left_flag=False,
    )

    # Required len = pred_len = 1 -> all 5 target positions are valid.
    assert len(seq) == 5

    x, y = seq.sequences_batch([0])
    assert x.shape == (1, 0, 1)
    np.testing.assert_array_equal(
        y.numpy(), np.array([10], dtype=np.float32).reshape(1, 1, 1)
    )


def test_seq_len_zero_padding_default_warmup_dense(dense_data):
    """
    With seq_len=0 and default warmup in padding mode, first index is skipped.
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=0,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        padding_mode="zero",
    )

    # Default index_start_offset = -(seq_len - 1) = 1
    assert len(seq) == 4
    np.testing.assert_array_equal(
        seq.get_index_array(),
        np.array([[1], [2], [3], [4]]),
    )

    x, y = seq.sequences_batch([0])
    assert x.shape == (1, 0, 1)
    np.testing.assert_array_equal(
        y.numpy(), np.array([11], dtype=np.float32).reshape(1, 1, 1)
    )


def test_seq_len_zero_padding_warmup_zero_dense(dense_data):
    """
    Explicit warmup_steps=0 with seq_len=0 should include index 0.
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=0,
        pred_len=1,
        label_len=0,
        padding_left_flag=True,
        padding_mode="edge",
        warmup_steps=0,
    )

    assert len(seq) == 5
    np.testing.assert_array_equal(
        seq.get_index_array(),
        np.array([[0], [1], [2], [3], [4]]),
    )

    x, y = seq.sequences_batch([0])
    assert x.shape == (1, 0, 1)
    np.testing.assert_array_equal(
        y.numpy(), np.array([10], dtype=np.float32).reshape(1, 1, 1)
    )


def test_seq_len_zero_strict_jagged(jagged_data):
    """
    seq_len=0 should work on jagged arrays and preserve per-unit target counts.
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=jagged_data,
        seq_len=0,
        pred_len=1,
        label_len=0,
        padding_left_flag=False,
    )

    indices = seq.get_index_array()
    u0_indices = indices[indices[:, 0] == 0]
    u1_indices = indices[indices[:, 0] == 1]

    # Unit 0 len=3 -> 3 windows; Unit 1 len=5 -> 5 windows.
    assert len(u0_indices) == 3
    assert len(u1_indices) == 5


# =========================================================================
# === PART 5: pred_len=0 Edge Cases ===
# =========================================================================


def test_pred_len_zero_strict_dense_returns_empty_target(dense_data):
    """
    pred_len=0 with label_len=0 should return empty targets in strict mode.
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=2,
        pred_len=0,
        label_len=0,
        padding_left_flag=False,
    )

    # Logic:
    # required_len = seq_len + pred_len + pred_offset = 2 + 0 + 0 = 2
    # Dense len=5 -> valid starts are [0, 1, 2, 3] => 4 windows.
    assert len(seq) == 4

    x, y = seq.sequences_batch([0])

    # Window start t=0:
    # x = indices [0, 1] -> [10, 11]
    # y length = label_len + pred_len = 0 -> empty target sequence.
    expected_x = np.array([10, 11], dtype=np.float32).reshape(1, 2, 1)
    expected_y = np.empty((1, 0, 1), dtype=np.float32)

    np.testing.assert_array_equal(x.numpy(), expected_x)
    np.testing.assert_array_equal(y.numpy(), expected_y)


def test_pred_len_zero_with_label_len_dense(dense_data):
    """
    pred_len=0 still returns decoder-label context when label_len > 0.
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=2,
        pred_len=0,
        label_len=1,
        padding_left_flag=False,
    )

    # Logic:
    # required_len = 2 + 0 + 0 = 2 -> starts [0, 1, 2, 3], so 4 windows.
    # y start formula:
    # y starts at t + seq_len - label_len = t + 2 - 1 = t + 1, with length 1.
    assert len(seq) == 4

    x, y = seq.sequences_batch([0])

    # Window start t=0:
    # x = [10, 11]
    # y = index [1] -> [11]
    expected_x = np.array([10, 11], dtype=np.float32).reshape(1, 2, 1)
    expected_y = np.array([11], dtype=np.float32).reshape(1, 1, 1)

    np.testing.assert_array_equal(x.numpy(), expected_x)
    np.testing.assert_array_equal(y.numpy(), expected_y)


def test_pred_len_zero_padding_zero_warmup_dense(dense_data):
    """
    pred_len=0 with left padding should zero-pad only encoder history.
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=dense_data,
        seq_len=3,
        pred_len=0,
        label_len=0,
        padding_left_flag=True,
        padding_mode="zero",
        warmup_steps=2,
    )

    # Logic:
    # warmup_steps=2 -> first start t=-2.
    # required_len = 3 + 0 + 0 = 3, len=5 -> starts [-2, -1, 0, 1, 2] => 5 windows.
    assert len(seq) == 5

    x0, y0 = seq.sequences_batch([0])

    # Window start t=-2:
    # x indices [-2, -1, 0] -> zero padding for negative positions => [0, 0, 10]
    # y length is 0 -> empty target
    expected_x0 = np.array([0, 0, 10], dtype=np.float32).reshape(1, 3, 1)
    expected_y0 = np.empty((1, 0, 1), dtype=np.float32)

    np.testing.assert_array_equal(x0.numpy(), expected_x0)
    np.testing.assert_array_equal(y0.numpy(), expected_y0)


def test_pred_len_zero_strict_jagged_counts(jagged_data):
    """
    pred_len=0 should preserve strict per-unit counts on jagged data.
    """
    seq = AkwardJaggedAutoregressiveSequencer(
        features=jagged_data,
        seq_len=2,
        pred_len=0,
        label_len=0,
        padding_left_flag=False,
    )

    indices = seq.get_index_array()
    u0_indices = indices[indices[:, 0] == 0]
    u1_indices = indices[indices[:, 0] == 1]

    # Logic:
    # Unit 0 len=3, required_len=2 -> starts [0, 1] => 2 windows.
    # Unit 1 len=5, required_len=2 -> starts [0, 1, 2, 3] => 4 windows.
    assert len(u0_indices) == 2
    assert len(u1_indices) == 4

    # First global window belongs to Unit 0 at start t=0.
    x, y = seq.sequences_batch([0])
    expected_x = np.array([10, 11], dtype=np.float32).reshape(1, 2, 1)
    expected_y = np.empty((1, 0, 1), dtype=np.float32)

    np.testing.assert_array_equal(x.numpy(), expected_x)
    np.testing.assert_array_equal(y.numpy(), expected_y)
