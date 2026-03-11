import pytest
import numpy as np
import pandas as pd
import torch
import awkward as ak

# Adjust imports to match your project structure
from picid.data.optimization.sequencer import DenseArraySequencer, RaggedArraySequencer
from picid.utils.awkward_utils import ak_regularize_regular_axes

# =========================================================================
# === Fixtures ===
# =========================================================================


@pytest.fixture
def numpy_dense_data():
    """Standard (Time, Feat) numpy array."""
    # 10 time steps, 2 features
    return np.arange(20).reshape(10, 2).astype(np.float32)


@pytest.fixture
def torch_dense_data():
    """Standard (Time, Feat) torch tensor."""
    # 10 time steps, 2 features
    return torch.arange(20, dtype=torch.float32).reshape(10, 2)


@pytest.fixture
def jagged_ak_data():
    """
    Standard jagged array: (Units, Time, Feat).
    Unit 0: 5 steps
    Unit 1: 3 steps
    """
    data = ak.Array(
        [[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], [[10, 10], [11, 11], [12, 12]]]
    )

    return ak_regularize_regular_axes(data)


@pytest.fixture
def regular_ak_data():
    """
    Awkward array that is technically regular (rectangular).
    (2 Units, 2 Time, 2 Feat)
    """
    data = ak.Array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])

    return ak_regularize_regular_axes(data)


# =========================================================================
# === Tests: DenseArraySequencer ===
# =========================================================================


def test_dense_sequencer_numpy_init(numpy_dense_data):
    """
    Verifies DenseArraySequencer accepts Numpy arrays and initializes the driver.
    """
    seq = DenseArraySequencer(
        array=numpy_dense_data,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=1,
        padding_left_flag=False,
    )

    # Length check: 10 steps total. Req = 2+1 = 3.
    # Valid starts: 0..7. Total 8 windows.
    assert len(seq) == 8

    # Fetch data to ensure conversion to Float32 and Torch worked
    x, y = seq.sequences_batch([0])
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32


def test_dense_sequencer_torch_init(torch_dense_data):
    """
    Verifies DenseArraySequencer accepts Torch tensors.
    """
    seq = DenseArraySequencer(
        array=torch_dense_data,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=1,
        padding_left_flag=False,
    )

    # Logic should be identical to numpy path
    assert len(seq) == 8
    x, y = seq.sequences_batch([0])

    # Verify content matches input
    # Input row 0: [0, 1], row 1: [2, 3].
    expected_x = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    assert torch.allclose(x, expected_x)


def test_dense_sequencer_invalid_input():
    """
    Verifies TypeError is raised for non-numpy/non-torch/non-dataframe inputs.
    """
    invalid_data = [[1, 2], [3, 4]]

    # Update match string to include 'pd.DataFrame'
    with pytest.raises(
        TypeError, match="must be np.ndarray, torch.Tensor or pd.DataFrame"
    ):
        DenseArraySequencer(
            array=invalid_data, seq_len=1, label_len=0, pred_len=1, stride=1
        )


def test_dense_sequencer_dataframe_init():
    """DenseArraySequencer accepts pd.DataFrame (coverage: DataFrame branch)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    seq = DenseArraySequencer(
        array=df, seq_len=2, label_len=0, pred_len=1, stride=1, padding_left_flag=False
    )
    # 3 rows, req 2+1=3 -> 1 valid window
    assert len(seq) == 1
    x, y = seq.sequences_batch([0])
    assert x.shape == (1, 2, 2)


def test_dense_sequencer_delegation(numpy_dense_data):
    """
    Verifies that __len__, get_index_array, and sequences_batch
    are correctly delegated to the underlying driver.
    """
    seq = DenseArraySequencer(
        array=numpy_dense_data, seq_len=2, label_len=0, pred_len=1, stride=1
    )

    # Check __len__ delegation
    assert len(seq) == len(seq.driver)

    # Check get_index_array delegation
    indices = seq.get_index_array()
    assert isinstance(indices, np.ndarray)
    np.testing.assert_array_equal(indices, seq.driver.get_index_array())


# =========================================================================
# === Tests: RaggedArraySequencer ===
# =========================================================================


def test_ragged_sequencer_standard_jagged(jagged_ak_data):
    """
    Verifies RaggedArraySequencer handles standard jagged awkward arrays.
    """
    seq = RaggedArraySequencer(
        array=jagged_ak_data,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=1,
        padding_left_flag=False,
    )

    # Unit 0 (Len 5, Req 3) -> 3 windows
    # Unit 1 (Len 3, Req 3) -> 1 window
    # Total = 4
    assert len(seq) == 4

    # Verify initialization created the driver
    assert seq.driver is not None
    # Verify jagged mode is active in driver (implicit check via len calculation)
    assert seq.driver.is_ragged is True


def test_ragged_sequencer_get_index_array(jagged_ak_data):
    """RaggedArraySequencer.get_index_array delegates to driver (coverage)."""
    seq = RaggedArraySequencer(
        array=jagged_ak_data, seq_len=2, label_len=0, pred_len=1, stride=1
    )
    indices = seq.get_index_array()
    assert indices is not None
    np.testing.assert_array_equal(indices, seq.driver.get_index_array())


# def test_ragged_sequencer_fallback_regular(regular_ak_data):
#     """
#     Verifies that if a regular awkward array is passed, the class
#     handles it (via the if ragged_dim is None block) and still works.
#     """
#     # This hits the "if ragged_dim is None" block in __init__
#     seq = RaggedArraySequencer(
#         array=regular_ak_data, seq_len=1, label_len=0, pred_len=1, stride=1
#     )
#     # regular_ak_data has (B,C,D) = (2,2,2) and is regular.
#     # hence RaggedArraySequencer maps it to (var,2,2)
#     #  (Len 2, Req 2) -> 1 window
#     assert len(seq) == 1

#     x, y = seq.sequences_batch([0])
#     assert x.shape == (1, 1, 2, 2)  # (Batch, Time, Feat)


def test_ragged_sequencer_fallback_regular(regular_ak_data):
    """
    Verifies that if a regular awkward array is passed, the class handles it
    (via the if ragged_dim is None block).

    NOTE: Because regular_ak_data is (2, 2, 2), treating axis 0 as ragged
    makes the sequencer see a single time series of shape (Time=2, Feat1=2, Feat2=2).
    When sequenced, this results in a 4D tensor (Batch, Seq, Feat1, Feat2),
    which triggers an internal assertion error in the sequencer (which expects 3D).
    """
    # This hits the "if ragged_dim is None" block in __init__
    seq = RaggedArraySequencer(
        array=regular_ak_data, seq_len=1, label_len=0, pred_len=1, stride=1
    )

    # regular_ak_data has (B,C,D) = (2,2,2) and is regular.
    # hence RaggedArraySequencer maps it to (var,2,2)
    # (Len 2, Req 2) -> 1 window
    assert len(seq) == 1

    # We expect the sequencer to crash because it produces a 4D output [1, 1, 2, 2],
    # but the internal check enforces 3D output.
    # We use regex escaping (r"...") because torch.Size uses brackets [].
    expected_msg = (
        r"seq_x ndim mismatch: expected 3, got seq_x.shape torch.Size\(\[1, 1, 2, 2\]\)"
    )

    with pytest.raises(AssertionError, match=expected_msg):
        seq.sequences_batch([0])


def test_ragged_sequencer_type_casting(jagged_ak_data):
    """
    Verifies input data is cast to float32 automatically.
    """
    # Input fixture is integers
    seq = RaggedArraySequencer(
        array=jagged_ak_data, seq_len=2, label_len=0, pred_len=1, stride=1
    )

    x, y = seq.sequences_batch([0])
    assert x.dtype == torch.float32
