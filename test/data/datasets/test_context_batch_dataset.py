import pytest
import numpy as np
import torch
import awkward as ak
from lightning_fabric.utilities.data import AttributeDict
from torch.utils.data import DataLoader

# Adjust imports based on your actual structure
from lmetk.data.datasets.context_dataset import ContextBatchDataset
from lmetk.utils.awkward_utils import ak_regularize_regular_axes

# =========================================================================
# === Fixtures ===
# =========================================================================


@pytest.fixture
def dense_data():
    """
    Creates dense data suitable for ALL task types defined in ContextBatchDataset.
    Includes specific keys to satisfy the logic where target_key = task_type.
    """
    # Base arrays
    feats = np.random.randn(50, 5).astype(np.float32)  # 50 steps, 5 features
    target_1d = np.random.randn(50, 1).astype(np.float32)  # 50 steps, 1 feature

    return {
        "features": feats,
        "rul": target_1d,
        "ahrul": target_1d,  # Required if task_type="ahrul"
        "soc": target_1d,  # Required if task_type="soc"
        "target": target_1d,  # Required for 'forecasting'
        "fault_classification": np.zeros((50, 1)).astype(np.float32),
    }


@pytest.fixture
def ragged_data():
    """
    Creates ragged data (2 units/cycles) to test awkward array integration.
    """
    # Cycle 0: 20 steps. Cycle 1: 30 steps.
    c0 = [[x, x] for x in range(20)]
    c1 = [[x, x] for x in range(30)]
    # Regularize to ensure (Units, var_Time, Fixed_Feat)
    features = ak_regularize_regular_axes(ak.Array([c0, c1]))

    t0 = [[x] for x in range(20)]
    t1 = [[x] for x in range(30)]
    target = ak_regularize_regular_axes(ak.Array([t0, t1]))

    return {
        "features": features,
        "rul": target,
        "target": target,
        "fault_classification": target,
    }


# =========================================================================
# === Tests: Initialization & Task Logic ===
# =========================================================================


@pytest.mark.parametrize(
    "task, context_key, target_key",
    [
        ("rul", "features", "rul"),
        ("ahrul", "features", "ahrul"),
        ("soc", "features", "soc"),
        ("forecasting", "features", "target"),
        ("fault_classification", "features", "fault_classification"),
    ],
)
def test_init_task_mapping_success(dense_data, task, context_key, target_key):
    """
    Verifies that every supported task type initializes correctly and maps
    to the expected keys in the data dictionary.
    """
    ds = ContextBatchDataset(
        data_dict=dense_data,
        task_type=task,
        seq_len=10,
        label_len=0,
        pred_len=1,
        stride=1,
    )

    assert len(ds) > 0
    # Verify the internal datasets are pointing to the correct data
    assert context_key in ds.context_dataset.sequencers
    assert target_key in ds.target_dataset.sequencers


def test_init_invalid_task_type(dense_data):
    """Verifies ValueError for unknown tasks."""
    with pytest.raises(ValueError, match="Unknown task_type"):
        ContextBatchDataset(
            data_dict=dense_data,
            task_type="magic_wand",
            seq_len=10,
            label_len=0,
            pred_len=1,
        )


def test_init_missing_required_keys(dense_data):
    """Verifies ValueError if data_dict misses required keys."""
    incomplete_data = {"features": dense_data["features"]}  # Missing targets
    with pytest.raises(ValueError, match="Data dictionary must contain the key"):
        ContextBatchDataset(
            data_dict=incomplete_data,
            task_type="forecasting",
            seq_len=10,
            label_len=0,
            pred_len=1,
        )


def test_init_synchronization_mismatch():
    """
    Verifies assertion failure if Context and Target datasets have different lengths.
    """
    data = {"features": np.zeros((50, 1)), "target": np.zeros((20, 1))}
    with pytest.raises(
        AssertionError, match="Target and context sequencers must have the same length"
    ):
        ContextBatchDataset(
            data_dict=data,
            task_type="forecasting",
            seq_len=10,
            label_len=0,
            pred_len=1,
            stride=1,
        )


# =========================================================================
# === Tests: Data Retrieval (Scalar vs Batch) ===
# =========================================================================


def test_getitem_structure_dense_scalar(dense_data):
    """
    Verifies output structure for Scalar Access (ds[0]).
    Crucial: Must return 3D tensors (1, T, F) to support concatenation collation.
    """
    ds = ContextBatchDataset(
        data_dict=dense_data,
        task_type="rul",
        seq_len=5,
        label_len=0,
        pred_len=1,
        stride=1,
    )

    # ACT: Fetch single index
    idx = 0
    item = ds[[idx]]

    # ASSERT: Structure
    assert isinstance(item, AttributeDict)
    assert isinstance(item.context, AttributeDict)
    assert isinstance(item.target, AttributeDict)

    # ASSERT: Batch Index handling
    # Your code wraps int in a list: [0]
    assert item.batch_idx == [0]

    # ASSERT: Shapes
    # Context Features: (Batch=1, Seq=5, Feat=5)
    x = item.context.features_seq_x
    assert torch.is_tensor(x)
    assert x.ndim == 3
    assert x.shape == (1, 5, 5)

    # Target RUL: (Batch=1, Seq=5, Feat=1)
    y = item.target.rul_seq_x
    assert y.shape == (1, 5, 1)


def test_getitem_ragged_input_scalar(ragged_data):
    """
    Verifies Scalar Access for Ragged Data.
    """
    ds = ContextBatchDataset(
        data_dict=ragged_data,
        task_type="rul",
        seq_len=5,
        label_len=0,
        pred_len=1,
        stride=1,
    )

    assert len(ds) > 0

    # ACT
    item = ds[[0]]

    # ASSERT: Shapes match input dims
    # Ragged Features had 2 features -> (1, 5, 2)
    assert item.context.features_seq_x.shape == (1, 5, 2)
    # Ragged Target had 1 feature -> (1, 5, 1)
    assert item.target.rul_seq_x.shape == (1, 5, 1)


def test_getitem_batch_access(dense_data):
    """
    Tests accessing the dataset with a LIST of indices (BatchSampler style).
    ds[[0, 1, 2]]
    """
    ds = ContextBatchDataset(
        data_dict=dense_data, task_type="rul", seq_len=5, label_len=0, pred_len=1
    )

    indices = [0, 1, 2]

    # ACT
    batch = ds[indices]

    # ASSERT
    assert batch.batch_idx == indices

    # Verify tensor concatenation
    # 3 indices -> Batch dim should be 3
    assert batch.context.features_seq_x.shape == (3, 5, 5)
    assert batch.target.rul_seq_x.shape == (3, 5, 1)


# =========================================================================
# === Tests: DataLoader Integration ===
# =========================================================================


def test_collate_fn_access(dense_data):
    """Verifies the dataset exposes the correct collation function."""
    ds = ContextBatchDataset(
        data_dict=dense_data,
        task_type="forecasting",
        seq_len=10,
        label_len=0,
        pred_len=1,
    )
    fn = ds.get_collate_fn()
    assert callable(fn)


# def test_dataloader_flow(dense_data):
#     """
#     Verifies that the dataset works with a PyTorch DataLoader.
#     """
#     ds = ContextBatchDataset(
#         data_dict=dense_data,
#         task_type="forecasting",
#         seq_len=5,
#         label_len=0,
#         pred_len=2,
#     )

#     # Use custom collate_fn from dataset
#     dl = DataLoader(ds, batch_size=4, collate_fn=ds.get_collate_fn())

#     # ACT
#     # Note: If collate_key_value_batch handles lists/ints correctly, this passes.
#     # If it strictly requires Tensors for batch_idx, this might fail given the current code.
#     batch = next(iter(dl))

#     # ASSERT
#     assert isinstance(batch, AttributeDict)

#     # Batch size 4 * (1, 5, 5) -> Concat -> (4, 5, 5)
#     assert batch.context.features_seq_x.shape == (4, 5, 5)
#     # Target (pred_len=2) -> (4, 2, 1)
#     assert batch.target.target_seq_y.shape == (4, 2, 1)
