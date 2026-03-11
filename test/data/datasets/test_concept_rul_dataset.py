"""
Tests for ConceptRULDataset with real synthetic data (attribute-accessible data_dict via Box).
Covers __len__, __getitem__ (beginning-of-sequence padding, mid-sequence, unit switch), get_collate_fn, init validation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from box import Box

from picid.data.datasets.concept_rul_dataset import ConceptRULDataset


def _make_concept_rul_data_dict(T: int, F: int, window_size: int, n_concepts: int = 2):
    """Build data_dict compatible with ConceptRULDataset (attribute access: .rul, .features, etc.)."""
    np.random.seed(42)
    data = {
        "features": np.random.randn(T, F).astype(np.float32),
        "rul": np.arange(T - 1, -1, -1, dtype=np.float32),
        "timestamps": np.arange(T, dtype=np.float64),
        "health_states": np.random.randint(0, 3, size=T, dtype=np.int32),
        "concepts": np.random.randn(T, n_concepts).astype(np.float32),
        "unit": np.repeat(np.arange(2), T // 2)[:T],
    }
    return Box(data)


@pytest.fixture
def concept_rul_data_dict():
    """Real synthetic data for ConceptRULDataset: T=50, F=4, window_size=10, stride=2."""
    return _make_concept_rul_data_dict(T=50, F=4, window_size=10, n_concepts=2)


@pytest.fixture
def concept_rul_dataset(concept_rul_data_dict):
    return ConceptRULDataset(
        data_dict=concept_rul_data_dict,
        window_size=10,
        stride=2,
    )


def test_concept_rul_len(concept_rul_dataset):
    """__len__ is X.shape[0] // stride."""
    assert len(concept_rul_dataset) == 50 // 2


def test_concept_rul_getitem_structure(concept_rul_dataset):
    """__getitem__ returns AttributeDict with features, rul, concepts, health_states."""
    out = concept_rul_dataset[5]
    assert (
        "features" in out
        and "rul" in out
        and "concepts" in out
        and "health_states" in out
    )
    assert torch.is_tensor(out["features"])
    assert torch.is_tensor(out["rul"])
    assert out["features"].ndim >= 1
    assert out["rul"].shape[-1] == 1 or out["rul"].numel() >= 1


def test_concept_rul_getitem_beginning_of_sequence(concept_rul_data_dict):
    """Beginning of sequence (i*stride < window_size-1) uses backward padding."""
    ds = ConceptRULDataset(data_dict=concept_rul_data_dict, window_size=10, stride=2)
    out = ds[0]
    assert out["features"].shape[0] == 10
    assert out["features"].shape[1] == 4


def test_concept_rul_getitem_mid_sequence_single_unit(concept_rul_data_dict):
    """Mid-sequence with single unit (no unit switch) uses contiguous slice."""
    data = _make_concept_rul_data_dict(T=50, F=4, window_size=10, n_concepts=2)
    data.unit = np.zeros(50, dtype=np.int32)
    ds = ConceptRULDataset(data_dict=data, window_size=10, stride=2)
    out = ds[10]
    assert out["features"].shape == (10, 4)


def test_concept_rul_getitem_rul_value_matches_index(concept_rul_dataset):
    """RUL value at index i equals df_Y[i*stride]."""
    i = 3
    out = concept_rul_dataset[i]
    expected_rul = concept_rul_dataset.df_Y[i * concept_rul_dataset.stride]
    assert out["rul"].item() == pytest.approx(float(expected_rul))


def test_concept_rul_get_collate_fn(concept_rul_dataset):
    """get_collate_fn returns default_collate."""
    from torch.utils.data._utils.collate import default_collate

    assert concept_rul_dataset.get_collate_fn() is default_collate


def test_concept_rul_init_requires_ndarray():
    """Init raises if any value in data_dict is not ndarray."""
    data = Box(
        {
            "features": np.zeros((20, 2)),
            "rul": np.zeros(20),
            "timestamps": np.zeros(20),
            "health_states": np.zeros(20),
            "concepts": np.zeros((20, 1)),
            "unit": np.zeros(20),
        }
    )
    data["rul"] = [1, 2, 3]
    with pytest.raises(AssertionError, match="must be numpy arrays"):
        ConceptRULDataset(data_dict=data, window_size=5, stride=1)


def test_concept_rul_getitem_unit_switch_branch(concept_rul_data_dict):
    """Mid-sequence with unit switch uses padding + contiguous slice (cond.size > 0)."""
    data = concept_rul_data_dict.copy()
    data.unit = np.concatenate([np.zeros(15), np.ones(35)])
    ds = ConceptRULDataset(data_dict=data, window_size=10, stride=2)
    out = ds[8]
    assert out["features"].shape == (10, 4)


def test_concept_rul_collate_integration(concept_rul_dataset):
    """Batch of indices collates to stacked tensors."""
    fn = concept_rul_dataset.get_collate_fn()
    batch = [concept_rul_dataset[i] for i in [0, 1, 2]]
    out = fn(batch)
    assert out["features"].shape[0] == 3
    assert out["rul"].shape[0] == 3
