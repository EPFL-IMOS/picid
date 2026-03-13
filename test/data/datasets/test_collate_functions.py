"""
Phase 1: Collation logic tests.

Tests custom collate_fn implementations with varying batch sizes.
Asserts padding, stacking, and dictionary merging for tensor-only and
tuple-value samples. Real data (synthetic tensors), no mocks.
"""

from __future__ import annotations

import pytest
import torch

from picid.data.datasets.collate_functions import (
    collate_key_value_batch,
    collate_identity,
)


# ---------------------------------------------------------------------------
# collate_key_value_batch: tensor-only batches (primary use case)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_collate_key_value_batch_tensor_only_shape_and_dtype(batch_size):
    """Batch of dicts with only tensor values: cat along dim 0 gives (B, ...) when each sample is (1, T, F)."""
    seq_len, n_features = 10, 5
    # Each "sample" is (1, seq_len, n_features) as returned by dataset __getitem__([i])
    batch = [
        {
            "features": torch.randn(1, seq_len, n_features, dtype=torch.float32),
            "target": torch.randn(1, seq_len, 1, dtype=torch.float32),
        }
        for _ in range(batch_size)
    ]
    out = collate_key_value_batch(batch)
    assert "features" in out and "target" in out
    assert out["features"].shape == (batch_size, seq_len, n_features)
    assert out["target"].shape == (batch_size, seq_len, 1)
    assert out["features"].dtype == torch.float32
    assert out["target"].dtype == torch.float32


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_collate_key_value_batch_value_integrity(batch_size):
    """Concatenation order: first sample's data is in out[0], etc."""
    batch = [
        {"x": torch.tensor([[float(i)]], dtype=torch.float32)}
        for i in range(batch_size)
    ]
    out = collate_key_value_batch(batch)
    assert out["x"].shape == (batch_size, 1)
    for i in range(batch_size):
        assert out["x"][i].item() == pytest.approx(float(i))


def test_collate_key_value_batch_heterogeneous_keys():
    """Different samples can have same keys; all are merged (cat along dim 0)."""
    batch = [
        {"a": torch.tensor([[1.0]]), "b": torch.tensor([[2.0]])},
        {"a": torch.tensor([[3.0]]), "b": torch.tensor([[4.0]])},
    ]
    out = collate_key_value_batch(batch)
    assert out["a"].shape == (2, 1)
    assert out["b"].shape == (2, 1)
    assert out["a"][0].item() == pytest.approx(1.0)
    assert out["a"][1].item() == pytest.approx(3.0)


def test_collate_key_value_batch_unsupported_type_raises():
    """Non-tensor, non-tuple values raise ValueError."""
    batch = [{"x": [1, 2, 3]}]
    with pytest.raises(ValueError, match="Unsupported value type"):
        collate_key_value_batch(batch)


# ---------------------------------------------------------------------------
# collate_identity: batch is list of "full batch" returns
# ---------------------------------------------------------------------------


def test_collate_identity_returns_first_element():
    """When using BatchSampler, batch is list of one full-batch item; identity returns it."""
    single_batch = {"features": torch.randn(4, 10, 5), "target": torch.randn(4, 10, 1)}
    batch_list = [single_batch]
    out = collate_identity(batch_list)
    assert out is single_batch
    assert out["features"].shape == (4, 10, 5)


# ---------------------------------------------------------------------------
# Real-world assertion: label index vs source
# ---------------------------------------------------------------------------


def test_collate_preserves_batch_dimension_for_downstream():
    """Collated batch has batch dim first (each sample (1, T, F) -> cat -> (B, T, F))."""
    batch_size = 5
    batch = [
        {
            "features": torch.randn(1, 10, 8),
            "label": torch.tensor([[i]], dtype=torch.long),
        }
        for i in range(batch_size)
    ]
    out = collate_key_value_batch(batch)
    assert out["features"].dim() == 3 and out["features"].size(0) == batch_size
    assert out["label"].shape == (batch_size, 1)
    for i in range(batch_size):
        assert out["label"][i].item() == i


def test_collate_key_value_batch_single_sample():
    """Single-sample batch returns dict with same keys, tensors unchanged in shape (1, ...)."""
    batch = [{"a": torch.tensor([[1.0, 2.0]]), "b": torch.tensor([[3.0]])}]
    out = collate_key_value_batch(batch)
    assert out["a"].shape == (1, 2)
    assert out["b"].shape == (1, 1)
    assert out["a"][0, 0].item() == pytest.approx(1.0)


def test_collate_identity_with_multiple_elements_returns_first():
    """collate_identity with list of 3 batch dicts returns the first."""
    a = {"x": torch.tensor([1.0])}
    b = {"x": torch.tensor([2.0])}
    c = {"x": torch.tensor([3.0])}
    out = collate_identity([a, b, c])
    assert out is a
    assert out["x"].item() == pytest.approx(1.0)


def test_collate_key_value_batch_many_keys():
    """Batch with many keys; all are present in output and stacked correctly."""
    batch = [{f"k{i}": torch.tensor([[float(i)]]) for i in range(5)} for _ in range(3)]
    out = collate_key_value_batch(batch)
    assert len(out) == 5
    for i in range(5):
        assert out[f"k{i}"].shape == (3, 1)


def test_collate_key_value_batch_mixed_tensor_and_tuple_keys():
    """Batch with both tensor and tuple values: tensor keys are collated; tuple path is executed."""
    batch = [
        {
            "a": torch.tensor([[1.0]]),
            "b": (torch.tensor([[0.0]]), torch.tensor([[1.0]])),
        },
        {
            "a": torch.tensor([[2.0]]),
            "b": (torch.tensor([[0.0]]), torch.tensor([[2.0]])),
        },
    ]
    out = collate_key_value_batch(batch)
    assert "a" in out
    assert out["a"].shape == (2, 1)
    assert out["a"][0].item() == pytest.approx(1.0)
    assert out["a"][1].item() == pytest.approx(2.0)
