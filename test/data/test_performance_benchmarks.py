"""
Phase 3: Bottleneck detection — __getitem__ timing.

Basic timing benchmarks to ensure data loading is not excessively slow.
Marked ``benchmark`` and ``slow``; default focused runs should exclude them
(see ``pyproject.toml`` ``[tool.pytest.ini_options]``).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from picid.data.datasets.sliding_window_batch_dataset import SlidingWindowBatchDataset
from test.fixtures.builders import make_standard_normal_2d

_BENCHMARK_DATA_SEED = 90210


@pytest.fixture
def medium_sliding_dataset():
    """Dataset with ~1000 windows for timing (deterministic RNG)."""
    T, F = 5000, 16
    data = {
        "f": make_standard_normal_2d(
            seed=_BENCHMARK_DATA_SEED, n_rows=T, n_cols=F
        ).astype(np.float32)
    }
    return SlidingWindowBatchDataset(
        data_dict=data,
        seq_len=50,
        label_len=0,
        pred_len=10,
        stride=5,
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_getitem_single_index_latency(medium_sliding_dataset):
    """Single __getitem__([i]) completes in reasonable time (< 100ms per call)."""
    ds = medium_sliding_dataset
    n_calls = 20
    start = time.perf_counter()
    for i in range(n_calls):
        _ = ds[[i % len(ds)]]
    elapsed = time.perf_counter() - start
    per_call = elapsed / n_calls
    assert per_call < 0.1, f"__getitem__ too slow: {per_call*1000:.1f} ms per call"


@pytest.mark.benchmark
@pytest.mark.slow
def test_getitem_batch_latency(medium_sliding_dataset):
    """Batch __getitem__([0:32]) completes in reasonable time."""
    ds = medium_sliding_dataset
    batch_size = 32
    indices = list(range(min(batch_size, len(ds))))
    start = time.perf_counter()
    for _ in range(10):
        _ = ds[indices]
    elapsed = time.perf_counter() - start
    per_batch = elapsed / 10
    assert (
        per_batch < 1.0
    ), f"Batch __getitem__ too slow: {per_batch*1000:.0f} ms per batch"
