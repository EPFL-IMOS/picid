import random
import numpy as np
import pytest

from picid.data.datasets.subset_sampling import (
    _intervals_from_indices,
    dirichlet_block_lengths,
    geometric_block_lengths,
    make_subset_blocks_indices,
)


def test_subset_blocks_deterministic_no_overlap_and_total_length():
    len_ds = 1000
    subset_ratio = 0.23
    subset_blocks = 7
    seed = 12345

    res = make_subset_blocks_indices(
        len_ds=len_ds,
        subset_ratio=subset_ratio,
        subset_seed=seed,
        subset_blocks=subset_blocks,
    )
    n_subset = res["n_subset"]
    block_lengths = res["block_lengths"]
    gaps = res["gaps"]
    intervals = res["intervals"]
    idx = res["seq_idx"]

    # determinism
    res2 = make_subset_blocks_indices(
        len_ds=len_ds,
        subset_ratio=subset_ratio,
        subset_seed=seed,
        subset_blocks=subset_blocks,
    )
    assert np.array_equal(idx, res2["seq_idx"])
    assert intervals == res2["intervals"]

    # total length is n_subset
    assert idx.size == n_subset
    assert sum(block_lengths) == n_subset

    # indices are sorted, unique, in range
    assert np.all(idx[:-1] <= idx[1:])
    assert idx.min() >= 0
    assert idx.max() < len_ds
    assert np.unique(idx).size == idx.size

    # exactly subset_blocks contiguous blocks
    recovered = _intervals_from_indices(idx)
    assert len(recovered) == subset_blocks

    # contiguity inside blocks + non-overlap between blocks
    for (lo, hi), L in zip(recovered, block_lengths):
        assert hi - lo + 1 == L
    for (a_lo, a_hi), (b_lo, b_hi) in zip(recovered, recovered[1:]):
        assert a_hi < b_lo  # strictly separated

    # gaps sum to free
    assert sum(gaps) == len_ds - n_subset
    assert all(g >= 0 for g in gaps)


def test_subset_blocks_edge_case_single_block():
    len_ds = 100
    subset_ratio = 0.15
    subset_blocks = 1
    seed = 7

    res = make_subset_blocks_indices(
        len_ds=len_ds,
        subset_ratio=subset_ratio,
        subset_seed=seed,
        subset_blocks=subset_blocks,
    )
    n_subset = res["n_subset"]
    block_lengths = res["block_lengths"]
    intervals = res["intervals"]
    idx = res["seq_idx"]

    assert idx.size == n_subset
    assert block_lengths == [n_subset]
    assert len(intervals) == 1
    lo, hi = intervals[0]
    assert np.array_equal(idx, np.arange(lo, hi + 1))


def test_subset_blocks_raises_when_more_blocks_than_points():
    len_ds = 50
    subset_ratio = 0.1  # int(50*0.1) == 5
    subset_blocks = 6
    seed = 0

    try:
        make_subset_blocks_indices(
            len_ds=len_ds,
            subset_ratio=subset_ratio,
            subset_seed=seed,
            subset_blocks=subset_blocks,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_subset_blocks_raises_when_len_ds_smaller_than_blocks():
    len_ds = 5
    subset_ratio = 0.8  # n_subset == 4
    subset_blocks = 7  # blocks > len_ds and > n_subset
    seed = 1

    with pytest.raises(ValueError):
        make_subset_blocks_indices(
            len_ds=len_ds,
            subset_ratio=subset_ratio,
            subset_seed=seed,
            subset_blocks=subset_blocks,
        )


def test_subset_blocks_exactly_one_block_per_point():
    len_ds = 6
    subset_ratio = 0.99  # n_subset == 5 with current rounding
    subset_blocks = len_ds - 1  # 5
    seed = 99

    res = make_subset_blocks_indices(
        len_ds=len_ds,
        subset_ratio=subset_ratio,
        subset_seed=seed,
        subset_blocks=subset_blocks,
    )

    idx = res["seq_idx"]
    intervals = res["intervals"]
    gaps = res["gaps"]
    block_lengths = res["block_lengths"]

    assert len(intervals) >= 1
    assert all(hi >= lo for lo, hi in intervals)

    # sizes and counts
    assert res["n_subset"] == len_ds - 1  # 5 points chosen
    assert len(block_lengths) == subset_blocks
    assert sum(block_lengths) == res["n_subset"]
    assert idx.size == res["n_subset"]

    # structure checks (allow contiguous blocks to merge when gaps are zero)
    recovered = _intervals_from_indices(idx)
    assert 1 <= len(recovered) <= subset_blocks
    assert np.unique(idx).size == idx.size
    assert idx.min() >= 0 and idx.max() < len_ds
    for lo, hi in recovered:
        assert hi >= lo
    assert sum(hi - lo + 1 for lo, hi in recovered) == res["n_subset"]

    # exactly one position of the dataset should be excluded
    missing = set(range(len_ds)) - set(idx.tolist())
    assert len(missing) == 1

    # gaps bookkeeping
    assert len(gaps) == subset_blocks + 1
    assert sum(gaps) == len_ds - res["n_subset"]  # 1 missing point
    assert all(g >= 0 for g in gaps)


def test_dirichlet_block_lengths_reproducible_and_valid():
    n_subset = 50
    k = 6
    seed = 77
    rng = random.Random(seed)
    lengths = dirichlet_block_lengths(n_subset=n_subset, k=k, rng=rng, alpha=30.0)

    assert len(lengths) == k
    assert sum(lengths) == n_subset
    assert all(L >= 1 for L in lengths)

    # determinism with same seed
    rng2 = random.Random(seed)
    lengths2 = dirichlet_block_lengths(n_subset=n_subset, k=k, rng=rng2, alpha=30.0)
    assert lengths == lengths2


def test_geometric_block_lengths_reproducible_and_valid():
    n_subset = 80
    k = 8
    seed = 123
    rng = random.Random(seed)
    lengths = geometric_block_lengths(n_subset=n_subset, k=k, rng=rng, p=0.45)

    assert len(lengths) == k
    assert sum(lengths) == n_subset
    assert all(L >= 1 for L in lengths)

    rng2 = random.Random(seed)
    lengths2 = geometric_block_lengths(n_subset=n_subset, k=k, rng=rng2, p=0.45)
    assert lengths == lengths2
