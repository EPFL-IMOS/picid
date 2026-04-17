import random
import numpy as np


def dirichlet_block_lengths(n_subset, k, rng, alpha=20.0):
    ws = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    s = sum(ws)
    raw = [w / s * n_subset for w in ws]

    lengths = [max(1, int(round(x))) for x in raw]
    diff = n_subset - sum(lengths)

    order = sorted(
        range(k),
        key=lambda i: raw[i] - lengths[i],
        reverse=(diff > 0),
    )
    for i in order[: abs(diff)]:
        lengths[i] += 1 if diff > 0 else -1

    return lengths


def geometric_block_lengths(n_subset, k, rng, p=0.5):
    raw = []
    for _ in range(k):
        x = 1
        while rng.random() > p:
            x += 1
        raw.append(x)

    s = sum(raw)
    scaled = [x / s * n_subset for x in raw]
    lengths = [max(1, int(round(x))) for x in scaled]

    diff = n_subset - sum(lengths)
    order = sorted(
        range(k),
        key=lambda i: scaled[i] - lengths[i],
        reverse=(diff > 0),
    )
    for i in order[: abs(diff)]:
        lengths[i] += 1 if diff > 0 else -1

    return lengths


def create_sequence_subset(sequence_length, subset_ratio, subset_seed):
    """Functionality to test reduced data scenarios."""
    rng = random.Random(subset_seed)  # independent RNG

    # Floor to 1 so very small segments always contribute at
    # least one sequence (avoids empty float64 index arrays).
    n_subset = max(1, int(sequence_length * subset_ratio))

    seq_idx = rng.sample(range(sequence_length), n_subset)
    seq_idx.sort()
    return np.array(seq_idx)


def make_subset_blocks_indices(
    *,
    len_ds: int,
    subset_ratio: float,
    subset_seed: int,
    subset_blocks: int,
    block_length_sampler=dirichlet_block_lengths,
    sampler_kwargs={"alpha": 30.0},
):
    rng = random.Random(subset_seed)
    sampler_kwargs = sampler_kwargs or {}

    if not (0 < subset_ratio < 1):
        raise ValueError("subset_ratio must be in (0,1)")

    n_subset = int(len_ds * subset_ratio)
    if n_subset <= 0:
        raise ValueError("subset_ratio produced n_subset <= 0")

    k = int(subset_blocks)
    if k <= 0:
        raise ValueError("subset_blocks must be positive")
    if k > n_subset:
        raise ValueError("subset_blocks cannot exceed n_subset")

    free = len_ds - n_subset
    if free < 0:
        raise ValueError("n_subset exceeds dataset length")

    # 1) sample block lengths
    block_lengths = block_length_sampler(
        n_subset=n_subset,
        k=k,
        rng=rng,
        **sampler_kwargs,
    )

    if len(block_lengths) != k or sum(block_lengths) != n_subset:
        raise ValueError("block_length_sampler returned invalid block lengths")

    # 2) sample gaps deterministically
    bars = sorted(rng.sample(range(free + k), k))
    b = [-1] + bars + [free + k]
    gaps = [b[i + 1] - b[i] - 1 for i in range(k + 1)]

    # 3) layout blocks
    intervals = []
    pos = gaps[0]
    for i, L in enumerate(block_lengths):
        lo, hi = pos, pos + L - 1
        intervals.append((lo, hi))
        pos = hi + 1 + gaps[i + 1]

    seq_idx = np.concatenate([np.arange(lo, hi + 1) for lo, hi in intervals])

    return {
        "seq_idx": seq_idx,
        "intervals": intervals,
        "block_lengths": block_lengths,
        "gaps": gaps,
        "n_subset": n_subset,
    }


def _intervals_from_indices(idx: np.ndarray):
    idx = np.asarray(idx)
    if idx.size == 0:
        return []
    d = np.diff(idx)
    cuts = np.where(d != 1)[0]
    starts = [idx[0]] + [idx[c + 1] for c in cuts]
    ends = [idx[c] for c in cuts] + [idx[-1]]
    return list(zip(starts, ends))
