"""Utils for awkward arrays."""

import awkward as ak
import numpy as np


def ak_unflatten_discontinous_groups(
    values: ak.Array, groups_index: ak.Array
) -> ak.Array:
    """
    Reshape `values` into a nested array based on `groups`,
    where `groups` may be discontinous.
    """
    assert len(values) == len(
        groups_index
    ), "Values and groups must have the same length"
    idx = ak.argsort(groups_index, stable=True)
    g_sorted = groups_index[idx]
    v_sorted = values[idx]

    counts = ak.run_lengths(g_sorted)
    batched = ak.unflatten(v_sorted, counts)
    return batched


def ak_find_var_dim(arr):
    """Find variable-length dimensions in an Awkward Array."""
    assert isinstance(arr, ak.Array)

    t = ak.type(arr)
    axes, axis = [], 0
    while hasattr(t, "content"):
        # unwrap OptionType without advancing axis
        while isinstance(t, ak.types.OptionType):
            t = t.content
        if isinstance(t, ak.types.ListType):
            axes.append(axis)
        # RegularType is fixed-size, do nothing
        t = t.content
        axis += 1
    return axes


def ragged_index_tuples(arr, ragged_dims) -> list[tuple]:
    """
    Return index tuples for the first `ragged_dims` axes of `arr`.
    Each tuple corresponds to one selection into the regular tail.
    """
    # get local indices at each ragged axis
    indices = [ak.local_index(arr, axis=ax) for ax in range(ragged_dims)]
    multi_index = ak.zip(indices)
    for i in range(ragged_dims - 1):
        multi_index = ak.flatten(multi_index)
    return ak.to_list(multi_index)


def iterate_ragged_sub_blocks(arr: ak.Array, ragged_dims: int):
    """Iterate over sub-blocks of a ragged array."""
    idx_tuples = ragged_index_tuples(arr, ragged_dims)
    for idx in idx_tuples:
        yield idx, arr[idx]


def blocks_to_ragged_array(coords: list[tuple], blocks: np.ndarray) -> ak.Array:
    # build nested dict
    root = {}
    for coord, block in zip(coords, blocks):
        d = root
        for i in coord[:-1]:
            d = d.setdefault(i, {})
        d[coord[-1]] = block  # keep as np.ndarray

    # recursively turn dicts into lists
    def tolist(d):
        if isinstance(d, dict):
            return [tolist(d[i]) for i in sorted(d)]
        return d  # already a numpy array

    nested = tolist(root)
    return ak.Array(nested)
