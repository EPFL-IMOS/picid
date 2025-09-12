"""Utils for awkward arrays."""

import awkward as ak


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
