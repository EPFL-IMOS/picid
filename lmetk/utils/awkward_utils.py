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
