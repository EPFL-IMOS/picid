"""Utilities for working with Awkward Arrays."""

import awkward as ak
import numpy as np
from typing import Iterable, List, Optional, Union


def ak_unflatten_discontinous_groups(
    values: ak.Array, groups_index: ak.Array
) -> ak.Array:
    """
    Reshape values into a nested array based on group labels.

    Parameters
    ----------
    values : ak.Array
        Flat Awkward Array containing the values to regroup.
    groups_index : ak.Array
        Group labels aligned with ``values``.

    Returns
    -------
    ak.Array
        Nested Awkward Array grouped by the sorted group labels.
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


def ak_find_var_dims(arr: ak.Array) -> List[int]:
    """
    Find variable-length dimensions in an Awkward Array.

    In ``4 * var * var * 3`` this returns ``[1, 2]``.

    Parameters
    ----------
    arr : ak.Array
        Input Awkward Array.

    Returns
    -------
    List[int]
        Indices of the variable-length dimensions.
    """
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


def find_singular_ragged_dim(
    arrays: Union[ak.Array, Iterable[ak.Array]],
) -> Optional[int]:
    """
    Find the single variable-length (ragged) dimension across one or more arrays.

    Returns None if all arrays are regular; returns the unique ragged axis index
    if exactly one; raises ValueError if more than one variable-length dimension
    is present across the arrays.

    Parameters
    ----------
    arrays : Union[ak.Array, Iterable[ak.Array]]
        One array or a collection of arrays to inspect.

    Returns
    -------
    Optional[int]
        The unique ragged axis, or ``None`` if the structure is fully regular.
    """
    if isinstance(arrays, ak.Array):
        arrays = [arrays]
    var_dims = [ak_find_var_dims(arr) for arr in arrays]
    unique_var_dims = set(x for row in var_dims for x in row)
    if len(unique_var_dims) == 0:
        # No ragged dimensions found; the data is regular in structure.
        return None
    elif len(unique_var_dims) == 1:
        # Exactly one ragged dimension found, as expected.
        return unique_var_dims.pop()
    else:
        raise ValueError(
            f"Expected at most one variable-length dimension, found {len(unique_var_dims)}: {unique_var_dims}"
        )


def ragged_index_tuples(arr: ak.Array, ragged_dims: int) -> list[tuple]:
    """
    Return index tuples for the leading ragged axes of an array.

    Parameters
    ----------
    arr : ak.Array
        Input Awkward Array.
    ragged_dims : int
        Number of leading ragged dimensions to index.

    Returns
    -------
    list[tuple]
        Tuple indices for the ragged axes.
    """
    # get local indices at each ragged axis
    indices = [ak.local_index(arr, axis=ax) for ax in range(ragged_dims)]
    multi_index = ak.zip(indices)
    for i in range(ragged_dims - 1):
        multi_index = ak.flatten(multi_index)
    return ak.to_list(multi_index)


def iterate_ragged_sub_blocks(arr: ak.Array, ragged_dims: int):
    """
    Iterate over sub-blocks of a ragged array.

    Parameters
    ----------
    arr : ak.Array
        Input Awkward Array.
    ragged_dims : int
        Number of ragged dimensions to iterate over.

    Yields
    ------
    tuple
        A pair of index tuple and sub-array view.
    """
    idx_tuples = ragged_index_tuples(arr, ragged_dims)
    for idx in idx_tuples:
        yield idx, arr[idx]


def blocks_to_ragged_array(coords: list[tuple], blocks: np.ndarray) -> ak.Array:
    """
    Construct a ragged Awkward Array from blocks of regular arrays.

    Parameters
    ----------
    coords : list[tuple]
        Coordinate tuples for each block.
    blocks : np.ndarray
        Regular block values to assemble.

    Returns
    -------
    ak.Array
        Ragged Awkward Array built from the supplied blocks.
    """

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


def ak_regularize_regular_axes(arr: ak.Array) -> ak.Array:
    """
    Convert all regular axes in an Awkward Array to regular.
    This is useful after constructing an Awkward Array from nested lists/dicts,
    where some axes may be irregular even if they are actually regular.
    Note:
        - Assume that innermost dimensions are regular (usually feature dim)

        Because:
        var dim
            can be rugged or not rugged
        int dim
            is always not rugged

    """
    # Dims that are currently flagged as variable, but may are not.
    # TODO: check if last dim is in fact not rugged
    if arr.ndim > 2:
        arr = ak.to_regular(arr, axis=-1)

    var_dims = ak_find_var_dims(arr)
    for d in var_dims:
        lengths = ak.ravel(ak.num(arr, axis=d))
        if ak.all(lengths == lengths[0]):
            arr = ak.to_regular(arr, axis=d)
    return arr


def get_ak_shape(arr: ak.Array) -> List[Union[int, str]]:
    """
    Get the shape of an Awkward Array, representing ragged dimensions as ``"var"``.

    Parameters
    ----------
    arr : ak.Array
        Input Awkward Array.

    Returns
    -------
    List[Union[int, str]]
        Shape description with ragged axes marked as ``"var"``.
    """
    t = ak.type(arr)
    shape = []

    # FIX: Handle the top-level ArrayType, which gives the first dimension's length.
    if isinstance(t, ak.types.ArrayType):
        shape.append(t.length)
        # Move to the inner type before starting the loop
        t = t.content

    while hasattr(t, "content"):
        # Handle optional types (missing values) by unwrapping them
        while isinstance(t, ak.types.OptionType):
            t = t.content

        if isinstance(t, ak.types.RegularType):
            shape.append(t.size)
        elif isinstance(t, ak.types.ListType):
            shape.append("var")

        # Move to the next inner dimension
        t = t.content

    if hasattr(t, "shape") and len(t.shape) > 0:
        shape.extend(t.shape)

    return shape


def ak_flatten_variable_dims(
    arr: ak.Array, axis: Optional[Union[int, List[int]]] = None
) -> ak.Array:
    """
    Flattens specified or all variable dimensions of an Awkward Array.

    This function is designed to handle the (N, var, F) -> (N*var_i, F)
    transformation you described.

    Parameters
    ----------
    arr
        The input Awkward Array.
    axis
        The dimension(s) to flatten.
        - If None (default): Automatically finds all variable (ragged)
          dimensions using `ak_find_var_dims` and flattens them.
        - If int: Flattens the specified axis.
        - If List[int]: Flattens all specified axes.

    Returns
    -------
    A new Awkward Array with the specified dimensions flattened.
    """

    axes_to_flatten: List[int]

    if axis is None:
        # Automatic mode: find all variable dimensions
        axes_to_flatten = ak_find_var_dims(arr)
        if not axes_to_flatten:
            # No variable dims found, nothing to do
            return arr
    elif isinstance(axis, int):
        axes_to_flatten = [axis]
    else:  # It's a List[int]
        axes_to_flatten = axis

    # Sort axes in reverse order to flatten from the innermost
    # specified axis outward. This preserves the indices of
    # the outer axes during the loop.
    # e.g., (N, var, var, F) and flatten [1, 2]
    # 1. Flatten axis 2 -> (N, var, var*F) (axis 1 is still 1)
    # 2. Flatten axis 1 -> (N, var*var*F)
    axes_to_flatten = sorted(axes_to_flatten, reverse=True)

    output_arr = arr
    for ax in axes_to_flatten:
        try:
            output_arr = ak.flatten(output_arr, axis=ax)
        except ValueError as e:
            # Handle cases where axis is out of bounds
            raise ValueError(
                f"Failed to flatten axis {ax}. "
                f"Array shape: {ak.type(arr)}, Error: {e}"
            ) from e

    return output_arr
