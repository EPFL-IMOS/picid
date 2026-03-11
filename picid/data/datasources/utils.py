from typing import Any


def convert_outer_list_to_inner(
    data_list: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """
    Transposes a list of dictionaries into a dictionary of lists.

    Parameters
    ----------
    data_list : list[dict[str, Any]]
        A list of dictionaries where every dictionary
        has the same keys.

    Returns
    -------
    dict[str, list[Any]]
        A dictionary where keys match the input dicts, and values
        are lists of the corresponding values from the input.
    """
    if not data_list:
        return {}

    # Ensure all dictionaries have the same keys.
    # We convert keys to a set for robust comparison (order-independent).
    first_keys = set(data_list[0].keys())

    for d in data_list:
        if set(d.keys()) != first_keys:
            raise ValueError("Not all dicts have the same keys!")

    # Concatenate values for each key using the keys from the first element
    # to maintain deterministic order (if relying on insertion order).
    # Since we validated all sets are equal, using data_list[0] keys is safe.
    keys = list(data_list[0].keys())

    stacked = {key: [d[key] for d in data_list] for key in keys}

    return stacked
