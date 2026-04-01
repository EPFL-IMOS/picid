"""
Shared PHMD split assembly helpers.

The PHMD datasets all follow the same high-level pattern: a fold payload is
grouped by unit, each unit is converted into a nested record, and the result is
reassembled into the split-aware container expected by the datasource layer.
This module keeps that orchestration out of the loader classes themselves.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import awkward as ak
import numpy as np
import pandas as pd

from picid.data.datasources.utils import convert_outer_list_to_inner

SplitPayload = dict[str, dict[str, list[Any]]]


def _normalize_unit_column(df: pd.DataFrame, unit_column: str) -> pd.Series:
    """
    Return the unit column as a one-dimensional series.

    Parameters
    ----------
    df : pandas.DataFrame
        Split dataframe to inspect.
    unit_column : str
        Column that identifies the physical unit.

    Returns
    -------
    pandas.Series
        Normalized one-dimensional unit label series.
    """

    unit_col_data = df[unit_column]
    if isinstance(unit_col_data, pd.DataFrame):
        # Some PHMD frames expose the unit column as a single-column dataframe.
        # Squeezing it here keeps the downstream boolean mask consistent.
        unit_col_data = unit_col_data.squeeze("columns")
    return unit_col_data


def _apply_debug_subset(
    unit_payload: dict[str, Any],
    debug_subset_range: tuple[int, int, int] | None,
) -> dict[str, Any]:
    """
    Slice awkward-array payloads for debug runs while preserving metadata.

    Parameters
    ----------
    unit_payload : dict[str, Any]
        Per-unit payload emitted by the PHMD unit conversion hook.
    debug_subset_range : tuple[int, int, int] | None
        Optional ``(n_cycles, cycle_len, feature_num)`` slice restriction.

    Returns
    -------
    dict[str, Any]
        Updated payload with awkward-array entries sliced for debug runs.
    """

    if debug_subset_range is None:
        return unit_payload

    n_cycles, cycle_len, feature_num = debug_subset_range
    updated_elem: dict[str, Any] = {}
    for subkey, value in unit_payload.items():
        # Only awkward arrays are resized during debug slicing; metadata and
        # scalar helper fields must stay untouched to preserve parity outputs.
        if isinstance(value, ak.Array):
            if value.ndim != 3:
                updated_elem[subkey] = value[:n_cycles]
            else:
                updated_elem[subkey] = value[:n_cycles, :cycle_len, :feature_num]
        else:
            updated_elem[subkey] = value
    return updated_elem


def build_split_payloads(
    main_fold: dict[str, pd.DataFrame],
    *,
    unit_column: str,
    feature_columns: list[str],
    target_column: str,
    process_unit: Callable[[pd.DataFrame, Any, list[str], str], dict[str, Any]],
    debug_subset_range: tuple[int, int, int] | None = None,
) -> tuple[SplitPayload, dict[str, Any]]:
    """
    Convert a split dataframe fold into the PHMD split container shape.

    Parameters
    ----------
    main_fold : dict[str, pandas.DataFrame]
        Primary task fold split by ``train``/``val``/``test``.
    unit_column : str
        Column used to group rows by physical unit.
    feature_columns : list[str]
        Feature columns passed to the per-unit conversion hook.
    target_column : str
        Target column passed to the per-unit conversion hook.
    process_unit : callable
        Per-unit conversion hook implemented by the concrete loader.
    debug_subset_range : tuple[int, int, int] | None, optional
        Optional slice parameters for debug-only payload reduction.

    Returns
    -------
    tuple[SplitPayload, dict[str, Any]]
        The reassembled split-aware payload and any extra unit-level metadata
        extracted from the per-unit records.
    """

    split_payloads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    additional_metadata_unit_name = defaultdict(list)
    additional_metadata_unit_id = defaultdict(list)

    for split_name in ("train", "val", "test"):
        df = main_fold[split_name]
        unit_series = _normalize_unit_column(df, unit_column)

        for unit_name in np.unique(unit_series):
            pre = df.isna().sum()
            mask = pd.Series((unit_series == unit_name), index=df.index)
            assert mask.dtype == bool and mask.ndim == 1 and len(mask) == len(df)
            assert not df.columns.duplicated().any(), "Duplicate columns found"

            df_unit = df.loc[mask]
            post = df_unit.isna().sum()
            assert (pre == post).all(), "Selection created new NaNs"

            unit_data = process_unit(
                df_unit=df_unit,
                unit_name=unit_name,
                features_col=feature_columns,
                target_col=target_column,
            )

            split_payloads[split_name].append(
                _apply_debug_subset(unit_data, debug_subset_range)
            )

            metadata = unit_data.get("metadata", {})
            metadata_unit_name = metadata.get("unit_name")
            metadata_unit_id = metadata.get("unit_id")

            if metadata_unit_name not in (None, {}):
                assert metadata_unit_id not in (
                    None,
                    {},
                ), f"Metadata unit id is invalid for unit name {metadata_unit_name}"
                additional_metadata_unit_name[split_name].append(metadata_unit_name)
                additional_metadata_unit_id[split_name].append(metadata_unit_id)

    train = convert_outer_list_to_inner(split_payloads["train"])
    val = convert_outer_list_to_inner(split_payloads["val"])
    test = convert_outer_list_to_inner(split_payloads["test"])

    out_dict: SplitPayload = {}
    all_keys = set(train) | set(val) | set(test)
    for key in all_keys:
        out_dict[key] = {
            "train": train.get(key, []),
            "val": val.get(key, []),
            "test": test.get(key, []),
        }

    if "metadata" in out_dict and "unit_metadata" not in out_dict:
        out_dict["unit_metadata"] = out_dict.pop("metadata")

    extra_meta: dict[str, Any] = {}
    if additional_metadata_unit_name:
        extra_meta["unit_names"] = additional_metadata_unit_name
    if additional_metadata_unit_id:
        extra_meta["unit_ids"] = additional_metadata_unit_id

    return out_dict, extra_meta
