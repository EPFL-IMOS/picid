"""Reporting helpers for heterogeneous split-container diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from io import StringIO
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from picid.data.data_objects.core.metadata_data_object import BaseDataObjectWithMetadata

SPLIT_ALIGNMENT_REPORT_HEADERS = [
    "key",
    "splits",
    "unit_counts",
    "sample_shapes",
    "schema_status",
    "sample_schema",
]


def describe_unit_payload(unit_list: list[Any]) -> str:
    """
    Summarize the first payload stored for one split.

    Parameters
    ----------
    unit_list : list[Any]
        Unit payloads stored for a single split.

    Returns
    -------
    str
        Compact textual description of the representative payload.
    """
    if not unit_list:
        return "empty"

    first = unit_list[0]
    if isinstance(first, (np.ndarray, pd.Series, pd.DataFrame)):
        return str(first.shape)
    if isinstance(first, ak.Array):
        try:
            return str(first.shape)
        except (AttributeError, TypeError, ValueError):
            try:
                return f"ak[{ak.type(first)}]"
            except Exception:
                return f"ak[len={len(first)}]"
    if isinstance(first, Mapping):
        return "mapping"
    if hasattr(first, "shape"):
        try:
            return str(first.shape)
        except (AttributeError, TypeError, ValueError):
            pass
    if hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
        return f"len={len(first)}"
    return type(first).__name__


def payload_schema_signature(payload: Any) -> Any:
    """
    Build a comparable schema signature for one unit payload.

    Parameters
    ----------
    payload : Any
        Unit payload whose schema should be summarized.

    Returns
    -------
    Any
        Comparable schema signature for the payload.
    """
    if isinstance(payload, BaseDataObjectWithMetadata):
        payload = dict(payload.items())
    if isinstance(payload, Mapping):
        return tuple(
            sorted(
                (key, payload_schema_signature(value)) for key, value in payload.items()
            )
        )
    return type(payload).__name__


def format_schema_signature(signature: Any) -> str:
    """
    Render a compact string representation of a schema signature.

    Parameters
    ----------
    signature : Any
        Schema signature returned by :func:`payload_schema_signature`.

    Returns
    -------
    str
        Human-readable representation of the schema signature.
    """
    if isinstance(signature, tuple):
        inner = ", ".join(
            f"{key}:{format_schema_signature(value)}" for key, value in signature
        )
        return "{" + inner + "}"
    return str(signature)


def collect_split_alignment_report(
    payloads: list[tuple[str, Mapping[str, list[Any]]]],
) -> dict[str, Any]:
    """
    Collect a non-strict report about split-key alignment and unit scales.

    Parameters
    ----------
    payloads : list[tuple[str, Mapping[str, list[Any]]]]
        Split-keyed payloads extracted from a split container.

    Returns
    -------
    dict[str, Any]
        Structured report describing split counts, schemas, and mismatches.
    """
    all_splits = sorted(
        {split for _, splits_data in payloads for split in splits_data.keys()}
    )

    rows = []
    split_name_sets: list[set[str]] = []
    schema_mismatches: list[dict[str, Any]] = []
    for data_key, splits_data in payloads:
        split_name_sets.append(set(splits_data.keys()))
        counts = {
            split: len(splits_data[split]) if split in splits_data else None
            for split in all_splits
        }
        shapes = {
            split: describe_unit_payload(splits_data[split])
            if split in splits_data
            else "-"
            for split in all_splits
        }
        schemas = {
            split: format_schema_signature(
                payload_schema_signature(splits_data[split][0])
            )
            if split in splits_data and splits_data[split]
            else "-"
            for split in all_splits
        }
        schema_status = {}
        for split in all_splits:
            if split not in splits_data or not splits_data[split]:
                schema_status[split] = "empty"
                continue
            reference_schema = payload_schema_signature(splits_data[split][0])
            if all(
                payload_schema_signature(payload) == reference_schema
                for payload in splits_data[split][1:]
            ):
                schema_status[split] = "homogeneous"
            else:
                schema_status[split] = "heterogeneous"
        rows.append(
            {
                "key": data_key,
                "splits": sorted(splits_data.keys()),
                "counts": counts,
                "shapes": shapes,
                "schemas": schemas,
                "schema_status": schema_status,
            }
        )
        for split, unit_list in splits_data.items():
            if not unit_list:
                continue
            expected_schema = payload_schema_signature(unit_list[0])
            for unit_idx, payload in enumerate(unit_list[1:], start=1):
                current_schema = payload_schema_signature(payload)
                if current_schema != expected_schema:
                    schema_mismatches.append(
                        {
                            "key": data_key,
                            "split": split,
                            "unit_index": unit_idx,
                            "expected_schema": format_schema_signature(expected_schema),
                            "actual_schema": format_schema_signature(current_schema),
                        }
                    )

    split_names_match = (
        len({tuple(sorted(split_names)) for split_names in split_name_sets}) <= 1
    )
    unit_counts_match = True
    for split in all_splits:
        present_counts = {
            row["counts"][split] for row in rows if row["counts"][split] is not None
        }
        if len(present_counts) > 1:
            unit_counts_match = False
            break

    unit_schema_match = not schema_mismatches
    is_consistent = split_names_match and unit_counts_match and unit_schema_match
    return {
        "rows": rows,
        "splits": all_splits,
        "split_names_match": split_names_match,
        "unit_counts_match": unit_counts_match,
        "unit_schema_match": unit_schema_match,
        "schema_mismatches": schema_mismatches,
        "is_consistent": is_consistent,
    }


def _split_alignment_table_rows(report: dict[str, Any]) -> list[list[str]]:
    """
    Build display rows for a split alignment report.

    Parameters
    ----------
    report : dict[str, Any]
        Structured report returned by :func:`collect_split_alignment_report`.

    Returns
    -------
    list[list[str]]
        Display-ready rows ordered like :data:`SPLIT_ALIGNMENT_REPORT_HEADERS`.
    """
    table_rows = []
    for row in report["rows"]:
        split_names = ",".join(row["splits"]) or "-"
        count_str = ", ".join(
            f"{split}={row['counts'][split] if row['counts'][split] is not None else '-'}"
            for split in report["splits"]
        )
        shape_str = ", ".join(
            f"{split}={row['shapes'][split]}" for split in report["splits"]
        )
        schema_status_str = ", ".join(
            f"{split}={row['schema_status'][split]}" for split in report["splits"]
        )
        schema_str = ", ".join(
            f"{split}={row['schemas'][split]}" for split in report["splits"]
        )
        table_rows.append(
            [
                row["key"],
                split_names,
                count_str,
                shape_str,
                schema_status_str,
                schema_str,
            ]
        )
    return table_rows


def build_split_alignment_report_table(report: dict[str, Any]) -> Table:
    """
    Build a live Rich table for split alignment diagnostics.

    Parameters
    ----------
    report : dict[str, Any]
        Structured report returned by :func:`collect_split_alignment_report`.

    Returns
    -------
    rich.table.Table
        Rich renderable for interactive terminal or notebook display.
    """
    table = Table(
        title="Split Alignment Report",
        header_style="bold cyan",
        pad_edge=True,
        show_lines=False,
        expand=False,
    )
    for header in SPLIT_ALIGNMENT_REPORT_HEADERS:
        table.add_column(
            header,
            min_width=len(header),
            no_wrap=header in {"key", "splits"},
            overflow="fold",
        )
    for table_row in _split_alignment_table_rows(report):
        table.add_row(*(str(value) for value in table_row))
    return table


def format_split_alignment_report(report: dict[str, Any]) -> str:
    """
    Format a compact ASCII table for split alignment diagnostics.

    Parameters
    ----------
    report : dict[str, Any]
        Structured report returned by :func:`collect_split_alignment_report`.

    Returns
    -------
    str
        Plain-text fallback representation for logs and exception messages.
    """
    if not report["rows"]:
        return "No split payloads available."

    table_rows = _split_alignment_table_rows(report)

    table = Table(
        box=box.ASCII,
        header_style="",
        pad_edge=True,
        show_lines=False,
        expand=False,
    )
    for header in SPLIT_ALIGNMENT_REPORT_HEADERS:
        table.add_column(header, no_wrap=True, overflow="fold")
    for table_row in table_rows:
        table.add_row(*(str(value) for value in table_row))

    render_width = max(
        80,
        sum(
            max(len(str(value)) for value in [header] + [row[idx] for row in table_rows])
            for idx, header in enumerate(SPLIT_ALIGNMENT_REPORT_HEADERS)
        )
        + (3 * (len(SPLIT_ALIGNMENT_REPORT_HEADERS) - 1))
        + 4,
    )
    buffer = StringIO()
    console = Console(
        color_system=None,
        file=buffer,
        force_terminal=False,
        highlight=False,
        legacy_windows=False,
        width=render_width,
    )
    console.print(table)

    lines = buffer.getvalue().rstrip().splitlines()
    if report["schema_mismatches"]:
        lines.append("")
        lines.append("schema_mismatches:")
        for mismatch in report["schema_mismatches"]:
            lines.append(
                "  "
                f"{mismatch['key']}/{mismatch['split']} unit {mismatch['unit_index']}: "
                f"expected {mismatch['expected_schema']} but got {mismatch['actual_schema']}"
            )
    return "\n".join(lines)
