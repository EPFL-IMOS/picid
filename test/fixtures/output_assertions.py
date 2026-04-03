"""Helpers for asserting structure of captured stdout and multi-section text."""

from __future__ import annotations

from collections.abc import Sequence


def assert_markers_ordered(text: str, markers: Sequence[str], *, msg: str = "") -> None:
    """Assert each marker appears in ``text`` and first occurrences are strictly increasing."""
    suffix = f" ({msg})" if msg else ""
    prev = -1
    for idx, marker in enumerate(markers):
        pos = text.find(marker)
        if pos == -1:
            raise AssertionError(f"marker {idx} not found: {marker!r}{suffix}")
        if pos <= prev:
            raise AssertionError(
                f"markers out of order: {markers[idx - 1]!r} at {prev}, "
                f"then {marker!r} at {pos}{suffix}"
            )
        prev = pos


def slice_between_markers(
    text: str,
    start_marker: str,
    end_marker: str | None = None,
) -> str:
    """Return ``text[start:end]`` where ``start`` is the first ``start_marker`` and ``end`` is the first ``end_marker`` after that (or end of string).

    The returned slice includes ``start_marker`` and excludes ``end_marker``.
    """
    i = text.find(start_marker)
    if i == -1:
        raise AssertionError(f"start marker not found: {start_marker!r}")
    if end_marker is None:
        return text[i:]
    j = text.find(end_marker, i + len(start_marker))
    if j == -1:
        raise AssertionError(f"end marker not found: {end_marker!r}")
    return text[i:j]


def kv_line_value(text: str, key: str) -> str | None:
    """Return the value part of the first line ``{key}: <value>`` (stripped), or ``None``."""
    prefix = f"{key}:"
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None
