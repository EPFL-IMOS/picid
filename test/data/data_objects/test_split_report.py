"""Tests for split-container validation report rendering."""

import awkward as ak
import numpy as np
from rich.console import Console
from rich.table import Table

from picid.data.data_objects.validation import (
    build_split_alignment_report_table,
    collect_split_alignment_report,
    format_split_alignment_report,
)


def _sample_report():
    return collect_split_alignment_report(
        [
            (
                "features",
                {
                    "train": [ak.Array(np.zeros((2, 3)))],
                    "val": [ak.Array(np.ones((1, 3)))],
                },
            ),
            (
                "target",
                {
                    "train": [ak.Array(np.zeros((2, 1)))],
                    "val": [ak.Array(np.ones((1, 1)))],
                },
            ),
        ]
    )


def test_build_split_alignment_report_table_returns_rich_table():
    """Split reports have a Rich renderable for interactive display."""
    table = build_split_alignment_report_table(_sample_report())

    assert isinstance(table, Table)
    assert table.title == "Split Alignment Report"


def test_build_split_alignment_report_table_renders_without_ansi_codes():
    """Recorded Rich output is styled by Rich, not pre-rendered as ASCII text."""
    console = Console(
        color_system=None,
        force_terminal=True,
        record=True,
        width=140,
    )

    console.print(build_split_alignment_report_table(_sample_report()))
    rendered = console.export_text()

    assert "Split Alignment Report" in rendered
    assert "features" in rendered
    assert "target" in rendered
    assert "sample_shapes" in rendered
    assert "\x1b[" not in rendered
    assert "+-" not in rendered


def test_format_split_alignment_report_draws_ascii_fallback_table():
    """Plain-text fallback remains stable for logs and exception messages."""
    report = _sample_report()

    rendered = format_split_alignment_report(report)
    lines = rendered.splitlines()

    assert lines[0].startswith("+")
    assert lines[-1].startswith("+")
    assert "| key" in rendered
    assert "sample_shapes" in rendered
    assert "sample_schema" in rendered
    assert "features" in rendered
    assert "target" in rendered
    assert "train=ak[" in rendered
    assert "val=ak[" in rendered
    assert "\x1b[" not in rendered
