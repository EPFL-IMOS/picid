"""Tests for split-container validation report rendering."""

import awkward as ak
import numpy as np

from picid.data.data_objects.validation import (
    collect_split_alignment_report,
    format_split_alignment_report,
)


def test_format_split_alignment_report_draws_rich_ascii_table():
    """Split reports render as a Rich table captured for plain-text logs."""
    report = collect_split_alignment_report(
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
