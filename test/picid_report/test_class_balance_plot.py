"""
Tests for picid_report.class_balance_plot.

Covers:
- compute_class_counts: pure label array → per-class integer counts
- generate_plot: stacked bar chart saved as PDF (no MZVAV data required)
- display_counts_table: rich table smoke test
- get_labels_for_ratio: integration test, skipped when MZVAV data is absent
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from picid_report.class_balance_plot import (
    CLASS_NAMES,
    NUM_CLASSES,
    compute_class_counts,
    display_counts_table,
    generate_plot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def all_normal_labels() -> np.ndarray:
    """200 samples all labelled class 0 (Normal)."""
    return np.zeros(200, dtype=int)


@pytest.fixture
def mixed_labels() -> np.ndarray:
    """120 samples: 60 Normal, 30 Damper, 20 Heating Coil, 10 Cooling Coil."""
    return np.array([0] * 60 + [1] * 30 + [2] * 20 + [3] * 10, dtype=int)


@pytest.fixture
def uniform_ratios() -> list[float]:
    return [0.1, 0.5, 1.0]


@pytest.fixture
def sample_counts_per_ratio(mixed_labels) -> dict[float, np.ndarray]:
    """Pre-computed counts for a small set of ratios, used for plot tests."""
    ratios = [0.1, 0.5, 1.0]
    return {r: compute_class_counts((mixed_labels * r).astype(int)) for r in ratios}


# ---------------------------------------------------------------------------
# compute_class_counts
# ---------------------------------------------------------------------------


class TestComputeClassCounts:
    """compute_class_counts returns a length-NUM_CLASSES int array."""

    def test_all_normal(self, all_normal_labels):
        counts = compute_class_counts(all_normal_labels)
        assert counts.shape == (NUM_CLASSES,)
        assert counts[0] == 200
        assert counts[1:].sum() == 0

    def test_mixed_sums_to_total(self, mixed_labels):
        counts = compute_class_counts(mixed_labels)
        assert counts.sum() == len(mixed_labels)

    def test_mixed_per_class(self, mixed_labels):
        counts = compute_class_counts(mixed_labels)
        assert counts[0] == 60
        assert counts[1] == 30
        assert counts[2] == 20
        assert counts[3] == 10

    def test_empty_returns_zeros(self):
        counts = compute_class_counts(np.array([], dtype=int))
        assert counts.shape == (NUM_CLASSES,)
        assert counts.sum() == 0

    def test_single_class_boundary(self):
        labels = np.full(50, 3, dtype=int)
        counts = compute_class_counts(labels)
        assert counts[3] == 50
        assert counts[:3].sum() == 0

    def test_dtype_is_int(self, mixed_labels):
        counts = compute_class_counts(mixed_labels)
        assert counts.dtype == int or np.issubdtype(counts.dtype, np.integer)

    def test_unknown_class_not_counted(self):
        labels = np.array([0, 1, 99], dtype=int)  # class 99 outside NUM_CLASSES
        counts = compute_class_counts(labels)
        # Only classes 0..NUM_CLASSES-1 are tracked; class 99 is ignored
        assert counts.sum() == 2


# ---------------------------------------------------------------------------
# generate_plot
# ---------------------------------------------------------------------------


class TestGeneratePlot:
    """generate_plot saves a PDF and returns the correct path."""

    def _make_counts(self) -> dict[float, np.ndarray]:
        ratios = [0.1, 0.5, 1.0]
        base = np.array([100, 20, 15, 5], dtype=int)
        return {r: (base * r).astype(int) for r in ratios}

    def test_returns_pdf_path(self, tmp_path):
        counts = self._make_counts()
        output_path = generate_plot([0.1, 0.5, 1.0], counts, str(tmp_path))
        assert output_path.suffix == ".pdf"
        assert output_path.exists()

    def test_output_in_correct_directory(self, tmp_path):
        counts = self._make_counts()
        output_path = generate_plot([0.1, 0.5, 1.0], counts, str(tmp_path))
        assert output_path.parent.resolve() == tmp_path.resolve()

    def test_filename_contains_mzvav(self, tmp_path):
        counts = self._make_counts()
        output_path = generate_plot([0.1, 0.5, 1.0], counts, str(tmp_path))
        assert "mzvav" in output_path.name.lower()

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "subdir" / "nested"
        counts = self._make_counts()
        output_path = generate_plot([0.1, 0.5, 1.0], counts, str(nested))
        assert output_path.exists()

    def test_single_ratio(self, tmp_path):
        counts = {1.0: np.array([80, 10, 7, 3], dtype=int)}
        output_path = generate_plot([1.0], counts, str(tmp_path))
        assert output_path.exists()

    def test_all_zero_counts_does_not_crash(self, tmp_path):
        counts = {0.001: np.zeros(NUM_CLASSES, dtype=int)}
        output_path = generate_plot([0.001], counts, str(tmp_path))
        assert output_path.exists()


# ---------------------------------------------------------------------------
# display_counts_table
# ---------------------------------------------------------------------------


class TestDisplayCountsTable:
    """display_counts_table renders a rich table without raising."""

    def test_smoke(self, capsys):
        counts = {
            0.1: np.array([10, 2, 1, 0], dtype=int),
            1.0: np.array([100, 20, 15, 5], dtype=int),
        }
        display_counts_table([0.1, 1.0], counts)

    def test_zero_class_shown_yellow(self):
        """When a class has zero samples it is displayed as yellow '0'."""
        counts = {0.001: np.array([5, 0, 0, 0], dtype=int)}
        # Just verify no exception; rich output is not easily captured in tests.
        display_counts_table([0.001], counts)

    def test_all_classes_covered(self):
        counts = {r: np.array([10, 5, 3, 2], dtype=int) for r in [0.5, 1.0]}
        display_counts_table([0.5, 1.0], counts)


# ---------------------------------------------------------------------------
# Integration: get_labels_for_ratio (skipped without MZVAV data)
# ---------------------------------------------------------------------------

_MZVAV_DATA_DIR = Path(os.environ.get("MZVAV_DATA_DIR", "")).expanduser()
_MZVAV_CSV = _MZVAV_DATA_DIR / "building" / "MZVAV-2-2.csv"
_MZVAV_AVAILABLE = _MZVAV_CSV.exists()


@pytest.mark.skipif(
    not _MZVAV_AVAILABLE,
    reason="MZVAV dataset not found; set MZVAV_DATA_DIR to the datasets root to enable",
)
class TestGetLabelsForRatioIntegration:
    """Integration tests that run the full MZVAV pipeline.

    Enable by setting MZVAV_DATA_DIR=/path/to/datasets (parent of building/).
    """

    def test_full_ratio_returns_labels(self):
        from picid_report.class_balance_plot import get_labels_for_ratio

        labels = get_labels_for_ratio(
            data_dir=str(_MZVAV_DATA_DIR),
            subset_ratio=1.0,
            subset_blocks=3,
            subset_seed=0,
        )
        assert labels.ndim == 1
        assert len(labels) > 0
        assert set(labels).issubset({0, 1, 2, 3})

    def test_small_ratio_fewer_samples(self):
        from picid_report.class_balance_plot import get_labels_for_ratio

        labels_full = get_labels_for_ratio(
            data_dir=str(_MZVAV_DATA_DIR),
            subset_ratio=1.0,
            subset_blocks=3,
            subset_seed=0,
        )
        labels_small = get_labels_for_ratio(
            data_dir=str(_MZVAV_DATA_DIR),
            subset_ratio=0.1,
            subset_blocks=3,
            subset_seed=0,
        )
        assert len(labels_small) < len(labels_full)

    def test_labels_match_num_classes(self):
        from picid_report.class_balance_plot import get_labels_for_ratio

        labels = get_labels_for_ratio(
            data_dir=str(_MZVAV_DATA_DIR),
            subset_ratio=1.0,
            subset_blocks=3,
            subset_seed=0,
        )
        counts = compute_class_counts(labels)
        assert counts.sum() == len(labels)
        assert len(counts) == NUM_CLASSES

    def test_full_ratio_contains_all_fault_classes(self):
        from picid_report.class_balance_plot import get_labels_for_ratio

        labels = get_labels_for_ratio(
            data_dir=str(_MZVAV_DATA_DIR),
            subset_ratio=1.0,
            subset_blocks=3,
            subset_seed=0,
        )
        # MZVAV with full training split must have at least some fault samples
        assert any(labels > 0), "Expected fault-class samples at ratio=1.0"
