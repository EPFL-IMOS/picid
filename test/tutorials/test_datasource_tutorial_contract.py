"""Contract tests for shared datasource tutorial report formatting."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import awkward as ak
import numpy as np

from test.fixtures.output_assertions import assert_markers_ordered
from tutorials.datasources._tutorial_cli import (
    STANDARD_DATASOURCE_TUTORIAL_SCRIPTS,
    TUTORIAL_SCRIPTS_WITH_NO_DOWNLOAD,
)
from tutorials.datasources._loader_introspection import (
    flatten_records,
    infer_feature_target_columns,
    ragged_summary,
    summarize_loader_metadata,
    unit_metadata_examples,
)
from tutorials.datasources._standard_report import (
    STANDARD_SECTIONS,
    build_standard_report,
    compute_feature_stats,
    compute_split_overview,
    compute_target_stats,
    format_dataset_specific_notes,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TUTORIALS_DATASOURCES = _REPO_ROOT / "tutorials" / "datasources"


def test_standard_datasource_tutorial_scripts_expose_common_cli_flags() -> None:
    """Each registered loader tutorial documents ``--data-dir``; download-capable ones add ``--no-download``."""
    for name in STANDARD_DATASOURCE_TUTORIAL_SCRIPTS:
        script = _TUTORIALS_DATASOURCES / name
        assert script.is_file(), f"missing tutorial script: {script}"
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"{name}: {proc.stderr or proc.stdout}"
        out = proc.stdout
        assert "--data-dir" in out, f"{name} help missing --data-dir"
        if name in TUTORIAL_SCRIPTS_WITH_NO_DOWNLOAD:
            assert "--no-download" in out, f"{name} help missing --no-download"


def test_standard_report_includes_all_section_headers() -> None:
    report = build_standard_report(
        loader_config_body="data_dir: /tmp/example",
        loader_metadata_body="loader: ExampleLoader",
        split_overview_body="train: 1 units",
        feature_stats_body="feature stats placeholder",
        target_stats_body="target stats placeholder",
        dataset_specific_notes_body="No extra notes.",
    )
    assert_markers_ordered(
        report,
        [f"\n{title}\n" for title in STANDARD_SECTIONS],
        msg="standard report section headers",
    )


def test_build_standard_report_has_no_leading_or_trailing_newlines() -> None:
    report = build_standard_report(
        loader_config_body="x",
        loader_metadata_body="y",
        split_overview_body="z",
        feature_stats_body="a",
        target_stats_body="b",
        dataset_specific_notes_body="final body line",
    )
    assert not report.startswith("\n")
    assert not report.endswith("\n")


def test_compute_split_overview_describes_splits() -> None:
    features = {
        "train": [np.zeros((10, 2), dtype=np.float32)],
        "test": [np.ones((5, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32)],
    }
    text = compute_split_overview(features)
    assert "train" in text
    assert "test" in text
    assert "units=1" in text
    assert "units=2" in text


def test_compute_feature_stats_returns_tabular_text() -> None:
    features = {
        "train": [np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)],
    }
    text = compute_feature_stats(features)
    assert "mean" in text.lower() or "col" in text.lower()
    assert "train" in text or "0" in text


def test_compute_target_stats_returns_text() -> None:
    targets = {
        "train": [np.array([[0.0], [1.0]], dtype=np.float32)],
    }
    text = compute_target_stats(targets)
    assert len(text.strip()) > 0


def test_standard_report_stats_handle_ragged_awkward_arrays() -> None:
    """Shared stats helpers must not crash on ragged awkward payloads."""
    ragged_features = ak.Array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0]],
        ]
    )
    ragged_targets = ak.Array(
        [
            [[0.0], [0.0]],
            [[1.0]],
        ]
    )
    features = {"train": [ragged_features]}
    targets = {"train": [ragged_targets]}

    split_text = compute_split_overview(features, targets_by_split=targets)
    feature_text = compute_feature_stats(features)
    target_text = compute_target_stats(targets)

    assert "train:" in split_text
    assert "train:" in feature_text
    assert "col_0" in feature_text
    assert "n_values=" in target_text


def test_format_dataset_specific_notes() -> None:
    assert "alpha" in format_dataset_specific_notes("alpha")
    out = format_dataset_specific_notes(["a", "b"])
    assert "a" in out and "b" in out


def test_flatten_records_numpy_and_zip_longest_metadata() -> None:
    """Missing per-unit metadata must not drop feature rows (zip_longest)."""
    container = {
        "features": {"train": [np.ones((3, 2), dtype=np.float32), np.zeros((2, 2))]},
        "target": {"train": [np.zeros((3, 1)), np.zeros((2, 1))]},
        "unit_metadata": {"train": [{"id": 0}]},
    }
    records = flatten_records(container)
    assert len(records) == 2
    assert records[0]["metadata"] == {"id": 0}
    assert records[1]["metadata"] == {}
    assert records[0]["features"].shape == (3, 2)
    assert records[1]["features"].shape == (2, 2)
    assert isinstance(records[0]["features"], np.ndarray)


def test_flatten_records_preserves_awkward_array() -> None:
    aw = ak.Array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    container = {
        "features": {"train": [aw]},
        "target": {"train": [np.zeros(3)]},
    }
    records = flatten_records(container)
    assert len(records) == 1
    assert isinstance(records[0]["features"], ak.Array)
    assert ak.to_numpy(records[0]["features"]).shape == (3, 2)


def test_unit_metadata_examples_handles_missing_metadata() -> None:
    container = {
        "features": {"train": [np.zeros((1, 1))]},
        "target": {"train": [np.zeros(1)]},
    }
    out = unit_metadata_examples(container)
    for split in ("train", "val", "test"):
        assert split in out
    assert "no unit metadata" in out["train"].lower()
    assert "no unit metadata" in out["val"].lower()


def test_ragged_summary_optional_for_non_ragged_payloads() -> None:
    uniform = [
        {"features": np.ones((4, 2)), "split": "train"},
        {"features": np.ones((4, 2)), "split": "train"},
    ]
    assert ragged_summary(uniform) is None


def test_ragged_summary_reports_variable_lengths() -> None:
    records = [
        {"features": np.ones((2, 3))},
        {"features": np.ones((7, 3))},
    ]
    text = ragged_summary(records)
    assert text is not None
    assert "2" in text and "7" in text


def test_summarize_loader_metadata() -> None:
    assert (
        "empty" in summarize_loader_metadata({}).lower()
        or summarize_loader_metadata({}) == "(empty)"
    )
    body = summarize_loader_metadata({"b": 1, "a": [1, 2]})
    assert "a:" in body and "b:" in body


def test_infer_feature_target_columns() -> None:
    meta = {"column_map": {"features": ["x", "y"], "target": ["z"]}}
    fc, tc = infer_feature_target_columns(meta, (10, 2), (10, 1))
    assert fc == ["x", "y"] and tc == ["z"]
    fc2, tc2 = infer_feature_target_columns({}, (5, 3), (5, 2))
    assert fc2 == ["feature_0", "feature_1", "feature_2"]
    assert tc2 == ["target_0", "target_1"]
