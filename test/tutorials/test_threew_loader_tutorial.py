"""Regression tests for ``tutorials/datasources/threew_loader.py``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from test.fixtures.output_assertions import (
    assert_markers_ordered,
    kv_line_value,
    slice_between_markers,
)
from tutorials.datasources import threew_loader as threew_tutorial


class _FakeThreeWLoaderForTutorial:
    """Minimal stand-in for ``ThreeWLoader`` so ``main()`` runs without real 3W data."""

    def __init__(
        self,
        *,
        data_dir: str = "",
        data_name: str = "",
        task_mode: str = "anomaly_detection",
        folds_file: str = "",
        validation_fold: int = 0,
        test_fold: int = 1,
        include_ova: bool = False,
        include_simulated_train: bool = True,
        export_event_class: bool = True,
        download: bool = True,
        **_: Any,
    ) -> None:
        self.data_path = data_dir
        self.task_mode = task_mode

    def load_data(self) -> None:
        return None

    def get_available_task_modes(self) -> list[str]:
        return ["anomaly_detection"]

    def get_data(self) -> dict[str, Any]:
        return {
            "features": {
                "train": [np.array([[1.0, 2.0]], dtype=np.float32)],
                "val": [np.array([[3.0]], dtype=np.float32)],
                "test": [np.array([[4.0]], dtype=np.float32)],
            },
            "target": {
                "train": [np.array([[[0.0]]], dtype=np.float32)],
                "val": [np.array([[[np.nan]]], dtype=np.float32)],
                "test": [np.array([[[0.0]]], dtype=np.float32)],
            },
            "metadata": {
                "train": [
                    {
                        "unit_name": "train/u1",
                        "class_label": 0,
                        "feature_columns": ["f0", "f1"],
                    }
                ],
                "val": [
                    {
                        "unit_name": "val/u1",
                        "class_label": 1,
                        "feature_columns": ["f0"],
                    }
                ],
                "test": [
                    {
                        "unit_name": "test/u1",
                        "class_label": 0,
                        "feature_columns": ["f0"],
                    }
                ],
            },
        }

    def get_meta_data(self) -> dict[str, Any]:
        return {
            "unit_names": {
                "train": ["train/u1"],
                "val": ["val/u1"],
                "test": ["test/u1"],
            }
        }


def _write_minimal_folds_csv(folds_dir: Path) -> None:
    folds_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [("0/WELL-NORMAL-0001.csv", 2, False)],
        columns=["instancia", "fold", "is_ova"],
    )
    df.to_csv(folds_dir / "folds_clf_02.csv", index=False)


def test_threew_tutorial_fails_fast_when_no_instances_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    If folds exist but no instance parquet can be loaded, ``main()`` must not
    finish by printing empty baseline statistics (e.g. "No instances found.").
    """
    folds_dir = tmp_path / "dataset" / "folds"
    folds_dir.mkdir(parents=True)
    df = pd.DataFrame(
        [("0/WELL-NORMAL-0001.csv", 2, False)],
        columns=["instancia", "fold", "is_ova"],
    )
    df.to_csv(folds_dir / "folds_clf_02.csv", index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "threew_loader.py",
            "--data-dir",
            str(tmp_path),
            "--folds-file",
            "dataset/folds/folds_clf_02.csv",
            "--no-download",
        ],
    )
    with pytest.raises((RuntimeError, FileNotFoundError)):
        threew_tutorial.main()


def test_resolve_data_dir_normalizes_cli_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI ``--data-dir`` must expand ``~`` and resolve to an absolute path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "mydata"
    target.mkdir()
    assert threew_tutorial.resolve_data_dir("~/mydata") == str(target.resolve())


def test_threew_tutorial_cli_defaults_to_show_all_ragged_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tutorial should show all ragged examples unless explicitly limited."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["threew_loader.py"],
    )
    args = threew_tutorial.parse_args()
    assert args.max_ragged_examples is None


def test_threew_tutorial_cli_accepts_show_all_ragged_examples_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward/explicit flag should map to unlimited ragged examples output."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["threew_loader.py", "--show-all-ragged-examples"],
    )
    args = threew_tutorial.parse_args()
    assert args.max_ragged_examples is None


def test_ragged_summary_returns_one_example_per_unit() -> None:
    """Ragged examples should include every record across splits by default."""
    records = [
        {
            "split": "train",
            "features": np.array([[1.0], [2.0]], dtype=np.float32),
            "target": np.array([[[0.0]], [[0.0]]], dtype=np.float32),
            "metadata": {"unit_name": "train/u1"},
        },
        {
            "split": "train",
            "features": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            "target": np.array([[[0.0]], [[0.0]], [[0.0]]], dtype=np.float32),
            "metadata": {"unit_name": "train/u2"},
        },
        {
            "split": "val",
            "features": np.array([[1.0]], dtype=np.float32),
            "target": np.array([[[1.0]]], dtype=np.float32),
            "metadata": {"unit_name": "val/u1"},
        },
        {
            "split": "test",
            "features": np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
            "target": np.array([[[1.0]], [[1.0]], [[1.0]], [[1.0]]], dtype=np.float32),
            "metadata": {"unit_name": "test/u1"},
        },
    ]

    _, examples = threew_tutorial.ragged_summary(records)
    assert len(examples) == len(records)


def test_loader_unit_metadata_and_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tutorial output should document loader metadata and one unit-metadata example per split."""
    folds_dir = tmp_path / "dataset" / "folds"
    _write_minimal_folds_csv(folds_dir)
    monkeypatch.setattr(threew_tutorial, "ThreeWLoader", _FakeThreeWLoaderForTutorial)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "threew_loader.py",
            "--data-dir",
            str(tmp_path),
            "--folds-file",
            "dataset/folds/folds_clf_02.csv",
            "--no-download",
        ],
    )
    threew_tutorial.main()
    out = capsys.readouterr().out

    assert_markers_ordered(
        out,
        [
            "LOADER METADATA",
            "UNIT METADATA EXAMPLES (ONE PER SPLIT)",
            "CURRENT TASK AND AVAILABLE TASKS",
            "BASELINE-STYLE CLASS DISTRIBUTION (INSTANCES)",
        ],
    )
    assert kv_line_value(out, "current_task") == "anomaly_detection"
    assert kv_line_value(out, "available_tasks") == "['anomaly_detection']"

    unit_block = slice_between_markers(
        out,
        "UNIT METADATA EXAMPLES (ONE PER SPLIT)",
        "BASELINE-STYLE CLASS DISTRIBUTION (INSTANCES)",
    )
    assert "train" in unit_block
    assert "val" in unit_block
    assert "test" in unit_block


def test_target_level_statistics_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tutorial output should include target-level missing/stats like feature-level section."""
    folds_dir = tmp_path / "dataset" / "folds"
    _write_minimal_folds_csv(folds_dir)
    monkeypatch.setattr(threew_tutorial, "ThreeWLoader", _FakeThreeWLoaderForTutorial)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "threew_loader.py",
            "--data-dir",
            str(tmp_path),
            "--folds-file",
            "dataset/folds/folds_clf_02.csv",
            "--no-download",
        ],
    )
    threew_tutorial.main()
    out = capsys.readouterr().out

    assert "TARGET-LEVEL STATISTICS AND MISSING RATIOS" in out
    assert "target_0" in out
    assert "Global target missing ratio:" in out
