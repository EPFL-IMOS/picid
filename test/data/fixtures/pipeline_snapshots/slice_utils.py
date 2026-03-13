"""
Phase 0: Helpers for pipeline snapshot generation and comparison.

- container_to_slice_manifest: build manifest (shapes, dtypes, slice paths) from container or split dict.
- compare_to_saved_slices: assert current manifest + numeric slices match saved.
- assert_preprocessing_time_within_bounds: fail if current time exceeds baseline * max_time_ratio.

Regenerate snapshots: uv run python test/scripts/snapshot/generate_pipeline_snapshots.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SLICE_MANIFEST_VERSION = "1.0"


def container_to_slice_manifest(
    data: Any,
    keys: list[str],
    out_dir: Path,
    max_rows_per_unit: int = 100,
    splits: list[str] | None = None,
    preprocessing_time_seconds: float | None = None,
) -> dict[str, Any]:
    """
    Produce a manifest (shapes, dtypes, slice paths) and write numeric slices to out_dir.

    data: SplitDatasetContainer or split_dict (split -> key -> list of arrays).
    keys: data keys to include (e.g. ["features", "target"]).
    out_dir: directory to write .npy slice files.
    max_rows_per_unit: cap rows per unit for slice (first N rows).
    splits: default ["train", "val", "test"].
    preprocessing_time_seconds: optional; stored in manifest for transformed stage.

    Returns manifest dict suitable for slices_manifest.json.
    """
    splits = splits or ["train", "val", "test"]
    if hasattr(data, "to_split_dict"):
        split_dict = data.to_split_dict()
    else:
        split_dict = data

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": SLICE_MANIFEST_VERSION,
        "splits": {},
    }
    if preprocessing_time_seconds is not None:
        manifest["preprocessing_time_seconds"] = preprocessing_time_seconds

    for split in splits:
        if split not in split_dict:
            continue
        manifest["splits"][split] = {}
        for key in keys:
            if key not in split_dict[split]:
                continue
            arrs = split_dict[split][key]
            if not isinstance(arrs, list):
                arrs = [arrs]
            unit_entries = []
            for unit_idx, arr in enumerate(arrs):
                arr = np.asarray(arr)
                shape = list(arr.shape)
                dtype = str(arr.dtype)
                n_rows = min(max_rows_per_unit, arr.shape[0]) if arr.size else 0
                slice_arr = (
                    arr[:n_rows]
                    if n_rows
                    else np.array([]).reshape((0,) + arr.shape[1:])
                )
                slice_name = f"{split}_{key}_{unit_idx}.npy"
                slice_path = out_dir / slice_name
                np.save(slice_path, slice_arr)
                unit_entries.append(
                    {
                        "shape": shape,
                        "dtype": dtype,
                        "slice_path": slice_name,
                    }
                )
            manifest["splits"][split][key] = unit_entries

    return manifest


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load slices_manifest.json."""
    with open(manifest_path, "r") as f:
        return json.load(f)


def compare_to_saved_slices(
    current_manifest: dict[str, Any],
    saved_manifest_path: Path,
    saved_slices_dir: Path,
    current_split_dict: dict[str, dict[str, Any]] | None = None,
    current_slices_dir: Path | None = None,
    max_rows_per_unit: int = 100,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Assert current_manifest and optionally numeric slices match saved.

    Compares shapes and dtypes from manifests. If current_split_dict is provided,
    numeric comparison uses in-memory slices from current data. If current_slices_dir
    is provided, numeric comparison uses slice files in that directory (same names as saved).
    """
    saved = load_manifest(saved_manifest_path)
    assert saved.get("schema_version") == current_manifest.get(
        "schema_version"
    ), "schema_version mismatch"
    for split, keys_dict in saved.get("splits", {}).items():
        assert split in current_manifest.get("splits", {}), f"missing split {split}"
        cur_split = current_manifest["splits"][split]
        for key, unit_entries in keys_dict.items():
            assert key in cur_split, f"missing key {key} in split {split}"
            cur_entries = cur_split[key]
            assert len(cur_entries) == len(
                unit_entries
            ), f"unit count mismatch split={split} key={key}"
            for unit_idx, (saved_entry, cur_entry) in enumerate(
                zip(unit_entries, cur_entries)
            ):
                assert (
                    saved_entry["shape"] == cur_entry["shape"]
                ), f"shape mismatch {split}/{key}/{unit_idx}"
                assert (
                    saved_entry["dtype"] == cur_entry["dtype"]
                ), f"dtype mismatch {split}/{key}/{unit_idx}"
                slice_name = saved_entry["slice_path"]
                saved_slice = np.load(saved_slices_dir / slice_name)
                if current_slices_dir and (current_slices_dir / slice_name).exists():
                    cur_slice = np.load(current_slices_dir / slice_name)
                    np.testing.assert_allclose(
                        saved_slice, cur_slice, rtol=rtol, atol=atol
                    )
                elif (
                    current_split_dict
                    and split in current_split_dict
                    and key in current_split_dict[split]
                ):
                    arrs = current_split_dict[split][key]
                    if not isinstance(arrs, list):
                        arrs = [arrs]
                    if unit_idx < len(arrs):
                        arr = np.asarray(arrs[unit_idx])
                        n_rows = min(max_rows_per_unit, arr.shape[0]) if arr.size else 0
                        cur_slice = (
                            arr[:n_rows]
                            if n_rows
                            else np.array([]).reshape((0,) + arr.shape[1:])
                        )
                        np.testing.assert_allclose(
                            saved_slice, cur_slice, rtol=rtol, atol=atol
                        )


def assert_preprocessing_time_within_bounds(
    current_time_seconds: float,
    saved_manifest_path: Path,
    max_time_ratio: float = 2.0,
    max_absolute_slack_seconds: float | None = None,
) -> None:
    """
    Load baseline from manifest; fail if current_time exceeds baseline * max_time_ratio
    (or baseline + max_absolute_slack_seconds if set).
    """
    saved = load_manifest(saved_manifest_path)
    baseline = saved.get("preprocessing_time_seconds")
    if baseline is None:
        return
    threshold = baseline * max_time_ratio
    if max_absolute_slack_seconds is not None:
        threshold = min(threshold, baseline + max_absolute_slack_seconds)
    assert current_time_seconds <= threshold, (
        f"Preprocessing time {current_time_seconds:.2f}s exceeds "
        f"baseline*{max_time_ratio} ({baseline:.2f}*{max_time_ratio}={threshold:.2f}s). "
        "Regenerate snapshots if the slowdown was intentional."
    )
