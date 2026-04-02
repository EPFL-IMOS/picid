#!/usr/bin/env python3
"""Tutorial: load Airbus Helicopter accelerometer data and inspect the predefined split."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from picid.data.datasources.airbus_helicopter import AirbusHelicopterLoader

from tutorials.datasources._tutorial_cli import (
    add_data_dir_argument,
    add_no_download_argument,
)
from tutorials.datasources._loader_introspection import (
    flatten_records,
    ragged_summary,
    summarize_loader_metadata,
    unit_metadata_examples,
)
from tutorials.datasources._standard_report import (
    build_standard_report,
    compute_feature_stats,
    compute_split_overview,
    compute_target_stats,
    format_dataset_specific_notes,
)


def _read_simple_yaml_top_level(path: Path) -> dict[str, str]:
    """Read simple top-level YAML key/value pairs without extra deps."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if raw_line.startswith(" ") or raw_line.startswith("\t"):
            continue
        match = re.match(r"^([A-Za-z0-9_]+)\s*:\s*(.*?)\s*$", line)
        if not match:
            continue
        key, value = match.groups()
        out[key] = value.strip().strip("'\"")
    return out


def _resolve_template(value: str, variables: dict[str, str]) -> str:
    """Resolve ${...} placeholders for the few patterns used in config files."""
    resolved = value
    changed = True
    while changed:
        changed = False
        for key, replacement in variables.items():
            token = "${" + key + "}"
            if token in resolved:
                resolved = resolved.replace(token, replacement)
                changed = True
    env_pattern = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}")
    while True:
        match = env_pattern.search(resolved)
        if not match:
            break
        env_key = match.group(1)
        env_val = os.environ.get(env_key, "")
        resolved = resolved[: match.start()] + env_val + resolved[match.end() :]
    return resolved


def resolve_data_dir(cli_data_dir: str | None) -> str:
    """Resolve data directory from CLI override or project config defaults."""
    if cli_data_dir:
        return str(Path(cli_data_dir).expanduser())

    repo_root = Path(__file__).resolve().parents[2]
    paths_cfg = _read_simple_yaml_top_level(
        repo_root / "configs" / "paths" / "default.yaml"
    )
    ds_cfg = _read_simple_yaml_top_level(
        repo_root / "configs" / "datasource" / "airbus_helicopter.yaml"
    )

    root_dir = paths_cfg.get("root_dir", "")
    root_dir = _resolve_template(
        root_dir,
        {
            "paths.root_dir": str(repo_root),
        },
    )
    if not root_dir:
        root_dir = os.environ.get("PROJECT_ROOT", str(repo_root))

    paths_data_dir = _resolve_template(
        paths_cfg.get("data_dir", "${paths.root_dir}/datasets"),
        {
            "paths.root_dir": root_dir,
        },
    )
    ds_data_dir = _resolve_template(
        ds_cfg.get("data_dir", "${paths.data_dir}"),
        {
            "paths.data_dir": paths_data_dir,
            "paths.root_dir": root_dir,
        },
    )
    if ds_data_dir:
        return str(Path(ds_data_dir).expanduser())
    return str((repo_root / "datasets").resolve())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Load Airbus Helicopter data with AirbusHelicopterLoader and print "
            "a standardized split/statistics report."
        )
    )
    add_data_dir_argument(
        parser,
        help=(
            "Directory containing training_healthy.h5, dataset_anomalies.h5, "
            "and ground_truth.csv. If omitted, resolved from "
            "configs/datasource/airbus_helicopter.yaml and configs/paths/default.yaml."
        ),
    )
    add_no_download_argument(
        parser,
        help="Do not download missing HDF5/CSV files from the public host.",
    )
    return parser.parse_args()


def _split_branch(container: Any, key: str) -> dict[str, list[Any]]:
    if isinstance(container, Mapping):
        raw = container.get(key)
    else:
        raw = getattr(container, key, None)
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, list[Any]] = {}
    for split, units in raw.items():
        out[str(split)] = list(units) if isinstance(units, list) else [units]
    return out


def _build_dataset_notes(meta: dict[str, Any] | None, container: Any) -> str:
    lines = [
        "PREDEFINED SPLIT NOTE",
        "",
        "Train: healthy sequences only from training_healthy.h5 (per-second targets are 0).",
        "Test: anomaly sequences from dataset_anomalies.h5; labels from ground_truth.csv.",
        "Validation: deep copy of the test split — same logical units and labels as test, "
        "but separate array objects (historical dataset / pipeline contract).",
    ]
    if meta and meta.get("dims_explanation"):
        lines.extend(["", str(meta["dims_explanation"])])

    records = flatten_records(container)
    ragged = ragged_summary(records)
    if ragged:
        lines.extend(["", ragged])

    um = unit_metadata_examples(container)
    lines.extend(["", "First unit_metadata example per split:"])
    for sp in ("train", "val", "test"):
        lines.append(f"  {sp}: {um[sp]}")

    return "\n".join(lines)


def main() -> None:
    """Run Airbus Helicopter loading tutorial and print structured output."""
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    auto_download = not args.no_download

    loader = AirbusHelicopterLoader(
        data_dir=data_dir,
        data_name="airbus_helicopter",
        task_mode="anomaly_detection",
        download=auto_download,
    )
    loader.load_data()
    container = loader.get_data()
    meta = loader.get_meta_data()

    features_by_split = _split_branch(container, "features")
    targets_by_split = _split_branch(container, "target")
    if not features_by_split:
        raise RuntimeError(
            "AirbusHelicopterLoader returned no features; check data_dir and files."
        )

    loader_config_body = "\n".join(
        [
            f"data_path: {loader.data_path}",
            f"data_name: {loader.data_name}",
            f"task_mode: {loader.task_mode}",
            f"download: {auto_download}",
        ]
    )
    loader_metadata_body = summarize_loader_metadata(meta)
    split_overview_body = compute_split_overview(
        features_by_split, targets_by_split=targets_by_split
    )
    feature_stats_body = compute_feature_stats(features_by_split)
    target_stats_body = compute_target_stats(targets_by_split)
    dataset_notes_body = format_dataset_specific_notes(
        _build_dataset_notes(meta, container)
    )

    report = build_standard_report(
        loader_config_body=loader_config_body,
        loader_metadata_body=loader_metadata_body,
        split_overview_body=split_overview_body,
        feature_stats_body=feature_stats_body,
        target_stats_body=target_stats_body,
        dataset_specific_notes_body=dataset_notes_body,
    )

    print("AIRBUS HELICOPTER TUTORIAL")
    print()
    print(report)


if __name__ == "__main__":
    main()
