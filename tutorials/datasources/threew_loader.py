#!/usr/bin/env python3
"""Tutorial: load Petrobras 3W and inspect baseline-style dataset statistics.

This script demonstrates:
1) Loading 3W via PICID's predefined-split datasource loader.
2) Reproducing key baseline exploration tables from the original 3W notebook.
3) Showing how ragged unit lengths are handled natively (no padding/truncation).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutorials.datasources._tutorial_cli import (
    add_data_dir_argument,
    add_no_download_argument,
    normalize_data_dir_override,
)
from picid.data.datasources.threew import ThreeWLoader

EVENT_NAMES: dict[int, str] = {
    0: "Normal",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
}


def _read_simple_yaml_top_level(path: Path) -> dict[str, str]:
    """Read simple top-level YAML key/value pairs without extra deps."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Ignore nested keys; tutorial only needs top-level scalar fields.
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
    # Resolve ${oc.env:VAR} patterns.
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
    normalized_cli = normalize_data_dir_override(cli_data_dir)
    if normalized_cli:
        return normalized_cli

    repo_root = Path(__file__).resolve().parents[2]
    paths_cfg = _read_simple_yaml_top_level(
        repo_root / "configs" / "paths" / "default.yaml"
    )
    ds_cfg = _read_simple_yaml_top_level(
        repo_root / "configs" / "datasource" / "threew.yaml"
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
            "Load Petrobras 3W with ThreeWLoader and print baseline-style "
            "dataset statistics with ragged-data diagnostics."
        )
    )
    add_data_dir_argument(
        parser,
        help=(
            "Root directory containing the local 3W dataset folder. "
            "If omitted, resolved from configs/datasource/threew.yaml "
            "and configs/paths/default.yaml."
        ),
    )
    parser.add_argument(
        "--folds-file",
        default="threew_dataset/folds/folds_clf_02.csv",
        help="Path to folds CSV relative to --data-dir.",
    )
    parser.add_argument(
        "--validation-fold",
        type=int,
        default=0,
        help="Fold id to use as validation split.",
    )
    parser.add_argument(
        "--test-fold",
        type=int,
        default=1,
        help="Fold id to use as test split.",
    )
    parser.add_argument(
        "--include-ova",
        action="store_true",
        help="Include one-vs-all rows from the folds file.",
    )
    parser.add_argument(
        "--exclude-simulated-train",
        action="store_true",
        help="Exclude rows with fold=-1 (simulated) from train split.",
    )
    add_no_download_argument(
        parser,
        help=(
            "Disable automatic 3W download when local data is missing. "
            "By default, the loader tries to download from the public Petrobras repo."
        ),
    )
    parser.add_argument(
        "--max-ragged-examples",
        type=int,
        default=None,
        help=(
            "Maximum number of ragged per-unit shape examples to print. "
            "By default, all units are shown."
        ),
    )
    parser.add_argument(
        "--show-all-ragged-examples",
        action="store_const",
        const=None,
        dest="max_ragged_examples",
        help=(
            "Explicit alias to print all ragged per-unit shape examples "
            "(default behavior)."
        ),
    )
    return parser.parse_args()


def resolve_threew_layout(
    data_dir: str, folds_file: str
) -> tuple[str, str, list[Path], Path]:
    """Resolve a valid (data_dir, folds_file) pair for common local layouts."""
    base = Path(data_dir).expanduser()

    data_dir_candidates = [
        base,
        base / "threew",
        base / "3w",
        base.parent if base.name in {"dataset", "threew_dataset"} else None,
    ]
    folds_file_candidates = [
        folds_file,
        "threew_dataset/folds/folds_clf_02.csv",
        "dataset/folds/folds_clf_02.csv",
        "folds/folds_clf_02.csv",
    ]

    attempted: list[Path] = []
    seen: set[tuple[Path, str]] = set()
    for data_root in data_dir_candidates:
        if data_root is None:
            continue
        for fold_rel in folds_file_candidates:
            key = (data_root, fold_rel)
            if key in seen:
                continue
            seen.add(key)
            candidate_path = data_root / fold_rel
            attempted.append(candidate_path)
            if candidate_path.exists():
                return str(data_root), fold_rel, attempted, candidate_path

    return str(base), folds_file, attempted, base / folds_file


def infer_source(instance_name: str) -> str:
    """Infer source category from original 3W naming convention."""
    stem = instance_name.upper()
    if stem.startswith("SIMULATED"):
        return "SIMULATED"
    if stem.startswith("DRAWN"):
        return "HAND_LABELED"
    return "REAL"


def summarize_loader_metadata(meta_data: Any) -> list[str]:
    """Build printable lines for loader ``meta_data`` (split counts when available)."""
    lines: list[str] = []
    if meta_data is None:
        lines.append("meta_data: (missing)")
        return lines
    if not isinstance(meta_data, dict):
        lines.append(f"meta_data: (unexpected type {type(meta_data).__name__})")
        return lines

    keys = sorted(meta_data.keys())
    lines.append("top-level keys: " + (", ".join(keys) if keys else "(none)"))

    split_keys = ("train", "val", "test")
    for field in ("unit_names", "unit_ids", "class_labels"):
        payload = meta_data.get(field)
        if not isinstance(payload, dict):
            continue
        parts: list[str] = []
        for split in split_keys:
            seq = payload.get(split)
            if isinstance(seq, (list, tuple)):
                parts.append(f"{split}={len(seq)}")
            else:
                parts.append(f"{split}=(n/a)")
        lines.append(f"{field} split counts: {', '.join(parts)}")

    # Any other dict-valued entries that look split-organized.
    for name in keys:
        if name in ("unit_names", "unit_ids", "class_labels"):
            continue
        payload = meta_data.get(name)
        if not isinstance(payload, dict):
            continue
        if not set(payload.keys()).issubset(set(split_keys)):
            continue
        parts = []
        for split in split_keys:
            seq = payload.get(split)
            if isinstance(seq, (list, tuple)):
                parts.append(f"{split}={len(seq)}")
            else:
                parts.append(f"{split}=(n/a)")
        lines.append(f"{name} split counts: {', '.join(parts)}")

    return lines


def _compact_value_for_metadata_line(value: Any, max_len: int = 96) -> str:
    """Deterministic short string for metadata values (avoid huge reprs)."""
    if isinstance(value, np.ndarray):
        s = f"ndarray(shape={value.shape}, dtype={value.dtype})"
    elif isinstance(value, (dict, list, tuple)):
        s = repr(value)
    else:
        s = repr(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def unit_metadata_examples(container: Any) -> dict[str, str]:
    """One compact example string per split using ``_resolve_split_unit_metadata``."""
    by_split = _resolve_split_unit_metadata(container)
    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        items = by_split.get(split) or []
        if not items:
            out[split] = "(no unit metadata for this split)"
            continue
        first = items[0]
        if not isinstance(first, dict):
            out[split] = _compact_value_for_metadata_line(first)
            continue
        parts = [
            f"{k}={_compact_value_for_metadata_line(first[k])}"
            for k in sorted(first.keys())
        ]
        out[split] = "{" + ", ".join(parts) + "}"
    return out


def _resolve_split_unit_metadata(
    container: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Return split-wise per-unit metadata from legacy or canonical container layouts."""
    unit_metadata = getattr(container, "unit_metadata", None)
    if isinstance(unit_metadata, dict):
        return {
            split: list(unit_metadata.get(split, []))
            for split in ("train", "val", "test")
        }

    metadata_payload = container.get("metadata", {})
    if isinstance(metadata_payload, dict) and set(metadata_payload.keys()).issubset(
        {"train", "val", "test"}
    ):
        return {
            split: list(metadata_payload.get(split, []))
            for split in ("train", "val", "test")
        }
    return {"train": [], "val": [], "test": []}


def flatten_records(container: dict[str, dict[str, list[Any]]]) -> list[dict[str, Any]]:
    """Build a unit-level list of records aligned across split payload keys."""
    records: list[dict[str, Any]] = []
    split_metadata = _resolve_split_unit_metadata(container)
    for split in ("train", "val", "test"):
        features_list = container.get("features", {}).get(split, [])
        metadata_list = split_metadata.get(split, [])
        target_list = container.get("target", {}).get(split, [])
        for idx, (features, metadata, target) in enumerate(
            zip(features_list, metadata_list, target_list)
        ):
            records.append(
                {
                    "split": split,
                    "unit_idx": idx,
                    "features": np.asarray(features),
                    "target": np.asarray(target),
                    "metadata": metadata,
                }
            )
    return records


def class_distribution(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Compute instance counts per class across all splits."""
    counter = Counter(int(r["metadata"]["class_label"]) for r in records)
    rows = []
    for class_id in sorted(counter):
        rows.append(
            {
                "CLASS_ID": class_id,
                "CLASS_NAME": EVENT_NAMES.get(class_id, f"Class {class_id}"),
                "INSTANCES": counter[class_id],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        total = int(df["INSTANCES"].sum())
        df["RATIO_%"] = (100.0 * df["INSTANCES"] / total).round(2)
    return df


def source_comparison(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Compute REAL/SIMULATED/HAND_LABELED instance counts by class and total."""
    class_source_counter: defaultdict[tuple[int, str], int] = defaultdict(int)
    for r in records:
        class_id = int(r["metadata"]["class_label"])
        unit_name = str(r["metadata"]["unit_name"])
        instance_name = unit_name.split("/", maxsplit=1)[-1]
        source = infer_source(instance_name)
        class_source_counter[(class_id, source)] += 1

    class_ids = sorted({class_id for class_id, _ in class_source_counter})
    source_order = ["REAL", "SIMULATED", "HAND_LABELED"]
    rows: list[dict[str, Any]] = []
    for class_id in class_ids:
        row = {
            "CLASS_ID": class_id,
            "CLASS_NAME": EVENT_NAMES.get(class_id, f"Class {class_id}"),
        }
        for source in source_order:
            row[source] = class_source_counter.get((class_id, source), 0)
        row["TOTAL"] = int(sum(row[source] for source in source_order))
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        total_row = {
            "CLASS_ID": "TOTAL",
            "CLASS_NAME": "TOTAL",
            "REAL": int(df["REAL"].sum()),
            "SIMULATED": int(df["SIMULATED"].sum()),
            "HAND_LABELED": int(df["HAND_LABELED"].sum()),
        }
        total_row["TOTAL"] = (
            total_row["REAL"] + total_row["SIMULATED"] + total_row["HAND_LABELED"]
        )
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    return df


def feature_stats(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Compute feature-level stats and missing ratios without densifying ragged data."""
    running: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "n_obs": 0.0,
            "n_missing": 0.0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": np.inf,
            "max": -np.inf,
            "n_instances": 0.0,
            "n_all_missing_instances": 0.0,
        }
    )

    total_feature_cells = 0
    total_missing_cells = 0

    for r in records:
        x = np.asarray(r["features"], dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n_timesteps, n_features = x.shape
        cols = list(r["metadata"].get("feature_columns", []))
        if len(cols) != n_features:
            cols = [f"feature_{i}" for i in range(n_features)]

        for col_idx, col_name in enumerate(cols):
            values = x[:, col_idx]
            missing_mask = np.isnan(values)
            n_missing = int(missing_mask.sum())
            n_total = int(values.size)
            n_valid = n_total - n_missing
            valid_values = values[~missing_mask]

            stats = running[col_name]
            stats["n_instances"] += 1
            stats["n_obs"] += n_total
            stats["n_missing"] += n_missing
            if n_missing == n_total:
                stats["n_all_missing_instances"] += 1
            if n_valid > 0:
                stats["sum"] += float(valid_values.sum())
                stats["sum_sq"] += float((valid_values**2).sum())
                stats["min"] = min(stats["min"], float(valid_values.min()))
                stats["max"] = max(stats["max"], float(valid_values.max()))

        total_feature_cells += int(n_timesteps * n_features)
        total_missing_cells += int(np.isnan(x).sum())

    rows = []
    for feature in sorted(running):
        s = running[feature]
        n_obs = int(s["n_obs"])
        n_missing = int(s["n_missing"])
        n_valid = n_obs - n_missing
        mean = np.nan
        std = np.nan
        if n_valid > 0:
            mean = s["sum"] / n_valid
            var = max((s["sum_sq"] / n_valid) - (mean**2), 0.0)
            std = var**0.5
        rows.append(
            {
                "FEATURE": feature,
                "OBS": n_obs,
                "MISSING": n_missing,
                "MISSING_%": round(100.0 * n_missing / n_obs, 3) if n_obs else np.nan,
                "MEAN": round(float(mean), 6) if np.isfinite(mean) else np.nan,
                "STD": round(float(std), 6) if np.isfinite(std) else np.nan,
                "MIN": round(float(s["min"]), 6) if np.isfinite(s["min"]) else np.nan,
                "MAX": round(float(s["max"]), 6) if np.isfinite(s["max"]) else np.nan,
                "ALL_MISSING_INSTANCES": int(s["n_all_missing_instances"]),
            }
        )
    feature_df = pd.DataFrame(rows)
    global_stats = {
        "feature_cells": total_feature_cells,
        "missing_cells": total_missing_cells,
        "n_instances": len(records),
    }
    return feature_df, global_stats


def target_stats(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Compute target-channel stats and missing ratios without densifying ragged data."""
    running: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "n_obs": 0.0,
            "n_missing": 0.0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": np.inf,
            "max": -np.inf,
            "n_instances": 0.0,
            "n_all_missing_instances": 0.0,
        }
    )

    total_target_cells = 0
    total_missing_cells = 0

    for r in records:
        y = np.asarray(r["target"], dtype=np.float64)
        if y.ndim == 0:
            y = y.reshape(1, 1)
        elif y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim == 2:
            pass
        else:
            y = y.reshape(y.shape[0], -1)

        n_timesteps, n_targets = y.shape
        cols = list(r["metadata"].get("target_columns", []))
        if len(cols) != n_targets:
            cols = [f"target_{i}" for i in range(n_targets)]

        for col_idx, col_name in enumerate(cols):
            values = y[:, col_idx]
            missing_mask = np.isnan(values)
            n_missing = int(missing_mask.sum())
            n_total = int(values.size)
            n_valid = n_total - n_missing
            valid_values = values[~missing_mask]

            stats = running[col_name]
            stats["n_instances"] += 1
            stats["n_obs"] += n_total
            stats["n_missing"] += n_missing
            if n_missing == n_total:
                stats["n_all_missing_instances"] += 1
            if n_valid > 0:
                stats["sum"] += float(valid_values.sum())
                stats["sum_sq"] += float((valid_values**2).sum())
                stats["min"] = min(stats["min"], float(valid_values.min()))
                stats["max"] = max(stats["max"], float(valid_values.max()))

        total_target_cells += int(n_timesteps * n_targets)
        total_missing_cells += int(np.isnan(y).sum())

    rows = []
    for target_name in sorted(running):
        s = running[target_name]
        n_obs = int(s["n_obs"])
        n_missing = int(s["n_missing"])
        n_valid = n_obs - n_missing
        mean = np.nan
        std = np.nan
        if n_valid > 0:
            mean = s["sum"] / n_valid
            var = max((s["sum_sq"] / n_valid) - (mean**2), 0.0)
            std = var**0.5
        rows.append(
            {
                "TARGET": target_name,
                "OBS": n_obs,
                "MISSING": n_missing,
                "MISSING_%": round(100.0 * n_missing / n_obs, 3) if n_obs else np.nan,
                "MEAN": round(float(mean), 6) if np.isfinite(mean) else np.nan,
                "STD": round(float(std), 6) if np.isfinite(std) else np.nan,
                "MIN": round(float(s["min"]), 6) if np.isfinite(s["min"]) else np.nan,
                "MAX": round(float(s["max"]), 6) if np.isfinite(s["max"]) else np.nan,
                "ALL_MISSING_INSTANCES": int(s["n_all_missing_instances"]),
            }
        )
    target_df = pd.DataFrame(rows)
    global_stats = {
        "target_cells": total_target_cells,
        "missing_cells": total_missing_cells,
        "n_instances": len(records),
    }
    return target_df, global_stats


def ragged_summary(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    """Summarize variable temporal lengths to show ragged behavior."""
    rows = []
    example_lines = []
    for split in ("train", "val", "test"):
        split_records = [r for r in records if r["split"] == split]
        lengths = [int(r["features"].shape[0]) for r in split_records]
        if not lengths:
            continue
        rows.append(
            {
                "SPLIT": split,
                "INSTANCES": len(lengths),
                "MIN_LEN": int(np.min(lengths)),
                "P50_LEN": int(np.median(lengths)),
                "MAX_LEN": int(np.max(lengths)),
                "UNIQUE_LENGTHS": int(len(set(lengths))),
            }
        )
        for r in split_records:
            x = np.asarray(r["features"])
            y = np.asarray(r["target"])
            unit_name = str(r["metadata"]["unit_name"])
            example_lines.append(
                f"  - {split:<5} | {unit_name:<35} | "
                f"features shape={tuple(x.shape)}, target shape={tuple(y.shape)}"
            )
    return pd.DataFrame(rows), example_lines


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    """Run 3W loading tutorial and print structured analysis output."""
    args = parse_args()

    resolved_data_dir = resolve_data_dir(args.data_dir)
    (
        resolved_data_dir,
        resolved_folds_file,
        attempted_folds_paths,
        expected_folds_path,
    ) = resolve_threew_layout(resolved_data_dir, args.folds_file)
    auto_download_enabled = not args.no_download
    if not expected_folds_path.exists() and not auto_download_enabled:
        attempted_paths_rendered = "\n".join(
            f"  - {str(path)}" for path in attempted_folds_paths
        )
        raise FileNotFoundError(
            "3W folds file not found.\n"
            f"Expected one of:\n{attempted_paths_rendered}\n\n"
            "Fix by either:\n"
            "  1) passing --data-dir to the 3W root that contains "
            "`threew_dataset/`, or\n"
            "  2) passing --folds-file relative to that data dir.\n"
            "Example:\n"
            "  python tutorials/datasources/threew_loader.py "
            "--data-dir /path/to/threew --folds-file "
            "threew_dataset/folds/folds_clf_02.csv"
        )

    loader = ThreeWLoader(
        data_dir=resolved_data_dir,
        data_name="threew",
        task_mode="anomaly_detection",
        folds_file=resolved_folds_file,
        validation_fold=args.validation_fold,
        test_fold=args.test_fold,
        include_ova=args.include_ova,
        include_simulated_train=not args.exclude_simulated_train,
        export_event_class=True,
        download=auto_download_enabled,
    )
    loader.load_data()
    container = loader.get_data()
    meta_data = loader.get_meta_data()

    records = flatten_records(container)
    if not records:
        raise RuntimeError(
            "ThreeWLoader returned zero instances; verify data layout, folds_file, "
            "and that instance parquet files exist under dataset/ or threew_dataset/."
        )
    class_df = class_distribution(records)
    source_df = source_comparison(records)
    feat_df, global_feature_stats = feature_stats(records)
    target_df, global_target_stats = target_stats(records)
    ragged_df, ragged_examples = ragged_summary(records)

    print_section("3W TUTORIAL - LOADER CONFIG")
    print(f"data_dir: {loader.data_path}")
    print(f"folds_file: {resolved_folds_file}")
    print(
        "splits: "
        f"val_fold={args.validation_fold}, test_fold={args.test_fold}, "
        f"include_ova={args.include_ova}, "
        f"include_simulated_train={not args.exclude_simulated_train}, "
        f"auto_download={auto_download_enabled}"
    )

    print_section("LOADER METADATA")
    for line in summarize_loader_metadata(meta_data):
        print(line)

    print_section("UNIT METADATA EXAMPLES (ONE PER SPLIT)")
    for split, example in unit_metadata_examples(container).items():
        print(f"{split}:")
        print(f"  {example}")

    print_section("CURRENT TASK AND AVAILABLE TASKS")
    print(f"current_task: {loader.task_mode}")
    available_task_fn = getattr(loader, "get_available_task_modes", None)
    if callable(available_task_fn):
        print(f"available_tasks: {available_task_fn()}")
    else:
        print("available_tasks: [anomaly_detection]")

    print_section("BASELINE-STYLE CLASS DISTRIBUTION (INSTANCES)")
    print(
        class_df.to_string(index=False) if not class_df.empty else "No instances found."
    )

    print_section("REAL vs SIMULATED vs HAND_LABELED (BY CLASS)")
    print(
        source_df.to_string(index=False)
        if not source_df.empty
        else "No source statistics available."
    )

    print_section("FEATURE-LEVEL STATISTICS AND MISSING RATIOS")
    print(feat_df.to_string(index=False) if not feat_df.empty else "No feature data.")
    if global_feature_stats["feature_cells"] > 0:
        missing_pct = (
            100.0
            * global_feature_stats["missing_cells"]
            / global_feature_stats["feature_cells"]
        )
        print()
        print(
            "Global feature missing ratio: "
            f"{global_feature_stats['missing_cells']} / "
            f"{global_feature_stats['feature_cells']} ({missing_pct:.3f}%)"
        )

    print_section("TARGET-LEVEL STATISTICS AND MISSING RATIOS")
    print(
        target_df.to_string(index=False) if not target_df.empty else "No target data."
    )
    if global_target_stats["target_cells"] > 0:
        target_missing_pct = (
            100.0
            * global_target_stats["missing_cells"]
            / global_target_stats["target_cells"]
        )
        print()
        print(
            "Global target missing ratio: "
            f"{global_target_stats['missing_cells']} / "
            f"{global_target_stats['target_cells']} ({target_missing_pct:.3f}%)"
        )

    print_section("RAGGED DATA DEMONSTRATION (NO MANUAL PADDING/TRUNCATION)")
    print(
        ragged_df.to_string(index=False)
        if not ragged_df.empty
        else "No ragged length statistics available."
    )
    print()
    print("Sample raw per-unit tensor shapes (length varies naturally by instance):")
    if args.max_ragged_examples is not None and args.max_ragged_examples <= 0:
        raise ValueError("--max-ragged-examples must be a positive integer.")
    ragged_examples_to_print = (
        ragged_examples
        if args.max_ragged_examples is None
        else ragged_examples[: args.max_ragged_examples]
    )
    for line in ragged_examples_to_print:
        print(line)
    if args.max_ragged_examples is None:
        print()
        print(
            "Note: all units are shown by default. To print fewer, use "
            "--max-ragged-examples N."
        )

    print_section("FRAMEWORK TAKEAWAYS")
    print(
        "- ThreeWLoader returns split payloads as list-per-instance, preserving native lengths."
    )
    print(
        "- PICID consumes ragged unit lists directly, so no preprocessing padding/truncation is required."
    )
    print(
        "- Metadata carries class labels and source names, enabling baseline analyses "
        "without custom dataset plumbing."
    )
    if "unit_names" in meta_data:
        print(
            "- Loader metadata includes split-wise unit_names/unit_ids/class_labels for "
            "downstream auditability."
        )


if __name__ == "__main__":
    main()
