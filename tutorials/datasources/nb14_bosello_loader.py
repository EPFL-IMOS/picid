#!/usr/bin/env python3
"""Tutorial: NB14 (NASA randomized) battery data with Bosello-style predefined splits."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutorials.datasources._tutorial_cli import add_data_dir_argument
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


def get_nb14_bosello_loader_class() -> type[Any]:
    """Return :class:`NB14Loader`; patch in tests to avoid real dataset IO."""
    from picid.data.datasources.nb14.loader import NB14Loader

    return NB14Loader


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


def resolve_paths_data_dir(cli_data_dir: str | None) -> str:
    """Resolve ``paths.data_dir`` for interpolating datasource templates."""
    if cli_data_dir:
        return str(Path(cli_data_dir).expanduser())

    repo_root = Path(__file__).resolve().parents[2]
    paths_cfg = _read_simple_yaml_top_level(
        repo_root / "configs" / "paths" / "default.yaml"
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
    return str(Path(paths_data_dir).expanduser())


def load_nb14_bosello_yaml() -> dict[str, str]:
    """Load ``configs/datasource/nb14_bosello.yaml`` top-level keys."""
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "configs" / "datasource" / "nb14_bosello.yaml"
    return _read_simple_yaml_top_level(path)


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


def _build_nb14_bosello_notes(container: Any) -> str:
    """Domain notes aligned with Bosello battery protocol (filtering, ah-RUL)."""
    bullets = [
        "filtering — Train/val/test battery lists in ``picid.data.datasources.nb14.loader`` "
        "omit several RW cells (commented ``train_names`` / ``val_names`` / ``test_names`` "
        "entries) so the PICID split matches the curated NASA subset used in the "
        "Bosello-style benchmark, rather than every channel in the randomized release.",
        "ah-RUL — With ``task_mode: ahRul`` (see ``configs/datasource/nb14_bosello.yaml``), "
        "per-cycle targets are future horizon RUL from ``RulHandler.prepare_y_future`` over "
        "SOH and discharge trajectories (absolute-horizon RUL supervision).",
    ]
    records = flatten_records(container)
    ragged = ragged_summary(records)
    if ragged:
        bullets.append(ragged)
    um = unit_metadata_examples(container)
    tail_lines = ["First unit_metadata example per split:"]
    for sp in ("train", "val", "test"):
        tail_lines.append(f"  {sp}: {um[sp]}")
    body_bullets = format_dataset_specific_notes(bullets)
    return body_bullets + "\n\n" + "\n".join(tail_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load NB14 with NB14Loader (Bosello splits) and print the standard "
            "datasource tutorial report."
        )
    )
    add_data_dir_argument(
        parser,
        help=(
            "Override datasets root used as ``paths.data_dir`` when resolving "
            "``configs/datasource/nb14_bosello.yaml``."
        ),
    )
    return parser.parse_args()


def main(*, data_dir_cli: str | None = None) -> None:
    """Run the tutorial; ``data_dir_cli`` overrides CLI when provided (e.g. tests)."""
    if data_dir_cli is None:
        args = parse_args()
        data_dir_cli = args.data_dir

    paths_data_dir = resolve_paths_data_dir(data_dir_cli)
    cfg = load_nb14_bosello_yaml()
    data_name = cfg.get("data_name", "NB14")
    task_mode = cfg.get("task_mode", "ahRul")

    Loader = get_nb14_bosello_loader_class()
    loader = Loader(
        data_dir=paths_data_dir,
        data_name=data_name,
        task_mode=task_mode,
    )
    loader.load_data()
    container = loader.get_data()
    meta = loader.get_meta_data()

    features_by_split = _split_branch(container, "features")
    targets_by_split = _split_branch(container, "target")
    if not features_by_split:
        raise RuntimeError(
            "NB14Loader returned no features; check data_dir and dataset layout."
        )

    loader_config_body = "\n".join(
        [
            f"data_name: {loader.data_name}",
            f"task_mode: {loader.task_mode}",
            f"data_dir (resolved): {paths_data_dir}",
            f"loader.data_path: {getattr(loader, 'data_path', paths_data_dir)}",
            "config: configs/datasource/nb14_bosello.yaml",
        ]
    )
    loader_metadata_body = summarize_loader_metadata(meta)
    split_overview_body = compute_split_overview(
        features_by_split, targets_by_split=targets_by_split
    )
    feature_stats_body = compute_feature_stats(features_by_split)
    target_stats_body = compute_target_stats(targets_by_split)
    dataset_notes_body = format_dataset_specific_notes(
        _build_nb14_bosello_notes(container)
    )

    report = build_standard_report(
        loader_config_body=loader_config_body,
        loader_metadata_body=loader_metadata_body,
        split_overview_body=split_overview_body,
        feature_stats_body=feature_stats_body,
        target_stats_body=target_stats_body,
        dataset_specific_notes_body=dataset_notes_body,
    )

    print("NB14 BOSSELLO TUTORIAL")
    print()
    print(report)


if __name__ == "__main__":
    main()
