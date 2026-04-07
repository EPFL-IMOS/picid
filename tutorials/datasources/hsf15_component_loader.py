#!/usr/bin/env python3
"""Tutorial: load HSF15 (PHMD) for a single subsystem component and print a standard report."""

from __future__ import annotations

import argparse
import ast
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

COMPONENT_NUM_CLASSES: dict[str, int] = {
    "accumulator": 4,
    "cooler": 3,
    "pump": 3,
    "valve": 4,
}


def get_hsf15_loader_class() -> type[Any]:
    """Return :class:`HSF15Loader`; isolated for tests via ``monkeypatch``."""
    from picid.data.datasources.phmd_hsf15 import HSF15Loader

    return HSF15Loader


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


def _parse_auxiliary_tasks(raw: str) -> list[Any]:
    s = raw.strip()
    if s.startswith("["):
        try:
            return list(ast.literal_eval(s))
        except (SyntaxError, ValueError):
            return []
    return []


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


def load_hsf15_component_yaml(component: str) -> dict[str, str]:
    """Load ``configs/datasource/hsf15_<component>.yaml`` top-level keys."""
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "configs" / "datasource" / f"hsf15_{component}.yaml"
    return _read_simple_yaml_top_level(path)


def resolve_cache_dir(component: str, paths_data_dir: str) -> str:
    """Resolve PHMD ``cache_dir`` from the component datasource YAML."""
    cfg = load_hsf15_component_yaml(component)
    cache_tmpl = cfg.get("cache_dir", "${paths.data_dir}/phmd_cache")
    resolved = _resolve_template(
        cache_tmpl,
        {
            "paths.data_dir": paths_data_dir,
        },
    )
    return str(Path(resolved).expanduser())


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


def _build_dataset_notes(
    component: str,
    num_classes: int,
    container: Any,
) -> str:
    lines = [
        "HSF15 COMPONENT NOTE",
        "",
        f"task_mode: {component}",
        f"num_classes: {num_classes}",
        "",
        "HSF15Loader reads PHMD-backed folds; task_mode selects the subsystem fault "
        "column (accumulator, cooler, pump, or valve). Features are nested per cycle "
        "(often awkward); this tutorial uses the same split/stat summaries as other "
        "PHM datasource tutorials.",
    ]
    records = flatten_records(container)
    ragged = ragged_summary(records)
    if ragged:
        lines.extend(["", ragged])

    um = unit_metadata_examples(container)
    lines.extend(["", "First unit_metadata example per split:"])
    for sp in ("train", "val", "test"):
        lines.append(f"  {sp}: {um[sp]}")
    return "\n".join(lines)


def parse_wrapper_args() -> argparse.Namespace:
    """Arguments shared by thin per-component wrapper scripts."""
    parser = argparse.ArgumentParser(
        description=(
            "Load HSF15 for a fixed subsystem component and print the standard "
            "datasource tutorial report."
        )
    )
    add_data_dir_argument(
        parser,
        help=(
            "Override datasets root used to resolve ``paths.data_dir`` when expanding "
            "cache_dir in configs/datasource/hsf15_<component>.yaml."
        ),
    )
    return parser.parse_args()


def parse_component_args() -> argparse.Namespace:
    """Arguments for ``hsf15_component_loader.py`` when run directly."""
    parser = argparse.ArgumentParser(
        description=(
            "Load HSF15 with HSF15Loader for one component and print the standard "
            "datasource tutorial report."
        )
    )
    parser.add_argument(
        "--component",
        choices=sorted(COMPONENT_NUM_CLASSES.keys()),
        default="cooler",
        help=(
            "Subsystem task_mode (must match configs/datasource/hsf15_<component>.yaml). "
            "Defaults to 'cooler' so the script is runnable with no extra flags."
        ),
    )
    add_data_dir_argument(
        parser,
        help=(
            "Override datasets root used to resolve ``paths.data_dir`` when expanding "
            "cache_dir in the component YAML."
        ),
    )
    return parser.parse_args()


def main_for_component(component: str, *, data_dir_cli: str | None = None) -> None:
    """Run the tutorial for a single HSF15 component (used by wrappers and tests)."""
    if component not in COMPONENT_NUM_CLASSES:
        raise ValueError(
            f"Unknown component {component!r}; expected one of {COMPONENT_NUM_CLASSES}."
        )

    paths_data_dir = resolve_paths_data_dir(data_dir_cli)
    cache_dir = resolve_cache_dir(component, paths_data_dir)
    cfg = load_hsf15_component_yaml(component)
    fold = int(cfg.get("fold", "0"))
    data_name = cfg.get("data_name", "HSF15")
    auxiliary_tasks = _parse_auxiliary_tasks(cfg.get("auxiliary_tasks", "[]"))

    num_classes = COMPONENT_NUM_CLASSES[component]

    Loader = get_hsf15_loader_class()
    loader = Loader(
        fold=fold,
        data_name=data_name,
        task_mode=component,
        auxiliary_tasks=auxiliary_tasks,
        cache_dir=cache_dir,
    )
    loader.load_data()
    container = loader.get_data()
    meta = loader.get_meta_data()

    features_by_split = _split_branch(container, "features")
    targets_by_split = _split_branch(container, "target")
    if not features_by_split:
        raise RuntimeError(
            "HSF15Loader returned no features; check PHMD cache, fold, and task_mode."
        )

    loader_config_body = "\n".join(
        [
            f"component: {component}",
            f"data_name: {loader.data_name}",
            f"task_mode: {loader.task_mode}",
            f"fold: {loader.fold}",
            f"cache_dir: {loader.cache_dir}",
            f"paths.data_dir (resolved): {paths_data_dir}",
            f"reference num_classes (from tutorial contract): {num_classes}",
        ]
    )
    loader_metadata_body = summarize_loader_metadata(meta)
    split_overview_body = compute_split_overview(
        features_by_split, targets_by_split=targets_by_split
    )
    feature_stats_body = compute_feature_stats(features_by_split)
    target_stats_body = compute_target_stats(targets_by_split)
    dataset_notes_body = format_dataset_specific_notes(
        _build_dataset_notes(component, num_classes, container)
    )

    report = build_standard_report(
        loader_config_body=loader_config_body,
        loader_metadata_body=loader_metadata_body,
        split_overview_body=split_overview_body,
        feature_stats_body=feature_stats_body,
        target_stats_body=target_stats_body,
        dataset_specific_notes_body=dataset_notes_body,
    )

    print("HSF15 TUTORIAL")
    print()
    print(report)


def main() -> None:
    """CLI entry when running ``hsf15_component_loader.py`` directly."""
    args = parse_component_args()
    main_for_component(args.component, data_dir_cli=args.data_dir)


if __name__ == "__main__":
    main()
