#!/usr/bin/env python3
"""Crawl configs/ and output config_map.json for the Experiment Generator UI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def crawl_config_group(config_dir: Path, group_name: str) -> dict:
    """Recursively crawl a config group directory.

    Returns a nested dict: dirs map to subdicts, .yaml files map to option
    names (stem). Files or dirs starting with '_' are skipped.
    """
    result: dict = {}
    if not config_dir.exists():
        return result

    for p in sorted(config_dir.iterdir()):
        if p.name.startswith("_"):
            continue
        if p.is_dir():
            sub = crawl_config_group(p, group_name)
            if sub:
                result[p.name] = sub
        elif p.suffix in (".yaml", ".yml"):
            result[p.stem] = None  # leaf option
    return result


def build_config_map(configs_dir: Path) -> dict:
    """Build the full config map from configs/ root."""
    config_map: dict = {}

    exp_dir = configs_dir / "experiment"
    if exp_dir.exists():
        config_map["experiment"] = crawl_config_group(exp_dir, "experiment")

    paths_dir = configs_dir / "paths"
    if paths_dir.exists():
        opts = [p.stem for p in sorted(paths_dir.glob("*.yaml")) if not p.name.startswith("_")]
        config_map["paths"] = opts

    debug_dir = configs_dir / "debug"
    if debug_dir.exists():
        opts = [p.stem for p in sorted(debug_dir.glob("*.yaml")) if not p.name.startswith("_")]
        config_map["debug"] = opts

    return config_map


def _flatten_exp(tree: dict, prefix: str) -> list[str]:
    """Flatten nested experiment tree to keys like 'phme20/prognostics/raw/xgboost'."""
    keys: list[str] = []
    for k, v in tree.items():
        path = f"{prefix}/{k}" if prefix else k
        if v is None:
            keys.append(path)
        elif isinstance(v, dict):
            keys.extend(_flatten_exp(v, path))
    return keys


def main(configs_dir: Path | None = None, output_dir: Path | None = None) -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    configs_dir = configs_dir or (project_root / "configs")
    output_dir = output_dir or (project_root / "docs" / "experiment-generator")

    output_dir.mkdir(parents=True, exist_ok=True)
    config_map = build_config_map(configs_dir)

    if "experiment" in config_map:
        config_map["experiment_keys"] = sorted(_flatten_exp(config_map["experiment"], ""))

    out_path = output_dir / "config_map.json"
    out_path.write_text(json.dumps(config_map, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate config_map.json for Experiment Generator")
    parser.add_argument("--configs", type=Path, default=None, help="Configs directory")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()
    main(configs_dir=args.configs, output_dir=args.output)
