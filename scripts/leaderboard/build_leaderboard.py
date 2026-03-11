#!/usr/bin/env python3
"""Build leaderboard_data.json from leaderboard_links.yaml.

Reads docs/leaderboard/leaderboard_links.yaml and writes
docs/leaderboard/leaderboard_data.json for the static leaderboard page.

Usage::

    uv run python scripts/leaderboard/build_leaderboard.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def build_leaderboard_data(links_yaml: Path) -> dict:
    """Read leaderboard_links.yaml and return structured leaderboard data."""
    raw = yaml.safe_load(links_yaml.read_text()) or {}
    datasets: list[dict] = []

    for dataset_name, entries in raw.items():
        if not entries:
            continue
        rows = []
        for e in entries:
            rows.append(
                {
                    "experiment_key": e.get("experiment_key", ""),
                    "submitter": e.get("submitter", ""),
                    "timestamp": e.get("timestamp", ""),
                    "gdrive_link": e.get("gdrive_link", ""),
                    "metrics": e.get("metrics", {}),
                    "notes": e.get("notes", ""),
                }
            )
        # Sort by first metric value ascending (lower = better for error metrics)
        if rows and rows[0]["metrics"]:
            first_metric = next(iter(rows[0]["metrics"]))
            rows.sort(key=lambda r: r["metrics"].get(first_metric, float("inf")))

        datasets.append({"dataset": dataset_name, "rows": rows})

    return {"datasets": datasets}


def main(links_yaml: Path | None = None, output_dir: Path | None = None) -> None:
    links_yaml = links_yaml or (PROJECT_ROOT / "docs" / "leaderboard" / "leaderboard_links.yaml")
    output_dir = output_dir or (PROJECT_ROOT / "docs" / "leaderboard")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = build_leaderboard_data(links_yaml)
    out_path = output_dir / "leaderboard_data.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    total = sum(len(d["rows"]) for d in data["datasets"])
    print(f"Wrote {out_path} ({total} submission(s) across {len(data['datasets'])} dataset(s))")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build leaderboard_data.json for the docs site.")
    p.add_argument("--links", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    main(links_yaml=args.links, output_dir=args.output)
