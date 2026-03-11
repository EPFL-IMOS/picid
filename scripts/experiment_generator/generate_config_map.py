#!/usr/bin/env python3
"""Generate config_map.json for the Experiment Generator web UI.

Delegates to picid.cli.generate_web. Run before mkdocs build.
"""
from __future__ import annotations

from pathlib import Path


def main(output_dir: Path | None = None) -> None:
    """Generate config_map.json (delegates to picid.cli.generate_web)."""
    from picid.cli.generate_web import main as generate_main

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = output_dir or (project_root / "docs" / "experiment-generator")
    generate_main(output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate config_map.json for Experiment Generator"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()
    main(output_dir=args.output)
