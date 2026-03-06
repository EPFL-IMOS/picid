#!/usr/bin/env python3
"""Build a submission message with full reproducibility info.

Usage::

    uv run python scripts/publish/build_submission_message.py <run_dir> [--gdrive-link URL] [--output FILE]

Reads REPRODUCE.md and run_metadata.yaml from <run_dir>, appends an optional
Google Drive link, and prints a self-contained submission message to stdout
(or writes it to --output).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_submission_message(run_dir: Path, gdrive_link: str = "") -> str:
    """Return a markdown submission message for the given run dir."""
    run_dir = Path(run_dir).resolve()
    parts: list[str] = []

    if (run_dir / "REPRODUCE.md").exists():
        parts.append((run_dir / "REPRODUCE.md").read_text())

    if (run_dir / "run_metadata.yaml").exists():
        parts.append(
            "## Run metadata\n```yaml\n"
            + (run_dir / "run_metadata.yaml").read_text().rstrip()
            + "\n```"
        )

    if gdrive_link:
        parts.append(f"## Artifacts (Google Drive)\n{gdrive_link}")

    return "\n\n---\n\n".join(parts) if parts else ""


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a submission message from a run dir.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", type=Path, help="Path to the run output directory")
    p.add_argument("--gdrive-link", default="", metavar="URL", help="Shareable Google Drive link")
    p.add_argument("--output", "-o", type=Path, default=None, help="Write to file instead of stdout")
    args = p.parse_args()

    msg = build_submission_message(args.run_dir, args.gdrive_link)

    if args.output:
        args.output.write_text(msg, encoding="utf-8")
    else:
        print(msg, end="" if not msg else "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
