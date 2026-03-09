#!/usr/bin/env python3
"""Compare two resolved config YAML files and print key-level differences.

Usage::

    uv run python scripts/reproducibility/compare_configs.py <config_a.yaml> <config_b.yaml>

Symbols:
  ~  key present in both but values differ
  -  key only in config A
  +  key only in config B
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from picid.utils.config_diff import diff_configs  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: compare_configs.py <config_a.yaml> <config_b.yaml>", file=sys.stderr)
        return 1

    result = diff_configs(sys.argv[1], sys.argv[2])

    for k, v1, v2 in result["different_values"]:
        print(f"~ {k}: {v1!r} -> {v2!r}")
    for k in result["only_in_first"]:
        print(f"- {k}  (only in first)")
    for k in result["only_in_second"]:
        print(f"+ {k}  (only in second)")

    total = len(result["different_values"]) + len(result["only_in_first"]) + len(result["only_in_second"])
    if total == 0:
        print("Configs are identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
