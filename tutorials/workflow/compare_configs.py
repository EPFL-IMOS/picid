#!/usr/bin/env python3
"""Tutorial: Compare configs between two experiments.

When metrics differ between runs, compare resolved configs to find what changed.

Canonical script: scripts/reproducibility/compare_configs.py

Run from project root:
    uv run python tutorials/workflow/compare_configs.py [config_a.yaml] [config_b.yaml]

If no args: uses two example configs from a recent run (if available).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent)
)

# Try to use picid.utils.config_diff if it exists
try:
    from picid.utils.config_diff import diff_configs
except ImportError:
    # Fallback: minimal inline implementation
    def _flatten(cfg, prefix=""):
        out = {}
        for k, v in cfg.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flatten(v, key))
            else:
                out[key] = v
        return out

    def diff_configs(path_a: str, path_b: str) -> dict:
        from omegaconf import OmegaConf

        a = OmegaConf.load(path_a)
        b = OmegaConf.load(path_b)
        fa = _flatten(OmegaConf.to_container(a, resolve=True))
        fb = _flatten(OmegaConf.to_container(b, resolve=True))
        return {
            "only_in_first": [k for k in fa if k not in fb],
            "only_in_second": [k for k in fb if k not in fa],
            "different_values": [
                (k, fa[k], fb[k]) for k in fa if k in fb and fa[k] != fb[k]
            ],
        }


def main():
    print("=" * 60)
    print("Tutorial: Compare Configs Between Experiments")
    print("=" * 60)

    if len(sys.argv) >= 3:
        path_a, path_b = sys.argv[1], sys.argv[2]
    else:
        # Find two configs from artifacts
        artifacts = PROJECT_ROOT / "artifacts"
        configs = list(artifacts.rglob("config_resolved.yaml"))[:2]
        if len(configs) < 2:
            print(
                "\nUsage: uv run python tutorials/workflow/compare_configs.py <config_a.yaml> <config_b.yaml>"
            )
            print(
                "Or:    uv run python scripts/reproducibility/compare_configs.py <config_a.yaml> <config_b.yaml>"
            )
            print("Or run an experiment first to generate configs.")
            sys.exit(1)
        path_a, path_b = str(configs[0]), str(configs[1])
        print(f"\nUsing: {path_a}")
        print(f"   vs: {path_b}")

    result = diff_configs(path_a, path_b)
    print("\nDifferences:")
    for k in result["only_in_first"]:
        print(f"  - {k} (only in first)")
    for k in result["only_in_second"]:
        print(f"  + {k} (only in second)")
    for k, v1, v2 in result["different_values"]:
        print(f"  ~ {k}: {v1!r} -> {v2!r}")
    if not (
        result["only_in_first"]
        or result["only_in_second"]
        or result["different_values"]
    ):
        print("  (none - configs are identical)")
    print("\nDone.")


if __name__ == "__main__":
    main()
