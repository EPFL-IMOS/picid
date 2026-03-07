#!/usr/bin/env python3
"""Reproduce an experiment from a run's saved config.

Loads config from run_dir/config_resolved.yaml and runs the pipeline with output
to run_dir/reproduce_<timestamp>. Use when you want to reproduce exactly from
the run's config (e.g. different library version comparison) rather than from
repo overrides.

Usage:
    uv run python scripts/reproducibility/reproduce_from_run.py <run_output_dir>
    uv run python scripts/reproducibility/reproduce_from_run.py $PROJECT_ROOT/artifacts/.../2026-03-06_22-42-27

Requires PROJECT_ROOT env var.
"""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/reproducibility/reproduce_from_run.py <run_output_dir>", file=sys.stderr)
        return 1

    run_dir = Path(sys.argv[1]).expanduser().resolve()
    config_path = run_dir / "config_resolved.yaml"

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    if not os.environ.get("PROJECT_ROOT"):
        print("PROJECT_ROOT env var required.", file=sys.stderr)
        return 1

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    repro_dir = run_dir / f"reproduce_{ts}"
    repro_dir.mkdir(parents=True, exist_ok=True)

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    # Override output paths for the reproduce run
    OmegaConf.update(cfg, "paths.output_dir", str(repro_dir))
    OmegaConf.update(cfg, "paths.ckpt_dir", str(repro_dir / "checkpoints"))
    OmegaConf.update(cfg, "paths.log_dir", str(repro_dir / "logs"))
    OmegaConf.update(cfg, "paths.plot_dir", str(repro_dir / "plots"))
    OmegaConf.update(cfg, "paths.eval_details", str(repro_dir / "eval_details"))
    OmegaConf.update(cfg, "paths.model_workdir", str(repro_dir / "model_workdir"))
    OmegaConf.update(cfg, "paths.model_cache_dir", str(repro_dir / "model_cache_dir"))
    OmegaConf.update(cfg, "paths.root_dir", os.environ["PROJECT_ROOT"])

    from picid.run import extras, run

    extras(cfg)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
