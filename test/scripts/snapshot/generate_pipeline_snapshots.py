#!/usr/bin/env python3
"""Generate pipeline snapshots for phme20 (real-data tests).

Output: {output_dir}/phme20/{loaded|transformed}/slices_manifest.json and *.npy
  Default output_dir: test/data/fixtures/pipeline_snapshots

Supports: test/pipeline/test_pipeline_phme20_snapshots.py
  - test_loaded_slices_match_saved
  - test_transformed_slices_match_saved

Uses only phme20 real data. Saves slices after load and after transforms;
records preprocessing_time_seconds for the transformed stage. Requires datasets.

Usage:
  uv run python test/scripts/snapshot/generate_pipeline_snapshots.py
  uv run python test/scripts/snapshot/generate_pipeline_snapshots.py --output-dir test/data/fixtures/pipeline_snapshots
  uv run python test/scripts/snapshot/generate_pipeline_snapshots.py --data-dir "$(pwd)/datasets"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root (test/scripts/snapshot/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from hydra.utils import instantiate

from picid.data.preprocessing.preprocessor import PreProcessor
from picid.transforms.base.transform_manager import ConfigTransformManager


# Config pairs: (datasource config path, transforms config path)
CONFIGS = {
    "phme20": (
        "configs/datasource/phme20.yaml",
        "configs/transforms/phme20/normalize_feature_target.yaml",
    ),
}

# Keys per config (must match datasource output; phme20 uses "rul" not "target")
CONFIG_KEYS = {
    "phme20": ["features", "rul"],
}


def load_config(path: str, overrides: dict | None = None) -> OmegaConf:
    """Load YAML config and optionally merge overrides (e.g. paths)."""
    cfg = OmegaConf.load(PROJECT_ROOT / path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def resolve_paths(cfg: OmegaConf, data_dir: Path | None) -> OmegaConf:
    """Resolve ${paths.data_dir} if present."""
    if data_dir is None:
        return cfg
    return OmegaConf.merge(
        cfg,
        OmegaConf.create({"paths": {"data_dir": str(data_dir)}}),
    )


def run_one(
    name: str,
    ds_path: str,
    tf_path: str,
    output_base: Path,
    data_dir: Path | None,
    keys: list[str],
    max_rows_per_unit: int,
) -> bool:
    """Run pipeline for one config; save loaded and transformed slices. Returns True on success."""
    output_base = output_base / name
    loaded_dir = output_base / "loaded"
    transformed_dir = output_base / "transformed"
    loaded_dir.mkdir(parents=True, exist_ok=True)
    transformed_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds_cfg = load_config(ds_path)
        tf_cfg = load_config(tf_path)
        if data_dir:
            ds_cfg = resolve_paths(ds_cfg, data_dir)
        datasource = instantiate(ds_cfg)
        transforms_manager = ConfigTransformManager(transforms_config=tf_cfg)
        preprocessor = PreProcessor(
            datasource=datasource,
            transforms=transforms_manager,
        )

        # Stage 1: load only
        datasource.load_data()
        datasource.split_data()
        container_loaded = datasource.get_data()

        from test.data.fixtures.pipeline_snapshots.slice_utils import (
            container_to_slice_manifest,
        )
        import json

        manifest_loaded = container_to_slice_manifest(
            container_loaded,
            keys=keys,
            out_dir=loaded_dir,
            max_rows_per_unit=max_rows_per_unit,
        )
        (loaded_dir / "slices_manifest.json").write_text(
            json.dumps(manifest_loaded, indent=2)
        )

        # Full pipeline and time transform step
        t0 = time.perf_counter()
        preprocessor.pipeline()
        transform_time = time.perf_counter() - t0

        container_transformed = preprocessor.data
        manifest_transformed = container_to_slice_manifest(
            container_transformed,
            keys=keys,
            out_dir=transformed_dir,
            max_rows_per_unit=max_rows_per_unit,
            preprocessing_time_seconds=round(transform_time, 4),
        )
        (transformed_dir / "slices_manifest.json").write_text(
            json.dumps(manifest_transformed, indent=2)
        )
        print(
            f"  {name}: loaded + transformed ok (transform_time={transform_time:.2f}s)"
        )
        return True
    except Exception as e:
        print(f"  {name}: failed - {e}")
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Generate pipeline snapshots for real-data tests."
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "test/data/fixtures/pipeline_snapshots",
    )
    default_data_dir = (
        PROJECT_ROOT / "datasets" if (PROJECT_ROOT / "datasets").exists() else None
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Override paths.data_dir for configs",
    )
    ap.add_argument("--max-rows", type=int, default=100)
    ap.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        default=None,
        help="Run only this config",
    )
    args = ap.parse_args()

    if getattr(args, "config", None):
        to_run = {args.config: CONFIGS[args.config]}
    else:
        to_run = CONFIGS

    success = 0
    for name, (ds_path, tf_path) in to_run.items():
        keys = CONFIG_KEYS.get(name, ["features", "target"])
        if run_one(
            name,
            ds_path,
            tf_path,
            args.output_dir,
            args.data_dir,
            keys,
            args.max_rows,
        ):
            success += 1

    print(f"Done: {success}/{len(to_run)} configs succeeded.")
    return 0 if success == len(to_run) else 1


if __name__ == "__main__":
    exit(main())
