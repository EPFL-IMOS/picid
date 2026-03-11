#!/usr/bin/env python3
"""
Pseudo-run: load data + run transforms only (no training).

Use this script to see loading and transform pipeline behaviour in action,
including the MetadataManifest and how new functionality is integrated.

Inspired by picid/run.py; runs only:
  - datasource load + split
  - transform manager + preprocessor.pipeline()

Usage (from project root; use `uv run` so the venv is used):

  # PHME20 datasource + its transforms
  uv run python pseudo_run.py experiment=phme20/prognostics/raw/linear_regression

  # UNIBO datasource + battery/unibo transforms (override transforms)
  uv run python pseudo_run.py experiment=unibo/base transforms=battery/unibo/combined_fit_predict

  # Disable cache to always run load + transforms
  uv run python pseudo_run.py experiment=phme20/prognostics/base transforms=phme20/normalize_feature_target cache.use_cache_after_loading=false cache.use_cache_after_transfroms=false

Requires: PROJECT_ROOT set (or run from project root; script sets it if missing).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on path and set PROJECT_ROOT for config resolution
_ROOT = Path(__file__).resolve().parent
if _ROOT not in sys.path:
    sys.path.insert(0, str(_ROOT))
if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = str(_ROOT)

import hydra
from omegaconf import DictConfig, OmegaConf

# Resolvers used by configs (same as picid/run.py) so Hydra can resolve e.g. hydra.run.dir
OmegaConf.register_new_resolver("flat", lambda s: s.replace("/", "+"))

from picid.data.preprocessing.preprocessor import PreProcessor
from picid.transforms.base.transform_manager import ConfigTransformManager


def _print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _print_manifest(container) -> None:
    """Print MetadataManifest entries for inspection."""
    manifest = getattr(container, "manifest", None)
    if manifest is None:
        print("  (no manifest on container)")
        return
    n = len(manifest)
    print(f"  Manifest has {n} entries (one per transform step):")
    for i, entry in enumerate(manifest):
        print(f"    [{i+1}] step_id={entry.step_id!r} category={entry.category!r}")
        print(f"        key={entry.key!r} split={entry.split!r}")
        if entry.payload:
            for k, v in entry.payload.items():
                print(f"        payload.{k}={v!r}")
    if n == 0:
        print("  (no entries yet)")


def run_load_and_transforms(cfg: DictConfig) -> None:
    """Run only: load datasource, build transforms, run preprocessor.pipeline()."""
    # Resolve paths so datasource and cache can find dirs
    for k, d in cfg.get("paths", {}).items():
        if isinstance(d, str) and "${" not in d:
            path = Path(d).expanduser().resolve()
            if not path.exists() and k in ("data_dir", "cache_path", "artifacts_dir"):
                path.mkdir(parents=True, exist_ok=True)
            print(f"  paths.{k}: {path}")

    _print_section("1. Instantiate datasource")
    print(f"  _target_: {cfg.datasource.get('_target_', '?')}")
    dataset_source = hydra.utils.instantiate(cfg.datasource)
    print(f"  -> {type(dataset_source).__name__}")

    _print_section("2. Instantiate transform manager")
    transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)
    names = transforms_manager.get_transform_names()
    print(f"  Transform names: {names}")

    _print_section("3. Preprocessor mode + pipeline")
    split_mode = dataset_source.get_split_mode()
    print(f"  split mode: {split_mode!r}")

    preprocessor = PreProcessor(
        datasource=dataset_source, transforms=transforms_manager
    )

    data_cache_path = (
        cfg.paths.cache_path
        if (cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms)
        else None
    )
    data_library = (
        Path(cfg.paths.root_dir) / "picid/data/datasources" if data_cache_path else None
    )
    transform_library = (
        Path(cfg.paths.root_dir) / "picid/transforms" if data_cache_path else None
    )
    cache_preprocessed = cfg.cache.use_cache_after_transfroms

    print(
        f"  cache: data_cache_path={data_cache_path!s}, cache_preprocessed={cache_preprocessed}"
    )
    preprocessor.pipeline(
        data_cache_path=data_cache_path,
        data_library_part_path=data_library,
        transform_library_part_path=transform_library,
        cache_preprocessed=cache_preprocessed,
    )

    _print_section("4. Result: container keys and manifest")
    data = preprocessor.data
    if data is None:
        print("  preprocessor.data is None")
        return
    split_dict = data.to_split_dict() if hasattr(data, "to_split_dict") else {}
    for split, keys in split_dict.items() if isinstance(split_dict, dict) else []:
        print(
            f"  split {split!r}: keys {list(keys.keys()) if isinstance(keys, dict) else keys}"
        )
        if isinstance(keys, dict):
            for k, v in keys.items():
                if hasattr(v, "shape"):
                    print(f"    {k}: shape={v.shape}")
                elif isinstance(v, list) and len(v) and hasattr(v[0], "shape"):
                    print(f"    {k}: list of {len(v)} arrays, first shape={v[0].shape}")
                else:
                    print(
                        f"    {k}: type={type(v).__name__} len={len(v) if hasattr(v, '__len__') else '?'}"
                    )

    print("\n  --- MetadataManifest ---")
    _print_manifest(data)

    _print_section("Done (no training)")
    print()


@hydra.main(version_base="1.3", config_path="configs", config_name="run.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point: compose config and run load + transforms only."""
    print("Pseudo-run: load + transforms only (no training)")
    try:
        from hydra.core.global_hydra import HydraConfig

        overrides = HydraConfig.get().runtime.choices
        print("Config choices (e.g. experiment, datasource):", dict(overrides))
    except Exception:
        pass
    run_load_and_transforms(cfg)


if __name__ == "__main__":
    main()
