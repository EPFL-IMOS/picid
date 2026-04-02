#!/usr/bin/env python3
"""Tutorial: compose PHME20 datasource config, load, and print a standard report."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tutorials.datasources._tutorial_cli import (  # noqa: E402
    add_data_dir_argument,
    normalize_data_dir_override,
)
from tutorials.datasources._loader_introspection import (  # noqa: E402
    flatten_records,
    ragged_summary,
    summarize_loader_metadata,
    unit_metadata_examples,
)
from tutorials.datasources._standard_report import (  # noqa: E402
    build_standard_report,
    compute_feature_stats,
    compute_split_overview,
    compute_target_stats,
    format_dataset_specific_notes,
)

_SECTION_WIDTH = 88


def _build_loader(cfg: DictConfig) -> Any:
    """Instantiate the composed datasource (patchable in tests)."""
    return instantiate(cfg.datasource)


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


def _print_banner_section(title: str, body: str) -> None:
    sep = "=" * _SECTION_WIDTH
    print()
    print(sep)
    print(title)
    print(sep)
    print(body.rstrip() if body.strip() else "(empty)")


def format_phme20_rul_context(loader: Any, container: Any) -> str:
    """PHME20 task_mode, PHMD fold, RUL vs fault targets, and unit identifier behavior."""
    fold = getattr(loader, "fold", None)
    task = getattr(loader, "task_mode", None)
    lines = [
        f"TASK MODE: {task}",
        f"PHMD FOLD (predefined split index): {fold}",
        "",
        "RUL target context: task_mode='rul' maps the PHMD per-timestep target column "
        "named 'rul' into SplitDatasetContainer.target. For fault detection, set "
        "task_mode='fault' in configs/datasource/phme20.yaml so the target column follows "
        "the fault label instead.",
        "",
        "PHMD filter: fold selects which predefined train/val/test assignment the PHMD "
        "repository returns for this challenge; it is not an in-loader random split.",
        "",
        "Unit identifiers: rows are grouped by meta_data['identifier']; PHME20Loader "
        "coerces the PHMD unit key to int, stores scalar unit_id, and exposes "
        "unit_name as 'Unit_<id>' for stable string labels.",
    ]
    um = unit_metadata_examples(container)
    lines.extend(["", "Example unit_metadata (first train unit):"])
    lines.append(f"  train: {um.get('train', '')}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compose PHME20 datasource config via Hydra, load via PHMD, and print "
            "a standard datasource tutorial report."
        )
    )
    add_data_dir_argument(
        parser,
        help=(
            "Override datasets root; set as Hydra ``paths.data_dir`` (PHMD cache uses "
            "``${paths.data_dir}/phmd_cache`` per configs/datasource/phme20.yaml)."
        ),
    )
    return parser.parse_args()


def _compose_phme20_config(*, paths_data_dir: str | None = None) -> DictConfig:
    overrides = [
        "datasource=phme20",
        "hydra/job_logging=default",
        "hydra/hydra_logging=default",
        f"paths.root_dir={_REPO_ROOT}",
    ]
    if paths_data_dir:
        overrides.append(f"paths.data_dir={paths_data_dir}")
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=str(_CONFIG_DIR),
    ):
        return compose(
            config_name="run",
            overrides=overrides,
        )


def _format_loader_config_summary(cfg: DictConfig, loader: Any) -> str:
    ds = cfg.datasource
    cfg_path = _CONFIG_DIR / "datasource" / "phme20.yaml"
    fold = OmegaConf.select(ds, "fold")
    aux = OmegaConf.select(ds, "auxiliary_tasks")
    aux_list = OmegaConf.to_container(aux, resolve=True) if aux is not None else []
    cache_dir = OmegaConf.select(ds, "cache_dir")
    return "\n".join(
        [
            f"config_file: {cfg_path}",
            f"data_name: {getattr(loader, 'data_name', '')}",
            f"task_mode: {getattr(loader, 'task_mode', '')}",
            f"fold (PHMD): {fold!r}",
            f"auxiliary_tasks: {aux_list!r}",
            f"cache_dir (config): {cache_dir!r}",
        ]
    )


def _build_dataset_notes(loader: Any, container: Any) -> str:
    lines = [
        "PHME20 uses PHMDMultiSourceLoader (PHME20Loader) with predefined splits.",
        f"task_mode={getattr(loader, 'task_mode', None)!r} selects _get_target_column() "
        "so the assembled target list aligns with that PHMD column.",
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


def main() -> None:
    """Compose phme20 datasource config, load, and print reports."""
    args = parse_args()
    cfg = _compose_phme20_config(
        paths_data_dir=normalize_data_dir_override(args.data_dir),
    )
    try:
        loader = _build_loader(cfg)
        loader.load_data()
        container = loader.get_data()
        meta = loader.get_meta_data()

        features_by_split = _split_branch(container, "features")
        targets_by_split = _split_branch(container, "target")
        if not features_by_split:
            raise RuntimeError(
                "PHME20Loader returned no features; check PHMD cache and config."
            )

        print("PHME20 DATASOURCE TUTORIAL")
        _print_banner_section(
            "PHME20 RUL / SPLIT CONTEXT",
            format_phme20_rul_context(loader, container),
        )

        loader_config_body = _format_loader_config_summary(cfg, loader)
        loader_metadata_body = summarize_loader_metadata(meta)
        split_overview_body = compute_split_overview(
            features_by_split, targets_by_split=targets_by_split
        )
        feature_stats_body = compute_feature_stats(features_by_split)
        target_stats_body = compute_target_stats(targets_by_split)
        dataset_notes_body = format_dataset_specific_notes(
            _build_dataset_notes(loader, container)
        )

        report = build_standard_report(
            loader_config_body=loader_config_body,
            loader_metadata_body=loader_metadata_body,
            split_overview_body=split_overview_body,
            feature_stats_body=feature_stats_body,
            target_stats_body=target_stats_body,
            dataset_specific_notes_body=dataset_notes_body,
        )
        print()
        print(report)
    finally:
        GlobalHydra.instance().clear()


if __name__ == "__main__":
    main()
