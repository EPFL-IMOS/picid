#!/usr/bin/env python3
"""Tutorial: compose XJTU-SY datasource config, load, and print a standard report."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
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


def _operating_condition_id(unit_name: str | None) -> str | None:
    """Map ``Bearing 1_3``-style names to operating-condition key ``1``."""
    if unit_name is None:
        return None
    s = str(unit_name).replace("Bearing", "").strip()
    if "_" not in s:
        return None
    head = s.split("_", 1)[0]
    return head if head.isdigit() else None


def format_operating_condition_coverage(container: Any) -> str:
    """Per-split counts of units under each XJTU-SY operating condition (1/2/3)."""
    lines: list[str] = ["OPERATING CONDITION COVERAGE"]
    records = flatten_records(container)
    by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    for r in records:
        sp = str(r.get("split", ""))
        if sp not in by_split:
            continue
        oc = _operating_condition_id(r.get("metadata", {}).get("unit_name"))
        if oc is not None:
            by_split[sp][oc] += 1
    for sp in ("train", "val", "test"):
        c = by_split[sp]
        parts = [f"OC{k}={c.get(str(k), 0)}" for k in (1, 2, 3)]
        lines.append(f"  {sp}: " + ", ".join(parts))
    return "\n".join(lines)


def format_split_protocol_context(loader: Any, container: Any) -> str:
    """Human-readable split protocol lines required by tutorial contract tests."""
    split_mode = getattr(loader, "split_mode", None)
    fold_id = getattr(loader, "fold_id", None)
    lines = [
        f"SPLIT MODE: {split_mode}",
        f"FOLD ID: {fold_id}",
        "",
        format_operating_condition_coverage(container),
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compose XJTU-SY datasource config via Hydra, load via PHMD, and print "
            "a standard datasource tutorial report."
        )
    )
    add_data_dir_argument(
        parser,
        help=(
            "Override datasets root; set as Hydra ``paths.data_dir`` (PHMD cache uses "
            "``${paths.data_dir}/phmd_cache`` per configs/datasource/xjtu_sy.yaml)."
        ),
    )
    return parser.parse_args()


def _compose_xjtu_sy_config(*, paths_data_dir: str | None = None) -> DictConfig:
    overrides = [
        "datasource=xjtu_sy",
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
    cfg_path = _CONFIG_DIR / "datasource" / "xjtu_sy.yaml"
    fold = OmegaConf.select(ds, "fold")
    use_ragged = OmegaConf.select(ds, "use_ragged")
    aux = OmegaConf.select(ds, "auxiliary_tasks")
    aux_list = OmegaConf.to_container(aux, resolve=True) if aux is not None else []
    return "\n".join(
        [
            f"config_file: {cfg_path}",
            f"data_name: {getattr(loader, 'data_name', '')}",
            f"task_mode: {getattr(loader, 'task_mode', '')}",
            f"fold (PHMD): {fold!r}",
            f"split_mode: {getattr(loader, 'split_mode', '')}",
            f"fold_id (CV protocol): {getattr(loader, 'fold_id', '')!r}",
            f"use_ragged: {use_ragged!r}",
            f"auxiliary_tasks: {aux_list!r}",
        ]
    )


def _build_dataset_notes(loader: Any, container: Any) -> str:
    lines = [
        "XJTU-SY uses PHMD transport with optional paper-style train/val/test tables.",
        "",
        "split_mode selects the evaluation family: "
        "'in_domain' runs 5-fold CV with all three operating conditions in each split; "
        "'domain_shift' holds out condition 3 for test while train/val use conditions 1–2.",
        f"Active protocol: split_mode={getattr(loader, 'split_mode', None)!r}, "
        f"fold_id={getattr(loader, 'fold_id', None)!r} "
        "(see split_assignments_by_mode in configs/datasource/xjtu_sy.yaml).",
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
    """Compose xjtu_sy datasource config, load, and print reports."""
    args = parse_args()
    cfg = _compose_xjtu_sy_config(
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
                "XJTU_SYLoader returned no features; check PHMD cache and config."
            )

        print("XJTU-SY DATASOURCE TUTORIAL")
        _print_banner_section(
            "SPLIT PROTOCOL CONTEXT",
            format_split_protocol_context(loader, container),
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
