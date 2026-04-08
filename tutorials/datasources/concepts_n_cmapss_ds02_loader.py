#!/usr/bin/env python3
"""Tutorial: load Concepts N-CMAPSS DS02 multisource config and inspect the layout."""

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

from picid.data.datasources.base.multi_source_loader import (  # noqa: E402
    MultiSourceLoader,
)

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


def _build_multisource_loader(cfg: DictConfig) -> MultiSourceLoader:
    """Instantiate the composed multisource datasource (patchable in tests)."""
    return instantiate(cfg.datasource)


def _split_branch(container: Any, key: str) -> dict[str, list[Any]]:
    if isinstance(container, dict):
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


def _format_multisource_overview(loader: MultiSourceLoader) -> str:
    aliases = ", ".join(loader.get_source_names())
    lines = [f"source_aliases (ordered): {aliases}"]
    splitter = loader.get_multisource_data_splitter()
    if splitter is None:
        lines.append("multisource_data_splitter: (none)")
    else:
        lines.extend(
            [
                f"sources_train: {list(splitter.sources_train)}",
                f"sources_val: {list(splitter.sources_val)}",
                f"sources_test: {list(splitter.sources_test)}",
            ]
        )
    return "\n".join(lines)


def _format_selected_units(cfg: DictConfig) -> str:
    ds = cfg.datasource
    lines: list[str] = []
    for alias in getattr(ds, "source_list", {}):
        if alias not in ds:
            continue
        sub = ds[alias]
        la = OmegaConf.select(sub, "load_arguments") or {}
        units = OmegaConf.select(la, "units")
        mode = OmegaConf.select(la, "mode")
        n_ds = OmegaConf.select(la, "n_DS")
        ulist = (
            OmegaConf.to_container(units, resolve=True) if units is not None else None
        )
        lines.append(f"{alias}: n_DS={n_ds!r}, mode={mode!r}, units={ulist!r}")
    return "\n".join(lines) if lines else "(no per-source load_arguments in config)"


def _format_requested_concepts(cfg: DictConfig) -> str:
    ds = cfg.datasource
    lines: list[str] = []
    seen: list[list[Any]] = []
    for alias in getattr(ds, "source_list", {}):
        if alias not in ds:
            continue
        sub = ds[alias]
        la = OmegaConf.select(sub, "load_arguments") or {}
        concepts = OmegaConf.select(la, "concepts")
        clist = (
            OmegaConf.to_container(concepts, resolve=True)
            if concepts is not None
            else []
        )
        if clist not in seen:
            seen.append(clist)
        lines.append(f"{alias}: {clist!r}")
    if len(seen) > 1:
        lines.append(
            "(note: concept lists differ across sources; values above are per alias.)"
        )
    elif len(seen) == 1:
        lines.append(f"shared_concepts: {seen[0]!r}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compose Concepts N-CMAPSS DS02 multisource config via Hydra, load, "
            "and print the datasource tutorial report."
        )
    )
    add_data_dir_argument(
        parser,
        help=(
            "Override datasets root; set as Hydra ``paths.data_dir`` (child N-CMAPSS "
            "sources resolve ``path: ${paths.data_dir}/N-CMAPSS``)."
        ),
    )
    return parser.parse_args()


def _compose_ds02_config(*, paths_data_dir: str | None = None) -> DictConfig:
    overrides = [
        "datasource=concepts_n_cmapss_ds02",
        # Avoid hydra_colorlog (optional); tutorial must run in minimal envs.
        "hydra/job_logging=default",
        "hydra/hydra_logging=default",
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


def _candidate_paths_data_dir_overrides(cli_data_dir: str | None) -> list[str | None]:
    """Ordered candidate ``paths.data_dir`` roots for DS02 location fallback."""
    normalized = normalize_data_dir_override(cli_data_dir)
    if normalized:
        return [normalized]

    # Default first: keep existing behavior.
    candidates: list[str | None] = [None]

    # Worktree convenience: if running from `<repo>/.worktrees/<branch>`,
    # also try `<repo>/datasets` where users typically stage heavy datasets once.
    parts = _REPO_ROOT.parts
    if ".worktrees" in parts:
        idx = parts.index(".worktrees")
        if idx >= 1:
            main_repo_root = Path(*parts[:idx])
            candidates.append(str((main_repo_root / "datasets").resolve()))
    return candidates


def _build_dataset_notes(
    loader: MultiSourceLoader,
    container: Any,
) -> str:
    lines = [
        "Concepts N-CMAPSS DS02 multisource preset (Depater et al. style splits).",
        f"Outer data_name={loader.data_name!r}, task_mode={loader.task_mode!r}.",
        "Child sources are separate N_CMAPSSDataSource instances; BySourceSplitter maps "
        "whole sources to train/val/test (no within-unit splitter on this datasource).",
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
    """Compose DS02 multisource datasource config, load, and print reports."""
    args = parse_args()
    cfg: DictConfig | None = None
    loader: MultiSourceLoader | None = None
    last_error: Exception | None = None
    for paths_data_dir in _candidate_paths_data_dir_overrides(args.data_dir):
        cfg = _compose_ds02_config(paths_data_dir=paths_data_dir)
        try:
            loader = _build_multisource_loader(cfg)
            loader.load_data()
            loader.split_data()
            break
        except FileNotFoundError as exc:
            last_error = exc
            loader = None
            continue

    if loader is None or cfg is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to initialize DS02 loader.")

    try:
        container = loader.get_data()
        meta = loader.get_meta_data()

        features_by_split = _split_branch(container, "features")
        targets_by_split = _split_branch(container, "target")
        if not features_by_split:
            raise RuntimeError(
                "MultiSourceLoader returned no features; check datasource composition."
            )

        print("CONCEPTS N-CMAPSS DS02 MULTISOURCE TUTORIAL")
        _print_banner_section(
            "MULTISOURCE OVERVIEW", _format_multisource_overview(loader)
        )
        _print_banner_section("SELECTED UNITS", _format_selected_units(cfg))
        _print_banner_section("REQUESTED CONCEPTS", _format_requested_concepts(cfg))

        cfg_path = _CONFIG_DIR / "datasource" / "concepts_n_cmapss_ds02.yaml"
        loader_config_body = "\n".join(
            [
                f"config_file: {cfg_path}",
                f"data_name: {loader.data_name}",
                f"task_mode: {loader.task_mode}",
                f"source_names: {list(loader.get_source_names())}",
            ]
        )
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
