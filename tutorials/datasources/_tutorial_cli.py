"""Shared CLI flags and registry for datasource tutorial scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

_DEFAULT_DATA_DIR_HELP = (
    "Override the PICID datasets root (``paths.data_dir``). When omitted, "
    "defaults follow ``configs/paths/default.yaml`` and the datasource YAML."
)


def add_data_dir_argument(
    parser: argparse.ArgumentParser,
    *,
    help: str | None = None,
) -> None:
    """Register ``--data-dir`` (optional override of ``paths.data_dir``)."""
    parser.add_argument(
        "--data-dir",
        default=None,
        help=help or _DEFAULT_DATA_DIR_HELP,
    )


def add_no_download_argument(
    parser: argparse.ArgumentParser,
    *,
    help: str | None = None,
) -> None:
    """Register ``--no-download`` for loaders that support disabling fetch."""
    parser.add_argument(
        "--no-download",
        action="store_true",
        help=help
        or "Do not download missing dataset files when the loader supports it.",
    )


def normalize_data_dir_override(value: str | None) -> str | None:
    """Return an absolute expanded path suitable for Hydra overrides, or None."""
    if value is None or not str(value).strip():
        return None
    return str(Path(value).expanduser().resolve())


# Tutorial entry points checked by ``test/tutorials/test_datasource_tutorial_contract.py``.
STANDARD_DATASOURCE_TUTORIAL_SCRIPTS: tuple[str, ...] = (
    "airbus_helicopter_loader.py",
    "concepts_n_cmapss_ds02_loader.py",
    "hsf15_accumulator_loader.py",
    "hsf15_component_loader.py",
    "hsf15_cooler_loader.py",
    "hsf15_pump_loader.py",
    "hsf15_valve_loader.py",
    "nb14_bosello_loader.py",
    "phme20_loader.py",
    "pronostia_loader.py",
    "threew_loader.py",
    "unibo21_bosello_loader.py",
    "xjtu_sy_loader.py",
)

# Subset whose loaders expose a meaningful automatic download toggle.
TUTORIAL_SCRIPTS_WITH_NO_DOWNLOAD: frozenset[str] = frozenset(
    {
        "airbus_helicopter_loader.py",
        "pronostia_loader.py",
        "threew_loader.py",
    }
)
