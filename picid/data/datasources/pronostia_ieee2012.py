"""Raw IEEE 2012 PRONOSTIA datasource backed by the challenge folder layout.

The goal of this loader is to expose the Kaggle/IEEE challenge dump with the same
high-level contract used by the existing PHMD-backed Pronostia datasource:

- one ragged unit per bearing
- two vibration channels per record
- an RUL target broadcast across each 2560-sample vibration window

This lets the existing Pronostia preprocessing and evaluation stack work without
special cases while keeping the split protocol explicit in config.
"""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
import re
import subprocess
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd

from picid.data.datasources.base.predefined_split_loader import (
    PredefinedSplitLoaderBase,
)

logger = logging.getLogger(__name__)

# The raw challenge dump names each window file as acc_00001.csv, acc_00002.csv, ...
# We sort on the numeric suffix to preserve the original temporal order.
ACC_FILE_RE = re.compile(r"acc_(\d+)\.csv$")
UNIT_DIR_RE = re.compile(r"^Bearing(\d+_\d+)$")

# The user-facing Kaggle layout exposes only these two subsets in scope for v1.
LEARNING_SET = "Learning_set"
FULL_TEST_SET = "Full_Test_Set"
SUPPORTED_SUBSETS = (LEARNING_SET, FULL_TEST_SET)

# Public mirror used to reconstruct the canonical IEEE PHM 2012 challenge layout
# without relying on PHMD as a transport layer.
DEFAULT_DOWNLOAD_URL = (
    "https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset"
)
DEFAULT_DOWNLOAD_REF = "master"

# Each CSV stores a 2560-sample vibration snapshot. We expose the channels in the
# same order used by the PHMD-backed Pronostia loader so raw-vs-PHMD comparisons
# stay interpretable without extra downstream remapping.
FEATURE_COLUMNS = ["vibration_horizontal", "vibration_vertical"]
TARGET_COLUMN = "rul"

# PRONOSTIA windows are sampled every 10 seconds.
RUNTIME_STEP_SECONDS = 10.0

# We reuse the existing PRONOSTIA bearing identifiers so the new raw datasource and
# the PHMD-backed datasource can be compared unit by unit.
UNIT_NAMES_TO_ID = {
    "1_1": (1, 1),
    "1_2": (1, 2),
    "1_3": (1, 3),
    "1_4": (1, 4),
    "1_5": (1, 5),
    "1_6": (1, 6),
    "1_7": (1, 7),
    "2_1": (2, 1),
    "2_2": (2, 2),
    "2_3": (2, 3),
    "2_4": (2, 4),
    "2_5": (2, 5),
    "2_6": (2, 6),
    "2_7": (2, 7),
    "3_1": (3, 1),
    "3_2": (3, 2),
    "3_3": (3, 3),
}
# Official challenge test bearings stop before failure; the hidden tail length is
# published as the actual RUL at the end of the observed sequence.
TEST_RULS = {
    "1_3": 5730.0,
    "1_4": 2900.0,
    "1_5": 1610.0,
    "1_6": 1460.0,
    "1_7": 7570.0,
    "2_3": 7530.0,
    "2_4": 1390.0,
    "2_5": 3090.0,
    "2_6": 1290.0,
    "2_7": 580.0,
    "3_3": 820.0,
}
# Total lifetime in seconds for each bearing. For learning bearings this is the
# full run-to-failure duration, and for official test bearings it is:
# observed_runtime_at_cutoff + hidden_rul.
PRONOSTIA_TOTAL_LIFE_SECONDS = {
    (1, 1): 28020.0,
    (1, 2): 8700.0,
    (1, 3): 23740.0,
    (1, 4): 14180.0,
    (1, 5): 24620.0,
    (1, 6): 24470.0,
    (1, 7): 22580.0,
    (2, 1): 9100.0,
    (2, 2): 7960.0,
    (2, 3): 19540.0,
    (2, 4): 7500.0,
    (2, 5): 23100.0,
    (2, 6): 7000.0,
    (2, 7): 2290.0,
    (3, 1): 5140.0,
    (3, 2): 16360.0,
    (3, 3): 4330.0,
}

# The public raw mirror is internally inconsistent for Bearing 1_4:
#
# - the official lifetime metadata (and PHMD targets) align with the first kept
#   observation after the normal one-window warm-up drop
# - but the PHMD feature payload only matches once the raw signal is shifted by
#   an additional 10 windows
#
# We resolve that conflict in favor of feature alignment, because features are the
# primary observed signal and targets are derived from the aligned observation
# sequence inside this loader.
EXTRA_FEATURE_ALIGNMENT_OFFSETS = {
    "1_4": 10,
}


class PronostiaIEEE2012Loader(PredefinedSplitLoaderBase):
    """Load the Kaggle/IEEE 2012 PRONOSTIA folder dump as a predefined split datasource."""

    def __init__(
        self,
        data_dir: str,
        split_assignments: dict[str, list[str]] | None = None,
        split_mode: str | None = None,
        fold_id: int | str | None = None,
        split_assignments_by_mode: dict[str, Any] | None = None,
        sample_range_size: int = 2560,
        download_if_missing: bool = True,
        download_url: str = DEFAULT_DOWNLOAD_URL,
        download_ref: str = DEFAULT_DOWNLOAD_REF,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_dir = str(Path(data_dir).expanduser())
        # The loader can be used in two equally valid ways:
        #
        # 1. Class-first scripts/notebooks pass explicit ``split_assignments``.
        # 2. Hydra configs pass ``split_mode`` plus a table of assignments for each mode.
        #
        # Supporting both keeps the loader convenient outside Hydra while still
        # letting the repo expose one canonical datasource config file whose split
        # policy is selected by a single parameter.
        self.split_mode = split_mode
        self.fold_id = fold_id
        self.split_assignments = self._resolve_split_assignments(
            split_assignments=split_assignments,
            split_mode=split_mode,
            fold_id=fold_id,
            split_assignments_by_mode=split_assignments_by_mode,
        )
        self.sample_range_size = int(sample_range_size)
        # The raw loader is intended to be usable directly as a class from a
        # notebook or script. These options let it prepare the canonical dataset
        # root on demand without requiring Hydra or PHMD.
        self.download_if_missing = bool(download_if_missing)
        self.download_url = str(download_url)
        self.download_ref = str(download_ref)

    def _resolve_split_assignments(
        self,
        *,
        split_assignments: dict[str, list[str]] | None,
        split_mode: str | None,
        fold_id: int | str | None,
        split_assignments_by_mode: dict[str, Any] | None,
    ) -> dict[str, list[str]]:
        """Resolve the configured split table into the loader's concrete split map.

        ``split_assignments`` remains the most explicit API and therefore wins when
        it is provided directly. The mode-based path exists so Hydra can keep a
        single canonical ``datasource=pronostia`` config and switch only the
        split-policy parameter when moving between in-domain and domain-shift runs.
        """
        if split_assignments is not None:
            return {
                split: list(unit_keys) for split, unit_keys in split_assignments.items()
            }

        if split_mode is None or split_assignments_by_mode is None:
            raise ValueError(
                "PronostiaIEEE2012Loader requires either explicit split_assignments "
                "or both split_mode and split_assignments_by_mode."
            )

        if split_mode not in split_assignments_by_mode:
            raise ValueError(
                "Unknown PRONOSTIA split_mode "
                f"{split_mode!r}. Available modes: {sorted(split_assignments_by_mode)}"
            )

        split_table_or_folds = split_assignments_by_mode[split_mode]

        # Support both:
        # 1. direct tables: {train: [...], val: [...], test: [...]}
        # 2. CV tables: {fold_1: {train: [...]}, ..., fold_5: {...}}
        #
        # The second form is what the repo uses now for five-fold CV, while the
        # first form remains accepted so scripts can still pass one explicit mode
        # table without inventing synthetic fold names.
        if {"train", "val", "test"}.issubset(split_table_or_folds):
            resolved_table = split_table_or_folds
        else:
            if fold_id is None:
                raise ValueError(
                    "PronostiaIEEE2012Loader requires fold_id when "
                    "split_assignments_by_mode contains multiple CV folds."
                )

            fold_key = (
                fold_id
                if isinstance(fold_id, str) and str(fold_id) in split_table_or_folds
                else f"fold_{fold_id}"
            )
            if fold_key not in split_table_or_folds:
                raise ValueError(
                    "Unknown PRONOSTIA fold_id "
                    f"{fold_id!r} for split_mode {split_mode!r}. "
                    f"Available folds: {sorted(split_table_or_folds)}"
                )
            resolved_table = split_table_or_folds[fold_key]

        return {
            split: list(unit_keys) for split, unit_keys in resolved_table.items()
        }

    def _load_data(self) -> dict[str, dict[str, list[Any]]]:
        # v1 intentionally mirrors the existing Pronostia prognostics path:
        # ragged unit lists with RUL targets only.
        if self.task_mode != "rul":
            raise NotImplementedError(
                "PronostiaIEEE2012Loader currently supports task_mode='rul' only."
            )

        data_root = self._ensure_data_root()

        discovered_units = self._discover_units(data_root)
        self._validate_split_assignments(discovered_units)

        split_records: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        for split_name in ("train", "val", "test"):
            for unit_key in self.split_assignments.get(split_name, []):
                subset_name, unit_dir = discovered_units[unit_key]
                split_records[split_name].append(
                    self._load_unit(
                        unit_key=unit_key,
                        subset_name=subset_name,
                        unit_dir=unit_dir,
                    )
                )

        out_dict: dict[str, dict[str, list[Any]]] = {}
        all_keys = set().union(
            *(record.keys() for records in split_records.values() for record in records)
        )
        for key in all_keys:
            out_dict[key] = {
                split_name: [record[key] for record in split_records[split_name]]
                for split_name in ("train", "val", "test")
            }

        metadata_splits = out_dict.get("metadata", {})
        unit_id_splits = out_dict.get("unit_id", {})
        self.meta_data = {
            "unit_ids": {
                split: unit_id_splits.get(split, [])
                for split in ("train", "val", "test")
            },
            "unit_names": {
                split: [meta["unit_name"] for meta in metadata_splits.get(split, [])]
                for split in ("train", "val", "test")
            },
            "features": deepcopy(FEATURE_COLUMNS),
            "identifier": "unit_id",
            "target": TARGET_COLUMN,
            "target_kind": TARGET_COLUMN,
            "dims_explanation": (
                "Features use ragged PRONOSTIA windows of shape "
                "(n_records, sample_range_size, 2); target broadcasts RUL "
                "per record as (n_records, sample_range_size, 1)."
            ),
        }
        return out_dict

    def _ensure_data_root(self) -> Path:
        """Return a valid PRONOSTIA root, downloading it first when permitted.

        The raw loader deliberately does not fall back to PHMD. Either the caller
        points at a real IEEE 2012 tree, or we prepare that tree from the public
        GitHub mirror, or we fail loudly with an actionable message.
        """
        data_root = Path(self.data_dir)
        if self._has_expected_layout(data_root):
            return data_root

        if not self.download_if_missing:
            raise FileNotFoundError(
                "PRONOSTIA raw data_dir is missing the expected "
                f"{LEARNING_SET}/ and {FULL_TEST_SET}/ folders at {data_root}. "
                "Automatic preparation is disabled because download_if_missing=False."
            )

        self._download_missing_dataset(data_root)

        if not self._has_expected_layout(data_root):
            raise FileNotFoundError(
                "Prepared PRONOSTIA dataset does not contain the expected "
                f"{LEARNING_SET}/ and {FULL_TEST_SET}/ folders at {data_root}."
            )
        return data_root

    def _has_expected_layout(self, data_root: Path) -> bool:
        """Check whether ``data_root`` already looks like a usable challenge dump."""
        return all(
            (data_root / subset_name).is_dir() for subset_name in SUPPORTED_SUBSETS
        )

    def _download_missing_dataset(self, data_root: Path) -> None:
        """Clone only the required PRONOSTIA subsets into ``data_root``.

        We still use the GitHub mirror as the source of truth, but we avoid pulling
        the entire repository history and every extra file. A sparse checkout keeps
        the local footprint limited to the two subsets this loader actually needs:
        ``Learning_set`` and ``Full_Test_Set``.
        """
        # A partially populated destination is more dangerous than a missing one
        # because it can silently mix files from different sources. Refuse it and
        # force the user to fix the directory intentionally.
        if data_root.exists() and any(data_root.iterdir()):
            raise FileNotFoundError(
                "PRONOSTIA data_dir exists but is missing the expected "
                f"{LEARNING_SET}/ and {FULL_TEST_SET}/ folders: {data_root}. "
                "Refusing to auto-download into a non-empty directory."
            )

        data_root.parent.mkdir(parents=True, exist_ok=True)
        clone_cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--branch",
            self.download_ref,
            self.download_url,
            str(data_root),
        ]

        # After the sparse clone we explicitly materialize only the challenge
        # subsets that are in scope for this datasource. This avoids downloading
        # the auxiliary Test_set and repository-level docs/assets.
        sparse_checkout_cmd = [
            "git",
            "-C",
            str(data_root),
            "sparse-checkout",
            "set",
            LEARNING_SET,
            FULL_TEST_SET,
        ]
        logger.info(
            "Preparing PRONOSTIA raw dataset at %s via %s", data_root, clone_cmd
        )
        try:
            subprocess.run(
                clone_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                sparse_checkout_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Could not prepare PRONOSTIA raw data because 'git' is not available. "
                f"Install git or place the dataset manually at {data_root}."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(
                "Could not download PRONOSTIA raw data from "
                f"{self.download_url}@{self.download_ref}. "
                f"git clone failed for destination {data_root}. "
                f"git stderr: {stderr or '<empty>'}"
            ) from exc

    def _discover_units(self, data_root: Path) -> dict[str, tuple[str, Path]]:
        # The split protocol is supplied via config, but discovery still needs to
        # scan both supported on-disk subsets so we can validate those assignments.
        discovered: dict[str, tuple[str, Path]] = {}
        for subset_name in SUPPORTED_SUBSETS:
            subset_dir = data_root / subset_name
            if not subset_dir.exists():
                continue
            for unit_dir in subset_dir.iterdir():
                if not unit_dir.is_dir():
                    continue
                match = UNIT_DIR_RE.match(unit_dir.name)
                if match is None:
                    continue
                unit_key = match.group(1)
                discovered[unit_key] = (subset_name, unit_dir)
        if not discovered:
            raise FileNotFoundError(
                f"No bearing folders found under {data_root}. "
                f"Expected {LEARNING_SET}/ and {FULL_TEST_SET}/."
            )
        return discovered

    def _validate_split_assignments(
        self, discovered_units: dict[str, tuple[str, Path]]
    ) -> None:
        # Config-driven splits are a feature here, so we validate aggressively to
        # catch typos, duplicate membership, or silently ignored bearings.
        configured_keys: list[str] = []
        for split_name in ("train", "val", "test"):
            configured_keys.extend(self.split_assignments.get(split_name, []))

        duplicates = sorted(
            {key for key in configured_keys if configured_keys.count(key) > 1}
        )
        if duplicates:
            raise ValueError(f"Units assigned to multiple splits: {duplicates}")

        unknown_units = sorted(set(configured_keys) - set(discovered_units))
        if unknown_units:
            raise ValueError(f"Configured units not found on disk: {unknown_units}")

        unassigned_units = sorted(set(discovered_units) - set(configured_keys))
        if unassigned_units:
            raise ValueError(
                f"Unassigned units present in data_dir: {unassigned_units}"
            )

    def _load_unit(
        self, *, unit_key: str, subset_name: str, unit_dir: Path
    ) -> dict[str, Any]:
        # The public IEEE/Kaggle dump includes one leading observation that the
        # legacy PHMD PRONOSTIA payload does not expose. To keep the raw loader
        # compatible with the framework's established PRONOSTIA semantics, we
        # intentionally drop that warm-up record here. This yields:
        #
        # - exact learning-set parity against the PHMD loader
        # - observable-prefix parity plus extra tail windows for the official
        #   full-test bearings
        acc_files = self._aligned_acc_files(self._sorted_acc_files(unit_dir), unit_key)
        windows = [self._read_acc_csv(path) for path in acc_files]
        features_np = np.stack(windows, axis=0).astype(np.float32, copy=False)

        unit_id = UNIT_NAMES_TO_ID.get(unit_key)
        if unit_id is None:
            raise KeyError(f"Unknown PRONOSTIA unit key: {unit_key}")

        hidden_rul_seconds = TEST_RULS.get(unit_key, 0.0)
        expected_total_life = float(PRONOSTIA_TOTAL_LIFE_SECONDS[unit_id])

        # Each file represents the system state at one observation point. We define
        # the target as the remaining useful life at the start of that record:
        #
        #   RUL(record_i) = total_life - elapsed_runtime_at_record_i
        #
        # For learning bearings, hidden_rul_seconds is zero and the final observed
        # record reaches RUL == 0.
        #
        # For official test bearings, the raw sequence stops early. The published
        # hidden RUL tells us how much life remains after the last observed record,
        # so the final target becomes that hidden tail instead of zero.
        runtime = (
            np.arange(1, len(acc_files) + 1, dtype=np.float32) * RUNTIME_STEP_SECONDS
        )
        rul = expected_total_life - runtime
        target_np = np.broadcast_to(
            rul[:, None, None],
            (len(acc_files), self.sample_range_size, 1),
        ).copy()

        return {
            "features": ak.from_regular(ak.from_numpy(features_np), axis=1),
            "target": ak.from_numpy(target_np),
            "unit_id": ak.from_numpy(np.array(unit_id)),
            "metadata": {
                "unit_id": unit_id,
                "unit_name": f"Bearing {unit_key}",
                "unit_length": len(acc_files),
                "features_columns": deepcopy(FEATURE_COLUMNS),
                "target_col": TARGET_COLUMN,
                "target_in_features": True,
                "task": self.task_mode,
                "dataset_subset": subset_name,
                # For learning bearings this is zero. For official test bearings it
                # is the RUL that remains beyond the last observed record.
                "hidden_rul_seconds": hidden_rul_seconds,
                # Keeping the expected total life in metadata makes notebook and test
                # comparisons easier to interpret without hardcoding the lookup twice.
                "expected_total_life_seconds": expected_total_life,
                # Most units use only the default warm-up drop (1 file). Bearing 1_4
                # needs an extra feature shift because the public raw mirror is offset
                # by 10 windows relative to the PHMD feature payload.
                "feature_alignment_extra_offset": EXTRA_FEATURE_ALIGNMENT_OFFSETS.get(
                    unit_key, 0
                ),
            },
        }

    def _aligned_acc_files(self, acc_files: list[Path], unit_key: str) -> list[Path]:
        """Drop the leading raw record so the raw loader matches PHMD semantics.

        Every PRONOSTIA bearing in the public mirror has many observations, so
        losing the first one is an intentional semantic alignment step rather than
        a data-loss accident. If a caller points at a malformed bearing with only
        one record, fail loudly instead of returning an empty unit.
        """
        total_shift = 1 + EXTRA_FEATURE_ALIGNMENT_OFFSETS.get(unit_key, 0)
        if len(acc_files) <= total_shift:
            raise ValueError(
                f"PRONOSTIA unit {unit_key} needs more than {total_shift} acc_*.csv "
                "files to apply the PHMD compatibility alignment."
            )
        # Default behavior: drop the leading warm-up record that PHMD does not
        # expose. Special case for Bearing 1_4: the raw mirror's feature files are
        # shifted forward by another 10 windows compared with the PHMD feature
        # payload. We therefore drop 11 files in total for that bearing:
        #
        # - 1 global warm-up window shared by all bearings
        # - 10 additional windows only for Bearing 1_4
        #
        # Targets are then regenerated from the remaining aligned observations, so
        # they remain synchronized with the features we actually expose.
        return acc_files[total_shift:]

    def _sorted_acc_files(self, unit_dir: Path) -> list[Path]:
        # The Kaggle dump can be read directly from filenames; no sidecar index is
        # required as long as the numeric suffix ordering is respected.
        indexed_paths: list[tuple[int, Path]] = []
        for path in unit_dir.iterdir():
            if not path.is_file():
                continue
            match = ACC_FILE_RE.match(path.name)
            if match is None:
                continue
            indexed_paths.append((int(match.group(1)), path))
        indexed_paths.sort(key=lambda item: item[0])
        if not indexed_paths:
            raise FileNotFoundError(f"No acc_*.csv files found in {unit_dir}")
        return [path for _, path in indexed_paths]

    def _read_acc_csv(self, path: Path) -> np.ndarray:
        # Raw acc_*.csv files do not ship headers, so we read the two vibration
        # columns positionally and validate that each file still matches the loader's
        # expected 2560-sample contract.
        #
        # In the IEEE/Kaggle dump the acceleration files include extra leading
        # metadata columns before the actual vibration signals. The real sensor
        # channels are the *last* two columns, not the first two.
        #
        # Their raw order is opposite to the PHMD-backed Pronostia feature order, so
        # we reverse them here. This keeps the raw loader faithful to the source
        # while still exposing features in the framework's established Pronostia
        # channel order.
        frame = pd.read_csv(path, header=None, sep=self._infer_delimiter(path))
        values = frame.to_numpy(dtype=np.float32, copy=True)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError(
                f"{path} must contain at least two acceleration columns, "
                f"found shape {values.shape}."
            )
        if values.shape[0] != self.sample_range_size:
            raise ValueError(
                f"{path} has {values.shape[0]} rows; "
                f"expected sample_range_size={self.sample_range_size}."
            )
        return values[:, -2:][:, ::-1]

    def _infer_delimiter(self, path: Path) -> str:
        """Infer whether a raw acceleration file uses comma or semicolon separators.

        The public mirror is mostly comma-delimited, but some challenge copies in the
        wild use semicolons. Sniffing the first record keeps the loader tolerant to
        both encodings while still producing the same numeric payload.
        """
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        semicolons = first_line.count(";")
        commas = first_line.count(",")
        if semicolons > commas:
            return ";"
        return ","
