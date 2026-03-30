"""Raw IEEE 2012 PRONOSTIA datasource backed by the challenge folder layout."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
import re
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd

from picid.data.datasources.base.predefined_split_loader import (
    PredefinedSplitLoaderBase,
)

logger = logging.getLogger(__name__)

ACC_FILE_RE = re.compile(r"acc_(\d+)\.csv$")
UNIT_DIR_RE = re.compile(r"^Bearing(\d+_\d+)$")
LEARNING_SET = "Learning_set"
FULL_TEST_SET = "Full_Test_Set"
FEATURE_COLUMNS = ["vibration_horizontal", "vibration_vertical"]
TARGET_COLUMN = "runtime_seconds"
RUNTIME_STEP_SECONDS = 10.0
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


class PronostiaIEEE2012Loader(PredefinedSplitLoaderBase):
    """Load the Kaggle/IEEE 2012 PRONOSTIA folder dump as a predefined split datasource."""

    def __init__(
        self,
        data_dir: str,
        split_assignments: dict[str, list[str]],
        use_ragged: bool = True,
        sample_range_size: int = 2560,
        derive_full_test_rul: bool = True,
        include_temperature: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_dir = str(Path(data_dir).expanduser())
        self.split_assignments = {
            split: list(unit_keys) for split, unit_keys in split_assignments.items()
        }
        self.use_ragged = use_ragged
        self.sample_range_size = int(sample_range_size)
        self.derive_full_test_rul = derive_full_test_rul
        self.include_temperature = include_temperature

    def _load_data(self) -> dict[str, dict[str, list[Any]]]:
        if self.task_mode != "rul":
            raise NotImplementedError(
                "PronostiaIEEE2012Loader currently supports task_mode='rul' only."
            )
        if not self.use_ragged:
            raise NotImplementedError(
                "PronostiaIEEE2012Loader currently supports use_ragged=True only."
            )
        if self.include_temperature:
            raise NotImplementedError(
                "Temperature ingestion is out of scope for PronostiaIEEE2012Loader."
            )

        data_root = Path(self.data_dir)
        if not data_root.exists():
            raise FileNotFoundError(f"PRONOSTIA data_dir does not exist: {data_root}")

        discovered_units = self._discover_units(data_root)
        self._validate_split_assignments(discovered_units)

        split_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
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
                "(n_records, sample_range_size, 2); target broadcasts runtime "
                "per record as (n_records, sample_range_size, 1)."
            ),
        }
        return out_dict

    def _discover_units(self, data_root: Path) -> dict[str, tuple[str, Path]]:
        discovered: dict[str, tuple[str, Path]] = {}
        for subset_name in (LEARNING_SET, FULL_TEST_SET):
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
            raise ValueError(f"Unassigned units present in data_dir: {unassigned_units}")

    def _load_unit(
        self, *, unit_key: str, subset_name: str, unit_dir: Path
    ) -> dict[str, Any]:
        if subset_name == FULL_TEST_SET and not self.derive_full_test_rul:
            raise NotImplementedError(
                "Full_Test_Set ingestion requires derive_full_test_rul=True in v1."
            )

        acc_files = self._sorted_acc_files(unit_dir)
        windows = [self._read_acc_csv(path) for path in acc_files]
        features_np = np.stack(windows, axis=0).astype(np.float32, copy=False)

        runtime = np.arange(len(acc_files), dtype=np.float32) * RUNTIME_STEP_SECONDS
        target_np = np.broadcast_to(
            runtime[:, None, None],
            (len(acc_files), self.sample_range_size, 1),
        ).copy()

        unit_id = UNIT_NAMES_TO_ID.get(unit_key)
        if unit_id is None:
            raise KeyError(f"Unknown PRONOSTIA unit key: {unit_key}")

        hidden_rul_seconds = TEST_RULS.get(unit_key, 0.0)
        expected_total_life = float(PRONOSTIA_TOTAL_LIFE_SECONDS[unit_id])

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
                "hidden_rul_seconds": hidden_rul_seconds,
                "expected_total_life_seconds": expected_total_life,
            },
        }

    def _sorted_acc_files(self, unit_dir: Path) -> list[Path]:
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
        frame = pd.read_csv(path, header=None)
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
        return values[:, :2]
