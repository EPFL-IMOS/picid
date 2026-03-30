#!/usr/bin/env python3
"""Compare the legacy PHMD PRONOSTIA loader against the raw IEEE 2012 loader.

This script is intentionally class-first: it does not rely on Hydra and can be
used as a standalone verification step while migrating away from the PHMD-backed
PRONOSTIA datasource.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pandas as pd

from picid.data.datasources.phmd_pronostia import PronostiaLoader
from picid.data.datasources.pronostia_ieee2012 import (
    PronostiaIEEE2012Loader,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_raw_dir() -> Path:
    return _repo_root() / "datasets" / "pronostia_ieee2012"


def _default_phmd_cache_dir() -> Path:
    return _repo_root() / "datasets" / "phmd_cache"


def _default_phmd_payload_cache() -> Path:
    return _default_phmd_cache_dir() / "datasets" / "PRONOSTIA" / "_picid_pronostia_fold0_rul_ragged.joblib"


IN_DOMAIN_SPLIT_ASSIGNMENTS = {
    "train": ["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "2_4", "3_1"],
    "val": ["1_5", "2_5", "3_2"],
    "test": ["1_6", "1_7", "2_6", "2_7", "3_3"],
}


@dataclass
class UnitRecord:
    split: str
    idx: int
    metadata: dict


def _index_units(container) -> dict[str, UnitRecord]:
    indexed: dict[str, UnitRecord] = {}
    for split in container.features.keys():
        for idx, meta in enumerate(container.unit_metadata.get(split, [])):
            indexed[meta["unit_name"]] = UnitRecord(split=split, idx=idx, metadata=meta)
    return indexed


def _slice_by_rule(values, *, rule: str, overlap: int):
    if overlap == 0:
        return values[:0]
    if rule == "prefix":
        return values[:overlap]
    if rule == "full":
        return values
    raise ValueError(f"Unknown comparison rule: {rule}")


def _series_allclose(lhs, rhs, *, atol: float) -> bool:
    if len(lhs) != len(rhs):
        return False
    for left_window, right_window in zip(lhs, rhs):
        if not np.allclose(ak.to_numpy(left_window), ak.to_numpy(right_window), atol=atol):
            return False
    return True


def _scalar_from_target_window(window) -> float:
    return float(np.asarray(window).reshape(-1)[0])


def _window_count(unit_features) -> int:
    return len(unit_features)


def _split_statistics_df(container) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        split_features = container.features.get(split, [])
        split_targets = container.target.get(split, [])
        window_counts = [_window_count(unit_features) for unit_features in split_features]
        target_starts = [
            _scalar_from_target_window(ak.to_numpy(unit_target[0]))
            for unit_target in split_targets
            if len(unit_target)
        ]
        target_ends = [
            _scalar_from_target_window(ak.to_numpy(unit_target[-1]))
            for unit_target in split_targets
            if len(unit_target)
        ]
        rows.append(
            {
                "split": split,
                "n_units": len(window_counts),
                "n_windows_total": int(sum(window_counts)),
                "windows_per_unit_min": int(min(window_counts, default=0)),
                "windows_per_unit_mean": (
                    round(float(np.mean(window_counts)), 2) if window_counts else 0.0
                ),
                "windows_per_unit_max": int(max(window_counts, default=0)),
                "target_start_s_max": float(max(target_starts, default=np.nan)),
                "target_end_s_min": float(min(target_ends, default=np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _print_split_membership(title: str, container) -> None:
    if title:
        print(title)
    for split in ("train", "val", "test"):
        unit_names = [
            meta["unit_name"] for meta in container.unit_metadata.get(split, [])
        ]
        print(f"{split}: {unit_names}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare legacy PHMD PRONOSTIA data against the raw IEEE 2012 loader."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=_default_raw_dir(),
        help="Prepared raw IEEE 2012 dataset root. The loader can clone it if missing.",
    )
    parser.add_argument(
        "--phmd-cache-dir",
        type=Path,
        default=_default_phmd_cache_dir(),
        help="PHMD cache root used by the legacy Pronostia loader.",
    )
    parser.add_argument(
        "--phmd-payload-cache",
        type=Path,
        default=_default_phmd_payload_cache(),
        help="Optional processed PHMD payload cache used to speed up repeated runs.",
    )
    parser.add_argument(
        "--no-download-raw",
        action="store_true",
        help="Disable automatic raw dataset preparation when --raw-data-dir is missing.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for floating-point comparisons.",
    )
    return parser


def main() -> int:
    args = _build_argument_parser().parse_args()

    phmd_loader = PronostiaLoader(
        fold=0,
        data_name="PRONOSTIA",
        task_mode="rul",
        cache_dir=str(args.phmd_cache_dir),
        payload_cache_path=str(args.phmd_payload_cache),
        use_ragged=True,
    )
    phmd_loader.load_data()
    phmd_data = phmd_loader.get_data()

    raw_loader = PronostiaIEEE2012Loader(
        data_name="pronostia_ieee2012_in_domain_compare",
        task_mode="rul",
        data_dir=str(args.raw_data_dir),
        split_assignments=IN_DOMAIN_SPLIT_ASSIGNMENTS,
        sample_range_size=2560,
        download_if_missing=not args.no_download_raw,
    )
    raw_loader.load_data()
    raw_data = raw_loader.get_data()

    phmd_index = _index_units(phmd_data)
    raw_index = _index_units(raw_data)
    common_units = sorted(set(phmd_index) & set(raw_index))
    if not common_units:
        print("No shared PRONOSTIA units were found between PHMD and raw loaders.")
        return 1

    print("=== Raw PRONOSTIA IEEE 2012 split statistics ===")
    print(_split_statistics_df(raw_data).to_string(index=False))

    print("\n=== PHMD split membership ===")
    _print_split_membership("", phmd_data)

    print("\n=== Raw IEEE 2012 in-domain split membership ===")
    _print_split_membership("", raw_data)

    comparison_rows = []
    for unit_name in common_units:
        phmd_ref = phmd_index[unit_name]
        raw_ref = raw_index[unit_name]
        phmd_features = phmd_data.features[phmd_ref.split][phmd_ref.idx]
        raw_features = raw_data.features[raw_ref.split][raw_ref.idx]
        phmd_target = phmd_data.target[phmd_ref.split][phmd_ref.idx]
        raw_target = raw_data.target[raw_ref.split][raw_ref.idx]

        phmd_n = len(phmd_features)
        raw_n = len(raw_features)
        dataset_subset = raw_ref.metadata["dataset_subset"]
        comparison_rule = "prefix" if dataset_subset == "Full_Test_Set" else "full"
        overlap = min(phmd_n, raw_n)

        aligned_phmd_features = _slice_by_rule(
            phmd_features, rule=comparison_rule, overlap=overlap
        )
        aligned_raw_features = _slice_by_rule(
            raw_features, rule=comparison_rule, overlap=overlap
        )
        aligned_phmd_target = _slice_by_rule(
            phmd_target, rule=comparison_rule, overlap=overlap
        )
        aligned_raw_target = _slice_by_rule(raw_target, rule=comparison_rule, overlap=overlap)

        same_overlap_features = _series_allclose(
            aligned_phmd_features, aligned_raw_features, atol=args.atol
        )
        same_overlap_targets = _series_allclose(
            aligned_phmd_target, aligned_raw_target, atol=args.atol
        )

        if dataset_subset == "Learning_set":
            expected_behavior_holds = (
                phmd_n == raw_n and same_overlap_features and same_overlap_targets
            )
        else:
            expected_behavior_holds = (
                raw_n >= phmd_n and same_overlap_features and same_overlap_targets
            )

        comparison_rows.append(
            {
                "unit_name": unit_name,
                "dataset_subset": dataset_subset,
                "comparison_rule": comparison_rule,
                "phmd_split": phmd_ref.split,
                "raw_split": raw_ref.split,
                "phmd_n_windows": phmd_n,
                "raw_n_windows": raw_n,
                "overlap_windows": overlap,
                "raw_extra_windows": max(raw_n - overlap, 0),
                "same_overlap_features": same_overlap_features,
                "same_overlap_targets": same_overlap_targets,
                "phmd_overlap_first_target_s": (
                    _scalar_from_target_window(ak.to_numpy(aligned_phmd_target[0]))
                    if overlap
                    else None
                ),
                "raw_overlap_first_target_s": (
                    _scalar_from_target_window(ak.to_numpy(aligned_raw_target[0]))
                    if overlap
                    else None
                ),
                "phmd_overlap_last_target_s": (
                    _scalar_from_target_window(ak.to_numpy(aligned_phmd_target[-1]))
                    if overlap
                    else None
                ),
                "raw_overlap_last_target_s": (
                    _scalar_from_target_window(ak.to_numpy(aligned_raw_target[-1]))
                    if overlap
                    else None
                ),
                "expected_behavior_holds": expected_behavior_holds,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    print("\n=== PHMD vs raw IEEE 2012 comparison ===")
    print(comparison_df.to_string(index=False))

    print("\n=== Expected-behavior summary ===")
    print(
        comparison_df.groupby(["dataset_subset", "comparison_rule"])[
            "expected_behavior_holds"
        ].agg(["sum", "count"]).reset_index().to_string(index=False)
    )

    failures = comparison_df.loc[~comparison_df["expected_behavior_holds"]]
    if not failures.empty:
        print("\nUnits requiring inspection:")
        print(failures.to_string(index=False))
        return 1

    print("\nAll shared PRONOSTIA units matched the expected PHMD-vs-raw behavior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
