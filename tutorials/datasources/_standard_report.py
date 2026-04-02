"""Shared text report layout for datasource tutorial scripts.

Tutorials should use :data:`STANDARD_SECTIONS` and :func:`build_standard_report`
so stdout structure stays consistent across datasets.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import awkward as ak

STANDARD_SECTIONS: tuple[str, ...] = (
    "LOADER CONFIG",
    "LOADER METADATA",
    "SPLIT OVERVIEW",
    "FEATURE-LEVEL STATISTICS",
    "TARGET-LEVEL STATISTICS",
    "DATASET-SPECIFIC NOTES",
)

_DEFAULT_SECTION_WIDTH = 88


def _section(title: str, body: str, *, width: int) -> str:
    sep = "=" * width
    content = body.rstrip()
    if not content:
        content = "(empty)"
    return f"\n{sep}\n{title}\n{sep}\n{content}\n"


def build_standard_report(
    *,
    loader_config_body: str,
    loader_metadata_body: str,
    split_overview_body: str,
    feature_stats_body: str,
    target_stats_body: str,
    dataset_specific_notes_body: str,
    width: int = _DEFAULT_SECTION_WIDTH,
) -> str:
    """Assemble the canonical multi-section tutorial report.

    The returned string has no leading or trailing newline characters, so
    ``print(report)`` adds exactly one final newline when writing to a TTY.
    """
    parts = [
        _section(STANDARD_SECTIONS[0], loader_config_body, width=width),
        _section(STANDARD_SECTIONS[1], loader_metadata_body, width=width),
        _section(STANDARD_SECTIONS[2], split_overview_body, width=width),
        _section(STANDARD_SECTIONS[3], feature_stats_body, width=width),
        _section(STANDARD_SECTIONS[4], target_stats_body, width=width),
        _section(STANDARD_SECTIONS[5], dataset_specific_notes_body, width=width),
    ]
    return "".join(parts).strip("\n")


def _as_float_array(unit: Any) -> np.ndarray:
    return np.asarray(unit, dtype=np.float64)


def _time_length_for_unit(unit: Any) -> int:
    """Best-effort temporal length for numpy/awkward unit payloads."""
    if isinstance(unit, ak.Array):
        return int(len(unit))
    return _time_length(_as_float_array(unit))


def _feature_matrix(unit: Any) -> np.ndarray | None:
    """Convert one feature unit into a 2D (time, feature) float matrix."""
    if isinstance(unit, ak.Array):
        if len(unit) == 0 or unit.ndim < 2:
            return None
        arr = unit
        try:
            # Collapse ragged temporal nesting while preserving last feature axis.
            while arr.ndim > 2:
                arr = ak.flatten(arr, axis=1)
            out = np.asarray(ak.to_numpy(arr), dtype=np.float64)
        except Exception:
            return None
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        if out.ndim != 2 or out.size == 0:
            return None
        return out

    arr = _as_float_array(unit)
    if arr.ndim < 2 or arr.size == 0:
        return None
    return arr.reshape(-1, arr.shape[-1])


def _target_vector(unit: Any) -> np.ndarray:
    """Flatten one target unit into a 1D float vector."""
    if isinstance(unit, ak.Array):
        try:
            return np.asarray(
                ak.to_numpy(ak.flatten(unit, axis=None)), dtype=np.float64
            )
        except Exception:
            return np.asarray([], dtype=np.float64)
    arr = _as_float_array(unit)
    return arr.reshape(-1)


def _time_length(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    return int(arr.shape[0]) if arr.ndim >= 1 else 1


def compute_split_overview(
    features_by_split: Mapping[str, Sequence[Any]],
    *,
    targets_by_split: Mapping[str, Sequence[Any]] | None = None,
) -> str:
    """Summarize unit counts and sequence lengths per split (features-aligned)."""
    lines: list[str] = []
    splits = sorted(features_by_split.keys())
    for sp in splits:
        feats = features_by_split[sp]
        n_units = len(feats)
        lengths = [_time_length_for_unit(u) for u in feats]
        total_t = int(sum(lengths))
        min_l = min(lengths) if lengths else 0
        max_l = max(lengths) if lengths else 0
        tgt_note = ""
        if targets_by_split is not None and sp in targets_by_split:
            tn = len(targets_by_split[sp])
            if tn != n_units:
                tgt_note = f", target_units={tn} (mismatch vs features)"
        lines.append(
            f"{sp}: units={n_units}, total_timesteps={total_t}, "
            f"min_len={min_l}, max_len={max_l}{tgt_note}"
        )
    return "\n".join(lines) if lines else "(empty)"


def compute_feature_stats(features_by_split: Mapping[str, Sequence[Any]]) -> str:
    """Per-column NaN-aware means/stds pooled within each split (2D feature blocks)."""
    rows: list[str] = []
    for sp in sorted(features_by_split.keys()):
        units = features_by_split[sp]
        if not units:
            rows.append(f"{sp}: no units")
            continue
        two_d = [a for u in units if (a := _feature_matrix(u)) is not None]
        if not two_d:
            rows.append(f"{sp}: no 2D feature arrays")
            continue
        n_feat = two_d[0].shape[-1]
        if any(a.shape[-1] != n_feat for a in two_d):
            rows.append(f"{sp}: inconsistent n_features across units; skipping stats")
            continue
        stacked = np.concatenate([a.reshape(-1, n_feat) for a in two_d], axis=0)
        means = np.nanmean(stacked, axis=0)
        stds = np.nanstd(stacked, axis=0)
        miss = int(np.isnan(stacked).sum())
        cells = int(stacked.size)
        miss_pct = 100.0 * miss / cells if cells else 0.0
        rows.append(
            f"{sp}: shape_aggregated={stacked.shape}, "
            f"missing_cells={miss}/{cells} ({miss_pct:.3f}%)"
        )
        for j in range(n_feat):
            rows.append(f"  col_{j}: mean={means[j]:.6g}, std={stds[j]:.6g}")
    return "\n".join(rows) if rows else "(empty)"


def compute_target_stats(targets_by_split: Mapping[str, Sequence[Any]]) -> str:
    """Scalar summaries per split (works for (T, C) or (T,) targets)."""
    rows: list[str] = []
    for sp in sorted(targets_by_split.keys()):
        units = targets_by_split[sp]
        if not units:
            rows.append(f"{sp}: no units")
            continue
        flat_parts = [v for u in units if (v := _target_vector(u)).size]
        if not flat_parts:
            rows.append(f"{sp}: empty targets")
            continue
        vec = np.concatenate(flat_parts)
        finite = vec[np.isfinite(vec)]
        rows.append(
            f"{sp}: n_values={vec.size}, finite={finite.size}, "
            f"nan={int(np.isnan(vec).sum())}, "
            f"mean={float(np.nanmean(vec)):.6g}, std={float(np.nanstd(vec)):.6g}"
        )
    return "\n".join(rows) if rows else "(empty)"


def format_dataset_specific_notes(notes: str | Iterable[str] | None) -> str:
    """Normalize freeform or bullet lines for the notes section body."""
    if notes is None:
        return "(none)"
    if isinstance(notes, str):
        s = notes.strip()
        return s if s else "(none)"
    lines = [str(x).strip() for x in notes]
    lines = [x for x in lines if x]
    if not lines:
        return "(none)"
    return "\n".join(f"- {x}" for x in lines)
