"""Shared text report layout for datasource tutorial scripts.

Tutorials should use :data:`STANDARD_SECTIONS` and :func:`build_standard_report`
so stdout structure stays consistent across datasets.
"""

from __future__ import annotations

from collections import Counter
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


def _target_matrix(unit: Any) -> np.ndarray | None:
    """Convert one target unit into a 2D (time, target_channel) float matrix."""
    if isinstance(unit, ak.Array):
        if len(unit) == 0:
            return None
        arr = unit
        try:
            # Collapse extra ragged temporal nesting while keeping channel axis.
            while arr.ndim > 2:
                arr = ak.flatten(arr, axis=1)
            out = np.asarray(ak.to_numpy(arr), dtype=np.float64)
        except Exception:
            return None
    else:
        out = _as_float_array(unit)

    if out.size == 0:
        return None
    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
        out = out.reshape(-1, 1)
    elif out.ndim > 2:
        out = out.reshape(out.shape[0], -1)
    if out.ndim != 2:
        return None
    return out


def _fmt_float(value: float) -> str:
    if np.isfinite(value):
        return f"{float(value):.6g}"
    return "nan"


def _time_length(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    return int(arr.shape[0]) if arr.ndim >= 1 else 1


def compute_split_overview(
    features_by_split: Mapping[str, Sequence[Any]],
    *,
    targets_by_split: Mapping[str, Sequence[Any]] | None = None,
) -> str:
    """Summarize split size/length stats and unit-level length diagnostics."""
    lines: list[str] = []
    splits = sorted(features_by_split.keys())
    for sp in splits:
        feats = features_by_split[sp]
        n_units = len(feats)
        lengths = [_time_length_for_unit(u) for u in feats]
        total_t = int(sum(lengths))
        min_l = min(lengths) if lengths else 0
        max_l = max(lengths) if lengths else 0
        mean_l = float(np.mean(lengths)) if lengths else 0.0
        p50_l = float(np.percentile(lengths, 50)) if lengths else 0.0
        p90_l = float(np.percentile(lengths, 90)) if lengths else 0.0
        unique_l = len(set(lengths)) if lengths else 0
        tgt_note = ""
        if targets_by_split is not None and sp in targets_by_split:
            tn = len(targets_by_split[sp])
            if tn != n_units:
                tgt_note = f", target_units={tn} (mismatch vs features)"
        lines.append(
            f"{sp}: units={n_units}, total_timesteps={total_t}, "
            f"min_len={min_l}, p50_len={p50_l:.1f}, p90_len={p90_l:.1f}, "
            f"max_len={max_l}, mean_len={mean_l:.1f}, unique_lens={unique_l}{tgt_note}"
        )
        if lengths:
            counts = Counter(lengths)
            top = ", ".join(
                f"{length}x{count}" for length, count in counts.most_common(5)
            )
            min_idx = int(np.argmin(lengths))
            max_idx = int(np.argmax(lengths))
            preview = ", ".join(
                f"unit_{idx}={lengths[idx]}" for idx in range(min(5, len(lengths)))
            )
            if len(lengths) > 5:
                preview += ", ..."
            lines.append(f"  length_hist_top: {top}")
            lines.append(
                f"  shortest_unit=unit_{min_idx}({lengths[min_idx]}), "
                f"longest_unit=unit_{max_idx}({lengths[max_idx]})"
            )
            lines.append(f"  unit_length_examples: {preview}")
    return "\n".join(lines) if lines else "(empty)"


def compute_feature_stats(features_by_split: Mapping[str, Sequence[Any]]) -> str:
    """Per-column NaN-aware stats pooled per split without materialized concat."""
    rows: list[str] = []
    for sp in sorted(features_by_split.keys()):
        units = features_by_split[sp]
        if not units:
            rows.append(f"{sp}: no units")
            continue
        n_feat: int | None = None
        n_obs: np.ndarray | None = None
        n_missing: np.ndarray | None = None
        running_sum: np.ndarray | None = None
        running_sum_sq: np.ndarray | None = None
        total_rows = 0
        valid_units = 0
        inconsistent_n_features = False

        for unit in units:
            mat = _feature_matrix(unit)
            if mat is None:
                continue
            if n_feat is None:
                n_feat = int(mat.shape[-1])
                n_obs = np.zeros(n_feat, dtype=np.int64)
                n_missing = np.zeros(n_feat, dtype=np.int64)
                running_sum = np.zeros(n_feat, dtype=np.float64)
                running_sum_sq = np.zeros(n_feat, dtype=np.float64)
            elif int(mat.shape[-1]) != n_feat:
                inconsistent_n_features = True
                break

            valid_units += 1
            n_rows = int(mat.shape[0])
            total_rows += n_rows
            missing_mask = np.isnan(mat)
            safe = np.where(missing_mask, 0.0, mat)

            assert n_obs is not None
            assert n_missing is not None
            assert running_sum is not None
            assert running_sum_sq is not None
            n_obs += n_rows
            n_missing += missing_mask.sum(axis=0).astype(np.int64)
            running_sum += safe.sum(axis=0)
            running_sum_sq += (safe * safe).sum(axis=0)

        if valid_units == 0 or n_feat is None:
            rows.append(f"{sp}: no 2D feature arrays")
            continue
        if inconsistent_n_features:
            rows.append(f"{sp}: inconsistent n_features across units; skipping stats")
            continue

        assert n_obs is not None
        assert n_missing is not None
        assert running_sum is not None
        assert running_sum_sq is not None
        n_valid = n_obs - n_missing
        means = np.divide(
            running_sum,
            n_valid,
            out=np.full(n_feat, np.nan, dtype=np.float64),
            where=n_valid > 0,
        )
        ex2 = np.divide(
            running_sum_sq,
            n_valid,
            out=np.full(n_feat, np.nan, dtype=np.float64),
            where=n_valid > 0,
        )
        variances = np.maximum(ex2 - np.square(means), 0.0)
        stds = np.sqrt(variances)
        miss = int(n_missing.sum())
        cells = int(total_rows * n_feat)
        miss_pct = 100.0 * miss / cells if cells else 0.0
        rows.append(
            f"{sp}: shape_aggregated=({total_rows}, {n_feat}), "
            f"missing_cells={miss}/{cells} ({miss_pct:.3f}%)"
        )
        for j in range(n_feat):
            rows.append(f"  col_{j}: mean={means[j]:.6g}, std={stds[j]:.6g}")
    return "\n".join(rows) if rows else "(empty)"


def compute_target_stats(targets_by_split: Mapping[str, Sequence[Any]]) -> str:
    """Per-unit target summaries (avoids cross-unit aggregation)."""
    rows: list[str] = []
    for sp in sorted(targets_by_split.keys()):
        units = targets_by_split[sp]
        if not units:
            rows.append(f"{sp}: no units")
            continue
        rows.append(f"{sp}: units={len(units)}")
        for idx, unit in enumerate(units):
            mat = _target_matrix(unit)
            if mat is None:
                rows.append(f"  unit_{idx}: empty target")
                continue
            n_t, n_c = int(mat.shape[0]), int(mat.shape[1])
            miss = int(np.isnan(mat).sum())
            cells = int(mat.size)
            rows.append(
                f"  unit_{idx}: shape={mat.shape}, missing_cells={miss}/{cells} "
                f"({(100.0 * miss / cells) if cells else 0.0:.3f}%)"
            )
            for col in range(n_c):
                values = mat[:, col]
                finite_mask = np.isfinite(values)
                finite_vals = values[finite_mask]
                nan_count = int(np.isnan(values).sum())
                mean = float(finite_vals.mean()) if finite_vals.size else np.nan
                std = float(finite_vals.std()) if finite_vals.size else np.nan
                rows.append(
                    f"    col_{col}: n_values={n_t}, finite={int(finite_mask.sum())}, "
                    f"nan={nan_count}, mean={_fmt_float(mean)}, std={_fmt_float(std)}"
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
