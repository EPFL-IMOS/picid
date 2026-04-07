"""Reusable helpers for introspecting datasource loader outputs in tutorials.

Keeps split containers, numpy, and awkward payloads handling consistent without
forcing irregular awkward data through ``numpy.asarray``.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import zip_longest
from typing import Any

import awkward as ak
import numpy as np

_SPLITS: tuple[str, ...] = ("train", "val", "test")


def _is_split_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    keys = set(value.keys())
    return bool(keys) and keys.issubset({"train", "val", "test"})


def _as_unit_list(seq: Any) -> list[Any]:
    if seq is None:
        return []
    if isinstance(seq, list):
        return seq
    return [seq]


def _split_branch(container: Any, key: str) -> dict[str, list[Any]]:
    if isinstance(container, Mapping):
        raw = container.get(key)
    else:
        raw = getattr(container, key, None)
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, list[Any]] = {}
    for split, units in raw.items():
        out[str(split)] = _as_unit_list(units)
    return out


def _resolve_split_unit_metadata(container: Any) -> dict[str, list[Any]]:
    unit_meta = getattr(container, "unit_metadata", None)
    if unit_meta is None and isinstance(container, Mapping):
        unit_meta = container.get("unit_metadata")
    if isinstance(unit_meta, dict) and unit_meta:
        return {sp: _as_unit_list(unit_meta.get(sp)) for sp in _SPLITS}
    if isinstance(container, Mapping):
        legacy = container.get("metadata")
        if isinstance(legacy, dict) and _is_split_mapping(legacy):
            return {sp: _as_unit_list(legacy.get(sp)) for sp in _SPLITS}
    return {sp: [] for sp in _SPLITS}


def _coerce_leaf_array(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, ak.Array):
        return value
    return np.asarray(value)


def _compact_value_line(value: Any, max_len: int = 96) -> str:
    if isinstance(value, np.ndarray):
        s = f"ndarray(shape={value.shape}, dtype={value.dtype})"
    elif isinstance(value, ak.Array):
        try:
            form = str(type(value.layout).__name__)
        except Exception:  # noqa: BLE001 — tutorial introspection only
            form = "Array"
        s = f"ak.Array(len={len(value)}, layout={form})"
    elif isinstance(value, (dict, list, tuple)):
        s = repr(value)
    else:
        s = repr(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def flatten_records(container: Any) -> list[dict[str, Any]]:
    """One dict per unit with split, unit_idx, features, target, metadata.

    Per-unit lists are aligned with :func:`itertools.zip_longest` so missing
    ``unit_metadata`` or ``target`` entries do not truncate feature rows.
    Awkward arrays are left as :class:`awkward.Array`; other sequence-like
    values become :class:`numpy.ndarray`.
    """
    features_by_split = _split_branch(container, "features")
    targets_by_split = _split_branch(container, "target")
    if not targets_by_split:
        targets_by_split = _split_branch(container, "targets")
    unit_meta_by_split = _resolve_split_unit_metadata(container)

    records: list[dict[str, Any]] = []
    for split in _SPLITS:
        feats = features_by_split.get(split, [])
        tgts = targets_by_split.get(split, [])
        metas = unit_meta_by_split.get(split, [])
        for idx, (feat, meta, tgt) in enumerate(
            zip_longest(feats, metas, tgts, fillvalue=None)
        ):
            if feat is None:
                continue
            meta_out: dict[str, Any] = {}
            if isinstance(meta, Mapping):
                meta_out = dict(meta)
            elif meta is not None:
                meta_out = {"_value": meta}
            records.append(
                {
                    "split": split,
                    "unit_idx": idx,
                    "features": _coerce_leaf_array(feat),
                    "target": _coerce_leaf_array(tgt) if tgt is not None else None,
                    "metadata": meta_out,
                }
            )
    return records


def summarize_loader_metadata(meta_data: Mapping[str, Any] | None) -> str:
    """Stable, human-readable summary of :meth:`get_meta_data` output."""
    if meta_data is None or len(meta_data) == 0:
        return "(empty)"
    lines = [
        f"{k}: {_compact_value_line(meta_data[k])}" for k in sorted(meta_data.keys())
    ]
    return "\n".join(lines)


def infer_feature_target_columns(
    metadata: Mapping[str, Any] | None,
    feature_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[list[str], list[str]]:
    """Resolve column names from ``column_map`` or synthetic ``feature_i`` / ``target_i``."""
    meta = dict(metadata) if metadata else {}
    cmap = meta.get("column_map") or {}
    if len(feature_shape) >= 2:
        n_feat = int(feature_shape[-1])
    elif len(feature_shape) == 1:
        n_feat = 1
    else:
        n_feat = 0

    if len(target_shape) >= 2:
        n_tgt = int(target_shape[-1])
    elif len(target_shape) == 1:
        n_tgt = 1
    else:
        n_tgt = 0

    feat_keys = cmap.get("features") or cmap.get("feature")
    feat_names = list(feat_keys) if isinstance(feat_keys, list) else []
    if len(feat_names) != n_feat:
        feat_names = [f"feature_{i}" for i in range(n_feat)]

    tgt_keys = cmap.get("target") or cmap.get("targets")
    tgt_names = list(tgt_keys) if isinstance(tgt_keys, list) else []
    if len(tgt_names) != n_tgt:
        tgt_names = [f"target_{i}" for i in range(n_tgt)]

    return feat_names, tgt_names


def _time_axis_length(features: Any) -> int:
    if features is None:
        return 0
    if isinstance(features, np.ndarray):
        if features.size == 0:
            return 0
        return int(features.shape[0]) if features.ndim >= 1 else 1
    if isinstance(features, ak.Array):
        return int(len(features))
    arr = np.asarray(features)
    if arr.size == 0:
        return 0
    return int(arr.shape[0]) if arr.ndim >= 1 else 1


def ragged_summary(records: list[Mapping[str, Any]]) -> str | None:
    """Describe variable per-unit time lengths, or ``None`` when not ragged."""
    lengths = [_time_axis_length(r.get("features")) for r in records]
    lengths = [L for L in lengths if L > 0]
    if len(lengths) < 2:
        return None
    if len(set(lengths)) <= 1:
        return None
    arr = np.asarray(lengths, dtype=np.int64)
    min_idx = int(np.argmin(arr))
    max_idx = int(np.argmax(arr))
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    mean = float(np.mean(arr))
    unique = int(np.unique(arr).size)
    return (
        f"ragged time lengths: min={int(arr[min_idx])}, p50={p50:.1f}, "
        f"p90={p90:.1f}, max={int(arr[max_idx])}, mean={mean:.1f}, "
        f"n_units={len(lengths)}, unique_lengths={unique}; "
        f"shortest_unit_idx={min_idx}, longest_unit_idx={max_idx}"
    )


def unit_metadata_examples(container: Any) -> dict[str, str]:
    """One short example line per split; missing metadata is labeled explicitly."""
    by_split = _resolve_split_unit_metadata(container)
    out: dict[str, str] = {}
    for split in _SPLITS:
        items = by_split.get(split) or []
        if not items:
            out[split] = "(no unit metadata for this split)"
            continue
        first = items[0]
        if not isinstance(first, Mapping):
            out[split] = _compact_value_line(first)
            continue
        parts = [f"{k}={_compact_value_line(first[k])}" for k in sorted(first.keys())]
        out[split] = "{" + ", ".join(parts) + "}"
    return out
