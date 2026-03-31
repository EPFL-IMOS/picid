"""
Provide datasource helpers for phmd payload cache.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from picid.utils.hash_utils import ensure_serializable

_BLOB_VERSION = 1


def save_phmd_payload_cache(
    path: str | Path,
    fingerprint: dict[str, Any],
    data_dict: dict[str, Any],
    meta_data: dict[str, Any],
) -> None:
    """
    Persist loader payload and serialisable fingerprint to ``path``.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file (e.g. ``.joblib``). Parent directories are created as needed.
    fingerprint : dict
        Value compatible with the loader's cache fingerprint (serialised).
    data_dict : dict
        Loader ``data_dict`` to restore.
    meta_data : dict
        Loader ``meta_data`` to restore.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _BLOB_VERSION,
        "fingerprint": ensure_serializable(fingerprint),
        "data_dict": data_dict,
        "meta_data": meta_data,
    }
    joblib.dump(payload, p)


def load_phmd_payload_cache(
    path: str | Path,
    fingerprint: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Load payload from ``path`` if the stored fingerprint matches.

    Parameters
    ----------
    path : str or pathlib.Path
        Cache file written by :func:`save_phmd_payload_cache`.
    fingerprint : dict
        Current loader fingerprint; must match the blob after
        :func:`~picid.utils.hash_utils.ensure_serializable`.

    Returns
    -------
    dict or None
        ``{"data_dict": ..., "meta_data": ...}`` on success, else ``None``.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    blob = joblib.load(p)
    if not isinstance(blob, dict):
        return None
    want = ensure_serializable(fingerprint)
    if blob.get("fingerprint") != want:
        return None
    data_dict = blob.get("data_dict")
    if not isinstance(data_dict, dict):
        return None
    meta = blob.get("meta_data")
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        return None
    return {"data_dict": data_dict, "meta_data": meta}
