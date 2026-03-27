"""Tests for PHMD processed-payload disk cache helpers."""

from __future__ import annotations

from pathlib import Path

from picid.data.datasources.base.phmd_payload_cache import (
    load_phmd_payload_cache,
    save_phmd_payload_cache,
)


def test_load_phmd_payload_cache_missing_file_returns_none(tmp_path: Path) -> None:
    """
    Return None when the cache file does not exist.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the missing path.
    """
    path = tmp_path / "missing.joblib"
    assert load_phmd_payload_cache(path, {"fold": 0}) is None


def test_save_and_load_phmd_payload_cache_roundtrip(tmp_path: Path) -> None:
    """
    Round-trip save then load restores ``data_dict`` and ``meta_data``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the cache file.
    """
    path = tmp_path / "blob.joblib"
    fingerprint = {"data_name": "X", "fold": 0, "task_mode": "rul"}
    data_dict = {"features": {"train": [1, 2]}}
    meta_data = {"k": "v"}
    save_phmd_payload_cache(path, fingerprint, data_dict, meta_data)
    loaded = load_phmd_payload_cache(path, fingerprint)
    assert loaded is not None
    assert loaded["data_dict"] == data_dict
    assert loaded["meta_data"] == meta_data


def test_load_phmd_payload_cache_fingerprint_mismatch_returns_none(
    tmp_path: Path,
) -> None:
    """
    Return None when the on-disk fingerprint does not match.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the cache file.
    """
    path = tmp_path / "blob.joblib"
    save_phmd_payload_cache(path, {"fold": 0}, {"features": {}}, {})
    assert load_phmd_payload_cache(path, {"fold": 1}) is None
