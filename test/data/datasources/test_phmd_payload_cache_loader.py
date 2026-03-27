"""Integration tests for PHMDMultiSourceLoader ``payload_cache_path``."""

from __future__ import annotations

from pathlib import Path

from pytest_mock import MockerFixture

from picid.data.datasources.phmd_xjtu_sy import XJTU_SYLoader


def test_phmd_loader_payload_cache_skips_second_load_data(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """
    Second loader instance must not call ``_load_data`` when cache is valid.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the cache file and PHMD ``cache_dir``.
    mocker : MockerFixture
        Pytest-mock fixture used to stub ``_load_data``.
    """
    cache_path = tmp_path / "payload.joblib"
    stub_dict = {
        "features": {"train": [], "val": [], "test": []},
        "rul": {"train": [], "val": [], "test": []},
    }
    calls: list[int] = []

    def fake_load_data(self) -> dict:
        """
        Increment call counter and return the stub split payload.

        Returns
        -------
        dict
            Stub ``data_dict``-shaped payload.
        """
        calls.append(1)
        return stub_dict

    mocker.patch.object(XJTU_SYLoader, "_load_data", fake_load_data)

    a = XJTU_SYLoader(
        fold=0,
        data_name="NB14",
        task_mode="rul",
        cache_dir=str(tmp_path),
        payload_cache_path=str(cache_path),
    )
    a.load_data()
    assert len(calls) == 1

    b = XJTU_SYLoader(
        fold=0,
        data_name="NB14",
        task_mode="rul",
        cache_dir=str(tmp_path),
        payload_cache_path=str(cache_path),
    )
    b.load_data()
    assert len(calls) == 1


def test_payload_cache_path_omitted_from_cache_fingerprint(tmp_path: Path) -> None:
    """
    ``get_cache_fingerprint()`` must not depend on ``payload_cache_path``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary PHMD ``cache_dir`` for both loaders.
    """
    common = {
        "fold": 0,
        "data_name": "NB14",
        "task_mode": "rul",
        "cache_dir": str(tmp_path),
    }
    a = XJTU_SYLoader(
        **common,
        payload_cache_path=str(tmp_path / "a.joblib"),
    )
    b = XJTU_SYLoader(
        **common,
        payload_cache_path=str(tmp_path / "b.joblib"),
    )
    assert a.get_cache_fingerprint() == b.get_cache_fingerprint()
