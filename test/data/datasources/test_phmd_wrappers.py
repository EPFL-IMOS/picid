"""
Tests for phmd_* loader wrappers (CBMv3, HSF15, Battery, Pronostia, etc.).
Uses conftest phmd mock; exercises init and, where possible, load_data with mocked _load_data.
"""

from __future__ import annotations

import pytest

# phmd is mocked in conftest; import loaders that depend on it
from picid.data.datasources.base.exceptions import DatasourceConfigurationError
from picid.data.datasources.phmd_cbmv3 import CBMv3Loader
from picid.data.datasources.phmd_hsf15 import HSF15Loader
from picid.data.datasources.base.phmd_loader import PHMDMultiSourceLoader
from picid.data.datasources.phmd_pronostia import (
    PronostiaLoader,
    UNIT_NAMES_TO_ID,
    TEST_RULS,
)
from picid.data.datasources.phmd_n_cmapss import PHMDNCMAPSSLoader
from picid.data.datasources.phmd_phme20 import PHME20Loader
from picid.data.datasources.phmd_pubd16 import Pubd16Loader
from picid.data.datasources.phmd_xjtu_sy import XJTU_SYLoader


def _minimal_phmd_kwargs(cache_dir="/tmp/phmd_cache"):
    return {
        "fold": 0,
        "data_name": "NB14",
        "task_mode": "rul",
        "cache_dir": cache_dir,
    }


def test_phmd_wrappers_reject_multisource_splitter():
    """PHMD wrappers should fail fast on incompatible splitter config."""
    loader_classes = (
        CBMv3Loader,
        HSF15Loader,
        PronostiaLoader,
        PHMDNCMAPSSLoader,
        PHME20Loader,
        Pubd16Loader,
        XJTU_SYLoader,
    )

    for loader_cls in loader_classes:
        with pytest.raises(
            DatasourceConfigurationError,
            match="does not accept multisource_data_splitter",
        ):
            loader_cls(
                **_minimal_phmd_kwargs(),
                multisource_data_splitter=object(),
            )


def test_phmd_cbmv3_loader_init():
    """CBMv3Loader initializes."""
    loader = CBMv3Loader(**_minimal_phmd_kwargs())
    assert loader.data_name == "NB14"


def test_phmd_hsf15_loader_init():
    """HSF15Loader initializes."""
    loader = HSF15Loader(**_minimal_phmd_kwargs())
    assert loader.data_name == "NB14"


def test_pronostia_loader_init():
    """PronostiaLoader initializes; constants are defined."""
    loader = PronostiaLoader(**_minimal_phmd_kwargs())
    assert loader.data_name == "NB14"
    assert "1_3" in UNIT_NAMES_TO_ID
    assert "1_3" in TEST_RULS
    assert isinstance(loader, PHMDMultiSourceLoader)


def test_phmd_n_cmapss_loader_init():
    """PHMDNCMAPSSLoader initializes."""
    loader = PHMDNCMAPSSLoader(**_minimal_phmd_kwargs())
    assert loader.data_name == "NB14"


def test_phmd_phme20_loader_init():
    """PHME20Loader initializes."""
    loader = PHME20Loader(**_minimal_phmd_kwargs())
    assert loader.data_name == "NB14"


def test_phmd_pubd16_loader_init():
    """Pubd16Loader initializes."""
    loader = Pubd16Loader(**_minimal_phmd_kwargs())
    assert loader.data_name == "NB14"


def test_phmd_xjtu_sy_loader_init():
    """XJTU_SYLoader initializes."""
    loader = XJTU_SYLoader(**_minimal_phmd_kwargs())
    assert loader.data_name == "NB14"
    assert isinstance(loader, PHMDMultiSourceLoader)
