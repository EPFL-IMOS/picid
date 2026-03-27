"""Characterization tests for AirbusHelicopterLoader."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from picid.data.data_objects import SplitDatasetContainer
from picid.data.datasources.base.exceptions import DatasourceConfigurationError
from picid.data.datasources.airbus_helicopter import AirbusHelicopterLoader


TRAIN_COUNT = 1677
TEST_COUNT = 594


@pytest.fixture
def airbus_kwargs(tmp_path):
    return {
        "data_dir": str(tmp_path),
        "data_name": "airbus_helicopter",
        "task_mode": "anomaly_detection",
        "download": False,
    }


def _make_airbus_row(split: str, idx: int) -> dict:
    return {
        "features": np.array([idx], dtype=np.float32),
        "target": np.array([1.0 if split == "test" else 0.0], dtype=np.float32),
        "unit_id": np.array([idx], dtype=np.int64),
        "metadata": {
            "unit_name": f"{split}_seq_{idx}",
            "n_cycles": 1,
            "split": split,
            "sampling_freq": 1024,
            "duration": 60,
            "dims_info": "60s x 1024Hz x 1feat",
        },
    }


@pytest.fixture
def mocked_airbus_read_data():
    train = [_make_airbus_row("train", idx) for idx in range(TRAIN_COUNT)]
    test = [_make_airbus_row("test", idx) for idx in range(TEST_COUNT)]
    return {"train": train, "test": test}


def test_airbus_loader_rejects_multisource_splitter(airbus_kwargs):
    """Constructor should fail fast on incompatible multisource configuration."""
    with pytest.raises(
        DatasourceConfigurationError,
        match="does not accept multisource_data_splitter",
    ):
        AirbusHelicopterLoader(
            **airbus_kwargs,
            multisource_data_splitter=object(),
        )


def test_airbus_loader_default_split_mode_is_within_units(airbus_kwargs):
    """Canonical split mode is within_units when no multisource splitter is configured."""
    loader = AirbusHelicopterLoader(**airbus_kwargs)
    assert loader.get_split_mode() == "within_units"


def test_airbus_loader_load_data_get_data_and_meta_data(
    airbus_kwargs, mocked_airbus_read_data
):
    """load_data/get_data/get_meta_data work from mocked read_data and never download."""
    with (
        patch.object(
            AirbusHelicopterLoader,
            "_download_missing_files",
            autospec=True,
        ) as mocked_download,
        patch.object(
            AirbusHelicopterLoader,
            "read_data",
            return_value=deepcopy(mocked_airbus_read_data),
        ),
    ):
        loader = AirbusHelicopterLoader(**airbus_kwargs)
        loader.load_data()

    mocked_download.assert_not_called()
    assert loader._is_loaded is True
    assert loader._is_splitted is True

    data = loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert set(data.keys()) == {"features", "target", "unit_id"}
    assert set(data["features"].keys()) == {"train", "val", "test"}
    assert len(data["features"]["train"]) == TRAIN_COUNT
    assert len(data["features"]["test"]) == TEST_COUNT
    assert len(data["features"]["val"]) == TEST_COUNT

    meta = loader.get_meta_data()
    assert set(meta.keys()) == {"unit_ids", "unit_names", "dims_explanation"}
    assert len(meta["unit_ids"]["train"]) == TRAIN_COUNT
    assert len(meta["unit_ids"]["test"]) == TEST_COUNT
    assert len(meta["unit_names"]["val"]) == TEST_COUNT
    assert meta["unit_names"]["train"][0] == "train_seq_0"
    assert meta["unit_names"]["test"][0] == "test_seq_0"
    assert "Ragged representation" in meta["dims_explanation"]


def test_airbus_loader_validation_split_is_cloned_from_test(
    airbus_kwargs, mocked_airbus_read_data
):
    """Validation split is a deep copy of test data inside _load_data."""
    with patch.object(
        AirbusHelicopterLoader,
        "read_data",
        return_value=deepcopy(mocked_airbus_read_data),
    ):
        loader = AirbusHelicopterLoader(**airbus_kwargs)
        loader.load_data()

    assert loader.data_dict["features"]["val"] is not loader.data_dict["features"]["test"]
    assert loader.data_dict["metadata"]["val"] is not loader.data_dict["metadata"]["test"]
    assert loader.data_dict["features"]["val"][0] is not loader.data_dict["features"]["test"][0]
    assert loader.data_dict["metadata"]["val"][0] is not loader.data_dict["metadata"]["test"][0]
    assert loader.data_dict["metadata"]["val"][0]["unit_name"] == "test_seq_0"
