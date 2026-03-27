from __future__ import annotations

from unittest.mock import MagicMock, patch

import awkward as ak
import numpy as np
import pytest

from picid.data.data_objects import SplitDatasetContainer
from picid.data.datasources.base.exceptions import DatasourceConfigurationError
from picid.data.datasources.nb14 import loader as nb14_loader
from picid.data.datasources.nb14.loader import NB14Loader, NasaRandomizedData, RulHandler


def test_rul_handler_prepare_y_future_nb14():
    """NB14 RUL interpolation should keep the expected future RUL ordering."""
    with patch("picid.data.datasources.nb14.prepare_rul_data.logger", MagicMock()):
        rul_handler = RulHandler()
        y_raw = rul_handler.prepare_y_future(
            battery_names=["batt-A-2.2-X"],
            battery_n_cycle=np.array([3]),
            y_soh=np.array([[2.2], [2.1], [2.0]]),
            current=np.array([[0, 10, 10, 0], [0, 10, 10, 0], [0, 10, 10, 0]]),
            time=np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
            capacity_threshold=None,
            capacity=2.2,
        )

    assert y_raw.shape == (3, 2)
    assert np.allclose(y_raw[:, 0], np.array([0.0, 10.0 / 2.2, 20.0 / 2.2]))
    assert np.allclose(y_raw[:, 1], np.array([20.0 / 2.2, 10.0 / 2.2, 0.0]))


@pytest.fixture
def mock_nasa_mat_file(monkeypatch):
    monkeypatch.setattr(
        "picid.data.datasources.nb14.nasa_random_data.download",
        lambda *args, **kwargs: None,
    )
    mock_raw_data = [
        {
            "type": "D",
            "voltage": [[4.2, 4.1]],
            "current": [[2.0, 2.0]],
            "temperature": [[30, 31]],
            "time": [[0, 3600]],
            "comment": "reference discharge",
        },
        {
            "type": "C",
            "voltage": [[]],
            "current": [[]],
            "temperature": [[]],
            "time": [[]],
            "comment": "charge",
        },
        {
            "type": "D",
            "voltage": [[4.0]],
            "current": [[1.5]],
            "temperature": [[33]],
            "time": [[0]],
            "comment": "discharge",
        },
        {
            "type": "C",
            "voltage": [[]],
            "current": [[]],
            "temperature": [[]],
            "time": [[]],
            "comment": "charge",
        },
        {
            "type": "D",
            "voltage": [[4.2, 4.1]],
            "current": [[1.8, 1.8]],
            "temperature": [[30, 31]],
            "time": [[0, 3600]],
            "comment": "reference discharge",
        },
    ]
    monkeypatch.setattr(
        "scipy.io.loadmat",
        lambda *args, **kwargs: {"data": [[[[mock_raw_data]]]]},
    )


def test_nasa_randomized_data_get_data(mock_nasa_mat_file):
    handler = NasaRandomizedData(data_path="fake/path")

    x, y, battery_n_cycle, time, current, initial_lengths = handler._get_data(
        names=["fake_battery_name"]
    )

    assert x.shape == (3, 2, 3)
    assert y.shape == (3,)
    assert time.shape == (3, 2)
    assert battery_n_cycle.shape == (1,)
    assert battery_n_cycle[0] == 3
    assert np.allclose(y, np.array([2.0, 1.8, 1.8]))
    assert np.allclose(current, [[2.0, 2.0], [1.5, 0.0], [1.8, 1.8]])
    assert initial_lengths == [2, 1, 2]


@pytest.fixture
def mock_nb14_loader_deps(monkeypatch):
    mock_nasa_instance = MagicMock()
    mock_rul_instance = MagicMock()
    mock_nasa_class = MagicMock(return_value=mock_nasa_instance)
    mock_rul_class = MagicMock(return_value=mock_rul_instance)

    monkeypatch.setattr(nb14_loader, "NasaRandomizedData", mock_nasa_class)
    monkeypatch.setattr(nb14_loader, "RulHandler", mock_rul_class)

    mock_names = {
        "train_names": [
            "Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/RW1",
            "Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/RW2",
        ],
        "val_names": [
            "Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/RW7"
        ],
        "test_names": [
            "Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/RW8"
        ],
    }
    mock_unit_ids = {
        name: i
        for i, name in enumerate(
            mock_names["train_names"]
            + mock_names["val_names"]
            + mock_names["test_names"]
        )
    }

    monkeypatch.setattr(nb14_loader, "train_names", mock_names["train_names"])
    monkeypatch.setattr(nb14_loader, "val_names", mock_names["val_names"])
    monkeypatch.setattr(nb14_loader, "test_names", mock_names["test_names"])
    monkeypatch.setattr(nb14_loader, "UNIT_NAMES_TO_ID", mock_unit_ids)

    mock_nasa_instance.get_discharge_whole_cycle_future.return_value = (
        np.random.rand(3, 10, 3),
        np.array([2.2, 2.1, 2.0]),
        np.random.rand(1, 10, 3),
        np.array([2.2]),
        np.random.rand(1, 10, 3),
        np.array([2.2]),
        np.array([2, 3]),
        np.array([1]),
        np.array([1]),
        np.random.rand(3, 10),
        np.random.rand(1, 10),
        np.random.rand(1, 10),
        np.random.rand(3, 10),
        np.random.rand(1, 10),
        np.random.rand(1, 10),
        [8, 5, 6],
        [7],
        [9],
    )
    mock_rul_instance.prepare_y_future.side_effect = [
        np.array([[0, 20.0], [1, 19.0], [2, 18.0]]),
        np.array([[0, 15.0]]),
        np.array([[0, 10.0]]),
    ]

    return mock_nasa_class, mock_rul_class, mock_nasa_instance, mock_rul_instance, mock_names


def test_nb14_loader_read_data(mock_nb14_loader_deps):
    mock_nasa_class, mock_rul_class, mock_nasa_instance, mock_rul_instance, mock_names = (
        mock_nb14_loader_deps
    )

    loader = NB14Loader(
        data_dir="fake/path",
        data_name="NB14",
        task_mode="prognostics",
    )
    loader.debug_subsample_rate = None

    data = loader.read_data()

    mock_nasa_class.assert_called_with("fake/path")
    mock_rul_class.assert_called_once()
    mock_nasa_instance.get_discharge_whole_cycle_future.assert_called_with(
        train_names=mock_names["train_names"],
        test_names=mock_names["test_names"],
        validation_names=mock_names["val_names"],
    )
    assert mock_rul_instance.prepare_y_future.call_count == 3

    assert len(data["train"]) == 2
    assert data["train"][0]["metadata"]["unit_name"] == mock_names["train_names"][0]
    assert data["train"][0]["metadata"]["n_cycles"] == 2
    assert np.allclose(
        np.unique(ak.to_numpy(ak.flatten(data["train"][0]["target"], axis=None))),
        [19.0, 20.0],
    )

    assert len(data["val"]) == 1
    assert data["val"][0]["metadata"]["unit_name"] == mock_names["val_names"][0]
    assert len(data["test"]) == 1
    assert data["test"][0]["metadata"]["unit_name"] == mock_names["test_names"][0]


def test_nb14_loader_uses_predefined_split_base_and_rejects_multisource_splitter(
    mock_nb14_loader_deps,
):
    loader = NB14Loader(
        data_dir="fake/path",
        data_name="NB14",
        task_mode="prognostics",
    )
    assert loader.get_split_mode() == "within_units"

    with pytest.raises(
        DatasourceConfigurationError,
        match="does not accept multisource_data_splitter",
    ):
        NB14Loader(
            data_dir="fake/path",
            data_name="NB14",
            task_mode="prognostics",
            multisource_data_splitter=object(),
        )

    _, _, mock_nasa_instance, mock_rul_instance, mock_names = mock_nb14_loader_deps

    loader = NB14Loader(
        data_dir="fake/path",
        data_name="NB14",
        task_mode="prognostics",
    )

    loader.load_data()
    data = loader.get_data()

    assert isinstance(data, SplitDatasetContainer)
    assert "features" in data
    assert "train" in data["features"]
    assert loader.get_data_name() == "NB14"
    assert loader.get_data_names() == ("NB14",)

    meta = loader.get_meta_data()
    assert meta["unit_names"]["train"] == mock_names["train_names"]
    assert meta["unit_ids"]["test"] == [3]

    loader.split_data()


@pytest.mark.parametrize(
    "validation_names, expected_call_count, expected_length",
    [
        (["fake_val_name"], 3, 18),
        (None, 2, 12),
    ],
)
def test_nasa_randomized_data_get_discharge_branches(
    validation_names, expected_call_count, expected_length
):
    mock_get_data = MagicMock()
    mock_return_tuple = (
        np.array([1]),
        np.array([2]),
        np.array([3]),
        np.array([4]),
        np.array([5]),
        [6],
    )
    mock_get_data.side_effect = [mock_return_tuple] * expected_call_count

    with patch.object(NasaRandomizedData, "_get_data", mock_get_data), patch(
        "picid.data.datasources.nb14.nasa_random_data.download", return_value=None
    ):
        handler = NasaRandomizedData(data_path="fake/path")
        result_tuple = handler.get_discharge_whole_cycle_future(
            train_names=["train"],
            test_names=["test"],
            validation_names=validation_names,
        )

    assert mock_get_data.call_count == expected_call_count
    assert len(result_tuple) == expected_length
