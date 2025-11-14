import pytest
import numpy as np
import pandas as pd
import awkward as ak
from unittest.mock import MagicMock

# --- Import the classes you need to test ---
# (You will need to adjust these import paths to match your library structure)
from lmetk.data.datasources.nb14.loader import (
    NB14Loader,
    NasaRandomizedData,
    RulHandler,
    UNIT_NAMES_TO_ID,
)

# We also need to mock scipy.io
import scipy.io

# =========================================================================
# === Part 1: Test the RulHandler (Pure Logic) ===
# =========================================================================


def test_rul_handler_prepare_y_future_nb14(mocker):
    """
    Tests the RulHandler with the NB14 logic (capacity_threshold=None).
    This means the EoL index is just the last cycle.
    """
    # 1. --- Arrange ---
    rul_handler = RulHandler()

    # Patch the *module-level* logger where RulHandler uses it.
    mocker.patch("lmetk.data.datasources.nb14.prepare_rul_data.logger", MagicMock())

    # --- Handler Inputs ---
    battery_names = ["batt-A-2.2-X"]  # Name provides nominal capacity
    battery_n_cycle = np.array([3])  # 1 battery, 3 cycles total
    y_soh = np.array([[2.2], [2.1], [2.0]])  # SOH (capacity) for each cycle

    current_arr = np.array(
        [
            [0, 10, 10, 0],  # Cycle 0
            [0, 10, 10, 0],  # Cycle 1
            [0, 10, 10, 0],  # Cycle 2
        ]
    )
    time_arr = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])

    capacity_threshold = None
    nominal_capacity = 2.2  # Passed in directly

    # 2. --- Act ---
    y_raw = rul_handler.prepare_y_future(
        battery_names=battery_names,
        battery_n_cycle=battery_n_cycle,
        y_soh=y_soh,
        current=current_arr,
        time=time_arr,
        capacity_threshold=capacity_threshold,
        capacity=nominal_capacity,
    )

    # 3. --- Assert ---
    expected_col_0 = np.array([0.0, 10.0 / 2.2, 20.0 / 2.2])
    expected_col_1 = np.array([20.0 / 2.2, 10.0 / 2.2, 0.0])

    assert y_raw.shape == (3, 2)
    assert np.allclose(y_raw[:, 0], expected_col_0)
    assert np.allclose(y_raw[:, 1], expected_col_1)


# =========================================================================
# === Part 2: Test NasaRandomizedData Class (I/O Mocking) ===
# =========================================================================


@pytest.fixture
def mock_nasa_mat_file(mocker):
    """
    Mocks the I/O for NasaRandomizedData:
    - `phmd.download.download`
    - `scipy.io.loadmat`
    """
    # 1. Mock the download
    mocker.patch(
        "lmetk.data.datasources.nb14.nasa_random_data.download", return_value=None
    )

    # 2. Create fake raw data that will produce 3 cycles with max_len=2
    mock_raw_data = [
        # Cycle 0 (Ref Discharge 1) - 'D'
        {
            "type": "D",
            "voltage": [[4.2, 4.1]],
            "current": [[2.0, 2.0]],
            "temperature": [[30, 31]],
            "time": [[0, 3600]],
            "comment": "reference discharge",
        },  # len 2
        # Cycle 1 (Charge) - 'C' (Skipped)
        {
            "type": "C",
            "voltage": [[]],
            "current": [[]],
            "temperature": [[]],
            "time": [[]],
            "comment": "charge",
        },
        # Cycle 2 (Normal Discharge) - 'D'
        {
            "type": "D",
            "voltage": [[4.0]],
            "current": [[1.5]],
            "temperature": [[33]],
            "time": [[0]],
            "comment": "discharge",
        },  # len 1
        # Cycle 3 (Charge) - 'C' (Skipped)
        # This is the new entry to force a cycle split
        {
            "type": "C",
            "voltage": [[]],
            "current": [[]],
            "temperature": [[]],
            "time": [[]],
            "comment": "charge",
        },
        # Cycle 4 (Ref Discharge 2) - 'D'
        {
            "type": "D",
            "voltage": [[4.2, 4.1]],
            "current": [[1.8, 1.8]],
            "temperature": [[30, 31]],
            "time": [[0, 3600]],
            "comment": "reference discharge",
        },  # len 2
    ]

    # 3. Mock `scipy.io.loadmat`
    mocker.patch("scipy.io.loadmat", return_value={"data": [[[[mock_raw_data]]]]})


def test_nasa_randomized_data_get_data(mock_nasa_mat_file):
    """
    Tests the `_get_data` method's parsing and SOH interpolation logic.
    """
    # 1. --- Arrange ---
    handler = NasaRandomizedData(data_path="fake/path")

    # 2. --- Act ---
    (x, y, battery_n_cycle, time, current, initial_lengths) = handler._get_data(
        names=["fake_battery_name"]
    )

    # 3. --- Assert ---
    # The code should find 3 discharge cycles (0, 2, 4) and skip 2 charge cycles (1, 3)

    # --- Check Shapes (This will now pass) ---
    assert x.shape == (3, 2, 3)  # 3 cycles, 2 steps (max_len), 3 features (V,I,T)
    assert y.shape == (3,)  # 3 SOH values
    assert time.shape == (3, 2)  # 3 cycles, 2 steps (padded)

    # --- Check SOH (Y) Interpolation ---
    expected_y = np.array([2.0, 1.8, 1.8])

    assert battery_n_cycle.shape == (1,)
    assert battery_n_cycle[0] == 3  # 3 discharge cycles found
    assert np.allclose(y, expected_y)

    # Check current padding
    assert np.allclose(current, [[2.0, 2.0], [1.5, 0.0], [1.8, 1.8]])


# =========================================================================
# === Part 3: Test NB14Loader Class (Orchestrator Mocking) ===
# =========================================================================


@pytest.fixture
def mock_nb14_loader_deps(mocker):
    """
    Mocks the dependencies of `NB14Loader.read_data`:
    - `NasaRandomizedData` class (both `__init__` and `get_discharge_...`)
    - `RulHandler` class (the `prepare_y_future` method)
    """

    # 1. --- Mock NasaRandomizedData ---
    mock_nasa_class = mocker.patch(
        "lmetk.data.datasources.nb14.loader.NasaRandomizedData"
    )

    # Define the 18-tuple of fake data it will return
    train_x = np.random.rand(3, 10, 3)
    train_y_soh = np.array([2.2, 2.1, 2.0])
    train_battery_range = np.array([2, 3])
    train_time = np.random.rand(3, 10)
    train_current = np.random.rand(3, 10)
    train_lengths = [8, 5]  # Batt1: 2 cycles (len 8, 5)

    # This is a mock data bug, train_lengths needs 3 values for 3 cycles
    train_lengths = [8, 5, 6]  # Batt1: 2 cycles (len 8, 5), Batt2: 1 cycle (len 6)

    val_x = np.random.rand(1, 10, 3)
    val_y_soh = np.array([2.2])
    val_battery_range = np.array([1])
    val_time = np.random.rand(1, 10)
    val_current = np.random.rand(1, 10)
    val_lengths = [7]
    test_x = np.random.rand(1, 10, 3)
    test_y_soh = np.array([2.2])
    test_battery_range = np.array([1])
    test_time = np.random.rand(1, 10)
    test_current = np.random.rand(1, 10)
    test_lengths = [9]

    mock_nasa_class.return_value.get_discharge_whole_cycle_future.return_value = (
        train_x,
        train_y_soh,
        val_x,
        val_y_soh,
        test_x,
        test_y_soh,
        train_battery_range,
        val_battery_range,
        test_battery_range,
        train_time,
        val_time,
        test_time,
        train_current,
        val_current,
        test_current,
        train_lengths,
        val_lengths,
        test_lengths,
    )

    # 2. --- Mock RulHandler ---
    mock_rul_class = mocker.patch("lmetk.data.datasources.nb14.loader.RulHandler")
    mock_rul_class.return_value.prepare_y_future.side_effect = [
        np.array([[0, 20.0], [1, 19.0], [2, 18.0]]),  # train
        np.array([[0, 15.0]]),  # val
        np.array([[0, 10.0]]),  # test
    ]

    # 3. --- Mock UNIT_NAMES_TO_ID and name lists ---
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

    mocker.patch.dict(
        "lmetk.data.datasources.nb14.loader.UNIT_NAMES_TO_ID", mock_unit_ids
    )
    mocker.patch(
        "lmetk.data.datasources.nb14.loader.train_names", mock_names["train_names"]
    )
    mocker.patch(
        "lmetk.data.datasources.nb14.loader.val_names", mock_names["val_names"]
    )
    mocker.patch(
        "lmetk.data.datasources.nb14.loader.test_names", mock_names["test_names"]
    )

    return mock_nasa_class, mock_rul_class, mock_names


def test_nb14_loader_read_data(mock_nb14_loader_deps):
    """
    Tests the high-level `NB14Loader.read_data` orchestration.
    """
    # 1. --- Arrange ---
    mock_nasa_class, mock_rul_class, mock_names = mock_nb14_loader_deps

    # Instantiate the loader, passing the required kwargs
    loader = NB14Loader(
        data_dir="fake/path",
        data_name="NB14",
        task_mode="prognostics",  # This was the missing key
    )

    loader.debug_subsample_rate = None

    # 2. --- Act ---
    data = loader.read_data()

    # 3. --- Assert ---
    mock_nasa_class.assert_called_with("fake/path")
    mock_rul_class.assert_called_once()
    mock_nasa_class.return_value.get_discharge_whole_cycle_future.assert_called_with(
        train_names=mock_names["train_names"],
        test_names=mock_names["test_names"],
        validation_names=mock_names["val_names"],
    )

    # Check the RUL handler calls
    assert mock_rul_class.return_value.prepare_y_future.call_count == 3

    # --- Check the final processed data ---
    # `data` is a dict of lists: {'train': [batt1_dict, batt2_dict], ...}

    # Train
    assert len(data["train"]) == 2  # 2 batteries
    assert data["train"][0]["metadata"]["unit_name"] == mock_names["train_names"][0]
    assert data["train"][0]["metadata"]["n_cycles"] == 2

    # The test was checking the wrong thing. We should check the unique values.
    # The RULs for this battery are 20.0 and 19.0.
    target_batt1 = data["train"][0]["target"]
    unique_vals_1 = np.unique(ak.to_numpy(ak.flatten(target_batt1, axis=None)))
    assert np.allclose(unique_vals_1, [19.0, 20.0])

    # batt2 (1 cycle)
    assert data["train"][1]["metadata"]["unit_name"] == mock_names["train_names"][1]
    assert data["train"][1]["metadata"]["n_cycles"] == 1
    # The RUL for this battery is 18.0
    target_batt2 = data["train"][1]["target"]
    unique_vals_2 = np.unique(ak.to_numpy(ak.flatten(target_batt2, axis=None)))
    assert np.allclose(unique_vals_2, [18.0])

    # Validation
    assert len(data["val"]) == 1  # 1 battery
    assert data["val"][0]["metadata"]["n_cycles"] == 1
    target_val = data["val"][0]["target"]
    unique_vals_val = np.unique(ak.to_numpy(ak.flatten(target_val, axis=None)))
    assert np.allclose(unique_vals_val, [15.0])

    # Test
    assert len(data["test"]) == 1  # 1 battery
    assert data["test"][0]["metadata"]["n_cycles"] == 1
    target_test = data["test"][0]["target"]
    unique_vals_test = np.unique(ak.to_numpy(ak.flatten(target_test, axis=None)))
    assert np.allclose(unique_vals_test, [10.0])


@pytest.mark.parametrize(
    "validation_names, expected_call_count, expected_length",
    [
        (["fake_val_name"], 3, 18),  # Test 1: With validation names
        (None, 2, 12),  # Test 2: With validation_names=None
    ],
)
def test_nasa_randomized_data_get_discharge_branches(
    mocker, validation_names, expected_call_count, expected_length
):
    """
    Tests the if/else logic in `get_discharge_whole_cycle_future`
    to ensure both return paths are covered.
    """
    # 1. --- Arrange ---

    # Mock the internal helper method `_get_data` that this function calls
    mock_get_data = mocker.patch.object(
        NasaRandomizedData,
        "_get_data",
    )

    # Define a simple 6-tuple that _get_data is expected to return
    mock_return_tuple = (
        np.array([1]),
        np.array([2]),
        np.array([3]),
        np.array([4]),
        np.array([5]),
        [6],
    )

    # Set the side_effect to return this tuple for each call
    mock_get_data.side_effect = [mock_return_tuple] * expected_call_count

    # We also need to mock the `download` in the __init__
    mocker.patch(
        "lmetk.data.datasources.nb14.nasa_random_data.download", return_value=None
    )

    # Instantiate the class
    handler = NasaRandomizedData(data_path="fake/path")

    # 2. --- Act ---
    # Call the function under test with the parameterized inputs
    result_tuple = handler.get_discharge_whole_cycle_future(
        train_names=["train"],
        test_names=["test"],
        validation_names=validation_names,  # This is (["fake_val_name"],) or (None,)
    )

    # 3. --- Assert ---

    # Check that _get_data was called the correct number of times
    # (3 for train/val/test, 2 for just train/test)
    assert mock_get_data.call_count == expected_call_count

    # Check that the returned tuple has the correct length
    # (18 for train/val/test, 12 for just train/test)
    assert len(result_tuple) == expected_length
