import pytest
import numpy as np
import pandas as pd
import awkward as ak
from unittest.mock import patch

from picid.data.datasources.base.exceptions import DatasourceConfigurationError

# --- Import the classes you need to test ---
# (You will need to adjust these import paths to match your library structure)
from picid.data.datasources.unibo.unibo_powertools_data import (
    UniboPowertoolsData,
    CycleCols,
    CapacityCols,
)
from picid.data.datasources.unibo.model_data_handler import ModelDataHandler

# Assuming your RulHandler is here, adjust path if needed
from picid.data.datasources.nb14.prepare_rul_data import RulHandler


# =========================================================================
# === Part 1: Test the RulHandler (Pure Logic) ===
# =========================================================================


def test_rul_handler_prepare_y_future():
    """
    Tests the core RUL calculation logic in isolation.
    We'll create a tiny battery with 3 cycles.
    - Nominal Capacity: 3.0 Ah
    - Threshold: 2.7 Ah
    - Cycle 0: SOH = 3.0 Ah
    - Cycle 1: SOH = 2.8 Ah
    - Cycle 2: SOH = 2.6 Ah (Fails here)
    """
    # 1. --- Arrange ---
    rul_handler = RulHandler()

    # --- Handler Inputs ---
    battery_names = ["batt-A-3.0-X"]  # Name provides nominal capacity
    battery_n_cycle = np.array([3])  # 1 battery, 3 cycles total

    # SOH (capacity) for each cycle
    y_soh = np.array([[3.0], [2.8], [2.6]])  # Cycle 0  # Cycle 1  # Cycle 2 (EoL)

    # Mock current (3 cycles, 4 timesteps)
    current_arr = np.array(
        [
            [0, 10, 10, 0],  # Cycle 0
            [0, 10, 10, 0],  # Cycle 1
            [0, 10, 10, 0],  # Cycle 2
        ]
    )

    # Mock time
    time_arr = np.array(
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]  # Cycle 0  # Cycle 1  # Cycle 2
    )

    # The failure threshold
    capacity_threshold = {3.0: 2.7}

    # 2. --- Act ---
    y_raw = rul_handler.prepare_y_future(
        battery_names=battery_names,
        battery_n_cycle=battery_n_cycle,
        y_soh=y_soh,
        current=current_arr,
        time=time_arr,
        capacity_threshold=capacity_threshold,
        capacity=None,  # Use capacity from name
    )

    # 3. --- Assert ---
    #
    # Manual calculation of the code's logic:
    # `integral` for one cycle = np.trapz(y=[10, 10], x=[1, 2]) = 10.
    # `Q_nom` = 3.0

    # RulHandler loop:
    # cycle 0 (i=0): integral=0. sum=0.  cap_int.append(0/3.0)   -> [0.0]
    # cycle 1 (i=4): integral=10. sum=10. cap_int.append(10/3.0) -> [0.0, 3.33]
    # cycle 2 (i=8): integral=10. sum=20. cap_int.append(20/3.0) -> [0.0, 3.33, 6.66]
    #
    # `capacity_integral_train` = [0.0, 3.33, 6.66]
    #
    # `index` (failure) = np.argmax([3.0, 2.8, 2.6] < 2.7) = 2
    # `Q_acc_EoL` = `capacity_integral_train[2]` = 6.66...
    #
    # RUL (y_future):
    # i = 0 (cycle 0): y = (20/3) - 0.0   = 20/3 = 6.66...
    # i = 1 (cycle 1): y = (20/3) - (10/3) = 10/3 = 3.33...
    # i = 2 (cycle 2): y = 0 (because i is not < index)

    expected_col_0 = np.array([0.0, 10.0 / 3.0, 20.0 / 3.0])
    expected_col_1 = np.array([20.0 / 3.0, 10.0 / 3.0, 0.0])

    assert y_raw.shape == (3, 2)
    assert np.allclose(y_raw[:, 0], expected_col_0)
    assert np.allclose(y_raw[:, 1], expected_col_1)


# =========================================================================
# === Part 2: Test the UniboPowertoolsData Class (I/O Mocking) ===
# =========================================================================


@pytest.fixture
def mock_unibo_csvs(mocker):
    """
    A pytest fixture that mocks the `pd.read_csv` and `phmd.download` calls
    inside the UniboPowertoolsData class.
    """
    # 1. Mock the dataset download
    mocker.patch(
        "picid.data.datasources.unibo.unibo_powertools_data.download", return_value=None
    )

    # 2. Create fake "cycle" (features) data
    # These are the 13 columns (0-12) from test_result.csv
    fake_cyc_df = pd.DataFrame(
        {
            "test_name": [
                "batt1",
                "batt1",
                "batt1",
                "batt1",
                "batt2",
                "batt2",
                "batt3",
            ],  # <-- ADDED batt3
            "record_id": [1, 2, 3, 4, 5, 6, 7],
            "time": [0, 1, 2, 0, 1, 2, 0],
            "step_time": [0, 1, 2, 0, 1, 2, 0],
            "line": [40, 40, 40, 40, 40, 40, 40],
            "voltage": [3.9, 3.8, 3.7, 3.9, 3.8, 3.7, 4.0],
            "current": [-5, -5, -5, -5, -5, -5, -5],
            "charging_capacity": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "discharging_capacity": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1],
            "wh_charging": [0, 0, 0, 0, 0, 0, 0],
            "wh_discharging": [0, 0, 0, 0, 0, 0, 0],
            "temperature": [30, 31, 32, 30, 31, 32, 29],
            "cycle_count": [1, 1, 1, 2, 1, 1, 1],  # batt1(c1,c2), batt2(c1), batt3(c1)
        }
    )

    # 3. Create fake "capacity" (labels) data
    # These are the 15 columns (0-14) from test_result_trial_end.csv
    fake_cap_df = pd.DataFrame(
        {
            "test_name": ["batt1", "batt1", "batt2", "batt3"],  # 0 <-- ADDED batt3
            "record_id": [3, 4, 6, 7],  # 1
            "time": [0, 0, 0, 0],  # 2
            "step_time": [0, 0, 0, 0],  # 3
            "line": [40, 40, 40, 40],  # 4
            "voltage": [0, 0, 0, 0],  # 5
            "current": [0, 0, 0, 0],  # 6
            "charging_capacity": [3.0, 2.9, 3.0, 3.0],  # 7
            "discharging_capacity": [0, 0, 0, 0],  # 8
            "wh_charging": [0, 0, 0, 0],  # 9
            "wh_discharging": [0, 0, 0, 0],  # 10
            "temperature": [0, 0, 0, 0],  # 11
            "cycle_count": [1, 2, 1, 1],  # 12
            "max_temperature": [0, 0, 0, 0],  # 13
            "average_tension": [0, 0, 0, 0],  # 14
        }
    )

    # 4. Mock the `pd.read_csv` call.
    mock_read_csv = mocker.patch(
        "pandas.read_csv",
        side_effect=[
            iter([fake_cyc_df]),  # First call returns cycle data
            iter([fake_cap_df]),  # Second call returns capacity data
        ],
    )

    return mock_read_csv


def test_unibo_data_loading(mock_unibo_csvs):
    """
    Tests that UniboPowertoolsData successfully loads and filters the mock CSVs.
    """
    # 1. --- Arrange & Act ---
    dataset = UniboPowertoolsData(base_path="fake/path", discharge_line=40)

    # 2. --- Assert ---
    assert mock_unibo_csvs.call_count == 2
    assert not dataset.cycle_raw.empty
    assert not dataset.cap_raw.empty
    assert len(dataset.discharge_cyc_raw) == 7  # 6 + 1 for batt3
    assert len(dataset.discharge_cap_raw) == 4  # 3 + 1 for batt3
    assert len(dataset.charge_cyc_raw) == 0


def test_unibo_prepare_data(mock_unibo_csvs):
    """
    Tests the `prepare_data` and `__get_cyc_and_cap` logic.
    """
    # 1. --- Arrange ---
    dataset = UniboPowertoolsData(
        base_path="fake/path", discharge_line=40, charge_line=40
    )

    # Define our splits. 'test' now has two batteries to make it ragged.
    train_names = ["batt1"]
    test_names = ["batt2", "batt3"]
    val_names = []

    # 2. --- Act ---
    dataset.prepare_data(train_names, test_names, val_names)

    # 3. --- Assert ---
    cyc_data, cap_data = dataset.get_discharge_data()

    # The 'cyc' data starts with 13 columns, and `__add_discharge_soc_pars`
    # adds 2, for a final total of 15.
    num_cyc_cols = 15
    # The 'cap' data starts with 15 columns, and `__add_discharge_soh_pars`
    # adds 5, for a final total of 20.
    num_cap_cols = 20

    # --- Check Train Split (batt1) ---
    # `batt1` had 2 cycles
    assert "train" in cyc_data
    assert len(cyc_data["train"]) == 2
    assert cyc_data["train"][0].shape == (3, num_cyc_cols)  # Cycle 1 had 3 timesteps
    assert cyc_data["train"][1].shape == (1, num_cyc_cols)  # Cycle 2 had 1 timestep

    assert "train" in cap_data
    assert cap_data["train"].shape[0] == 2  # 2 cycles
    assert cap_data["train"][0].shape == (num_cap_cols,)  # Check shape
    assert (
        cap_data["train"][0][CapacityCols.CORRESPONDING_CHARGING_CAPACITY] == 3.0
    )  # SOH for cycle 1
    assert (
        cap_data["train"][1][CapacityCols.CORRESPONDING_CHARGING_CAPACITY] == 2.9
    )  # SOH for cycle 2

    # --- Check Test Split (batt2, batt3) ---
    # `batt2` had 1 cycle, `batt3` had 1 cycle. Total 2.
    assert "test" in cyc_data
    assert len(cyc_data["test"]) == 2  # 1 cycle from batt2, 1 from batt3
    assert cyc_data["test"][0].shape == (
        2,
        num_cyc_cols,
    )  # batt2 Cycle 1 had 2 timesteps
    assert cyc_data["test"][1].shape == (
        1,
        num_cyc_cols,
    )  # batt3 Cycle 1 had 1 timestep

    assert "test" in cap_data
    assert cap_data["test"].shape[0] == 2  # 2 cycles
    assert cap_data["test"][0].shape == (num_cap_cols,)  # Check shape
    assert (
        cap_data["test"][0][CapacityCols.CORRESPONDING_CHARGING_CAPACITY] == 3.0
    )  # SOH for batt2 cycle 1

    # --- Check Val Split (empty) ---
    assert "val" not in cyc_data


# =========================================================================
# === Part 3: Test the ModelDataHandler Class (Object Mocking) ===
# =========================================================================


def test_model_handler_get_discharge_whole_cycle_future(mocker):
    """
    Tests the ModelDataHandler's main processing method.
    We check for correct padding and sign-flipping of the current.
    """

    # 1. --- Arrange ---
    # We will mock the helper method `__get_whole_cycle_soh_x_y` to
    # control exactly what it returns.

    # Define the fake data it will return
    # Features (X) - [V, I, T]
    raw_x_train = [np.array([[3.9, -5, 30], [3.8, -5, 31]]), np.array([[3.7, -5, 32]])]
    raw_x_val = [np.array([[4.0, -4, 25]])]
    raw_x_test = [np.array([[2.0, -1, 20]])]
    # SOH (Y) - [Corresponding_Charging_Capacity]
    y_train = np.array([[3.0], [2.9]])
    y_val = np.array([[3.0]])
    y_test = np.array([[2.5]])
    # Time
    time_train = [np.array([[0], [1]]), np.array([[0]])]
    time_val = [np.array([[0]])]
    time_test = [np.array([[0]])]

    # Patch the helper method (assert_called at end verifies patch was used)
    mock_get_soh = mocker.patch.object(
        ModelDataHandler,
        "_ModelDataHandler__get_whole_cycle_soh_x_y",
        # The processing order is train, test, val.
        side_effect=[
            # 1. Train X, Y
            (raw_x_train, y_train),
            # 2. Train Time
            (time_train, None),
            # 3. Test X, Y
            (raw_x_test, y_test),
            # 4. Test Time
            (time_test, None),
            # 5. Val X, Y
            (raw_x_val, y_val),
            # 6. Val Time
            (time_val, None),
        ],
    )

    # We need to create a minimal fake `data["cap"]` array.
    num_cap_cols = 20  # An approximation is fine for this mock

    # Create fake capacity data for the 'train' split
    fake_cap_train = np.empty((2, num_cap_cols), dtype=object)
    fake_cap_train[0, CapacityCols.TEST_NAME] = "batt1"
    fake_cap_train[1, CapacityCols.TEST_NAME] = "batt2"

    # Create fake capacity data for the 'val' split
    fake_cap_val = np.empty((1, num_cap_cols), dtype=object)
    fake_cap_val[0, CapacityCols.TEST_NAME] = "batt3"

    # Create fake capacity data for the 'test' split
    fake_cap_test = np.empty((1, num_cap_cols), dtype=object)
    fake_cap_test[0, CapacityCols.TEST_NAME] = "batt4"

    # Mock the dataset object (it's only used for its .get methods in __init__)
    mock_dataset = mocker.MagicMock(spec=UniboPowertoolsData)
    mock_dataset.get_discharge_data.return_value = (
        # `cyc_all` (index 0)
        {"train": "fake", "val": "fake", "test": "fake"},
        # `cap_all` (index 1)
        {"train": fake_cap_train, "val": fake_cap_val, "test": fake_cap_test},
    )
    mock_dataset.get_charge_data.return_value = ({}, {})

    # Instantiate the handler
    handler = ModelDataHandler(
        dataset=mock_dataset,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT, CycleCols.TEMPERATURE],
    )
    # We must mock the scaler creation as we didn't provide real data
    mocker.patch.object(handler, "_ModelDataHandler__assign_scalers", return_value=None)

    # 2. --- Act ---
    # Call the function under test
    (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        train_range,
        val_range,
        test_range,
        train_time,
        val_time,
        test_time,
        train_current,
        val_current,
        test_current,
        train_len,
        val_len,
        test_len,
    ) = handler.get_discharge_whole_cycle_future(
        train_names=["batt1", "batt2"],  # Names must match length of y_train
        test_names=["batt4"],
        val_names=["batt3"],  # Names must match length of y_val
    )

    # 3. --- Assert ---
    # The function will unpack:
    # train_y = y_train
    # val_y = y_val
    # test_y = y_test

    # --- Check SOH (Y) data ---
    assert np.all(train_y == y_train)  # (3.0, 2.9) == (3.0, 2.9)
    assert np.all(val_y == y_val)  # (3.0) == (3.0)
    assert np.all(test_y == y_test)  # (2.5) == (2.5)

    # Max length is 2 (from the first train cycle)

    # --- Check Padding (Shape) ---
    assert train_x.shape == (2, 2, 3)  # (n_cycles, max_len, n_features)
    assert val_x.shape == (1, 2, 3)
    assert test_x.shape == (1, 2, 3)

    assert train_time.shape == (2, 2, 1)  # (n_cycles, max_len, 1)
    assert val_time.shape == (1, 2, 1)
    assert test_time.shape == (1, 2, 1)

    # --- Check the CRITICAL current flip ---
    # train_current should be the (padded) current, but positive
    expected_train_current = np.array(
        [[5.0, 5.0], [5.0, 0.0]]  # Cycle 1  # Cycle 2 (padded)
    )
    assert np.all(train_current == expected_train_current)

    # val_current
    expected_val_current = np.array([[4.0, 0.0]])  # Cycle 1 (padded)
    assert np.all(val_current == expected_val_current)

    # test_current
    expected_test_current = np.array([[1.0, 0.0]])  # Cycle 1 (padded)
    assert np.all(test_current == expected_test_current)

    # --- Check other outputs ---
    assert val_len == [1]
    assert train_len == [2, 1]
    assert test_len == [1]

    # --- Check battery range (this will now pass) ---
    assert np.all(train_range == np.array([1, 2]))
    assert np.all(val_range == np.array([1]))
    assert np.all(test_range == np.array([1]))

    mock_get_soh.assert_called()


# =========================================================================
# === Part 4: UNIBO21Loader anomaly_detection task_mode ===
# =========================================================================


def test_unibo_loader_anomaly_detection_process_split_returns_binary_target():
    """
    When task_mode is anomaly_detection and rul_anomaly_fraction is set,
    _process_split converts RUL to binary (0=normal, 1=anomaly) per unit.
    """
    from picid.data.datasources.unibo.loader import UNIBO21Loader

    loader = UNIBO21Loader(
        data_dir="/tmp",
        data_name="UNIBO21",
        task_mode="anomaly_detection",
        rul_anomaly_fraction=0.3,
    )
    # RUL: 10, 8, 6, 4, 2, 0 -> max=10, threshold=0.3*10=3 -> anomaly where RUL<=3
    # So binary target should be [0,0,0,1,1,1]
    x = np.zeros((6, 2, 3), dtype=np.float32)  # 6 cycles, len 2, 3 features
    y = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 0.0])
    battery_range = [6]
    names = ["000-DM-3.0-4019-S"]  # must be in UNIT_NAMES_TO_ID
    # Flat list of per-cycle lengths (one int per cycle)
    valid_lengths = [2, 2, 2, 2, 2, 2]

    result = loader._process_split(
        x=x,
        y=y,
        battery_range=battery_range,
        names=names,
        valid_lengths=valid_lengths,
        flatten=True,
    )
    assert len(result) == 1
    target = result[0]["target"]
    # Awkward array: flatten to numpy for assertion
    if hasattr(target, "to_numpy"):
        arr = np.asarray(target.to_numpy()).ravel()
    else:
        arr = np.asarray(ak.ravel(target)).ravel()
    assert np.isin(
        arr, [0.0, 1.0]
    ).all(), f"Expected only 0 and 1, got {np.unique(arr)}"
    assert 0.0 in arr and 1.0 in arr, "Expected both labels present"


def test_unibo_loader_anomaly_detection_absolute_threshold():
    """Absolute threshold: anomaly when RUL <= rul_anomaly_threshold."""
    from picid.data.datasources.unibo.loader import UNIBO21Loader

    loader = UNIBO21Loader(
        data_dir="/tmp",
        data_name="UNIBO21",
        task_mode="anomaly_detection",
        rul_anomaly_threshold=5.0,
    )
    y = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 0.0])  # RUL <= 5 -> last three are anomaly
    x = np.zeros((6, 2, 3), dtype=np.float32)
    result = loader._process_split(
        x=x,
        y=y,
        battery_range=[6],
        names=["000-DM-3.0-4019-S"],
        valid_lengths=[2, 2, 2, 2, 2, 2],
        flatten=True,
    )
    target = result[0]["target"]
    arr = np.asarray(ak.ravel(target)).ravel()
    assert np.isin(arr, [0.0, 1.0]).all()
    # 10,8,6 > 5 -> 0; 4,2,0 <= 5 -> 1
    assert arr[0] == 0.0 and arr[-1] == 1.0


def test_unibo_loader_cache_fingerprint_captures_effective_anomaly_defaults():
    from picid.data.datasources.unibo.loader import UNIBO21Loader

    loader = UNIBO21Loader(
        data_dir="/tmp",
        data_name="UNIBO21",
        task_mode="anomaly_detection",
    )

    assert loader.get_cache_fingerprint()["rul_anomaly_fraction"] == 0.3
    assert loader.get_cache_fingerprint()["rul_anomaly_threshold"] is None


def test_unibo_loader_rejects_multisource_splitter_and_uses_predefined_splits():
    """UNIBO21 is predefined-split only, so a splitter config is ignored."""
    from picid.data.data_objects import SplitDatasetContainer
    from picid.data.datasources.unibo.loader import UNIBO21Loader

    mock_data = {
        "train": [
            {
                "features": np.ones((1, 2, 3)),
                "target": np.ones((1, 2, 1)),
                "metadata": {
                    "unit_name": "000-DM-3.0-4019-S",
                    "unit_id": 0,
                    "n_cycles": 1,
                },
            }
        ],
        "val": [
            {
                "features": np.ones((1, 2, 3)) * 2,
                "target": np.ones((1, 2, 1)) * 2,
                "metadata": {
                    "unit_name": "002-DM-3.0-4019-S",
                    "unit_id": 2,
                    "n_cycles": 1,
                },
            }
        ],
        "test": [
            {
                "features": np.ones((1, 2, 3)) * 3,
                "target": np.ones((1, 2, 1)) * 3,
                "metadata": {
                    "unit_name": "003-DM-3.0-4019-S",
                    "unit_id": 3,
                    "n_cycles": 1,
                },
            }
        ],
    }

    with (
        patch.object(UNIBO21Loader, "read_data", return_value=mock_data),
        pytest.raises(
            DatasourceConfigurationError,
            match="does not accept multisource_data_splitter",
        ),
    ):
        UNIBO21Loader(
            data_dir="/tmp",
            data_name="UNIBO21",
            task_mode="regression",
            multisource_data_splitter=object(),
        )

    with patch.object(UNIBO21Loader, "read_data", return_value=mock_data):
        loader = UNIBO21Loader(
            data_dir="/tmp",
            data_name="UNIBO21",
            task_mode="regression",
        )
        loader.load_data()

    data = loader.get_data()
    assert isinstance(data, SplitDatasetContainer)
    assert loader.get_split_mode() == "within_units"
    assert loader.get_data_name() == "UNIBO21"
    assert loader.get_data_names() == ("UNIBO21",)
