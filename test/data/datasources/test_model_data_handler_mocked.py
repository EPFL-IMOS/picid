"""
Tests for unibo.model_data_handler with mocked dataset (no real data loading).
"""

from __future__ import annotations

import numpy as np
import pytest

from picid.data.datasources.unibo.model_data_handler import ModelDataHandler
from picid.data.datasources.unibo.unibo_powertools_data import CycleCols, CapacityCols


def _make_cap(n_cycles: int, test_name: str, n_cols: int = 20):
    """Cap array with TEST_NAME column set for get_discharge_whole_cycle_future."""
    cap = np.zeros((n_cycles, n_cols), dtype=object)
    cap[:, CapacityCols.TEST_NAME] = test_name
    return cap


@pytest.fixture
def mock_dataset():
    """Dataset mock returning minimal charge/discharge structure for __init__ and scalers."""
    from unittest.mock import MagicMock

    # train: one "unit" with 10 steps, 20 cols (CycleCols etc.)
    train_cyc = [np.random.randn(10, 20).astype(np.float64)]
    mock = MagicMock()
    mock.get_charge_data.return_value = (
        {"train": train_cyc, "val": [], "test": []},
        {
            "train": np.zeros((1, 20)),
            "val": np.zeros((0, 20)),
            "test": np.zeros((0, 20)),
        },
    )
    mock.get_discharge_data.return_value = (
        {"train": train_cyc, "val": [], "test": []},
        {
            "train": np.zeros((1, 20)),
            "val": np.zeros((0, 20)),
            "test": np.zeros((0, 20)),
        },
    )
    return mock


@pytest.fixture
def mock_dataset_train_test_val():
    """Mock with train, test, and val non-empty for get_discharge_* and get_discharge_whole_cycle_future."""
    from unittest.mock import MagicMock

    n_cols = 20
    train_cyc = [np.random.randn(8, n_cols).astype(np.float64)]
    test_cyc = [np.random.randn(6, n_cols).astype(np.float64)]
    val_cyc = [np.random.randn(5, n_cols).astype(np.float64)]
    train_cap = _make_cap(1, "train", n_cols)
    test_cap = _make_cap(1, "test", n_cols)
    val_cap = _make_cap(1, "val", n_cols)
    mock = MagicMock()
    mock.get_charge_data.return_value = (
        {"train": train_cyc, "val": val_cyc, "test": test_cyc},
        {
            "train": np.zeros((1, n_cols)),
            "val": np.zeros((1, n_cols)),
            "test": np.zeros((1, n_cols)),
        },
    )
    mock.get_discharge_data.return_value = (
        {"train": train_cyc, "val": val_cyc, "test": test_cyc},
        {"train": train_cap, "val": val_cap, "test": test_cap},
    )
    return mock


def test_model_data_handler_init_with_mock_dataset(mock_dataset):
    """ModelDataHandler __init__ with mocked get_charge_data/get_discharge_data."""
    handler = ModelDataHandler(
        dataset=mock_dataset,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT],
    )
    assert handler.train_charge_cyc is not None
    assert handler.train_discharge_cyc is not None
    assert handler.val_discharge_cyc is not None


def test_model_data_handler_assign_scalers_and_get_scalers(mock_dataset):
    """__assign_scalers and get_scalers with minimal train data (coverage)."""
    handler = ModelDataHandler(
        dataset=mock_dataset,
        x_indices=[CycleCols.VOLTAGE],
    )
    handler._ModelDataHandler__assign_scalers()
    charge_s, discharge_s = handler.get_scalers()
    assert charge_s is not None
    assert discharge_s is not None
    assert len(charge_s) == 1
    assert len(discharge_s) == 1


# --- Legacy getters (cover get_discharge_whole_cycle, single_step, multiple_step, future, keep_only) ---


def test_model_data_handler_get_discharge_whole_cycle_soh(mock_dataset_train_test_val):
    """get_discharge_whole_cycle(soh=True) uses SOH branch and discharge_scalers."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT, CycleCols.TEMPERATURE],
    )
    handler._ModelDataHandler__assign_scalers()
    train_x, train_y, test_x, test_y = handler.get_discharge_whole_cycle(soh=True)
    assert train_x.shape[0] == 1 and test_x.shape[0] == 1
    assert train_y.shape == (1, 2) and test_y.shape == (1, 2)


def test_model_data_handler_get_discharge_whole_cycle_soc(mock_dataset_train_test_val):
    """get_discharge_whole_cycle(soh=False) uses SOC branch and pads train_y/test_y."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT],
    )
    handler._ModelDataHandler__assign_scalers()
    train_x, train_y, test_x, test_y = handler.get_discharge_whole_cycle(soh=False)
    assert train_x.ndim == 3 and train_y.ndim == 3
    assert train_x.shape[0] == 1 and test_x.shape[0] == 1


def test_model_data_handler_get_discharge_whole_cycle_multiple_output_soh(
    mock_dataset_train_test_val,
):
    """get_discharge_whole_cycle(multiple_output=True, soh=True) repeats SOH per timestep."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT],
    )
    handler._ModelDataHandler__assign_scalers()
    train_x, train_y, test_x, test_y = handler.get_discharge_whole_cycle(
        multiple_output=True, soh=True
    )
    assert train_y.shape[0] == 1 and train_y.shape[1] == train_x.shape[1]


def test_model_data_handler_get_discharge_single_step_soh(mock_dataset_train_test_val):
    """get_discharge_single_step(soh=True) flattens to 2D and uses discharge_scalers."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT],
    )
    handler._ModelDataHandler__assign_scalers()
    train_x, train_y, test_x, test_y = handler.get_discharge_single_step(soh=True)
    assert train_x.ndim == 2 and train_y.ndim == 2
    assert train_x.shape[0] == 8 and test_x.shape[0] == 6


def test_model_data_handler_get_discharge_single_step_soc(mock_dataset_train_test_val):
    """get_discharge_single_step(soh=False) uses SOC branch."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT],
    )
    handler._ModelDataHandler__assign_scalers()
    train_x, train_y, test_x, test_y = handler.get_discharge_single_step(soh=False)
    assert train_x.ndim == 2 and train_y.ndim == 2


def test_model_data_handler_get_discharge_multiple_step_soc(
    mock_dataset_train_test_val,
):
    """get_discharge_multiple_step(steps=3, soh=False) produces 3D sliding windows."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT],
    )
    handler._ModelDataHandler__assign_scalers()
    train_x, train_y, test_x, test_y = handler.get_discharge_multiple_step(
        steps=3, soh=False
    )
    assert train_x.ndim == 3 and train_x.shape[1] == 3


def test_model_data_handler_get_discharge_multiple_step_soh(
    mock_dataset_train_test_val,
):
    """get_discharge_multiple_step(steps=2, soh=True) uses SOH branch."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE],
    )
    handler._ModelDataHandler__assign_scalers()
    train_x, train_y, test_x, test_y = handler.get_discharge_multiple_step(
        steps=2, soh=True
    )
    assert train_x.ndim == 3
    assert train_y.ndim in (2, 3)  # SOH: one value per window or repeated


def test_model_data_handler_get_discharge_multiple_step_multiple_output(
    mock_dataset_train_test_val,
):
    """get_discharge_multiple_step(multiple_output=True) covers multiple_output branch."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT],
    )
    handler._ModelDataHandler__assign_scalers()
    train_x, train_y, test_x, test_y = handler.get_discharge_multiple_step(
        steps=2, soh=False, multiple_output=True
    )
    assert train_x.ndim == 3 and train_y.ndim == 3


def test_model_data_handler_get_discharge_grouped_multiple_steps(
    mock_dataset_train_test_val,
):
    """get_discharge_grouped_multiple_steps calls whole_cycle then __whole_cycle_to_multiple_step."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT],
    )
    handler._ModelDataHandler__assign_scalers()
    train_x, train_y, test_x, test_y = handler.get_discharge_grouped_multiple_steps(
        steps=2, multiple_output=True
    )
    assert train_x.ndim >= 3 and train_y.ndim >= 3  # 4D when multiple_output + grouped


def test_model_data_handler_get_discharge_whole_cycle_future(
    mock_dataset_train_test_val,
):
    """get_discharge_whole_cycle_future returns 18-tuple; train/test present."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT, CycleCols.STEP_TIME],
    )
    handler._ModelDataHandler__assign_scalers()
    out = handler.get_discharge_whole_cycle_future(
        train_names=["train"],
        test_names=["test"],
    )
    assert len(out) == 18
    assert out[0] is not None and out[4] is not None  # train x, test x
    assert out[1] is not None and out[5] is not None  # train y, test y


def test_model_data_handler_get_discharge_whole_cycle_future_with_val(
    mock_dataset_train_test_val,
):
    """get_discharge_whole_cycle_future with val_names includes validation in output."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT, CycleCols.STEP_TIME],
    )
    handler._ModelDataHandler__assign_scalers()
    out = handler.get_discharge_whole_cycle_future(
        train_names=["train"],
        test_names=["test"],
        val_names=["val"],
    )
    assert len(out) == 18
    assert out[2] is not None and out[3] is not None  # val x, val y


def test_model_data_handler_keep_only_capacity(mock_dataset_train_test_val):
    """keep_only_capacity selects first column of y; all branches."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT],
    )
    handler._ModelDataHandler__assign_scalers()
    _, train_y, _, _ = handler.get_discharge_whole_cycle(soh=True)
    out = handler.keep_only_capacity(train_y, is_multiple_output=False)
    assert out.ndim == 1 or (out.ndim == 2 and out.shape[-1] == 1)
    out_multi = handler.keep_only_capacity(
        np.random.randn(2, 5, 2), is_multiple_output=True
    )
    assert out_multi.shape == (2, 5)  # y[:, :, 0] drops last dim
    out_grouped = handler.keep_only_capacity(
        np.random.randn(2, 3, 4, 2),
        is_multiple_output=True,
        is_grouped_multiple_step=True,
    )
    assert out_grouped.shape == (2, 3, 4)  # y[:, :, :, 0] drops last dim
    out_grouped_single = handler.keep_only_capacity(
        np.random.randn(2, 3, 2),
        is_multiple_output=False,
        is_grouped_multiple_step=True,
    )
    assert out_grouped_single.shape == (2, 3)  # y[:, :, 0] drops last dim


def test_model_data_handler_keep_only_time(mock_dataset_train_test_val):
    """keep_only_time selects second column of y; all branches."""
    handler = ModelDataHandler(
        dataset=mock_dataset_train_test_val,
        x_indices=[CycleCols.VOLTAGE, CycleCols.CURRENT],
    )
    handler._ModelDataHandler__assign_scalers()
    _, train_y, _, _ = handler.get_discharge_whole_cycle(soh=True)
    out = handler.keep_only_time(train_y, is_multiple_output=False)
    assert out is not None
    assert out.ndim == 1 or (out.ndim == 2 and out.shape[-1] == 1)
    out_multi = handler.keep_only_time(
        np.random.randn(2, 5, 2), is_multiple_output=True
    )
    assert out_multi.shape == (2, 5)  # y[:, :, 1] drops last dim
    out_grouped = handler.keep_only_time(
        np.random.randn(2, 3, 4, 2),
        is_multiple_output=True,
        is_grouped_multiple_step=True,
    )
    assert out_grouped.shape == (2, 3, 4)  # y[:, :, :, 1] drops last dim
