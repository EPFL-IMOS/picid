"""
Phase 1: FitPredictTaskDataset — shape, dtype, and pipeline consistency.

Uses synthetic 3D arrays (n_tasks, n_samples, n_features) for real assertions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from picid.data.datasets.fit_predict_dataset import FitPredictTaskDataset


@pytest.fixture
def fit_predict_data_dict(synthetic_fit_predict_3d):
    """data_dict with 'features' and target key for regression/forecasting."""
    X = synthetic_fit_predict_3d["X"]
    y = synthetic_fit_predict_3d["y"]
    return {"features": X, "target": y, "regression": y, "rul": y}


@pytest.mark.parametrize("task_type", ["regression", "rul", "forecasting"])
def test_fit_predict_init_and_length(fit_predict_data_dict, task_type):
    """Dataset initializes and __len__ equals n_tasks."""
    target_key = "target" if task_type == "forecasting" else task_type
    data = {
        "features": fit_predict_data_dict["features"],
        target_key: fit_predict_data_dict[target_key],
    }
    ds = FitPredictTaskDataset(
        data_dict=data,
        task_type=task_type,
        meta_data_dict={},
    )
    assert len(ds) == data["features"].shape[0]


def test_fit_predict_getitem_shape_and_dtype(fit_predict_data_dict):
    """__getitem__(task_list) returns context/target with correct shape and dtype."""
    ds = FitPredictTaskDataset(
        data_dict=fit_predict_data_dict,
        task_type="regression",
        meta_data_dict={},
    )
    # BatchSampler gives list of indices; dataset expects task = list[int]
    item = ds[[0]]
    assert "context" in item and "target" in item
    # (1, n_samples, n_features) and (1, n_samples, n_targets)
    assert item["context"].dim() == 3
    assert item["target"].dim() == 3
    assert item["context"].dtype in (torch.float32, torch.float64)
    assert item["target"].dtype in (torch.float32, torch.float64)
    n_samples = fit_predict_data_dict["features"].shape[1]
    n_features = fit_predict_data_dict["features"].shape[2]
    assert item["context"].shape == (1, n_samples, n_features)
    assert item["target"].shape == (
        1,
        n_samples,
        fit_predict_data_dict["target"].shape[2],
    )


def test_fit_predict_task_idx_and_desc(fit_predict_data_dict):
    """task_idx and task_desc match the requested task index."""
    ds = FitPredictTaskDataset(
        data_dict=fit_predict_data_dict,
        task_type="regression",
        meta_data_dict={},
    )
    item = ds[[2]]
    assert item["task_idx"].item() == 2
    assert "3 of" in item["task_desc"] and "4" in item["task_desc"]


def test_fit_predict_missing_key_raises():
    """Missing required key raises ValueError."""
    with pytest.raises(ValueError, match="must contain the key"):
        FitPredictTaskDataset(
            data_dict={"features": np.random.randn(2, 10, 5).astype(np.float32)},
            task_type="regression",
            meta_data_dict={},
        )


def test_fit_predict_unknown_task_type_raises(fit_predict_data_dict):
    """Unknown task_type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown task_type"):
        FitPredictTaskDataset(
            data_dict=fit_predict_data_dict,
            task_type="unknown_task",
            meta_data_dict={},
        )


def test_fit_predict_convert_units_to_tasks_list_of_arrays():
    """With convert_units_to_tasks=True, list of 3D arrays is concatenated along task dim."""
    X_list = [np.random.randn(1, 20, 4).astype(np.float32) for _ in range(3)]
    y_list = [np.random.randn(1, 20, 1).astype(np.float32) for _ in range(3)]
    ds = FitPredictTaskDataset(
        data_dict={"features": X_list, "rul": y_list},
        task_type="rul",
        meta_data_dict={},
        convert_units_to_tasks=True,
    )
    assert len(ds) == 3
    assert ds.X.shape == (3, 20, 4)
    assert ds.y.shape == (3, 20, 1)


def test_fit_predict_convert_units_to_tasks_false_single_task():
    """With convert_units_to_tasks=False, list of (1,N,F) arrays concatenated along sample dim, one task."""
    X_list = [np.random.randn(1, 10, 4).astype(np.float32) for _ in range(2)]
    y_list = [np.random.randn(1, 10, 1).astype(np.float32) for _ in range(2)]
    ds = FitPredictTaskDataset(
        data_dict={"features": X_list, "rul": y_list},
        task_type="rul",
        meta_data_dict={},
        convert_units_to_tasks=False,
    )
    assert len(ds) == 1
    assert ds.X.shape == (1, 20, 4)
    assert ds.y.shape == (1, 20, 1)


def test_fit_predict_subset_range_applied_to_getitem():
    """subset_range slices context/target in __getitem__."""
    X = np.random.randn(2, 50, 4).astype(np.float32)
    y = np.random.randn(2, 50, 1).astype(np.float32)
    ds = FitPredictTaskDataset(
        data_dict={"features": X, "rul": y},
        task_type="rul",
        meta_data_dict={},
        subset_range=(0, 10, 1),
    )
    item = ds[[0]]
    assert item["context"].shape == (1, 10, 4)
    assert item["target"].shape == (1, 10, 1)


def test_fit_predict_get_unit_id_includes_unit_id():
    """When get_unit_id=True and unit_id in data_dict, __getitem__ includes unit_id."""
    X = np.random.randn(2, 30, 4).astype(np.float32)
    y = np.random.randn(2, 30, 1).astype(np.float32)
    unit_id = np.arange(2 * 30, dtype=np.float32).reshape(2, 30, 1)
    ds = FitPredictTaskDataset(
        data_dict={"features": X, "rul": y, "unit_id": unit_id},
        task_type="rul",
        meta_data_dict={},
        dataset_cfg={"get_unit_id": True},
    )
    item = ds[[0]]
    assert "unit_id" in item
    assert item["unit_id"].shape == (1, 30, 1) or item["unit_id"].numel() == 30


def test_fit_predict_classification_task_types():
    """All classification task types initialize and return correct keys."""
    X = np.random.randn(2, 20, 4).astype(np.float32)
    y_cls = np.random.randint(0, 3, size=(2, 20, 1)).astype(np.float32)
    for task_type in [
        "classification",
        "health_states",
        "concepts",
        "fault_classification",
    ]:
        key = (
            "fault_classification" if task_type == "fault_classification" else task_type
        )
        data = {"features": X, key: y_cls}
        ds = FitPredictTaskDataset(
            data_dict=data, task_type=task_type, meta_data_dict={}
        )
        assert len(ds) == 2
        item = ds[[0]]
        assert "context" in item and "target" in item


def test_fit_predict_inconsistent_tasks_raises():
    """X and y with different task dims raise AssertionError."""
    X = np.random.randn(3, 20, 4).astype(np.float32)
    y = np.random.randn(2, 20, 1).astype(np.float32)
    with pytest.raises(AssertionError, match="Inconsistent number of tasks"):
        FitPredictTaskDataset(
            data_dict={"features": X, "rul": y},
            task_type="rul",
            meta_data_dict={},
        )


def test_fit_predict_subset_range_with_unit_id():
    """subset_range and get_unit_id together: unit_id is sliced in __getitem__."""
    X = np.random.randn(1, 50, 4).astype(np.float32)
    y = np.random.randn(1, 50, 1).astype(np.float32)
    unit_id = np.arange(50, dtype=np.float32).reshape(1, 50, 1)
    ds = FitPredictTaskDataset(
        data_dict={"features": X, "rul": y, "unit_id": unit_id},
        task_type="rul",
        meta_data_dict={},
        dataset_cfg={"get_unit_id": True},
        subset_range=(0, 10, 1),
    )
    item = ds[[0]]
    assert "unit_id" in item
    assert item["unit_id"].shape == (10, 1)


def test_fit_predict_inconsistent_timesteps_raises():
    """X and y with different time steps raise AssertionError."""
    X = np.random.randn(2, 30, 4).astype(np.float32)
    y = np.random.randn(2, 20, 1).astype(np.float32)
    with pytest.raises(AssertionError, match="time steps"):
        FitPredictTaskDataset(
            data_dict={"features": X, "rul": y},
            task_type="rul",
            meta_data_dict={},
        )


def test_fit_predict_get_collate_fn(fit_predict_data_dict):
    """get_collate_fn returns default_collate (coverage)."""
    from torch.utils.data.dataloader import default_collate

    ds = FitPredictTaskDataset(
        data_dict=fit_predict_data_dict,
        task_type="regression",
        meta_data_dict={},
    )
    assert ds.get_collate_fn() is default_collate
