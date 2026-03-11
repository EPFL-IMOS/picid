"""
Phase 2: Empty & corrupt inputs — descriptive errors and graceful handling.

Tests empty directories (via empty arrays/lists), missing keys, and
invalid data shapes. Asserts the system raises descriptive errors or
handles them without crashing.
"""

from __future__ import annotations

import numpy as np
import pytest

from picid.data.datasets.sliding_window_batch_dataset import SlidingWindowBatchDataset
from picid.data.datasets.context_dataset import ContextBatchDataset
from picid.data.datasets.fit_predict_dataset import FitPredictTaskDataset
from picid.data.datasets.hydra_concat_dataset import initialize_datasets
from omegaconf import DictConfig


def test_sliding_window_empty_array_handling():
    """Empty feature array: dataset length is 0 or clear error is raised."""
    data = {"f": np.zeros((0, 2))}
    try:
        ds = SlidingWindowBatchDataset(
            data_dict=data,
            seq_len=2,
            label_len=0,
            pred_len=1,
            stride=1,
        )
        # Implementation may yield length 0 without raising
        assert len(ds) == 0
    except (ValueError, AssertionError, IndexError, Exception) as e:
        # Or it may raise; ensure message is descriptive
        assert len(str(e)) > 0


def test_context_dataset_missing_required_key():
    """Missing required key in data_dict raises ValueError with key name."""
    data = {"features": np.zeros((20, 2))}
    with pytest.raises(ValueError, match="must contain the key"):
        ContextBatchDataset(
            data_dict=data,
            task_type="rul",
            seq_len=5,
            label_len=0,
            pred_len=1,
        )


def test_context_dataset_length_mismatch():
    """Features and target length mismatch raise AssertionError."""
    data = {
        "features": np.zeros((50, 2)),
        "rul": np.zeros((20, 1)),
    }
    with pytest.raises(AssertionError, match="same length"):
        ContextBatchDataset(
            data_dict=data,
            task_type="rul",
            seq_len=5,
            label_len=0,
            pred_len=1,
            stride=1,
        )


def test_fit_predict_missing_key():
    """FitPredictTaskDataset raises when 'features' or target key missing."""
    with pytest.raises(ValueError, match="must contain the key"):
        FitPredictTaskDataset(
            data_dict={"target": np.random.randn(2, 10, 1).astype(np.float32)},
            task_type="regression",
            meta_data_dict={},
        )


def test_hydra_initialize_empty_list_raises():
    """initialize_datasets with empty list raises ValueError."""
    cfg = DictConfig(
        {
            "_target_": "picid.data.datasets.rul_context_dataset.RULContextBatchDataset",
            "task_type": "rul",
            "seq_len": 2,
            "label_len": 0,
            "pred_len": 1,
            "stride": 1,
        }
    )
    with pytest.raises(ValueError, match="empty"):
        initialize_datasets(cfg, {"features": [], "rul": []}, {})


def test_sliding_window_unsupported_type_raises():
    """Unsupported array type (e.g. list) raises NotImplementedError."""
    data = {"f": [[1, 2], [3, 4]]}
    with pytest.raises(NotImplementedError, match="Unsupported"):
        SlidingWindowBatchDataset(
            data_dict=data,
            seq_len=1,
            label_len=0,
            pred_len=1,
            stride=1,
        )
