"""
Phase 1: HydraConcatDataset and initialize_datasets.

Tests initialization from data_dict of lists of arrays, shape consistency,
and list-indexing through BaseVectorizedConcatDataset.
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import DictConfig

from picid.data.datasets.hydra_concat_dataset import (
    initialize_datasets,
    HydraConcatDataset,
)


@pytest.fixture
def minimal_rul_dataset_cfg():
    """Minimal Hydra-style config for RULContextBatchDataset per unit."""
    return DictConfig(
        {
            "_target_": "picid.data.datasets.rul_context_dataset.RULContextBatchDataset",
            "task_type": "rul",
            "seq_len": 5,
            "label_len": 0,
            "pred_len": 0,
            "stride": 1,
            "meta_data_dict": {},
        }
    )


@pytest.fixture
def multi_unit_data_list(synthetic_multi_unit_rul):
    """data_dict as lists of arrays + metadata for Hydra."""
    # RULContextBatchDataset expects pred_len=1; we pass meta_data_dict with concat_dataset_index
    return {
        "features": synthetic_multi_unit_rul["features"],
        "rul": synthetic_multi_unit_rul["rul"],
    }


def test_initialize_datasets_lengths_must_match():
    """initialize_datasets raises if list lengths differ."""
    data_dict = {
        "features": [np.zeros((10, 2)), np.zeros((10, 2))],
        "rul": [np.zeros((10, 1))],
    }
    meta = {}
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
    with pytest.raises(AssertionError, match="same length"):
        initialize_datasets(cfg, data_dict, meta)


def test_initialize_datasets_empty_list_raises():
    """Empty list in data_dict raises ValueError."""
    data_dict = {"features": [], "rul": []}
    meta = {}
    cfg = DictConfig(
        {
            "_target_": "picid.data.datasets.rul_context_dataset.RULContextBatchDataset",
            "task_type": "rul",
            "seq_len": 2,
            "label_len": 0,
            "pred_len": 1,
            "stride": 1,
            "meta_data_dict": {},
        }
    )
    with pytest.raises(ValueError, match="empty"):
        initialize_datasets(cfg, data_dict, meta)


def test_hydra_concat_dataset_init_and_length(
    multi_unit_data_list, minimal_rul_dataset_cfg
):
    """HydraConcatDataset builds and total length is sum of child lengths."""
    meta = {
        "unit_ids": {"train": [0, 1, 2]},
        "unit_names": {"train": ["u0", "u1", "u2"]},
        "current_data_split": "train",
    }
    # RULContextBatchDataset requires pred_len=1 and meta_data_dict with concat_dataset_index (injected by initialize_datasets)
    concat = HydraConcatDataset(
        dataset_cfg=minimal_rul_dataset_cfg,
        data_dict=multi_unit_data_list,
        meta_data_dict=meta,
    )
    total_len = sum(len(concat.datasets[i]) for i in range(len(concat.datasets)))
    assert len(concat) == total_len


def test_hydra_concat_getitem_list_indexing(
    multi_unit_data_list, minimal_rul_dataset_cfg
):
    """List indexing returns batched dict; shapes have batch dim."""
    meta = {
        "unit_ids": {"train": [0, 1, 2]},
        "unit_names": {"train": ["u0", "u1", "u2"]},
        "current_data_split": "train",
    }
    concat = HydraConcatDataset(
        dataset_cfg=minimal_rul_dataset_cfg,
        data_dict=multi_unit_data_list,
        meta_data_dict=meta,
    )
    indices = [0, 1]
    batch = concat[indices]
    assert "features" in batch and "rul" in batch
    assert batch["features"].dim() == 3
    assert batch["features"].size(0) == 2


def test_hydra_concat_get_collate_fn(multi_unit_data_list, minimal_rul_dataset_cfg):
    """get_collate_fn returns callable."""
    meta = {
        "unit_ids": {"train": [0, 1, 2]},
        "unit_names": {"train": ["u0", "u1", "u2"]},
        "current_data_split": "train",
    }
    concat = HydraConcatDataset(
        dataset_cfg=minimal_rul_dataset_cfg,
        data_dict=multi_unit_data_list,
        meta_data_dict=meta,
    )
    fn = concat.get_collate_fn()
    assert callable(fn)


def test_initialize_datasets_returns_same_type(
    multi_unit_data_list, minimal_rul_dataset_cfg
):
    """initialize_datasets returns list of datasets all of the same type."""
    meta = {
        "unit_ids": {"train": [0, 1, 2]},
        "unit_names": {"train": ["u0", "u1", "u2"]},
        "current_data_split": "train",
    }
    datasets = initialize_datasets(minimal_rul_dataset_cfg, multi_unit_data_list, meta)
    assert len(datasets) == 3
    for ds in datasets:
        assert type(ds) is type(datasets[0])


def test_initialize_datasets_non_list_raises():
    """initialize_datasets raises TypeError for non-list value (except metadata DictConfig)."""
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
    data_dict = {"features": [np.zeros((10, 2))], "rul": np.zeros((10, 1))}
    with pytest.raises(TypeError, match="must be a list"):
        initialize_datasets(cfg, data_dict, {})


def test_non_vectorized_hydra_concat_deprecation(
    multi_unit_data_list, minimal_rul_dataset_cfg
):
    """NonVectorizedHydraConcatDataset issues DeprecationWarning and get_collate_fn works."""
    from picid.data.datasets.hydra_concat_dataset import NonVectorizedHydraConcatDataset

    meta = {
        "unit_ids": {"train": [0, 1, 2]},
        "unit_names": {"train": ["u0", "u1", "u2"]},
        "current_data_split": "train",
    }
    with pytest.warns(DeprecationWarning, match="deprecated"):
        concat = NonVectorizedHydraConcatDataset(
            dataset_cfg=minimal_rul_dataset_cfg,
            data_dict=multi_unit_data_list,
            meta_data_dict=meta,
        )
    assert callable(concat.get_collate_fn())
    assert len(concat.datasets) == 3
