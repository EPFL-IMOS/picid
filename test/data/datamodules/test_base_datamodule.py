"""
Phase 1: BaseDataModule — dataloader creation, batch sizes, subset.

Uses real datasets (e.g. SlidingWindowBatchDataset) to assert train/val/test
dataloaders and batch size handling (integer and "full").
"""

from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import DataLoader

from picid.data.datamodules.base import BaseDataModule
from picid.data.datasets.sliding_window_batch_dataset import SlidingWindowBatchDataset


@pytest.fixture
def tiny_sliding_dataset():
    """Single-modality sliding window dataset, 20 windows."""
    data = {"f": np.random.randn(50, 4).astype(np.float32)}
    return SlidingWindowBatchDataset(
        data_dict=data,
        seq_len=5,
        label_len=0,
        pred_len=2,
        stride=2,
    )


@pytest.fixture
def base_datamodule(tiny_sliding_dataset):
    """BaseDataModule with same dataset for train/val/test (for test isolation)."""
    return BaseDataModule(
        dataset_train=tiny_sliding_dataset,
        dataset_val=tiny_sliding_dataset,
        dataset_test=tiny_sliding_dataset,
        train_batch_size=4,
        val_batch_size=2,
        test_batch_size=2,
        shuffle_train=True,
        shuffle_val=False,
        shuffle_test=False,
        num_workers=0,
        use_batch_sampler=True,
    )


def test_train_dataloader_returns_loader(base_datamodule):
    """train_dataloader() returns a DataLoader."""
    dl = base_datamodule.train_dataloader()
    assert isinstance(dl, DataLoader)


def test_train_dataloader_batch_shape(base_datamodule):
    """First batch has expected batch size (when use_batch_sampler=True)."""
    dl = base_datamodule.train_dataloader()
    batch = next(iter(dl))
    assert "f_seq_x" in batch
    assert batch["f_seq_x"].shape[0] == 4


def test_val_and_test_dataloaders(base_datamodule):
    """val and test dataloaders return batches of size 2."""
    val_batch = next(iter(base_datamodule.val_dataloader()))
    test_batch = next(iter(base_datamodule.test_dataloader()))
    assert val_batch["f_seq_x"].shape[0] == 2
    assert test_batch["f_seq_x"].shape[0] == 2


def test_full_batch_size_datamodule(tiny_sliding_dataset):
    """train_batch_size='full' uses full dataset as one batch."""
    dm = BaseDataModule(
        dataset_train=tiny_sliding_dataset,
        dataset_val=tiny_sliding_dataset,
        dataset_test=tiny_sliding_dataset,
        train_batch_size="full",
        val_batch_size="full",
        test_batch_size="full",
        num_workers=0,
        use_batch_sampler=True,
    )
    batch = next(iter(dm.train_dataloader()))
    assert batch["f_seq_x"].shape[0] == len(tiny_sliding_dataset)


def test_invalid_batch_size_raises(tiny_sliding_dataset):
    """Non-positive or non-'full' string batch size raises."""
    with pytest.raises(AssertionError):
        BaseDataModule(
            dataset_train=tiny_sliding_dataset,
            dataset_val=tiny_sliding_dataset,
            dataset_test=tiny_sliding_dataset,
            train_batch_size=0,
            val_batch_size=2,
            test_batch_size=2,
            num_workers=0,
            use_batch_sampler=True,
        )


def test_invalid_batch_size_string_raises(tiny_sliding_dataset):
    """Batch size string other than 'full' raises."""
    with pytest.raises(AssertionError, match="full"):
        BaseDataModule(
            dataset_train=tiny_sliding_dataset,
            dataset_val=tiny_sliding_dataset,
            dataset_test=tiny_sliding_dataset,
            train_batch_size="half",
            val_batch_size=2,
            test_batch_size=2,
            num_workers=0,
            use_batch_sampler=True,
        )


def test_shuffle_val_warning(tiny_sliding_dataset, caplog):
    """shuffle_val=True logs a warning."""
    with caplog.at_level("WARNING"):
        BaseDataModule(
            dataset_train=tiny_sliding_dataset,
            dataset_val=tiny_sliding_dataset,
            dataset_test=tiny_sliding_dataset,
            train_batch_size=2,
            val_batch_size=2,
            test_batch_size=2,
            shuffle_train=True,
            shuffle_val=True,
            num_workers=0,
            use_batch_sampler=True,
        )
    assert "validation" in caplog.text.lower() or "shuffl" in caplog.text.lower()


def test_shuffle_train_false_warning(tiny_sliding_dataset, caplog):
    """shuffle_train=False logs a warning."""
    with caplog.at_level("WARNING"):
        BaseDataModule(
            dataset_train=tiny_sliding_dataset,
            dataset_val=tiny_sliding_dataset,
            dataset_test=tiny_sliding_dataset,
            train_batch_size=2,
            val_batch_size=2,
            test_batch_size=2,
            shuffle_train=False,
            num_workers=0,
            use_batch_sampler=True,
        )
    assert "shuffl" in caplog.text.lower() or "generalization" in caplog.text.lower()


def test_wrong_dataset_type_raises(tiny_sliding_dataset):
    """Passing a non-BaseDataset/ConcatDataset type raises TypeError."""

    class NotADataset:
        pass

    with pytest.raises(TypeError, match="BaseDataset|BaseConcatDataset"):
        BaseDataModule(
            dataset_train=NotADataset(),
            dataset_val=tiny_sliding_dataset,
            dataset_test=tiny_sliding_dataset,
            train_batch_size=2,
            val_batch_size=2,
            test_batch_size=2,
            num_workers=0,
            use_batch_sampler=True,
        )


def test_subset_range_applied(tiny_sliding_dataset):
    """subset_range creates Subset for train/val/test when start < len."""
    from torch.utils.data import Subset

    dm = BaseDataModule(
        dataset_train=tiny_sliding_dataset,
        dataset_val=tiny_sliding_dataset,
        dataset_test=tiny_sliding_dataset,
        train_batch_size=2,
        val_batch_size=2,
        test_batch_size=2,
        num_workers=0,
        use_batch_sampler=True,
        subset_range=(0, 5, 1),
    )
    assert isinstance(dm.dataset_train, Subset)
    assert len(dm.dataset_train) == 5


def test_subset_range_skipped_when_start_gt_len(tiny_sliding_dataset):
    """subset_range is skipped when range start > dataset length."""
    dm = BaseDataModule(
        dataset_train=tiny_sliding_dataset,
        dataset_val=tiny_sliding_dataset,
        dataset_test=tiny_sliding_dataset,
        train_batch_size=2,
        val_batch_size=2,
        test_batch_size=2,
        num_workers=0,
        use_batch_sampler=True,
        subset_range=(1000, 1010, 1),
    )
    assert dm.dataset_train is tiny_sliding_dataset


def test_val_subset_ratio(tiny_sliding_dataset, caplog):
    """val_subset_ratio < 1 subsets the validation dataset in _create_dataloader."""
    with caplog.at_level("INFO"):
        dm = BaseDataModule(
            dataset_train=tiny_sliding_dataset,
            dataset_val=tiny_sliding_dataset,
            dataset_test=tiny_sliding_dataset,
            train_batch_size=4,
            val_batch_size=2,
            test_batch_size=2,
            num_workers=0,
            use_batch_sampler=True,
            val_subset_ratio=0.3,
            subset_seed=42,
        )
        dl = dm.val_dataloader()
        assert dl is not None
        assert hasattr(dl, "__iter__")
    assert "Subset" in caplog.text or "subset" in caplog.text.lower()


def test_test_subset_ratio(tiny_sliding_dataset, caplog):
    """test_subset_ratio < 1 subsets the test dataset in _create_dataloader."""
    with caplog.at_level("INFO"):
        dm = BaseDataModule(
            dataset_train=tiny_sliding_dataset,
            dataset_val=tiny_sliding_dataset,
            dataset_test=tiny_sliding_dataset,
            train_batch_size=4,
            val_batch_size=2,
            test_batch_size=2,
            num_workers=0,
            use_batch_sampler=True,
            test_subset_ratio=0.4,
            subset_seed=42,
        )
        dl = dm.test_dataloader()
        assert dl is not None
        assert hasattr(dl, "__iter__")
    assert "Subset" in caplog.text or "subset" in caplog.text.lower()


def test_train_subset_ratio(tiny_sliding_dataset, caplog):
    """train_subset_ratio < 1 subsets the dataset in _create_dataloader."""
    with caplog.at_level("INFO"):
        dm = BaseDataModule(
            dataset_train=tiny_sliding_dataset,
            dataset_val=tiny_sliding_dataset,
            dataset_test=tiny_sliding_dataset,
            train_batch_size=2,
            val_batch_size=2,
            test_batch_size=2,
            num_workers=0,
            use_batch_sampler=True,
            train_subset_ratio=0.5,
            subset_seed=42,
        )
        dl = dm.train_dataloader()
    assert "Subset" in caplog.text or "subset" in caplog.text.lower()
    batch = next(iter(dl))
    assert batch["f_seq_x"].shape[0] <= 2


def test_use_batch_sampler_false_with_collate_fn(tiny_sliding_dataset):
    """use_batch_sampler=False uses collate_fn from dataset (covers else branch in _create_dataloader)."""
    dm = BaseDataModule(
        dataset_train=tiny_sliding_dataset,
        dataset_val=tiny_sliding_dataset,
        dataset_test=tiny_sliding_dataset,
        train_batch_size=2,
        val_batch_size=2,
        test_batch_size=2,
        num_workers=0,
        use_batch_sampler=False,
    )
    assert dm.collate_fn_train is not None
    dl = dm.train_dataloader()
    assert dl.batch_size == 2
    assert dl.collate_fn is not None


def test_drop_last_when_remainder_one():
    """drop_last=True when len(dataset_train) % batch_size == 1."""
    # SlidingWindow length: (T - seq_len - pred_len) // stride + 1; use T so len=9, 9%4==1
    data = {"f": np.random.randn(23, 4).astype(np.float32)}
    ds = SlidingWindowBatchDataset(
        data_dict=data,
        seq_len=5,
        label_len=0,
        pred_len=2,
        stride=2,
        padding_left_flag=False,
    )
    n = len(ds)
    batch_size = 4
    assert n % batch_size == 1, f"expected len%batch_size==1, got len={n}"
    dm = BaseDataModule(
        dataset_train=ds,
        dataset_val=ds,
        dataset_test=ds,
        train_batch_size=batch_size,
        val_batch_size=2,
        test_batch_size=2,
        num_workers=0,
        use_batch_sampler=True,
    )
    dl = dm.train_dataloader()
    batches = list(iter(dl))
    assert all(b["f_seq_x"].shape[0] == batch_size for b in batches)


def test_train_dataloader_cached(base_datamodule):
    """Second call to train_dataloader returns same loader (line 319)."""
    dl1 = base_datamodule.train_dataloader()
    dl2 = base_datamodule.train_dataloader()
    assert dl1 is dl2


def test_val_dataloader_cached(base_datamodule):
    """Second call to val_dataloader returns same loader."""
    dl1 = base_datamodule.val_dataloader()
    dl2 = base_datamodule.val_dataloader()
    assert dl1 is dl2


def test_test_dataloader_cached(base_datamodule):
    """Second call to test_dataloader returns same loader."""
    dl1 = base_datamodule.test_dataloader()
    dl2 = base_datamodule.test_dataloader()
    assert dl1 is dl2


def test_prepare_data_no_op(base_datamodule):
    """prepare_data() is a no-op (pass)."""
    base_datamodule.prepare_data()


def test_setup_no_op(base_datamodule):
    """setup(stage) is a no-op (pass)."""
    base_datamodule.setup(stage="fit")


def test_val_dataloader_full_batch_size(tiny_sliding_dataset):
    """val_batch_size='full' uses full validation dataset as one batch."""
    dm = BaseDataModule(
        dataset_train=tiny_sliding_dataset,
        dataset_val=tiny_sliding_dataset,
        dataset_test=tiny_sliding_dataset,
        train_batch_size=4,
        val_batch_size="full",
        test_batch_size=2,
        num_workers=0,
        use_batch_sampler=True,
    )
    dl = dm.val_dataloader()
    batch = next(iter(dl))
    assert batch["f_seq_x"].shape[0] == len(tiny_sliding_dataset)


def test_test_dataloader_full_batch_size(tiny_sliding_dataset):
    """test_batch_size='full' uses full test dataset as one batch."""
    dm = BaseDataModule(
        dataset_train=tiny_sliding_dataset,
        dataset_val=tiny_sliding_dataset,
        dataset_test=tiny_sliding_dataset,
        train_batch_size=4,
        val_batch_size=2,
        test_batch_size="full",
        num_workers=0,
        use_batch_sampler=True,
    )
    dl = dm.test_dataloader()
    batch = next(iter(dl))
    assert batch["f_seq_x"].shape[0] == len(tiny_sliding_dataset)


def test_create_dataloader_without_batch_sampler(tiny_sliding_dataset):
    """_create_dataloader with use_batch_sampler=False uses batch_size and collate_fn (line 319)."""
    dm = BaseDataModule(
        dataset_train=tiny_sliding_dataset,
        dataset_val=tiny_sliding_dataset,
        dataset_test=tiny_sliding_dataset,
        train_batch_size=4,
        val_batch_size=2,
        test_batch_size=2,
        num_workers=0,
        use_batch_sampler=False,
    )
    loader = dm._create_dataloader(
        dm.dataset_train,
        batch_size=4,
        shuffle=True,
        drop_last=False,
        collate_fn=dm.collate_fn_train,
        subset_ratio=None,
        split_name="train",
    )
    assert loader.batch_size == 4
    assert loader.collate_fn is not None


def test_profiled_loader_path(tiny_sliding_dataset):
    """With profiler_cfg, train_dataloader uses profiled_loader (covers profiler_cfg branch)."""
    dm = BaseDataModule(
        dataset_train=tiny_sliding_dataset,
        dataset_val=tiny_sliding_dataset,
        dataset_test=tiny_sliding_dataset,
        train_batch_size=4,
        val_batch_size=2,
        test_batch_size=2,
        num_workers=0,
        use_batch_sampler=True,
        profiler_cfg=type(
            "Cfg",
            (),
            {
                "wait": 0,
                "warmup": 0,
                "active": 100,
                "repeat": 0,
                "activities": ["CPU"],
                "trace_path": "/tmp/trace.json",
            },
        )(),
    )
    dl = dm.train_dataloader()
    # Consume one batch to exercise profiled generator (schedule stays active, no exit)
    batch = next(iter(dl))
    assert isinstance(batch, dict)
