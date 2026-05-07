"""Base datamodule implementation for picid experiment pipelines."""

import inspect
import logging
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Subset,
)

from picid.data.datasets.base import (
    BaseConcatDataset,
    BaseDataset,
    BaseVectorizedConcatDataset,
)
from picid.data.datasets.hydra_concat_dataset import NonVectorizedHydraConcatDataset

logger = logging.getLogger(__name__)


class BaseDataModule(LightningDataModule):
    """
    Lightning data module for split-aware RUL datasets.

    Parameters
    ----------
    dataset_train : BaseDataset | BaseConcatDataset
        Training split dataset.
    dataset_val : BaseDataset | BaseConcatDataset
        Validation split dataset.
    dataset_test : BaseDataset | BaseConcatDataset
        Test split dataset.
    train_batch_size : int | str
        Training batch size or ``"full"``.
    val_batch_size : int | str
        Validation batch size or ``"full"``.
    test_batch_size : int | str
        Test batch size or ``"full"``.
    shuffle_train : bool, default=True
        Whether to shuffle the training loader.
    shuffle_val : bool, default=False
        Whether to shuffle the validation loader.
    shuffle_test : bool, default=False
        Whether to shuffle the test loader.
    num_workers : int, default=0
        Number of loader workers.
    prefetch_factor : int, default=2
        Prefetch factor for loader workers.
    pin_memory : bool, default=True
        Whether to pin host memory.
    use_batch_sampler : bool, default=True
        Whether to use a batch sampler instead of direct batching.
    subset_range : tuple[int, int, int] | None, optional
        Optional debugging subset range.
    train_subset_ratio : float | None, optional
        Fractional subset for the training split.
    val_subset_ratio : float | None, optional
        Fractional subset for the validation split.
    test_subset_ratio : float | None, optional
        Fractional subset for the test split.
    subset_seed : int, default=42
        Random seed used for subsetting.
    profiler_cfg : Any, optional
        Optional profiler configuration.
    """

    _data: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]

    def __init__(
        self,
        # datasets: Dict[str, Dataset],
        dataset_train,
        dataset_val,
        dataset_test,
        train_batch_size: int | str,
        val_batch_size: int | str,
        test_batch_size: int | str,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        shuffle_test: bool = False,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        use_batch_sampler: bool = True,
        subset_range: Optional[
            Tuple[int, int, int]
        ] = None,  # Used for debugging purposes
        train_subset_ratio: Optional[float] = None,
        val_subset_ratio: Optional[float] = None,
        test_subset_ratio: Optional[float] = None,
        subset_seed: int = 42,
        profiler_cfg=None,
    ):
        # for ds in ["train", "val", "test"]:
        #     if ds not in datasets:
        #         assert False, (
        #             f"Missing {ds} dataset in the provided datasets dictionary."
        #         )

        super().__init__()

        for size in [train_batch_size, val_batch_size, test_batch_size]:
            if isinstance(size, str):
                assert size in ["full"], f"Batch size must be  'full', got {size}"
            else:
                assert (
                    isinstance(size, int) and size > 0
                ), f"Batch size must be a positive integer, got {size}"

        for shuffle in [shuffle_val, shuffle_test]:
            if shuffle:
                logging.warning(
                    "Shuffling validation and test datasets is not recommended. "
                    "This may lead to unexpected results, especially in evaluation metrics."
                )

        if not shuffle_train:
            logging.warning(
                "Shuffling the training dataset is recommended for better model generalization."
            )

        # Automatically collect all init args
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)

        # Remove 'self' and 'datasets' from the list
        hparams = {
            arg: values[arg]
            for arg in args
            if arg
            not in ["self", "datasets", "dataset_train", "dataset_val", "dataset_test"]
        }

        self.profiler_cfg = profiler_cfg
        self.profiler = None
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.use_batch_sampler = use_batch_sampler
        self.train_subset_ratio = train_subset_ratio
        self.val_subset_ratio = val_subset_ratio
        self.test_subset_ratio = test_subset_ratio
        self.subset_seed = subset_seed

        # We need them because the debug subset does not have this function.
        self.collate_fn_train = (
            dataset_train.get_collate_fn()
            if (hasattr(dataset_train, "get_collate_fn") and not use_batch_sampler)
            else None
        )
        self.collate_fn_val = (
            dataset_val.get_collate_fn()
            if (hasattr(dataset_val, "get_collate_fn") and not use_batch_sampler)
            else None
        )
        self.collate_fn_test = (
            dataset_test.get_collate_fn()
            if (hasattr(dataset_test, "get_collate_fn") and not use_batch_sampler)
            else None
        )

        # for ds_name, ds in [
        #     ("dataset_train", dataset_train),
        #     ("dataset_val", dataset_val),
        #     ("dataset_test", dataset_test),
        # ]:
        #     if subset_range is not None:
        #         if len(range(*subset_range)) > len(ds):
        #             raise ValueError(
        #                 f"Subset range end index {subset_range} exceeds dataset length {len(ds)} for {ds_name}."
        #             )

        self.dataset_train = (
            dataset_train
            if subset_range is None or (subset_range[0] > dataset_train.__len__())
            else Subset(dataset_train, range(*subset_range))
        )

        self.dataset_train = (
            dataset_train
            if (subset_range is None) or (subset_range[0] > dataset_train.__len__())
            else Subset(dataset_train, range(*subset_range))
        )
        self.dataset_val = (
            dataset_val
            if (subset_range is None) or (subset_range[0] > dataset_val.__len__())
            else Subset(dataset_val, range(*subset_range))
        )
        self.dataset_test = (
            dataset_test
            if (subset_range is None) or (subset_range[0] > dataset_test.__len__())
            else Subset(dataset_test, range(*subset_range))
        )

        self.save_hyperparameters(
            hparams, ignore=["datasets", "dataset_train", "dataset_val", "dataset_test"]
        )

        for ds_name, ds in [
            ("dataset_train", dataset_train),
            ("dataset_val", dataset_val),
            ("dataset_test", dataset_test),
        ]:
            if not isinstance(
                ds,
                (
                    BaseDataset,
                    BaseConcatDataset,
                    BaseVectorizedConcatDataset,
                    NonVectorizedHydraConcatDataset,
                ),
            ):
                raise TypeError(
                    f"{ds_name} must be a subclass of BaseDataset or BaseConcatDataset, got {type(ds)}"
                )

        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Download and pre-process the underlying data.

        This calls the `prepare_data` function of the underlying reader. All
        previously completed preparation steps are skipped. It is called
        automatically by `pytorch_lightning` and executed on the first GPU in
        distributed mode.

        Parameters
        ----------
        *args
            Ignored. Only for adhering to parent class interface.
        **kwargs
            Ignored. Only for adhering to parent class interface.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load all splits into memory and optionally apply feature extraction.

        The splits are placed inside the `data`
        property. If a split is empty, a tuple of empty tensors with the correct
        number of dimensions is created as a placeholder. This ensures compatibility
        with higher-order data modules.

        If the data module was constructed with a `feature_extractor` argument,
        the feature windows are passed to the feature extractor. The resulting,
        new features may be re-windowed.

        Parameters
        ----------
        stage : str | None
            Ignored; kept for the Lightning interface.
        """
        pass

    def _create_dataloader(
        self,
        ds,
        batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=None,
        subset_ratio=None,
        split_name=None,
    ):
        """
        Build and return a DataLoader for the given dataset split.

        Parameters
        ----------
        ds : torch.utils.data.Dataset
            Dataset to wrap.
        batch_size : int
            Number of samples per batch.
        shuffle : bool, optional
            Whether to shuffle samples each epoch.
        drop_last : bool, optional
            Whether to drop the last incomplete batch.
        collate_fn : callable or None, optional
            Custom collate function passed to DataLoader.
        subset_ratio : float or None, optional
            If set, randomly subsample this fraction of the dataset.
        split_name : str or None, optional
            Name of the split (used for logging).

        Returns
        -------
        torch.utils.data.DataLoader
            Configured DataLoader ready for iteration.
        """
        loader_args = dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

        if subset_ratio is not None and 0 < subset_ratio < 1:
            rng = random.Random(self.subset_seed)  # independent RNG
            len_ds = len(ds)
            n_subset = int(len_ds * subset_ratio)
            subset_indices = rng.sample(range(len_ds), n_subset)
            subset_indices.sort()

            ds = Subset(ds, subset_indices)
            logger.info(
                f"Subsetting {split_name} dataset sequences: using {n_subset} out of {len_ds} sequences."
            )
            logger.info(f"Subset indices start with: {subset_indices[:100]}")

        if self.use_batch_sampler:
            assert collate_fn is None, "collate_fn not supported with use_batch_sampler"

            sampler = BatchSampler(
                RandomSampler(ds) if shuffle else SequentialSampler(ds),
                batch_size=batch_size,
                drop_last=drop_last,
            )

            loader_args["sampler"] = sampler
            # otherwise it adds a dimension with size 1 x ...
            loader_args["batch_size"] = None

        else:
            assert (
                collate_fn is not None
            ), "collate_fn must be provided if not using batch_sampler"

            loader_args["batch_size"] = batch_size
            loader_args["shuffle"] = shuffle
            loader_args["drop_last"] = drop_last
            loader_args["collate_fn"] = collate_fn

        loader = DataLoader(ds, **loader_args)

        if self.profiler_cfg is not None:
            return self.profiled_loader(loader)
        else:
            return loader

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a `torch.utils.data.DataLoader` for training.

        The data loader is configured to shuffle the data. The `pin_memory` option is
        activated to achieve maximum transfer speed to the GPU. The data loader is also
        configured to drop the last batch of the data if it would only contain one
        sample.

        The whole split is held in memory. Therefore, the `num_workers` are set to
        zero which uses the main process for creating batches.

        Parameters
        ----------
        *args
            Ignored. Only for adhering to parent class interface.
        **kwargs
            Ignored. Only for adhering to parent class interface.

        Returns
        -------
        DataLoader
            Training data loader.
        """

        if self.hparams.train_batch_size == "full":
            batch_size = len(self.dataset_train)
        else:
            batch_size = self.hparams.train_batch_size

        drop_last = len(self.dataset_train) % batch_size == 1

        if self._train_loader is None:
            loader = self._create_dataloader(
                self.dataset_train,
                batch_size=batch_size,
                shuffle=self.hparams.shuffle_train,
                drop_last=drop_last,
                collate_fn=self.collate_fn_train,
                subset_ratio=self.train_subset_ratio,
                split_name="train",
            )
            self._train_loader = loader
        else:
            loader = self._train_loader

        return loader

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a `torch.utils.data.DataLoader` for validation.

        The data loader is configured to leave the data unshuffled. The `pin_memory`
        option is activated to achieve maximum transfer speed to the GPU.

        The whole split is held in memory. Therefore, the `num_workers` are set to
        zero which uses the main process for creating batches.

        Parameters
        ----------
        *args
            Ignored. Only for adhering to parent class interface.
        **kwargs
            Ignored. Only for adhering to parent class interface.

        Returns
        -------
        DataLoader
            Validation data loader.
        """

        if self.hparams.val_batch_size == "full":
            batch_size = len(self.dataset_val)
        else:
            batch_size = self.hparams.val_batch_size

        if self._val_loader is None:
            loader = self._create_dataloader(
                self.dataset_val,
                batch_size=batch_size,
                shuffle=self.hparams.shuffle_val,
                collate_fn=self.collate_fn_val,
                subset_ratio=self.val_subset_ratio,
                split_name="val",
            )
            self._val_loader = loader
        else:
            loader = self._val_loader

        return loader

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a `torch.utils.data.DataLoader` for testing.

        The data loader is configured to leave the data unshuffled. The `pin_memory`
        option is activated to achieve maximum transfer speed to the GPU.

        The whole split is held in memory. Therefore, the `num_workers` are set to
        zero which uses the main process for creating batches.

        Parameters
        ----------
        *args
            Ignored. Only for adhering to parent class interface.
        **kwargs
            Ignored. Only for adhering to parent class interface.

        Returns
        -------
        DataLoader
            Test data loader.
        """
        if self.hparams.test_batch_size == "full":
            batch_size = len(self.dataset_test)
        else:
            batch_size = self.hparams.test_batch_size

        if self._test_loader is None:
            loader = self._create_dataloader(
                self.dataset_test,
                batch_size=batch_size,
                shuffle=self.hparams.shuffle_test,
                collate_fn=self.collate_fn_test,
                subset_ratio=self.test_subset_ratio,
                split_name="test",
            )
            self._test_loader = loader
        else:
            loader = self._test_loader

        return loader

    def profiled_loader(self, dl):
        """
        Wrap a DataLoader with a PyTorch profiler and return a profiled iterator.

        Parameters
        ----------
        dl : torch.utils.data.DataLoader
            DataLoader to profile.

        Returns
        -------
        callable
            Zero-argument callable that runs the loader under the profiler schedule
            configured in ``self.profiler_cfg``.
        """
        from torch.profiler import (
            ProfilerActivity,
            profile,
            record_function,
            schedule,
            ProfilerAction,
        )

        def profiled_loader():
            """Run the wrapped DataLoader under the configured profiler schedule."""
            sched = schedule(
                wait=self.profiler_cfg.wait,
                warmup=self.profiler_cfg.warmup,
                active=self.profiler_cfg.active,
                repeat=self.profiler_cfg.repeat,
            )

            acts = [getattr(ProfilerActivity, a) for a in self.profiler_cfg.activities]
            self.profiler = profile(
                activities=acts,
                schedule=sched,
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            )
            self.profiler.start()

            try:
                for batch in dl:
                    with record_function("dataloader_iteration"):
                        yield batch
                    self.profiler.step()

                    # ask the schedule where we are
                    state = sched(self.profiler.step_num)
                    if state == ProfilerAction.NONE:
                        self.profiler.stop()
                        logger.warning(
                            "Profiler has finished, terminating program. Printing results..."
                        )
                        logger.warning(
                            "\n%s",
                            self.profiler.key_averages().table(
                                sort_by="self_cpu_time_total", row_limit=10
                            ),
                        )
                        self.profiler.export_chrome_trace(self.profiler_cfg.trace_path)
                        logger.warning(
                            f"Profiler trace written to {self.profiler_cfg.trace_path}"
                        )
                        sys.exit(0)  # terminate program after profiler is done
            finally:
                # ensures clean shutdown if interrupted
                if self.profiler is not None:
                    self.profiler.stop()

        return profiled_loader()
