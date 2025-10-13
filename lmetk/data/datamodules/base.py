import inspect
import logging
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

from lmetk.data.datasets.base import (
    BaseConcatDataset,
    BaseDataset,
    BaseVectorizedConcatDataset,
)
from lmetk.data.datasets.hydra_concat_dataset import NonVectorizedHydraConcatDataset

logger = logging.getLogger(__name__)


class BaseDataModule(LightningDataModule):
    """
    Data module for the RUL datasets.
    This is a similar module to the one in rul_datasets but simplified.
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

        self.dataset_train = (
            dataset_train
            if subset_range is None
            else Subset(dataset_train, range(*subset_range))
        )
        self.dataset_val = (
            dataset_val
            if subset_range is None
            else Subset(dataset_val, range(*subset_range))
        )
        self.dataset_test = (
            dataset_test
            if subset_range is None
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

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load all splits as tensors into memory and optionally apply feature extractor.

        The splits are placed inside the [data][rul_datasets.core.RulDataModule.data]
        property. If a split is empty, a tuple of empty tensors with the correct
        number of dimensions is created as a placeholder. This ensures compatibility
        with higher-order data modules.

        If the data module was constructed with a `feature_extractor` argument,
        the feature windows are passed to the feature extractor. The resulting,
        new features may be re-windowed.

        Args:
            stage: Ignored. Only for adhering to parent class interface.
        """
        pass

    def _create_dataloader(
        self, ds, batch_size, shuffle=True, drop_last=False, collate_fn=None
    ):

        loader_args = dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

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
        Create a [data loader][torch.utils.data.DataLoader] for the training split.

        The data loader is configured to shuffle the data. The `pin_memory` option is
        activated to achieve maximum transfer speed to the GPU. The data loader is also
        configured to drop the last batch of the data if it would only contain one
        sample.

        The whole split is held in memory. Therefore, the `num_workers` are set to
        zero which uses the main process for creating batches.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.

        Returns:
            The training data loader
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
            )
            self._train_loader = loader
        else:
            loader = self._train_loader

        return loader

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a [data loader][torch.utils.data.DataLoader] for the validation split.

        The data loader is configured to leave the data unshuffled. The `pin_memory`
        option is activated to achieve maximum transfer speed to the GPU.

        The whole split is held in memory. Therefore, the `num_workers` are set to
        zero which uses the main process for creating batches.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.

        Returns:
            The validation data loader
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
            )
            self._val_loader = loader
        else:
            loader = self._val_loader

        return loader

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a [data loader][torch.utils.data.DataLoader] for the test split.

        The data loader is configured to leave the data unshuffled. The `pin_memory`
        option is activated to achieve maximum transfer speed to the GPU.

        The whole split is held in memory. Therefore, the `num_workers` are set to
        zero which uses the main process for creating batches.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.

        Returns:
            The test data loader
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
            )
            self._test_loader = loader
        else:
            loader = self._test_loader

        return loader

    def profiled_loader(self, dl):

        from torch.profiler import (
            ProfilerActivity,
            profile,
            record_function,
            schedule,
            ProfilerAction,
        )

        def profiled_loader():

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
