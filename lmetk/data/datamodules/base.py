from typing import Dict, List, Optional, Tuple, Any

import inspect
import numpy as np
import pytorch_lightning as pl
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from lmetk.data.datasets.base import BaseConcatDataset, BaseDataset

import logging
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
        subset_range: Optional[Tuple[int, int, int]] = None, # Used for debugging purposes
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
                assert isinstance(size, int) and size > 0, f"Batch size must be a positive integer, got {size}"

        for shuffle in [shuffle_val, shuffle_test]:
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
        hparams = {arg: values[arg] for arg in args if arg not in ['self', 'datasets', 'dataset_train', 'dataset_val', 'dataset_test']}


        # We need them because the debug subset does not have this function.
        collate_fn_train = getattr(dataset_train, "collate_fn", None)
        collate_fn_val = getattr(dataset_val, "collate_fn", None)
        collate_fn_test = getattr(dataset_test, "collate_fn", None)

        self.dataset_train = dataset_train if subset_range is None else Subset(dataset_train, range(*subset_range))
        self.dataset_val = dataset_val if  subset_range is None else Subset(dataset_val, range(*subset_range))
        self.dataset_test = dataset_test if subset_range is None else Subset(dataset_test, range(*subset_range))

        if subset_range is not None:
            for ds_attr, collate_fn in [
                ("dataset_train", collate_fn_train),
                ("dataset_val", collate_fn_val),
                ("dataset_test", collate_fn_test),
            ]:
                ds = getattr(self, ds_attr)
                if isinstance(ds, Subset):
                    ds.collate_fn = collate_fn
        
        self.save_hyperparameters(hparams, ignore=["datasets",  'dataset_train', 'dataset_val', 'dataset_test'])

        for ds_name, ds in [("dataset_train", dataset_train), ("dataset_val", dataset_val), ("dataset_test", dataset_test)]:
            if not isinstance(ds, (BaseDataset, BaseConcatDataset)):
                raise TypeError(f"{ds_name} must be a subclass of BaseDataset or BaseConcatDataset, got {type(ds)}")

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

        drop_last = len(self.dataset_train) % self.hparams.train_batch_size == 1
        loader = DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            shuffle=self.hparams.shuffle_train,
            drop_last=drop_last,
            pin_memory=True,
            collate_fn=self.dataset_train.collate_fn
        )

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

        return DataLoader(
            self.dataset_val,
            batch_size=batch_size,
            shuffle=self.hparams.shuffle_val,
            pin_memory=True,
            collate_fn=self.dataset_val.collate_fn,
        )

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

        return DataLoader(
            self.dataset_test,
            batch_size=batch_size,
            shuffle=self.hparams.shuffle_test,
            pin_memory=True,
            collate_fn=self.dataset_test.collate_fn,
        )
