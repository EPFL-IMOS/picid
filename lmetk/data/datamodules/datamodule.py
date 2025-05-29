from typing import Dict, List, Optional, Tuple, Any

import inspect
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from lmetk.data.datamodules.utils import collate_fn

class BaseDataModule(pl.LightningDataModule):
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
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        shuffle_test: bool = False,
    ):
        # for ds in ["train", "val", "test"]:
        #     if ds not in datasets:
        #         assert False, (
        #             f"Missing {ds} dataset in the provided datasets dictionary."
        #         )

        super().__init__()

        # Automatically collect all init args
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        
        # Remove 'self' and 'datasets' from the list
        hparams = {arg: values[arg] for arg in args if arg not in ['self', 'datasets', 'dataset_train', 'dataset_val', 'dataset_test']}

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        
        self.save_hyperparameters(hparams, ignore=["datasets",  'dataset_train', 'dataset_val', 'dataset_test'])

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
        drop_last = len(self.dataset_train) % self.hparams.train_batch_size == 1
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.hparams.train_batch_size,
            shuffle=self.hparams.shuffle_train,
            drop_last=drop_last,
            pin_memory=True,
            # collate_fn=collate_fn,
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
        return DataLoader(
            self.dataset_val,
            batch_size=self.hparams.val_batch_size,
            shuffle=self.hparams.shuffle_val,
            pin_memory=True,
            collate_fn=collate_fn,
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
        return DataLoader(
            self.dataset_test,
            batch_size=self.hparams.test_batch_size,
            shuffle=self.hparams.shuffle_test,
            pin_memory=True,
            collate_fn=collate_fn,
        )
