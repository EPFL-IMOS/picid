
from typing import Dict, List, Optional, Tuple, Any, Callable, cast, Union, Literal

from box import Box
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    get_worker_info,
    Dataset,
)

from lmetk.data.datasets.abstract_classes import AbstractTimeSeriesDataSource






class ConceptRULDataset(Dataset):

    def __init__(self, 
                data_split: Dict[str, np.ndarray],
                window_size: int,
                stride: int,):

        self.df_Y = data_split.labels
        self.X = data_split.features
        self.unit= data_split.unit
        self.concepts = data_split.concepts

        self.window_size = window_size
        self.stride = stride
        super().__init__()

    def __len__(self):
        return self.X.shape[0] // self.stride #Floor total length divided by stride

    def __getitem__(self, i):

        # X is the normalized data
        

        #Sequence is entirely within the data
        if (i * self.stride >= self.window_size - 1):

            unit = self.unit[i * self.stride - self.window_size + 1:i * self.stride + 1]  #Unit vector for desired sequence

            cond = np.where(unit != unit[0])[0] #Index of first different value

            if cond.size == 0: #If there is no index with different value
                i_start = i * self.stride - self.window_size + 1
                x = self.X[i_start:i * self.stride + 1, :].T
            else:
                counter = cond[0] #Find first index of switch
                padding = self.X[i * self.stride -(self.window_size - counter) + 1].reshape(-1, 1).repeat(counter, 1)
                x = self.X[i * self.stride - (self.window_size - counter) + 1:i * self.stride + 1, :].T
                x = np.concatenate((padding, x), 1)

            #Beginning of Sequence (backward filling)
        else:
            padding = self.X[0].reshape(-1, 1).repeat(self.window_size - i * self.stride - 1, 1)
            x = self.X[0:i * self.stride + 1, :].T
            x = np.concatenate((padding, x), 1)

        x = x.ravel()

        return torch.Tensor(x).squeeze(), torch.Tensor([self.df_Y[i * self.stride]]).squeeze(), torch.Tensor(self.concepts[i * self.stride])




class RulDataModule(pl.LightningDataModule):


    _data: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]

    def __init__(
        self,
        datasets: Dict[str, ConceptRULDataset],
        batch_size: int,
    ):
        
        for ds in ["train", "val", "test"]:
            if ds not in datasets:
                assert False, f"Missing {ds} dataset in the provided datasets dictionary."
        
        super().__init__()
        hparams = dict(
            batch_size=batch_size,
        )

        self.save_hyperparameters(hparams)



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
        self.data_source.pipeline()


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
        dataset = self.to_dataset("dev")
        drop_last = len(dataset) % self.batch_size == 1
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
            pin_memory=True,
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
            self.to_dataset("val"),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
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
            self.to_dataset("test"),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )