import logging
from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
from numpy import ndarray
from torch.utils.data import ConcatDataset, Dataset

logger = logging.getLogger(__name__)


class BaseVectorizedConcatDataset(ConcatDataset):
    """
    A dataset that concatenates multiple datasets and supports list indexing.
    This class extends the PyTorch ConcatDataset to allow for list indexing,
    enabling the retrieval of multiple samples at once."""

    def __getitem__(self, idx: List[int] | np.ndarray) -> dict:
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            idx_np = np.asarray(idx)
            if len(self.datasets) == 1:
                # trivial case, all indices belong to the only dataset
                out = [self.datasets[0][idx]]
                return self.get_collate_fn()(out)

            dataset_ids = np.searchsorted(self.cumulative_sizes, idx_np, side="right")
            local_idx = idx_np.copy()
            local_idx[dataset_ids > 0] -= np.asarray(self.cumulative_sizes)[
                dataset_ids[dataset_ids > 0] - 1
            ]

            # group indices per sub-dataset
            out = []
            for ds_id in np.unique(dataset_ids):
                mask = dataset_ids == ds_id
                sub_idx = local_idx[mask].tolist()
                out.append(
                    self.datasets[ds_id][sub_idx]
                )  # assumes sub-datasets handle list[int]

            return self.get_collate_fn()(out)
        else:
            raise NotImplementedError(
                "Your data module / data loader does not support list indexing, but you are using BaseVectorizedConcatDataset. "
                "Please enable list indexing in your data module / data loader or switch to BaseConcatDataset."
            )

    @abstractmethod
    def get_collate_fn(self) -> Callable:
        """
        Return the collate function used for vectorized indexing.

        Returns
        -------
        Callable
            Collate function that combines a list of samples into a batch.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class BaseDataset(Dataset, ABC):
    def __init__(self, data_dict: dict[str, ndarray | list[ndarray]], **kwargs):
        super().__init__()

        if kwargs:
            logger.warning(
                f"Unused parameters in BaseDataset __init__: {list(kwargs.keys())}"
            )

    @abstractmethod
    def get_collate_fn(self) -> Callable:
        """
        Return the collate function used by the dataset.

        Returns
        -------
        Callable
            Collate function that combines a list of samples into a batch.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class BaseConcatDataset(ConcatDataset, ABC):
    @abstractmethod
    def get_collate_fn(self) -> Callable:
        """
        Return the collate function used by the concatenated dataset.

        Returns
        -------
        Callable
            Collate function that combines a list of samples into a batch.
        """
        raise NotImplementedError("Subclasses must implement this method.")
