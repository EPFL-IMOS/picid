

from torch.utils.data import Dataset, ConcatDataset
from abc import ABC, abstractmethod

"""
IMPORTANT: 

Maybe later on we need a collate function factory to create different collate functions
depending on the dataset type and task type.

We have omitted this so far to keep the code simple and focused on the task at hand.

"""
from numpy import ndarray

import logging
logger = logging.getLogger(__name__)

class BaseDataset(Dataset, ABC):


    def __init__(self, 
                 data_dict: dict[str, ndarray | list[ndarray]], 
                 **kwargs):
        super().__init__()

        if kwargs:
            logger.warning(f"Unused parameters in BaseDataset __init__: {list(kwargs.keys())}")
        self.data_dict = data_dict


    @abstractmethod
    def get_collate_fn():
        """
        Collate function to be implemented by subclasses.
        This function should combine a list of samples into a batch.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class BaseConcatDataset(ConcatDataset, ABC):

    @abstractmethod
    def get_collate_fn():
        """
        Collate function to be implemented by subclasses.
        This function should combine a list of samples into a batch.
        """
        raise NotImplementedError("Subclasses must implement this method.")