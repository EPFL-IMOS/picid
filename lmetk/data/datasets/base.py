

from torch.utils.data import Dataset, ConcatDataset
from abc import ABC, abstractmethod

"""
IMPORTANT: 

Maybe later on we need a collate function factory to create different collate functions
depending on the dataset type and task type.

We have omitted this so far to keep the code simple and focused on the task at hand.

"""

class BaseDataset(Dataset, ABC):

    @staticmethod
    @abstractmethod
    def collate_fn(batch):
        """
        Collate function to be implemented by subclasses.
        This function should combine a list of samples into a batch.
        
        Args:
            batch (list): List of samples to collate.
        
        Returns:
            The collated batch.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class BaseConcatDataset(ConcatDataset, ABC):

    @staticmethod
    @abstractmethod
    def collate_fn(batch):
        """
        Collate function to be implemented by subclasses.
        This function should combine a list of samples into a batch.
        
        Args:
            batch (list): List of samples to collate.
        
        Returns:
            The collated batch.
        """
        raise NotImplementedError("Subclasses must implement this method.")