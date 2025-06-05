from typing import Dict, List
import torch
from torch.utils.data import ConcatDataset
from hydra.utils import instantiate


class HydraConcatDataset(ConcatDataset):
    def __init__(self, 
                 dataset_cfg,
                 data_split: Dict[str, List[torch.Tensor]], **kwargs):
        """
        Args:
            dataset_cfg: the dataset configuration.
            tensor_dict: dictionary of tensors.
        """

        # Fetch the first item from the dict
        first_key = next(iter(data_split))
        first_item = data_split[first_key]

        # Ensure all tensors in the dict have the same length
        assert all(len(data_split[key]) == len(first_item) for key in data_split), \
            "All tensors in the tensor_dict must have the same length."

        datasets = []
    
        for id_dataset in range(len(first_item)):    
            # Create a dictionary of tensors for the current dataset
            tensors = {key: data_split[key][id_dataset] for key in data_split}

            # Create a dataset for each key in the tensor_dict
            dataset = instantiate(dataset_cfg, data_split=tensors)
            datasets.append(dataset)

        super().__init__(datasets)