from typing import Dict, List
from numpy import ndarray
import torch
from hydra.utils import instantiate
import omegaconf

from lmetk.data.datasets.base import BaseConcatDataset
from torch.utils.data._utils.collate import default_collate

class HydraConcatDataset(BaseConcatDataset):
    def __init__(self, 
                 dataset_cfg,
                 data_dict: dict[str, ndarray | list[ndarray]], 
                 **kwargs):
        """
        Args:
            dataset_cfg: the dataset configuration.
            tensor_dict: dictionary of tensors.
            data_sict: dictionary of data.
                {
                "features":  list[ndarray],
                "target":  list[ndarray],
                "other_named_tensor": list[ndarray],
        }
        """
    
        # Checking the inputs
        lenghts = []
        for key, val in data_dict.items():
            if not isinstance(val, list) and not isinstance(val, omegaconf.listconfig.ListConfig):
                raise TypeError(f"Data in '{key}' must be a list, got {type(val)}.")
            if len(val) == 0:
                raise ValueError(f"Data list in '{key}' is empty.")

            lenghts.append(len(val))
        
        assert all(lenght == lenghts[0] for lenght in lenghts), "All lists must have the same length."
                    
        datasets = []
        # For every tensor in the list, create a dataset
        for idx in range(lenghts[0]):
            sub_data_dict = {}
            
            # Go ovet the all keys and collect the corresponding tensors
            for key, val in data_dict.items():
                sub_data_dict[key] = val[idx]

            dataset = instantiate(dataset_cfg, data_dict=sub_data_dict)
            datasets.append(dataset)


        super().__init__(datasets)

    def get_collate_fn(self):
        return default_collate