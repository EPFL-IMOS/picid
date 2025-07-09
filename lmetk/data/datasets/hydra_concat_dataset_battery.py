from typing import Dict, List
import torch
from torch.utils.data import ConcatDataset
from hydra.utils import instantiate
import omegaconf

class HydraConcatDataset(ConcatDataset):
    def __init__(self, 
                 dataset_cfg,
                 data_dict: dict,
                 split: str,
                 **kwargs):
        """
        Args:
            dataset_cfg: the dataset configuration.
            tensor_dict: dictionary of tensors.
            data_sict: dictionary of data.
                {
                "features": {
                    "train": list[ndarray],
                    "val": list[ndarray],
                    "test": list[ndarray]
                },
                "target": {
                    "train": list[ndarray],
                    "val": list[ndarray],
                    "test": list[ndarray]
                },
                "other_named_tensor": {
                    ...
                }
            
            split: split to cosider

        }
        """
    
        # Checking the inputs
        lenghts = []
        for key, val in data_dict.items():
            
            if split not in val:
                raise KeyError(f"Split '{split}' not found in data_dict['{key}']. Available splits: {list(val.keys())}")
            if not isinstance(val[split], list) and not isinstance(val[split], omegaconf.listconfig.ListConfig):
                raise TypeError(f"Data for split '{split}' in '{key}' must be a list, got {type(val[split])}.")
            if len(val[split]) == 0:
                raise ValueError(f"Data list for split '{split}' in '{key}' is empty.")
            
            # # Go over the list and check that all elements have the same length
            # for tensor in val[split]:
            #     if len(tensor) != len(val[split][0]):
            #         raise ValueError(f"All tensors in the list for split '{split}' in '{key}' must have the same length.")

            lenghts.append(len(val[split]))
        
        assert all(lenght == lenghts[0] for lenght in lenghts), "All lists must have the same length."
                    
        datasets = []
        # For every tensor in the list, create a dataset
        for idx in range(lenghts[0]):
            split_data_dict = {}
            
            # Go ovet the all keys and collect the corresponding tensors
            for key, val in data_dict.items():
                split_data_dict[key] = val[split][idx]
            
            dataset = instantiate(
                dataset_cfg, 
                context=split_data_dict['features'],
                target= split_data_dict['target'],
                time_features=split_data_dict['timestamps'],
                )
            datasets.append(dataset)


        super().__init__(datasets)