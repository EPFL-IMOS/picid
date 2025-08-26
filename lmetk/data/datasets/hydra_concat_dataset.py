from typing import Any

import omegaconf
from hydra.utils import instantiate
from numpy import ndarray
from torch.utils.data._utils.collate import default_collate

from lmetk.data.datasets.base import BaseConcatDataset


class HydraConcatDataset(BaseConcatDataset):
    def __init__(
        self,
        dataset_cfg,
        data_dict: dict[str, ndarray | list[ndarray]],
        meta_data_dict: dict[str, Any],
        **kwargs,
    ):
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
        self.meta_data_dict = meta_data_dict
        # Checking the inputs
        lenghts = []
        for key, val in data_dict.items():
            if isinstance(
                val, omegaconf.dictconfig.DictConfig
            ):
                assert key == 'metadata', f"Only 'metadata' key is allowed to be omegaconf.dictconfig.DictConfig but got {key} to be omegaconf.dictconfig.DictConfig."

                val.keys()


            elif not isinstance(val, list) and not isinstance(
                val, omegaconf.listconfig.ListConfig
            ):
                raise TypeError(f"Data in '{key}' must be a list, got {type(val)}.")
            if len(val) == 0:
                raise ValueError(f"Data list in '{key}' is empty.")

            lenghts.append(len(val))

        assert all(
            lenght == lenghts[0] for lenght in lenghts
        ), "All lists must have the same length."

        datasets = []
        # For every tensor in the list, create a dataset
        for idx in range(lenghts[0]):
            sub_data_dict = {}

            # Go ovet the all keys and collect the corresponding tensors
            for key, val in data_dict.items():
                sub_data_dict[key] = val[idx]

            dataset = instantiate(
                dataset_cfg,
                data_dict=sub_data_dict,
                meta_data_dict=self.meta_data_dict,
            )
            datasets.append(dataset)

        for ds in datasets:
            assert type(ds) is type(
                datasets[0]
            ), f"All datasets must be of the same type. Found {type(ds)} and {type(datasets[0])}."

        self.collate_fn = (
            datasets[0].get_collate_fn()
            if hasattr(datasets[0], "get_collate_fn")
            else default_collate
        )

        super().__init__(datasets)

    def get_collate_fn(self):
        return self.collate_fn
