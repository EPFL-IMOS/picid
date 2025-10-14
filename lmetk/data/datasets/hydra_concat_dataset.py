import warnings
from typing import Any, Callable

import omegaconf
from hydra.utils import instantiate
from numpy import ndarray
from torch.utils.data._utils.collate import default_collate

from lmetk.data.datasets.base import BaseVectorizedConcatDataset
from torch.utils.data import ConcatDataset


def initialize_datasets(dataset_cfg, data_dict, meta_data_dict):
    """
    Initializes datasets based on the provided configuration and data using hydra's `instantiate` function.

    This function takes a dataset configuration, a dictionary of data, and a dictionary
    of metadata, and creates a list of datasets. It ensures that the data provided is
    valid and consistent, and that all datasets created are of the same type.

    Args:
        dataset_cfg (omegaconf.DictConfig): The configuration for the dataset instantiation.
        data_dict (dict): A dictionary where keys are data identifiers and values are either
            lists or `omegaconf.listconfig.ListConfig` objects containing the data.
            The special key "metadata" is allowed to be an `omegaconf.dictconfig.DictConfig`.
        meta_data_dict (dict): A dictionary containing metadata to be passed to the dataset.
    Returns:
        list: A list of instantiated datasets.
    Raises:
        AssertionError: If the "metadata" key is not an `omegaconf.dictconfig.DictConfig`.
        TypeError: If any value in `data_dict` is not a list or `omegaconf.listconfig.ListConfig`.
        ValueError: If any list in `data_dict` is empty.
        AssertionError: If all lists in `data_dict` do not have the same length.
        AssertionError: If the instantiated datasets are not of the same type.
    """
    lenghts = []
    for key, val in data_dict.items():
        if isinstance(val, omegaconf.dictconfig.DictConfig):
            assert key == "metadata", (
                f"Only 'metadata' key is allowed to"
                f" be omegaconf.dictconfig.DictConfig but got {key}"
                " to be omegaconf.dictconfig.DictConfig."
            )

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

        # Go over the all keys and collect the corresponding tensors
        for key, val in data_dict.items():
            sub_data_dict[key] = val[idx]

        # TODO, also in the run.py: Patch to allow association of subunit in the concat dataset
        meta_data_dict = meta_data_dict.copy()
        meta_data_dict["concat_dataset_index"] = idx

        dataset = instantiate(
            dataset_cfg,
            data_dict=sub_data_dict,
            meta_data_dict=meta_data_dict,
        )
        datasets.append(dataset)

    for ds in datasets:
        assert type(ds) is type(
            datasets[0]
        ), f"All datasets must be of the same type. Found {type(ds)} and {type(datasets[0])}."

    return datasets


class HydraConcatDataset(BaseVectorizedConcatDataset):
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
        datasets = initialize_datasets(dataset_cfg, data_dict, meta_data_dict)

        self.collate_fn = (
            datasets[0].get_collate_fn()
            if hasattr(datasets[0], "get_collate_fn")
            else default_collate
        )

        super().__init__(datasets)

    def get_collate_fn(self) -> Callable:
        return self.collate_fn


class NonVectorizedHydraConcatDataset(ConcatDataset):
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
        datasets = initialize_datasets(dataset_cfg, data_dict, meta_data_dict)

        warnings.warn(
            "NonVectorizedHydraConcatDataset is deprecated and will be removed in a future release. "
            "Please use HydraConcatDataset instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.collate_fn = (
            datasets[0].get_collate_fn()
            if hasattr(datasets[0], "get_collate_fn")
            else default_collate
        )

        super().__init__(datasets)

    def get_collate_fn(self) -> Callable:
        return self.collate_fn
