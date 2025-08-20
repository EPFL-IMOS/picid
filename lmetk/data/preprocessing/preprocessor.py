import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import hydra
import numpy as np
from numpy import ndarray
from omegaconf import DictConfig, OmegaConf

from lmetk.data.preprocessing.base import PreProcessorInterface
from lmetk.transforms.base.data_transform import DataTransform


logger = logging.getLogger(__name__)


# TODO: refactor this logic
def swap_first_second_level(
    data: dict[str, ndarray | list[ndarray]],
    populate_metadata: bool = False,
) -> dict[str, ndarray | list[ndarray]]:
    """
    Swap the first and second level of a nested dictionary.
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary.")

    swapped = {}
    for key, value in data.items():
        if isinstance(value, dict):
            if (key == 'metadata') and (populate_metadata==True): 
                continue
            
            # If value is a dict, check if its values are also dicts (deeper than 2 levels)
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    raise ValueError(
                        "Error while swapping dictionary. Dictionary is deeper than 2 levels."
                    )
                if swapped.get(sub_key) is None:
                    swapped[sub_key] = {}
                swapped[sub_key][key] = sub_value
        else:
            raise ValueError(
                "Error while swapping dictionary. First level value must be a dict."
            )
    
    assert set(swapped.keys()) == {"train", "val", "test"}, "Swapped dictionary must contain solely 'train', 'val', and 'test' keys."
    
    if populate_metadata and "metadata" in data:
        for key in swapped.keys():
            if "metadata" not in swapped[key]:
                swapped[key]["metadata"] = data['metadata']
            else:
                raise ValueError(
                    "While swapping dictionary in the preprocessor the error occured: the metadata already exists in the second level."
                )
            
    return swapped


class PreProcessor(PreProcessorInterface):
    """Preprocessor of data."""

    def __init__(self, mode, datasource, transforms, **kwargs):
        super().__init__(datasource=datasource, transforms=transforms)

        assert mode in [
            "per_unit",
            "cross_unit",
        ], f"Unsupported mode: {mode}. Supported modes are 'per_unit' and 'cross_unit'."
        self.mode = mode
        self.data_lst: Optional[List[Any]] = None
        self._is_preprocessed = False

    def get_meta_data_dict(self):
        if hasattr(self.datasource, "get_meta_data") and callable(
            getattr(self.datasource, "get_meta_data", None)
        ):
            meta_data_dict = self.datasource.get_meta_data()
        else:
            meta_data_dict = {}
        return meta_data_dict

    def get_processed_data_dict(
        self, return_splits_on_first_level: bool = False
    ) -> dict[str, ndarray | list[ndarray]]:
        """
        Returns the final processed data.

        The returned dictionary is structured as follows:
        {
            "features": {
                "train": ndarray | list[ndarray],
                "val": ndarray | list[ndarray],
                "test": ndarray | list[ndarray]
            },
            "target": {
                "train": ndarray | list[ndarray],
                "val": ndarray | list[ndarray],
                "test": ndarray | list[ndarray]
            },
            "other_named_tensor": {
                ...
            }
        }

        If return_splits_on_first_level is True, the function will return the splits at the first level.
        """

        if not self._is_preprocessed or self.data_lst is None:
            raise RuntimeError(
                "Data has not been preprocessed. Please run the pipeline() first."
            )

        if isinstance(self.data_lst, list) and len(self.data_lst) == 1:
            if return_splits_on_first_level:
                # TODO: This became very convoluted. 
                # 1. Case when matadata dict has train/val/test splits
                metadata_ = self.data_lst[0].get('metadata', {})
                # Case of XJTU dataset or any other multisource phmd where splits are given and we manually put matadata under each unit and its split.
                if set(metadata_.keys()) == {"train", "val", "test"}:
                    return swap_first_second_level(self.data_lst[0], populate_metadata=False)
                
                else:
                    # Otherweise like in the ETTh dataset, we should populate the populate_metadata
                    return swap_first_second_level(self.data_lst[0], populate_metadata=True)
            else:
                return self.data_lst[0]

        elif isinstance(self.data_lst, list):
            # TODO: HAVE NOT TESTED
            # If multiple sources, return as a list
            raise ValueError("Not yet implemented, call R.T. to fix.")

        else:
            raise ValueError("Processed data is in an unexpected format.")

    # TODO: that is needed at least for evaluator, extend it for multisource maybe
    def get_transform_names(
        self,
    ):
        return self.source_transform_names_lst

    def prepare_data(self) -> None:
        """Prepare data from the datasource and store as instance variables."""
        data_lst = self.datasource.get_data()

        if hasattr(self.datasource, "get_meta_data") and callable(
            getattr(self.datasource, "get_meta_data", None)
        ):
            meta_data_lst = self.datasource.get_meta_data()
        else:
            meta_data_lst = [{}]

        # Ensure all outputs are lists for consistent processing
        if not isinstance(data_lst, list):
            data_lst = [data_lst]

        if not isinstance(meta_data_lst, list):
            meta_data_lst = [meta_data_lst]

        self.data_lst = data_lst
        self.meta_data_lst = meta_data_lst

    def apply_transforms(self, data_lst: List[Any], transforms) -> List[Any]:
        """
        Apply a list of transforms to each data item in data_lst.
        Args:
            data_lst: List of data items.
            transforms_lst: List of lists of DataTransform objects, one list per data item.
        Returns:
            List of transformed data items.
        """
        if not all(isinstance(t, DataTransform) for t in transforms.values()):
            raise ValueError(
                "Every element of transforms_lst must be a DataTransform object."
            )

        transformed_data_lst = []
        for current_data in data_lst:
            for transform_name, transform in transforms.items():
                # Removed detailed print statements for each transform application.

                logger.info(
                    f"Applying transform: '{transform_name}' (Strategy: {transform.strategy.__class__.__name__})"
                )
                start_time = time.time()
                current_data = transform.forward(current_data)
                end_time = time.time()
                logger.info(
                    f" transform: '{transform.transform_name}' - completed in {end_time - start_time:.4f} seconds."
                )

            transformed_data_lst.append(current_data)
        return transformed_data_lst

    def pipeline(self):
        """Run the preprocessing pipeline."""
        # Removed "Starting preprocessing pipeline..." print.
        self.prepare_data()
        # if multisource_data_splitter do not exist, then we work with separate sources
        assert (
            self.datasource.get_split_mode() == self.mode
        ), f"Expected split mode from preprocessor: '{self.mode}', but got dataset split mode: '{self.datasource.get_split_mode()}'"
        self.data_lst = self.apply_transforms(
            self.data_lst, self.transforms.get_transforms()
        )
        self._is_preprocessed = True

        return self.data_lst


def key_exists_anywhere(data, target_key):
    # Check if object has dict-like methods
    if hasattr(data, "keys") and hasattr(data, "values") and hasattr(data, "items"):
        # First check if the target key exists at this level
        if target_key in data:
            return True, data[target_key]

        # Then recursively search in all values
        for value in data.values():
            found, val = key_exists_anywhere(value, target_key)
            if found:
                return found, val

    elif isinstance(data, (list, tuple)):
        for item in data:
            found, val = key_exists_anywhere(item, target_key)
            if found:
                return found, val

    return False, None
