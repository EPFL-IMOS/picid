"""Preprocessor for data."""

import json
import os
import time
import hydra
import torch

import torch


class PreProcessor():
    """Preprocessor for datasets.

    Parameters
    ----------
    dataset : list
        List of data objects.
    data_dir : str
        Path to the directory containing the data.
    transforms_config : DictConfig, optional
        Configuration parameters for the transforms (default: None).
    **kwargs : optional
        Optional additional arguments.
    """

    def __init__(self, dataset, data_dir, transforms_config=None, **kwargs):
        self.dataset = dataset
        self.preprocessing_time = 0
        if transforms_config is not None:
            self.transforms_applied = True
            pre_transform = self.instantiate_pre_transform(
                data_dir, transforms_config
            )
            # Record the time taken for preprocessing
            start_time = time.time()
            super().__init__(
                self.processed_data_dir, None, pre_transform, **kwargs
            )
            end_time = time.time()
            self.preprocessing_time = end_time - start_time
            #self.save_transform_parameters()
            #self.load(self.processed_paths[0])
        else:
            self.transforms_applied = False
            super().__init__(data_dir, None, None, **kwargs)
            self.load(data_dir + "/processed/data.pt")

        self.data_list = [self.get(idx) for idx in range(len(self))]
        # Some datasets have fixed splits, and those are stored as split_idx during loading
        # We need to store this information to be able to reproduce the splits afterwards
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx

    

    def instantiate_pre_transform(
        self, data_dir, transforms_config
    ) -> torch_geometric.transforms.Compose:
        """Instantiate the pre-transforms.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.

        Returns
        -------
        torch_geometric.transforms.Compose
            Pre-transform object.
        """
        pre_transforms_dict = hydra.utils.instantiate(transforms_config)
        # Make sure the dict is ordered
        pre_transforms_dict = {
            key: DataTransform(**value)
            for key, value in transforms_config.items()
        }
        pre_transforms = <Function to compose multiple transfoms>(
            list(pre_transforms_dict.values())
        )
       
        return pre_transforms


    def save_transform_parameters(self) -> None:
        """Save the transform parameters."""
        # Check if root/params_dict.json exists, if not, save it
        path_transform_parameters = os.path.join(
            self.processed_data_dir, "path_transform_parameters_dict.json"
        )
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transforms_parameters, f, indent=4)
        else:
            # If path_transform_parameters exists, check if the transform_parameters are the same
            with open(path_transform_parameters) as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != self.transforms_parameters:
                raise ValueError(
                    "Different transform parameters for the same data_dir"
                )

            print(
                f"Transform parameters are the same, using existing data_dir: {self.processed_data_dir}"
            )

    def process(self) -> None:
        """Method that processes the data."""
        if isinstance(self.dataset, <some dataset class like sw_dataset>):
            data_list = [
                self.dataset.get(idx) for idx in range(len(self.dataset))
            ]
        
        elif isinstance(self.dataset, <some dataset class like sw_batched_dataset>):
            data_list = [self.dataset[idx] for idx in range(len(self.dataset))]
        
        elif isinstance(self.dataset, <instance object>):
            data_list = [self.dataset]
        elif isinstance(self.dataset, list):
            data_list = self.dataset
        else:
            raise ValueError(
                "Dataset must be a torch_geometric.data.Dataset or a list of data objects"
            )

        self.data_list = (
            [self.pre_transform(d) for d in data_list]
            if self.pre_transform is not None
            else data_list
        )
        # Transform 

        assert isinstance(self._data, )
        

    