import itertools
import logging
from collections.abc import Sequence
from typing import Optional, Tuple

import numpy as np
import torch
from lightning_fabric.utilities.data import AttributeDict
from numpy import ndarray
from torch.utils.data._utils.collate import default_collate

from picid.model.definitions import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    FORECASTING_TASKS,
)
from picid.data.datasets.base import BaseDataset
from picid.utils.assertions import assert_list_of_ndarray_or_nd_array_for_dims

logger = logging.getLogger(__name__)


class FitPredictTaskDataset(BaseDataset):
    """
    Dataset for fit-predict tasks with multiple tasks, each represented by 3D arrays.

    Parameters
    ----------
    data_dict : dict[str, np.ndarray | list[np.ndarray]]
        Dictionary containing 'X' and 'y' arrays or lists of arrays.
        - 'X': 3D array or list of 3D arrays of context features, shape
          (n_tasks, n_samples, n_features).
        - 'y': 3D array or list of 3D arrays of target values, shape
          (n_tasks, n_samples, n_targets).
    meta_data_dict : dict[str, any]
        Metadata associated with the dataset.
    convert_units_to_tasks : bool, optional
        If True, concatenates lists of arrays along the task dimension.
        If False, assumes all arrays belong to the same task and squeezes the
        first dimension.
    **kwargs
        Additional keyword arguments passed to BaseDataset.

    Raises
    ------
    ValueError
        If 'X' or 'y' is missing from data_dict.
    AssertionError
        If input arrays do not have the expected dimensions or if task/time
        step counts are inconsistent.

    Examples
    --------
    X = np.random.randn(5, 100, 10)  # 5 tasks, 100 samples, 10 features
    y = np.random.randn(5, 100, 1)   # 5 tasks, 100 samples, 1 target
    ds = FitPredictTaskDataset({'X': X, 'y': y}, meta_data_dict={})

    Returns
    -------
    AttributeDict
        For each task containing:
        - task_idx: Index of the task.
        - task_num: Total number of tasks.
        - task_desc: Description string.
        - target: Target array for the task.
        - context: Context array for the task.
    """

    def __init__(
        self,
        data_dict: dict[str, ndarray | list[ndarray]],
        task_type: str,
        meta_data_dict: dict[str, any],
        convert_units_to_tasks: bool = False,
        subset_range: Optional[
            Tuple[int, int, int]
        ] = None,  # Used for debugging purposes
        dataset_cfg: Optional[dict] = None,
        **kwargs,
    ):
        # Additional default values
        self.get_unit_id = False

        if dataset_cfg is not None and dataset_cfg.get("get_unit_id", None):
            self.get_unit_id = True

        # Make sure that the data_dict contains the required keys
        if task_type in REGRESSION_TASKS:
            required_keys = ["features", task_type]  # "time_features", "target"
            context_key = "features"
            target_key = task_type

        elif task_type in FORECASTING_TASKS:
            required_keys = ["features", "target"]  # "time_features", "target"
            context_key = "features"
            target_key = "target"

        elif task_type in CLASSIFICATION_TASKS:
            required_keys = ["features", task_type]
            context_key = "features"
            target_key = task_type

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        for key in required_keys:
            if key not in data_dict:
                raise ValueError(f"Data dictionary must contain the key: '{key}'")

        X = data_dict[context_key]
        y = data_dict[target_key]

        self.convert_units_to_tasks = convert_units_to_tasks
        self.subset_range = subset_range  # Used for debugging purposes

        # make sure target is 3 dimensional and context features 3
        # This is because its a standard multi-target fit predict task (2) and 2 dimensional,
        # and the first dimension is the task dimension
        assert_list_of_ndarray_or_nd_array_for_dims(X, 3)
        assert_list_of_ndarray_or_nd_array_for_dims(y, 3)

        def handle_multi_units(input_data, name: str):
            # Check if x and y are lists of arrays, if its the case for both, concatenate and log a warning
            if isinstance(input_data, Sequence) and all(
                isinstance(x, ndarray) for x in input_data
            ):
                if self.convert_units_to_tasks:
                    # Assert that all arrays have matching dimensions for dim 2 and 3
                    first_shape = input_data[0].shape
                    for arr in input_data:
                        assert (
                            arr.shape[1] == first_shape[1]
                            and arr.shape[2] == first_shape[2]
                        ), (
                            f"All arrays in {name} must have matching dimensions for axis 1 and 2. "
                            f"Expected ({first_shape[1]}, {first_shape[2]}), got ({arr.shape[1]}, {arr.shape[2]})"
                        )
                    input_data_c = np.concatenate(input_data, axis=0)
                else:
                    # In this case we assume everything goes to the same task
                    assert all(x.shape[0] == 1 for x in input_data), (
                        f"For the multi-task setting, given that convert_units_to_tasks is false,"
                        f"Expected all arrays in {name} to have shape (1, ...), "
                        f"but got shapes: {[x.shape for x in input_data]}"
                    )
                    squeezed_data = [x.squeeze(axis=0) for x in input_data]
                    input_data_c = np.concatenate(squeezed_data, axis=0)
                    input_data_c = input_data_c[np.newaxis, ...]  # add task dimension

                logger.warning(
                    f"Concatenated list of arrays for {name}. Shape before: "
                    f"{len(input_data)}, shape of {name}[0]: {input_data[0].shape}, "
                    f"shape after: {input_data_c.shape}, dtype: {input_data_c.dtype}"
                )

                return input_data_c

            else:
                return input_data

        self.X = handle_multi_units(X, "X")
        self.y = handle_multi_units(y, "y")
        self.meta_data_dict = meta_data_dict

        assert (
            self.X.shape[0] == self.y.shape[0]
        ), f"Inconsistent number of tasks: {self.X.shape[0]} != {self.y.shape[0]}"

        assert (
            self.X.shape[1] == self.y.shape[1]
        ), f"Inconsistent number of time steps: {self.X.shape[1]} != {self.y.shape[1]}"

        if self.get_unit_id:
            unit_id = data_dict.get("unit_id", None)
            self.unit_id = handle_multi_units(unit_id, "unit_id")
            assert (
                self.unit_id is not None
            ), "Unit ID must be provided if get_unit_id is True. Otherwise set dataset_cfg.get_unit_id to False."
            # Check that 0 and 1 dimension match the number of tasks and samples
            assert (
                self.unit_id.shape[0] == self.X.shape[0]
                and self.unit_id.shape[1] == self.X.shape[1]
            ), (
                f"Unit ID must have the same shape as X in the first two dimensions. "
                f"Expected ({self.X.shape[0]}, {self.X.shape[1]}), got ({self.unit_id.shape[0]}, {self.unit_id.shape[1]})"
            )

        self.n_tasks = len(self.y)  # first dimension is the task dimension

        super().__init__(data_dict, **kwargs)

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, task: list[int]):
        # get the integer from the list, as the dataloader provides a list of indices
        # Because we use batch sampler with batch size 1 in the DataModule
        task = task[0]
        X = np.expand_dims(self.X[task], axis=0)  # (1, n_samples, n_features)
        y = np.expand_dims(self.y[task], axis=0)  # (1, n_samples, n_outputs)

        if self.subset_range is not None:
            unique_y = np.unique(y)
            X = X[:, slice(*self.subset_range), :]
            y = y[:, slice(*self.subset_range), :]
            y = np.asarray(
                list(itertools.islice(itertools.cycle(unique_y), len(y.flatten()))),
                dtype=y.dtype,
            ).reshape(y.shape)

        d = AttributeDict(
            {
                "task_idx": torch.tensor(task),
                "task_num": torch.tensor(self.n_tasks),
                "task_desc": f"Task {task + 1} of {self.n_tasks}",
                "target": torch.from_numpy(y),
                "context": torch.from_numpy(X),
            }
        )

        if self.get_unit_id:
            unit_id = self.unit_id[task]
            # TODO: potential bug here when tasks > 1, check!
            if self.subset_range is not None:
                unit_id = unit_id[slice(*self.subset_range), :]
            d["unit_id"] = torch.from_numpy(unit_id)

        return AttributeDict(d)

    def get_collate_fn(self):
        return default_collate
