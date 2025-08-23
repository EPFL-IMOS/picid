
from lightning_fabric.utilities.data import AttributeDict
from numpy import ndarray
from torch.utils.data._utils.collate import default_collate

from lmetk.data.datasets.base import BaseDataset


class FitPredictTaskDataset(BaseDataset):
    def __init__(
        self,
        # context: torch.Tensor,
        # target: torch.Tensor,
        # time_features: torch.Tensor,
        data_dict: dict[str, ndarray | list[ndarray]],
        meta_data_dict: dict[str, any],
        **kwargs,
    ):
        # Make sure that the data_dict contains the required keys
        required_keys = ["X", "y"]
        for key in required_keys:
            if key not in data_dict:
                raise ValueError(f"Data dictionary must contain the key: '{key}'")

        self.X = data_dict["X"]
        self.y = data_dict["y"]

        # make sure target is 2 dimensional and context features 3
        if self.y.ndim != 2:
            raise ValueError("y must be 2-dimensional")
        if self.X.ndim != 3:
            raise ValueError("X must be 3-dimensional")

        assert (
            self.X.shape[0] == self.y.shape[0]
        ), f"Inconsistent number of tasks: {self.X.shape[0]} != {self.y.shape[0]}"

        assert (
            self.X.shape[1] == self.y.shape[1]
        ), f"Inconsistent number of time steps: {self.X.shape[1]} != {self.y.shape[1]}"

        self.n_tasks = len(self.y)  # first dimension is the task dimension

        super().__init__(data_dict, **kwargs)

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, task: int):
        d = AttributeDict(
            {
                "task_idx": task,
                "target": self.y[task],
                "context": self.X[task],
            }
        )

        return AttributeDict(d)

    def get_collate_fn(self):
        return default_collate
