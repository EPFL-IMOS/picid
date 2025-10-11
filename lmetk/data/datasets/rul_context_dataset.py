# @override decorator
import torch
from typing import Dict, List, override

from lightning_fabric.utilities.data import AttributeDict
from numpy import ndarray

from lmetk.data.datasets.collate_functions import collate_key_value_batch
from lmetk.data.datasets.context_dataset import ContextBatchDataset


class RULContextBatchDataset(ContextBatchDataset):
    """Creates a dataset for Remaining Useful Life (RUL) prediction.

    This class specializes `ContextBatchDataset` for a common RUL prediction
    scenario: predicting the RUL at the end of a given sequence of operational
    data. It uses a sliding window to generate input sequences (`X`) and
    corresponding target sequences (`y`).

    For a given input feature sequence of length `seq_len`, the corresponding
    target RUL sequence also has length `seq_len`. In a typical training loop,
    the target label is then taken as the **last value** of this RUL sequence.

    Args:
        data_dict (dict[str, ndarray | list[ndarray]]): A dictionary mapping string
            keys (e.g., 'features', 'RUL') to the time-series data arrays. The
            arrays can be NumPy arrays, PyTorch Tensors, or Awkward Arrays.
        task_type (str): A string identifier for the task type (e.g., 'regression'),
            used by parent classes.
        seq_len (int): The length of the sliding window, defining the number of
            historical time steps in each input sample.
        label_len (int): The length of the "label" or "overlap" portion of the
            sequence, used by the underlying sequencer. It defines the starting
            point of the target sequence relative to the input sequence.
        pred_len (int): The prediction horizon. **Note: This argument is ignored
            and internally overridden to 0** to make the target window align
            perfectly with the input window.
        stride (int, optional): The step size the sliding window takes across the
            data between generating consecutive sequences. Defaults to 1.
        **kwargs: Additional keyword arguments passed to the parent `ContextBatchDataset`.

    Example:
        >>> import numpy as np
        >>> # Example data: 100 time steps, 3 features
        >>> features = np.random.rand(100, 3)
        >>> # RUL counts down from 99 to 0
        >>> rul = np.arange(99, -1, -1).reshape(-1, 1)
        >>> data = {'features': features, 'RUL': rul}
        ...
        >>> # Create a dataset with a window size of 20
        >>> dataset = RULContextBatchDataset(
        ...     data_dict=data,
        ...     task_type='regression',
        ...     seq_len=20,
        ...     label_len=10,
        ...     pred_len=5  # This will be forced to 0
        ... )
        ...
        >>> # Get the first sample from the dataset
        >>> first_sample = dataset[0]
        >>> input_features = first_sample['features']
        >>> target_rul_sequence = first_sample['RUL']
        ...
        >>> print(f"Input features shape: {input_features.shape}")
        Input features shape: (20, 3)
        >>> print(f"Target RUL sequence shape: {target_rul_sequence.shape}")
        Target RUL sequence shape: (20, 1)
        ...
        >>> # In training, the target is the last value of the RUL sequence.
        >>> # Corresponds to the RUL at the end of the 20-step feature window.
        >>> final_target_rul = target_rul_sequence[-1]
        >>> print(f"Final target RUL value: {final_target_rul}")
        Final target RUL value: [80]

    Yields:
        A dictionary where keys correspond to those in `data_dict` and values
        are the sequenced NumPy arrays for a given sample index.
    """

    def __init__(
        self,
        data_dict: dict[str, ndarray | list[ndarray]],
        task_type: str,
        seq_len: int,  # window size
        label_len: int,  # overlap size
        pred_len: int,  # prediction size
        stride: int = 1,
        # TODO: What is our strategy for having def. params? everythewe? I vote everywhere.
        padding_left_flag: int = 0,
        return_index_flag: bool = False,
        warmup_steps: int = None,
        **kwargs,
    ):
        assert pred_len == 0, "pred_len must be 0 for RUL prediction."

        self.meta_data_dict = kwargs["meta_data_dict"]

        # Inject the unit name/id extraction functions into the meta_data_dict
        self.unid_id = self.extract_unit_id()
        self.unid_name = self.extract_unit_name()
        # self.meta_data_dict[key]
        super().__init__(
            data_dict=data_dict,
            task_type=task_type,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
            padding_left_flag=padding_left_flag,
            return_index_flag=return_index_flag,
            warmup_steps=warmup_steps,
            **kwargs,
        )

    def extract_unit_id(self):
        return self.meta_data_dict["unit_ids"][
            self.meta_data_dict["current_data_split"]
        ][self.meta_data_dict["concat_dataset_index"]]

    def extract_unit_name(self):
        return self.meta_data_dict["unit_names"][
            self.meta_data_dict["current_data_split"]
        ][self.meta_data_dict["concat_dataset_index"]]

    @override
    def __getitem__(self, batch_idx: List[int]):
        target: Dict = self.target_dataset.__getitem__(batch_idx)
        context: Dict = self.context_dataset.__getitem__(batch_idx)
        target_seq_x = target["rul_seq_x"]
        context_seq_x = context["features_seq_x"]
        # TO FINISH: we have now "unit_id" and "unit_name" in batch, how to collate it?
        d = AttributeDict(
            {
                "batch_idx": torch.tensor(batch_idx),
                # It has to b be on the one level down because we should have for every element of the batch one id, otherwise we get errors when we shuffle dataset
                "unit_id": torch.tensor(
                    [self.unid_id] * len(batch_idx)
                ),  # * len(batch_idx)
                # How can we actually collate strings?
                # "unit_name": (self.unid_name),
                # (B=1, seq_len, C) -> (B=1, C),
                # take the last time step,
                # collate fn will concatenate over the batch dim
                "rul": target_seq_x[:, -1],
                # (B, seq_len, C)
                "features": context_seq_x,
            }
        )

        ragged_indices = context.get("idx_sel", None)
        if ragged_indices is not None:
            d["ragged_indices"] = ragged_indices
        return AttributeDict(d)

    def get_collate_fn(self):
        return collate_key_value_batch


# import awkward as ak
# import numpy as np

# arr = ak.Array({
#     "level1": [
#         {"a": 1, "b": 4, "c": 6},
#         {"a": 2, "b": 5},
#         {"a": 3, "b": None, "c": None}
#     ]
# })
# print(arr)

# index = np.array([("level1", "b"), ("level1", "c")])

# print(index.shape)
