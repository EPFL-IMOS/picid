# @override decorator
import torch
from typing import Dict, List, Union, Optional, override

from lightning_fabric.utilities.data import AttributeDict
from numpy import ndarray

from picid.data.datasets.collate_functions import collate_key_value_batch
from picid.data.datasets.context_dataset import ContextBatchDataset


class RULContextBatchDataset(ContextBatchDataset):
    """Creates a dataset for Remaining Useful Life (RUL) prediction.

    This class specializes `ContextBatchDataset` for a common RUL prediction
    scenario: predicting the RUL at the end of a given sequence of operational
    data. It uses a sliding window to generate input sequences (`X`) and
    corresponding target sequences (`y`).

    For a given input feature sequence of length `seq_len`, the corresponding
    target RUL sequence also has length `seq_len`. In a typical training loop,
    the target label is then taken as the **last value** of this RUL sequence.

    Parameters
    ----------
    data_dict : dict[str, ndarray | list[ndarray]]
        A dictionary mapping string keys (e.g., 'features', 'RUL') to the
        time-series data arrays. The arrays can be NumPy arrays, PyTorch
        Tensors, or Awkward Arrays.
    task_type : str
        A string identifier for the task type (e.g., 'regression'), used by
        parent classes.
    seq_len : int
        The length of the sliding window, defining the number of historical
        time steps in each input sample.
    label_len : int
        The length of the "label" or "overlap" portion of the sequence, used
        by the underlying sequencer. It defines the starting point of the
        target sequence relative to the input sequence.
    pred_len : int
        The prediction horizon. **Note: This argument is ignored and
        internally overridden to 0** to make the target window align
        perfectly with the input window.
    stride : int, optional
        The step size the sliding window takes across the data between
        generating consecutive sequences. Defaults to 1.
    **kwargs
        Additional keyword arguments passed to the parent `ContextBatchDataset`.

    Examples
    --------
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

    Yields
    ------
    dict
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
        get_unit_id: bool = False,
        padding_left_flag: Union[bool, int] = True,
        warmup_steps: int = None,
        subset_ratio: Optional[float] = None,
        subset_seed: int = 42,
        subset_blocks: Optional[int] = None,
        **kwargs,
    ):
        assert pred_len == 0, "pred_len must be 0 for RUL prediction."
        # It is makes sense in the context of RUL as we want to predict the RUL at the end of the sequence x, seq_y is not used in this dataset

        self.meta_data_dict = kwargs["meta_data_dict"]

        self.get_unit_id = get_unit_id
        # Inject the unit name/id extraction functions into the meta_data_dict
        if self.get_unit_id:
            self.unid_id = self.extract_unit_id()
            self.unid_name = self.extract_unit_name()

        super().__init__(
            data_dict=data_dict,
            task_type=task_type,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
            padding_left_flag=padding_left_flag,
            warmup_steps=warmup_steps,
            subset_ratio=subset_ratio,
            subset_seed=subset_seed,
            subset_blocks=subset_blocks,
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
                "batch_idx": torch.tensor(batch_idx).unsqueeze(1),  # (B,1)
                # Out targets are (B,F) or (B,T,F), in the case of RUL here we have (B,F)
                "rul": target_seq_x[
                    :, -1
                ],  # Collate fn will concatenate over the batch dim take the last time step; # (B=1, seq_len, C) -> (B=1, C),
                "features": context_seq_x,  # (B, seq_len, C)
            }
        )

        if self.get_unit_id:
            # (B,F)
            d["unit_id"] = torch.tensor([self.unid_id] * len(batch_idx))
            # How can we actually collate strings?
            # "unit_name": (self.unid_name),
        return AttributeDict(d)

    def get_collate_fn(self):
        return collate_key_value_batch
