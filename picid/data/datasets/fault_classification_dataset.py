# @override decorator
import torch
from typing import Dict, List, override

from lightning_fabric.utilities.data import AttributeDict
from picid.data.datasets.rul_context_dataset import RULContextBatchDataset


class FaultClassificationBatchDataset(RULContextBatchDataset):
    """Creates a dataset for Fault Classification.

    This class specializes `ContextBatchDataset` for a common fault classification
    scenario: predicting the fault class at the end of a given sequence of operational
    data. It uses a sliding window to generate input sequences (`X`) and
    corresponding target sequences (`y`).

    For a given input feature sequence of length `seq_len`, the corresponding
    target fault class  sequence also has length `seq_len`. In a typical training loop,
    the target label is then taken as the **last value** of this fault class sequence.

    Parameters
    ----------
    data_dict : dict[str, ndarray | list[ndarray]]
        A dictionary mapping string keys (e.g., 'features', 'fault_class') to the
        time-series data arrays. The arrays can be NumPy arrays, PyTorch Tensors,
        or Awkward Arrays.
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
    >>> # Fault classes are integers from 0 to 4
    >>> fault_classes = np.random.randint(0, 5, size=(100, 1))
    >>> data = {'features': features, 'fault_class': fault_classes}
    ...
    >>> # Create a dataset with a window size of 20
    >>> dataset = FaultClassificationBatchDataset(
    ...     data_dict=data,
    ...     task_type='fault_classification',
    ...     seq_len=20,
    ...     label_len=10,
    ...     pred_len=5  # This will be forced to 0
    ... )
    ...
    >>> # Get the first sample from the dataset
    >>> first_sample = dataset[0]
    >>> input_features = first_sample['features']
    >>> target_fault_class_sequence = first_sample['fault_class']
    ...
    >>> print(f"Input features shape: {input_features.shape}")
    Input features shape: (20, 3)
    >>> print(f"Target fault class sequence shape: {target_fault_class_sequence.shape}")
    Target fault class sequence shape: (20, 1)
    ...
    >>> # In training, the target is the last value of the fault class sequence.
    >>> # Corresponds to the fault class at the end of the 20-step feature window.
    >>> final_target_fault_class = target_fault_class_sequence[-1]
    >>> print(f"Final target fault class value: {final_target_fault_class}")
    Final target fault class value: [4]

    Yields
    ------
    dict
        A dictionary where keys correspond to those in `data_dict` and values
        are the sequenced NumPy arrays for a given sample index.
    """

    @override
    def __getitem__(self, batch_idx: List[int]):
        target: Dict = self.target_dataset.__getitem__(batch_idx)
        context: Dict = self.context_dataset.__getitem__(batch_idx)
        target_seq_x = target[f"{self.task_type}_seq_x"]
        context_seq_x = context["features_seq_x"]

        # TO FINISH: we have now "unit_id" and "unit_name" in batch, how to collate it?
        d = AttributeDict(
            {
                "batch_idx": torch.tensor(batch_idx).unsqueeze(1),  # (B,1)
                # Out targets are (B,F) or (B,T,F), in the case of RUL here we have (B,F); F is a number of classes
                f"{self.task_type}": target_seq_x[
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
