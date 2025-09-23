from typing import List

from numpy import ndarray
import awkward as ak
from lmetk.data.datasets.base import BaseDataset
from lmetk.data.datasets.collate_functions import forecasting_collate_fn
from lmetk.data.optimization.sequencer import AutoregressiveSequencer
from lmetk.data.optimization.ak_jagged_sequencer import (
    AkwardJaggedAutoregressiveSequencer,
)
from lmetk.data.utils import to_torch_tensor
from lmetk.data.data_objects.data import DatasetContainer


class SlidingWindowBatchDataset(BaseDataset):
    """Efficient implementation of Standard 'sliding window' Time Series Dataset Class."""

    def __init__(
        self,
        data_dict: dict[str, ndarray | list[ndarray]],
        seq_len: int,  # window size
        label_len: int,  # overlap size
        pred_len: int,  # prediction size
        stride: int,
        **kwargs,
    ):
        # Make sure that the data_dict contains the required keys
        required_keys = ["features", "time_features"]
        for key in required_keys:
            if key not in data_dict:
                raise ValueError(f"Data dictionary must contain the key: '{key}'")
        # Workout data_dict here:
        data_types = {}
        for key, val in data_dict.items():
            data_types[key] = type(val)

        assert len(set(data_types.values())) == 1, "Expected the same types of input"
        data_type = list(data_types.values())[-1]

        if data_type == ndarray:

            # Convert the data_dict["features"] and data_dict["time_features"] to torch.Tensor
            features = to_torch_tensor(data_dict["features"])
            time_features = to_torch_tensor(data_dict["time_features"])

            assert (
                to_torch_tensor(data_dict["features"]).shape[0]
                == to_torch_tensor(data_dict["time_features"]).shape[0]
            ), "Features and time features must have the same length"

            # Add here the JaggedSequenceDataset

            # TODO: Make sure that the stride covers all the data
            self.feature_sequencer = AutoregressiveSequencer(
                features=features,
                time_features=time_features,
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                stride=stride,
            )
        elif data_type == ak.Array:
            """The following logic should come here"""
            # ATTENTION BUG ON PURPOSE WITH time_features=data_dict["features"]
            self.feature_sequencer = AkwardJaggedAutoregressiveSequencer(
                features=data_dict["features"],
                time_features=data_dict["features"],
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                stride=stride,
            )
        else:
            raise NotImplementedError(
                "Not implemented yet, SlidingWindonBatchDataset assumed to work with ndawway of with awkward ragged array"
            )

    def __len__(self):
        return self.feature_sequencer.__len__()  # len(self.feature_sequencer)

    def __getitem__(self, batch_idx: List[int]):
        """
        Get a batch of sequences based on the provided indices.
        Args:
            batch_idx (List[int]): Indices of the sequences to retrieve.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - seq_x: Input sequences.
                - seq_y: Target sequences.
                - seq_x_time: Time features for input sequences.
                - seq_y_time: Time features for target sequences.
        """
        seq_x, seq_y, seq_x_time, seq_y_time = self.feature_sequencer.sequences_batch(
            batch_idx
        )
        return seq_x, seq_y, seq_x_time, seq_y_time

    def get_collate_fn(self):
        return forecasting_collate_fn
