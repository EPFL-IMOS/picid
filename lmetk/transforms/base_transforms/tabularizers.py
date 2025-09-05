# tabularize:
#   # Concatenate time-features to the feature set
#   transform:
#     _target_: lmetk.transforms.base_transforms.tabularizers.TimeseriesTabularizer
#     select_features:
#       - time_features: t
#       - features: t
#       - target: history

#   metadata:
#     apply_to: ['features', 'time_features', 'target']
#     assign_to: features

import logging
from typing import Any, Dict

import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf

# Assuming this import path is correct for your project
from lmetk.data.data_objects.data import OrderedInput
from lmetk.data.optimization.sequencer import AutoregressiveSequencer
from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeseriesTabularizer(BaseTransform, ConcatinatedFitMultiSourceTransformMixin):
    r"""Extracts time features from the input data.

    This transform processes time-series data to generate additional features
    based on the timestamp, using the `time_features` utility.

    Parameters
    ----------
    freq : str
        The frequency string (e.g., 'h' for hourly, 'd' for daily)
        to be used by the `time_features` function.
    **kwargs : optional
        Additional parameters for the base transform.
    """

    def __init__(
        self,
        select_features: dict[str, str],
        timestep_dimension: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int,
        **kwargs,
    ):

        # Call the BaseTransform constructor.
        # Your BaseTransform doesn't take specific 'apply_to'/'fit_on' in its init,
        # so just pass any generic kwargs along.
        super().__init__()

        sf = OmegaConf.to_container(select_features, resolve=True)
        merged_sf = {}
        for d in sf:
            merged_sf.update(d)

        self.select_features = merged_sf
        self.timestep_dimension = timestep_dimension

        # check if horizon in merged_sf, then there should be only one key and its horizon
        # Because currently, the horizon is in there, its the target transform which should only return y
        if "horizon" in merged_sf.values():
            assert (
                len(merged_sf) == 1
            ), "If 'horizon' is present, it must be the only key."

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride

        # Store initial kwargs for __repr__, if BaseTransform doesn't already do this comprehensively
        # (your provided BaseTransform doesn't show _init_kwargs in its __init__,
        # but your original TimePFNTimeFeaturesTransform did, so keeping it consistent for your repr)
        self._init_kwargs = kwargs

    def __repr__(self):
        # Ensure 'freq' is included in the representation, along with any other init args
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        # Build a string representation for select_features dict
        if isinstance(self.select_features, dict):
            select_repr = ", ".join(
                f"{k}={v!r}" for k, v in self.select_features.items()
            )
            args += f", select_features={{ {select_repr} }}"
        else:
            args += f", select_features={self.select_features!r}"

        return f"{self.__class__.__name__}({args})"

    def transform_data(self, data: OrderedInput, metadata: Dict) -> Any:
        r"""Applies the tabularization to the input data.

        Make sure that the input data to this transform is of shape (N, T, ...)
        Where N is the number of samples and T is the task number.

        Parameters
        ----------
        data_segment : Any
            The input data segment to be tabularized.

        Returns
        -------
        np.ndarray
            A NumPy array containing the extracted time features.
        """
        data_segment = data
        N_samples = None
        feature_data = {}

        for key, selection in self.select_features.items():

            data = data_segment[key]

            if N_samples is not None:
                assert (
                    N_samples == data.shape[0]
                ), f"Inconsistent number of features across segments: {N_samples} != {data.shape[0]}"

            seq = AutoregressiveSequencer(
                features=torch.from_numpy(data),
                time_features=None,
                seq_len=self.seq_len,
                label_len=self.label_len,
                pred_len=self.pred_len,
                stride=self.stride,
            )

            def collect_sequences(collect_idx):
                seqs = []
                for i in range(len(seq)):
                    results = seq.sequences_batch(i)
                    seqs.append(results[collect_idx])
                return torch.stack(seqs, dim=0)

            if selection == "t":
                seqs = collect_sequences(1)  # Collect seq_y
                feature_data[key] = seqs

            if selection == "history":
                seqs = collect_sequences(0)  # Collect seq_x
                feature_data[key] = seqs

            if selection == "horizon":
                seqs = collect_sequences(1)  # Collect seq_x
                feature_data[key] = seqs

        # Now reshape in batch task dimension.
        for key, selection in self.select_features.items():
            if selection == "t":
                feature_data[key] = rearrange(feature_data[key], "b t ... -> t b ...")
            elif selection == "history":
                feature_data[key] = rearrange(feature_data[key], "b h 1 -> 1 b h")
            elif selection == "horizon":
                feature_data[key] = rearrange(feature_data[key], "b t 1 -> t b 1")

        n_tasks = set([values.shape[0] for key, values in feature_data.items()])

        assert (
            len(n_tasks) == 2 or len(n_tasks) == 1
        ), f"Inconsistent number of task dimensions across features. Can be either one or {max(n_tasks)} tasks, but got {n_tasks}"

        # In mode history, we want to concatenate the history up to the present time point.
        # The history remains the same for any forecasting timepoint t+h (h=horizon)

        """
        Current model of the naming convention:

                                        t
                                        |
                    --------------------------
           Target   |  history | p | horizon |
                    --------------------------
           Context  |  history | p | horizon |
                    --------------------------
                        ---> time axis
        """

        if max(n_tasks) != min(n_tasks) and (
            "history" in self.select_features.values()
        ):

            for key, selection in self.select_features.items():
                if selection == "history":
                    feature_data[key] = repeat(
                        feature_data[key], "1 b h -> t b h", t=max(n_tasks)
                    )

        # Return the values as a NumPy array.
        out = [data for key, data in feature_data.items()]
        if len(out) > 1:
            out = torch.concat(
                [data for key, data in feature_data.items()], dim=2
            ).numpy()
            return out
        else:
            return out[0].numpy()

    # Keeping your original __call__ and transform methods for compatibility.
    # They simply delegate to the new transform_data method.
    def __call__(self, data: Any) -> Any:
        return self.transform_data(data, None)
