# tabularize:
#   # Concatenate time-features to the feature set
#   transform:
#     _target_: picid.transforms.base_transforms.tabularizers.TimeseriesTabularizer
#     select_features:
#       - time_features: t
#       - features: t
#       - target: history

#   metadata:
#     apply_to: ['features', 'time_features', 'target']
#     assign_to: features

"""Tabularize time-series tensors for forecasting feature pipelines."""

import logging
from typing import Any, Dict, Optional

import omegaconf
import awkward as ak
import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf

# Assuming this import path is correct for your project
from picid.data.data_objects import NamedTransformInput
from picid.data.datasets.subset_sampling import (
    create_sequence_subset,
    make_subset_blocks_indices,
)
from picid.data.optimization.sequencer import DenseArraySequencer, RaggedArraySequencer
from picid.transforms.base.base_transform.base_transform import RaggedOrDenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeseriesTabularizer(NoFitPerSegmentMixin, RaggedOrDenseTransform):
    r"""
    Tabularize selected history, present, and horizon features.

    This transform can be used to extract time-based features alongside the
    existing forecasting inputs. In earlier configs the ``freq`` keyword was
    commonly passed through to the underlying ``time_features`` utility (for
    example hourly ``"h"`` or daily ``"d"`` data); that pass-through remains
    available through ``**kwargs`` when the downstream transform stack expects
    it.

    Parameters
    ----------
    select_features : dict[str, str]
        Mapping from input key to selection mode.
    timestep_dimension : int
        Timestep dimension used by downstream consumers.
    seq_len : int
        Input history length.
    label_len : int
        Decoder label length.
    pred_len : int
        Prediction horizon length.
    stride : int
        Sliding-window stride.
    subset_ratio : float, optional
        Fraction of sequences to keep when subsetting.
    subset_seed : int, default=42
        Random seed used for deterministic subsetting.
    subset_blocks : int, optional
        Number of subset blocks for grouped sampling.
    pred_offset : int, default=0
        Forecast offset applied when building sequences.
    padding_left_flag : bool, default=True
        Whether sequence padding should be left-aligned.
    warmup_steps : int, optional
        Number of warmup steps passed to the sequencer.
    **kwargs
        Extra keyword arguments accepted for API compatibility and forwarded to
        the broader transform setup, including optional ``freq`` configuration.
    """

    def __init__(
        self,
        select_features: dict[str, str],
        timestep_dimension: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int,
        subset_ratio: Optional[float] = None,
        subset_seed: int = 42,
        subset_blocks: Optional[int] = None,
        pred_offset: int = 0,
        padding_left_flag: bool = True,
        warmup_steps: int = None,
        **kwargs,
    ):
        """
        Instantiate a tabularizer (see class docstring for behavior overview).

        Parameters
        ----------
        select_features : dict[str, str]
            Mapping from input key to selection mode.
        timestep_dimension : int
            Timestep dimension used by downstream consumers.
        seq_len : int
            Input history length.
        label_len : int
            Decoder label length.
        pred_len : int
            Prediction horizon length.
        stride : int
            Sliding-window stride.
        subset_ratio : float, optional
            Fraction of sequences to keep when subsetting.
        subset_seed : int
            Random seed used for deterministic subsetting.
        subset_blocks : int, optional
            Number of subset blocks for grouped sampling.
        pred_offset : int
            Forecast offset applied when building sequences.
        padding_left_flag : bool
            Whether sequence padding should be left-aligned.
        warmup_steps : int, optional
            Number of warmup steps passed to the sequencer.
        **kwargs
            Extra keyword arguments for API compatibility (for example ``freq``).
        """
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
        self.subset_ratio = subset_ratio
        self.subset_seed = subset_seed
        self.pred_offset = pred_offset
        self.padding_left_flag = padding_left_flag
        self.warmup_steps = warmup_steps
        self.subset_blocks = subset_blocks

        # Store initial kwargs for __repr__, if BaseTransform doesn't already do this comprehensively
        # (your provided BaseTransform doesn't show _init_kwargs in its __init__,
        # but the base transform pattern does, so keeping it consistent for your repr)
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

    def _create_sequencer(self, arr, sequencer_params, key):
        """
        Create the right sequencer implementation for one array.

        Parameters
        ----------
        arr : Any
            Input array to sequence.
        sequencer_params : dict
            Keyword arguments forwarded to the sequencer.
        key : str
            Input key associated with ``arr``.

        Returns
        -------
        DenseArraySequencer or RaggedArraySequencer
            Sequencer matching the input array type.
        """
        if isinstance(arr, (np.ndarray, torch.Tensor)):
            seq = DenseArraySequencer(arr, **sequencer_params)

        elif isinstance(arr, ak.Array):
            seq = RaggedArraySequencer(arr, **sequencer_params)
        else:
            if isinstance(arr, omegaconf.ListConfig):
                raise NotImplementedError(
                    f"Unsupported type for key '{key}': {type(arr)}. "
                    "Most likely you should use HydraConcat to instantiate a separate iterator for each unit, as it looks like you are trying to past multiunit data into the single unit dataset class."
                )
            else:
                raise NotImplementedError(
                    f"Unsupported type for key '{key}': {type(arr)}"
                )
        return seq

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> Any:
        r"""
        Apply tabularization to the input data.

        Make sure that the input data to this transform is of shape (N, T, ...)
        Where N is the number of samples and T is the task number.

        Parameters
        ----------
        data : NamedTransformInput
            The input data segment to be tabularized.
        metadata : dict
            Split-specific metadata used during sequencing.

        Returns
        -------
        np.ndarray
            A NumPy array containing the extracted time features and other
            tabularized selections.
        """
        data_segment = data
        N_samples = None
        feature_data = {}
        _split = metadata["mode"]

        for key, selection in self.select_features.items():
            data = data_segment[key]

            if N_samples is not None:
                assert (
                    N_samples == data.shape[0]
                ), f"Inconsistent number of features across segments: {N_samples} != {data.shape[0]}"

            def collect_sequences(collect_idx):
                sequencer_params = dict(
                    seq_len=self.seq_len,
                    label_len=self.label_len,
                    pred_len=self.pred_len,
                    stride=self.stride,
                    pred_offset=self.pred_offset,
                    padding_left_flag=self.padding_left_flag,
                    warmup_steps=self.warmup_steps,
                )

                seq = self._create_sequencer(data, sequencer_params, key)

                subset_ratio = self.subset_ratio
                if subset_ratio is not None and 0 < subset_ratio < 1:
                    len_ds = len(seq)
                    if self.subset_blocks is not None and (
                        int(len_ds * subset_ratio) > self.subset_blocks
                    ):
                        res = make_subset_blocks_indices(
                            len_ds=len_ds,
                            subset_ratio=subset_ratio,
                            subset_seed=self.subset_seed,
                            subset_blocks=self.subset_blocks,
                        )
                        seq_idx = res["seq_idx"]
                    else:
                        seq_idx = create_sequence_subset(
                            len_ds,
                            subset_ratio=subset_ratio,
                            subset_seed=self.subset_seed,
                        )

                    logger.info(
                        f"Subsetting sequences: using {len(seq_idx)} out of {len(seq)} sequences."
                    )
                    logger.info(f"Subset indices start with: {seq_idx[:100]}")
                else:
                    seq_idx = np.arange(len(seq))

                results = seq.sequences_batch(seq_idx)
                return results[collect_idx]

            if selection == "t":
                seqs = collect_sequences(1)  # Collect seq_y
                feature_data[key] = seqs

            if selection == "present":
                seqs = collect_sequences(0)  # Collect seq_x
                feature_data[key] = seqs

            if selection == "history":
                seqs = collect_sequences(0)  # Collect seq_x
                feature_data[key] = seqs

            if selection == "horizon":
                seqs = collect_sequences(1)  # Collect seq_y
                feature_data[key] = seqs

        # Now reshape in batch task dimension.
        # Here we directly create a 3D tensor with the task dimension as the first dimension.
        # In terms of tabularization, this mmeans we create all task tables in one go.
        for key, selection in self.select_features.items():
            if selection == "t":
                # We want to select all contextual features at the prediction time point t.
                feature_data[key] = rearrange(feature_data[key], "b t f -> t b f")
            elif selection == "present":
                # Present is defined as the last time point of the history window.
                # b = batch h = history (time steps) f = features
                # We want to select all contextual features at the prediction time point t.
                feature_data[key] = rearrange(
                    feature_data[key][:, -1, None], "b 1 f -> 1 b f"
                )
            elif selection == "history":
                # b = batch h = history (time steps) f = features
                # We add the task dimension as the first dimension and flatten h and f
                # because we want all of history for all features in the tabular vector.
                feature_data[key] = rearrange(feature_data[key], "b h f -> 1 b (h f)")
            elif selection == "horizon":
                # Since the tabularization only allows one target (scalar), we map the horizon
                # to the task dimension. This way, we create new tasks for each horizon step.
                feature_data[key] = rearrange(feature_data[key], "b t f -> t b f")
            else:
                raise ValueError(f"Unknown selection type: {selection}")

        n_tasks = set([values.shape[0] for key, values in feature_data.items()])

        assert (
            len(n_tasks) == 2 or len(n_tasks) == 1
        ), f"Inconsistent number of task dimensions across features. Can be either one or {max(n_tasks)} tasks, but got {n_tasks}"

        # In mode history, we want to concatenate the history up to the present time point.
        # The history remains the same for any forecasting timepoint t+h (h=horizon)

        """
        Current model of the naming convention:
        naming scheme p = present

                                        t
                                        |
                    --------------------------
           Target   |  history | p | horizon |
                    --------------------------
           Context1 |  history | p | horizon |
           Context2 |  history | p | horizon |
                    --------------------------
                        ---> time axis
        """

        # That means some features have task dimension 1 (history) while other
        # have bigger task dimension due to the fact that we create directly the 3D table for all tasks.
        # Therefore we need to pad the task dimensions of the history features to the max task dimension.
        if max(n_tasks) != min(n_tasks) and (
            "history" in self.select_features.values()
            or "present" in self.select_features.values()
        ):
            for key, selection in self.select_features.items():
                if selection in ["history", "present"]:
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
        """
        Run :meth:`transform_data` without split metadata.

        Parameters
        ----------
        data : Any
            Input forwarded to :meth:`transform_data`.

        Returns
        -------
        Any
            Output of :meth:`transform_data`.
        """
        return self.transform_data(data, None)
