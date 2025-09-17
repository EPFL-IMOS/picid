import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import hydra
import omegaconf
import torch
from lmetk.transforms.base.base_transform import DenseTransform, RaggedTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceSpecialTransformMixin
from lmetk.transforms.battery.utils import get_mode_change_right_indices
import pandas as pd
import hydra

# Set up a logger for better feedback
logger = logging.getLogger(__name__)


import torch
from lmetk.data.optimization.sequencer import AutoregressiveSequencer
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.utils.awkward_utils import (
    ak_find_var_dims,
    ak_regularize_regular_axes,
    ragged_index_tuples,
)
from lmetk.transforms.base.multisource import find_singular_ragged_dim
import copy
from collections import defaultdict


class SpecialOneTransform(RaggedTransform, NoFitMultiSourceSpecialTransformMixin):
    """
    An adapter for sklearn.preprocessing.MinMaxScaler, conforming to BaseTransform interface.
    Handles various input data types by converting them to numpy.ndarray internally
    and converting back to the original type.
    """

    def fit_data(self, data: Any, metadata: Dict):
        """This transform does not require fitting."""
        pass

    def transform_data(self, data: Any, metadata: Dict) -> Any:

        # Ensure that all the keys have same dim 0
        assert len(set([len(val) for val in data.values()])) == 1, (
            f"All keys in data must have the same length at the first dimension. "
            f"Found lengths: {[len(val) for val in data.values()]}"
        )

        var_dim = find_singular_ragged_dim(data)
        multi_index = ragged_index_tuples(next(iter(data.values())), var_dim)

        # Merge metadata from the segment with the overall metadata
        data_metadata = copy.deepcopy(metadata)
        data_metadata.update(data.metadata or {})

        out_segment = defaultdict(list)
        iterator = multi_index

        for coord in iterator:
            for k in data.keys():
                np_data = data[k][coord].to_numpy()
                out_segment[k].append(np_data)

        return out_segment
