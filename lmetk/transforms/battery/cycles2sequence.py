from typing import Any, Dict
import numpy as np
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
from lmetk.data.datasources.nb14.utils import unflatten_cycles
import logging

logger = logging.getLogger(__name__)


class Cycles2Sequence(DenseTransform, NoFitMultiSourceTransformMixin):
    """
    Compress cycles to the mean/std representations of the cycles for NB1 dataset
    """

    def __init__(self, win_len=None, pad_value=0, key="features"):
        super().__init__()
        self.win_len = win_len
        self.key = key

        if pad_value == "np.nan":
            pad_value = np.nan

        self.pad_value = pad_value

    def transform_data(self, data: Any, metadata: Dict) -> Any:
        """Transforms a single data segment using the internal ."""

        # Get target if available, otherwise it's None
        flattened_y = data.get("target")

        features, target = unflatten_cycles(
            flattened_x=data[self.key],
            flattened_y=flattened_y,
            cycle_indices=data["metadata"]["cycle_indices"],
            win_len=self.win_len,
            pad_value=self.pad_value,
        )

        logger.info(f"Features shape after convertion to sequences: {features.shape}")

        output = {self.key: features}

        if target is not None:
            # Ensure target is 2D: (N, 1)
            if target.ndim == 3 and target.shape[2] == 1:
                target = target.squeeze(axis=2)
            elif target.ndim == 1:
                target = target.reshape(-1, 1)
            elif target.ndim == 0:
                target = np.array([target]).reshape(-1, 1)
            logger.info(f"Target shape after convertion to sequences: {target.shape}")
            output["target"] = target

        return output

    def fit_data(
        self,
        data: Any,
        metadata: Dict,
    ):

        pass
