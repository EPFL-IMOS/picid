from typing import Any, Dict, Optional
import numpy as np
from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
import logging

logger = logging.getLogger(__name__)


class Sequence2Statistics(BaseTransform, NoFitMultiSourceTransformMixin):
    """
    Transforms sequence data by computing the mean and standard deviation
    for each cycle, ignoring a specified padding value and np.nan.

    Input shape: (N_cycles, N_win, win_len, channels)
    Output shape: (N_cycles, channels * 2)
    """
    def __init__(self, padding_value: Optional[float] = None):
        """
        Args:
            padding_value (Optional[float]): A numerical value to be treated
                as padding and ignored during calculations. np.nan is always ignored.
        """
        super().__init__()
        self.padding_value = padding_value

    def transform_data(self, data: Any, metadata: Dict) -> Any:
        """
        Transforms a single data segment by calculating statistics.
        """
        features = data['features']
        logger.info(f"Original feature shape for statistics: {features.shape}")

        # Create a working copy to handle padding, ensuring dtype supports nan
        working_features = features.astype(float)

        # NEW: If a padding_value is specified, replace it with np.nan
        if self.padding_value is not None:
            working_features[working_features == self.padding_value] = np.nan
            logger.info(f"Replaced padding value '{self.padding_value}' with np.nan.")

        # 1. Calculate mean using the modified data
        mean_per_cycle = np.nanmean(working_features, axis=(1, 2))

        # 2. Calculate standard deviation using the modified data
        std_per_cycle = np.nanstd(working_features, axis=(1, 2))
        
        # 3. Clean up potential NaNs in std output
        std_per_cycle = np.nan_to_num(std_per_cycle, nan=0.0)

        # 4. Concatenate mean and std to form the new feature vector
        statistical_features = np.concatenate([mean_per_cycle, std_per_cycle], axis=1)

        logger.info(f"New statistical feature shape: {statistical_features.shape}")

        # 5. Return the calculated features
        return statistical_features

    def fit_data(self, data: Any, metadata: Dict):
        """This transform is stateless and does not require fitting."""
        pass