from typing import Any, Dict, Optional
import numpy as np
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
import logging

logger = logging.getLogger(__name__)


class Sequence2Statistics(DenseTransform, NoFitMultiSourceTransformMixin):
    """
    Transforms sequence data by computing the mean and standard deviation
    for each cycle, ignoring a specified padding value and np.nan.

    Input shape: (N_cycles, cycle_len (var), channels)
    Output shape: (N_cycles, channels * 2)
    """

    def __init__(
        self,
        padding_value: Optional[float] = None,
    ):
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
        # (N, win_len, channels)
        features = data["features"]

        # Check how many dim in just 3 them unsqueeze 1st dim
        if features.ndim == 2:
            logger.debug(f"Input features shape: {features.shape}")
            # unsqueeze 0 dim to have (sw, win_len, channels). For example (sw of single cycle, win len, channels)
            features = features[np.newaxis, np.newaxis, :, :]
            logger.debug(
                f"Reshaped features to: {features.shape}, unsqueezed two dim to have (N_cycles=1, N_win=1, win_len, channels)"
            )

        logger.debug(f"Original feature shape for statistics: {features.shape}")

        # Create a working copy to handle padding, ensuring dtype supports nan
        working_features = features.astype(float)

        # NEW: If a padding_value is specified, replace it with np.nan
        if self.padding_value is not None:
            working_features[working_features == self.padding_value] = np.nan
            logger.debug(f"Replaced padding value '{self.padding_value}' with np.nan.")

        # 1. Calculate mean using the modified data
        mean_per_cycle = np.nanmean(working_features, axis=(1, 2))

        # 2. Calculate standard deviation using the modified data
        std_per_cycle = np.nanstd(working_features, axis=(1, 2))

        # 3. Clean up potential NaNs in std output
        std_per_cycle = np.nan_to_num(std_per_cycle, nan=0.0)

        # 4. Concatenate mean and std to form the new feature vector
        # statistical_features = np.concatenate([mean_per_cycle, std_per_cycle], axis=1)
        statistical_features = np.concatenate(
            [mean_per_cycle.T, std_per_cycle.T], axis=0
        ).squeeze(1)

        logger.debug(f"New statistical feature shape: {statistical_features.shape}")

        assert statistical_features.ndim == 1, "Output should be 1D array per cycle."
        # 5. Return the calculated features
        return statistical_features

    def fit_data(self, data: Any, metadata: Dict):
        """This transform is stateless and does not require fitting."""
        pass
