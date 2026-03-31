from typing import Any, Dict, Optional
import numpy as np
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin
import logging

logger = logging.getLogger(__name__)


class Sequence2Statistics(NoFitPerSegmentMixin, DenseTransform):
    """
    Summarize each input cycle with per-channel mean and standard deviation.

    Parameters
    ----------
    padding_value : float, optional
        Value that should be treated as padding and ignored.
    """

    def __init__(
        self,
        padding_value: Optional[float] = None,
    ):
        """
        Remember the optional padding marker used during aggregation.

        Parameters
        ----------
        padding_value : float, optional
            Value that should be treated as padding and ignored.
        """
        super().__init__()
        self.padding_value = padding_value

    def transform_data(self, data: Any, metadata: Dict) -> Any:
        """
        Collapse a single sequence into mean and standard deviation values.

        Parameters
        ----------
        data : Any
            Mapping containing exactly one sequence-like entry.
        metadata : dict
            Metadata dictionary preserved by the interface.

        Returns
        -------
        numpy.ndarray
            One-dimensional vector containing mean and standard deviation
            summaries.
        """
        # (N, win_len, channels)
        assert len(data) == 1, "Data dictionary must contain exactly one entry."
        _, features = next(iter(data.items()))

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
        """
        Skip fitting because the transform is stateless.

        Parameters
        ----------
        data : Any
            Training data, ignored.
        metadata : dict
            Metadata dictionary, ignored.
        """
        pass
