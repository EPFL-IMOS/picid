from typing import Any, Dict
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin
import logging

logger = logging.getLogger(__name__)


class ExtractSequenceStartTime(NoFitPerSegmentMixin, DenseTransform):
    """
    Extracts the first time step from a time-feature sequence to represent
    the starting time of the event.

    This transform is robust and handles both 3D (N, timesteps, channels)
    and 4D (N, windows, timesteps, channels) input shapes.
    """

    def __init__(self, key="time_features"):
        super().__init__()
        self.key = key

    def transform_data(self, data: Any, metadata: Dict) -> Any:
        time_features = data[self.key]

        # Check the number of dimensions of the input array
        if time_features.ndim == 4:
            # Handle 4D input: (N, windows, timesteps, channels)
            # Select the first timestep of the first window.
            start_time_feature = time_features[:, 0, 0, :]
            logger.info("Detected 4D input. Extracting start time from first window.")

        elif time_features.ndim == 3:
            # Handle 3D input: (N, timesteps, channels)
            # Select the first timestep of the sequence.
            start_time_feature = time_features[:, 0, :]
            logger.info("Detected 3D input. Extracting first timestep.")

        else:
            # Handle unsupported shapes
            raise ValueError(
                f"Unsupported input dimensions for time_features: {time_features.ndim}. "
                "This transform only supports 3D or 4D arrays."
            )

        logger.info(f"Added 'start_time_feature' with shape {start_time_feature.shape}")

        return start_time_feature

    def fit_data(self, data: Any, metadata: Dict):
        """This transform is stateless and does not require fitting."""
        pass
