import logging
from typing import Any, Dict, List

import numpy as np
from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import RaggedOrDenseTransform
from lmetk.transforms.base.multisource import (
    StatelessPerSegmentMixin,
)
from lmetk.data.data_objects.utils import convert_to_numpy

logger = logging.getLogger(__name__)


class RULTaskUpToFirstZero(RaggedOrDenseTransform, StatelessPerSegmentMixin):

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Transforms the input data by truncating all arrays up to and including the first occurrence of zero in the 'target' array.
        This method searches for the first zero in the 'target' array, then truncates 'target' and all other arrays in the input data dictionary to this length.
        It ensures that the last element of the truncated 'target' is zero and that all other arrays are truncated to match the new length of 'target'.

        Args:
            data: Dictionary containing numpy arrays to concatenate
            metadata: Dictionary containing auxiliary metadata (unused in this transform)

        Returns:
            Concatenated numpy array
        Raises:
            AssertionError: If 'target' key is missing, no zero is found in 'target',
                            the last element after truncation is not zero,
                            or if there are no other keys to process.
        """
        assert "target" in data, "Missing 'target' key in data"

        # Identify how many zeros in target
        target = convert_to_numpy(data["target"])
        target.shape
        # Get number of zeros
        num_zeros = np.sum(target == 0)
        logger.info(f"Number of zeros in target: {num_zeros}")

        # Find the first zero occurance and cut everything after
        first_zero_idx = np.argmax(target == 0)
        if first_zero_idx == 0:
            logger.warning("No zero found in target, keeping the entire array")
            return {"target": target, "features": convert_to_numpy(data["features"])}

        assert first_zero_idx > 0, "No zero found in target"
        # Shift idx to include zero
        first_zero_idx += 1
        target = target[:first_zero_idx]

        logger.info(f"Cut target to first zero occurrence: {target.shape}")
        assert target[-1] == 0, "Last element of target is not zero after cutting"
        # Check the target has exactly one zero
        assert np.sum(target == 0) == 1, "Target does not contain exactly one zero"

        # Create the output dictinary
        out_dict = {"target": target}

        keys = list(data.keys())
        keys.remove("target")

        assert len(keys) > 0, "No keys to process other than target"
        logger.info(f"The values of {keys} going to be truncated to satisfy ")

        assert (len(keys) == 1) and (
            keys[0] == "features"
        ), "Currently support solely traget/features pair"
        key = keys[0]
        value = convert_to_numpy(data[key])
        value = value[:first_zero_idx]
        assert target.shape[0] == value.shape[0], "Shape mismatch after cutting"
        out_dict["features"] = value

        return out_dict
