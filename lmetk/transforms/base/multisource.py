from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from numpy import ndarray

import logging

logger = logging.getLogger(__name__)


class ConcatinatedFitMultiSourceTransformMixin(ABC):
    """
    Stateless mixin for transforming data from multiple sources.
    """

    def __init__(self):
        """
        Initialize the MultiSourceTransformMixin.
        """
        super().__init__()
        # Check if the class is a subclass of TransformerMixin
        if not hasattr(self, "fit_data") or not hasattr(self, "transform_data"):
            raise TypeError(
                "MultiSourceTransformMixin requires the class to implement fit and transform methods."
            )

    def fit_multi_source(
        self,
        data_segments: list[ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Get all train sources and fit the scalers to them.
        """
        # Go over all data segments dict keys and concatenate the data from all sources
        train = {}
        for key, val in data_segments.items():
            train[key] = np.concatenate(val, axis=0)
        self.fit_data(train, metadata=metadata)

    def transform_multi_source(
        self,
        data_segments: list[ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list[ndarray]:
        """
        Transform the data from all sources using the fitted scalers.
        """
        # TODO: below the case when each of the source of multisource has its own metadata, make it work if needed
        # # Check that metadata dict is not empty
        # if len(metadata.keys()) > 0:
        #     for key in metadata.keys():
        #         # TODO: check whenever metadata should have the same length as data_segments?
        #         if len(metadata[key]) != len(data_segments):
        #             logger.info(f"The length of the auxiliary data is not the same as the length of the data segments. {len(metadata[key])} != {len(data_segments)}")

        #     # Transform the data from all sources using the fitted scalers.
        #     out = []
        #     for idx in range(len(data_segments)):
        #         # TODO: In case we need metadata per each segment
        #         # aux_data_per_segment = {
        #         #     key: metadata[key][idx] for key in metadata.keys()
        #         # }
        #         out.append(
        #             self.transform_data(
        #                 data_segment=data_segments[idx],
        #                 metadata=metadata,
        #             )
        #         )

        # else:
        # Check that data_segment values of the dict has the same lengths

        lengths = [len(val) for val in data_segments.values()]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"The lengths of the data segments are not the same. Lengths: {lengths}"
            )

        out = []
        for i in range(lengths[0]):
            data_segment = {}

            # Construct current data segment from each source
            for key, val in data_segments.items():
                data_segment[key] = val[i]

            # Transform the current data segment
            out.append(self.transform_data(data_segment, metadata=metadata))

        return out
