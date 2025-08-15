from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from numpy import ndarray


class BaseMultiSourceTransformMixin(ABC):
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
        mode: str,
        auxiliary_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Get all train sources and fit the scalers to them.
        """

        train = np.concatenate(data_segments, axis=0)
        self.fit_data(train, mode=mode, auxiliary_data=auxiliary_data)

    def transform_multi_source(
        self,
        data_segments: list[ndarray],
        mode: str,
        auxiliary_data: Optional[Dict[str, Any]] = None,
    ) -> list[ndarray]:
        """
        Transform the data from all sources using the fitted scalers.
        """

        # Check that auxiliary_data dict is not empty
        if len(auxiliary_data.keys()) > 0:
            for key in auxiliary_data.keys():
                len(auxiliary_data[key]) == len(
                    data_segments
                ), f"The length of the auxiliary data is not the same as the length of the data segments. {len(auxiliary_data[key])} != {len(data_segments)}"

            # Transform the data from all sources using the fitted scalers.
            out = []
            for idx in range(len(data_segments)):
                aux_data_per_segment = {
                    key: auxiliary_data[key][idx] for key in auxiliary_data.keys()
                }
                out.append(
                    self.transform_data(
                        data_segment=data_segments[idx],
                        mode=mode,
                        auxiliary_data=aux_data_per_segment,
                    )
                )

        else:
            out = [
                self.transform_data(
                    data_segment, mode=mode, auxiliary_data=auxiliary_data
                )
                for data_segment in data_segments
            ]

        return out
        return out
