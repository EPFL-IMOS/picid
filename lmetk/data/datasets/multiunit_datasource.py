import pandas as pd
from typing import List, Optional, Any, Tuple

from abc import ABC

from lmetk.data.datasets.abstract_classes import AbstractTimeSeriesDataSource




class MultiUnitDataSource(AbstractTimeSeriesDataSource, ABC):
    """
    Combines multiple AbstractTimeSeriesDataSource instances into one unified datasource.
    """
    def __init__(
        self,
        sources: List[AbstractTimeSeriesDataSource],
        load_arguments: Any,
        data_splitter: Any,
        scaler: Any,
    ):
        # Store underlying sources
        self.sources = sources
        super().__init__(load_arguments, data_splitter, scaler)
        # Execute load, split, transform pipeline
        self._pipeline()

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Any]]:
        """
        Load and concatenate data and timestamps from all sources.
        Returns combined data, combined timestamps, and an optional mask.
        """
        data_frames = []
        timestamp_frames = []
        masks = []

        for src in self.sources:
            # Ensure each source has its data loaded but not yet split or scaled
            raw_data, raw_timestamps, mask = src._load_data()
            data_frames.append(raw_data)
            timestamp_frames.append(raw_timestamps)
            masks.append(mask)

        # Concatenate along the first axis, resetting indices
        combined_data = pd.concat(data_frames, ignore_index=True)
        combined_timestamps = pd.concat(timestamp_frames, ignore_index=True)

        # If masks are used, combine or discard
        combined_mask = None
        # e.g., combined_mask = pd.concat([m for m in masks if m is not None], ignore_index=True)

        return combined_data, combined_timestamps, combined_mask

    def __repr__(self) -> str:
        return (
            f"MultiUnitDataSource(sources={len(self.sources)} units, "
            f"load_arguments={self.load_arguments}, "
            f"data_splitter={self.data_splitter}, scaler={self.scaler})"
        )
