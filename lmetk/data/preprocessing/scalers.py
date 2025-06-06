from typing import Dict

import numpy as np
from lmetk.data.datasources.base import TimeSeriesDatasetInterface


class MultiSourceTransformMixin():
    """
    Stateless mixin for transforming data from multiple sources.
    """
    def __init__(self):
        """
        Initialize the MultiSourceTransformMixin.
        """

        # Check if the class is a subclass of TransformerMixin
        if not hasattr(self, 'fit') or not hasattr(self, 'transform'):
            raise TypeError("MultiSourceTransformMixin requires the class to implement fit and transform methods.")


    def fit_sources(self, sources_dict: Dict[str, TimeSeriesDatasetInterface], feature_key: str) -> None:
        """
        Get all train sources and fit the scalers to them.
        """
        train_splits = []
        for source_name, source in sources_dict.items():
            if hasattr(source, "get_data"):
                train_splits.append(source.get_data()[feature_key])
            else:
                raise ValueError(f"Source {source_name} does not implement get_data method.")
            
        train = np.concatenate(train_splits, axis=0)
        self.fit(train)

    def transform_sources(self, sources_dict: Dict[str, TimeSeriesDatasetInterface], feature_key: str) -> None:
        """
        Transform the data from all sources using the fitted scalers.
        """
        transformed_sources = {}
        for source_name, source in sources_dict.items():
            if hasattr(source, "get_data"):
                data = source.get_data()
                transformed_data = self.transform(data)
                transformed_sources[source_name] = transformed_data
            else:
                raise ValueError(f"Source {source_name} does not implement get_data method.")
