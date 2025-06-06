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
            if hasattr(source, "get_data_dict"):
                train_splits.append(source.get_data_dict()[feature_key])
            else:
                raise ValueError(f"Source {source_name} does not implement get_data method.")
            
        train = np.concatenate(train_splits, axis=0)
        self.fit(train)

    def transform_sources(self, sources_dict: Dict[str, TimeSeriesDatasetInterface], feature_key: str) -> None:
        """
        Transform the data from all sources using the fitted scalers.
        """

        for source_name, source in sources_dict.items():

            assert not source._is_split, \
                f"Source {source_name} is already split. This logic is currently not supported. "
            
            transformed_data_dict = {}
            if hasattr(source, "get_data_dict"):
                data = source.get_data_dict()[feature_key]
                transformed_data = self.transform(data)
                transformed_data_dict[feature_key] = transformed_data
            else:
                raise ValueError(f"Source {source_name} does not implement get_data method.")
            
            # Update the source's data_dict with the transformed data
            if hasattr(source, "update_data_dict"):
                source.update_data_dict(transformed_data_dict)
            else:
                raise ValueError(f"Source {source_name} does not implement update_data_dict method.")
