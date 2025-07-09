import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from typing import Any, List, Union, Tuple, Type, Optional, Dict
from lmetk.transforms.utils import _convert_to_numpy, _convert_from_numpy
from numpy import ndarray
from lmetk.transforms.base_transform import BaseTransform 
from abc import ABC


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
        if not hasattr(self, 'fit_data') or not hasattr(self, 'transform_data'):
            raise TypeError("MultiSourceTransformMixin requires the class to implement fit and transform methods.")


    def fit_multi_source(self, data_segments: list[ndarray], 
                               mode: str,
                               auxiliary_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Get all train sources and fit the scalers to them.
        """
            
        train = np.concatenate(data_segments, axis=0)
        self.fit_data(train, mode=mode, auxiliary_data=auxiliary_data)

    def transform_multi_source(self, data_segments: list[ndarray], 
                               mode: str,
                               auxiliary_data: Optional[Dict[str, Any]] = None,
                               ) -> list[ndarray]:
        """
        Transform the data from all sources using the fitted scalers.
        """
        
        # Check that auxiliary_data dict is not empty
        if len(auxiliary_data.keys()) >0:
            for key in auxiliary_data.keys():
                len(auxiliary_data[key]) == len(data_segments), f"The length of the auxiliary data is not the same as the length of the data segments. {len(auxiliary_data[key])} != {len(data_segments)}"
        
            # Transform the data from all sources using the fitted scalers.
            out = []
            for idx in range(len(data_segments)):
                aux_data_per_segment  = {key: auxiliary_data[key][idx] for key in auxiliary_data.keys()}
                out.append(self.transform_data(data_segment=data_segments[idx], mode=mode, auxiliary_data=aux_data_per_segment))


        
        else:
            out = [self.transform_data(data_segment, mode=mode, auxiliary_data=auxiliary_data) for data_segment in data_segments] 
        
        return out
        

class ConstantScaler(BaseTransform, BaseMultiSourceTransformMixin):
    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor
    
    def fit_data(self, 
                data_segment: ndarray, 
                mode: str,
                auxiliary_data: Optional[Dict[str, Any]] = None):
        """This transform do not require fitting."""
        return self

    def transform_data(self, 
                       data_segment: ndarray, 
                       mode: str,
                       auxiliary_data: Optional[Dict[str, Any]] = None,
                       ):
        return data_segment * self.factor
    

class N_CMAPSSScaler_Descriptors(BaseTransform, BaseMultiSourceTransformMixin): #BaseEstimator, TransformerMixin, MultiSourceTransformMixin):

    def __init__(self, scaling="standard"):

        super().__init__()
        self.scaling = scaling


    def fit_data(self, 
                data_segment: ndarray, 
                mode: str,
                auxiliary_data: Optional[Dict[str, Any]] = None):
        """This transform do not require fitting."""
        return self

    def transform_data(self, 
                       data_segment: ndarray, 
                       mode: str,
                       auxiliary_data: Optional[Dict[str, Any]] = None,
                       ):
        scaling = self.scaling
        if scaling == "standard":
            W_mean_1457 = np.array(
                [1.6362398e04, 5.4544175e-01, 6.1369926e01, 4.8835443e02]
            )
            W_std_1457 = np.array(
                [8.1254497e03, 1.2108228e-01, 1.8272049e01, 1.9934254e01]
            )
            data_segment = (data_segment - W_mean_1457) / W_std_1457 
            #W = (W - W_mean_1457) / W_std_1457


        elif scaling == "min-max":
            # min-max scaling in [-1, 1] range
            W_min_1457 = np.array(
                [3.0020000e03, 2.0002499e-01, 2.3730299e01, 4.2319705e02]
            )
            W_max_1457 = np.array(
                [3.5011000e04, 7.3987198e-01, 8.7362656e01, 5.2488281e02]
            )
            data_segment = 2 * (data_segment - W_min_1457) / (W_max_1457 - W_min_1457) - 1 
            #W = 2 * (W - W_min_1457) / (W_max_1457 - W_min_1457) - 1
        return data_segment


class N_CMAPSSScaler_Features(BaseTransform, BaseMultiSourceTransformMixin): #BaseEstimator, TransformerMixin, MultiSourceTransformMixin):

    def __init__(self, scaling="standard"):
        self.scaling = scaling


    def fit_data(self, 
                data_segment: ndarray, 
                mode: str,
                auxiliary_data: Optional[Dict[str, Any]] = None):
        """This transform do not require fitting."""
        return self

    def transform_data(self, data_segment: ndarray, 
                       mode: str,
                       auxiliary_data: Optional[Dict[str, Any]] = None):
        scaling = self.scaling
        if scaling == "standard":
            # standard scaling
            X_mean_1457 = np.array(
                [
                    5.6809174e02,
                    1.3297987e03,
                    1.6363848e03,
                    1.1242614e03,
                    1.2669851e01,
                    9.8758097e00,
                    1.2862692e01,
                    1.5640701e01,
                    2.3289339e02,
                    2.3705125e02,
                    9.8489847e00,
                    1.9629447e03,
                    8.2361680e03,
                    2.4945383e00,
                ]
            )

            X_std_1457 = np.array(
                [
                    2.0833166e01,
                    6.7058266e01,
                    1.2113522e02,
                    6.1629440e01,
                    2.8704438e00,
                    2.4181337e00,
                    2.9141707e00,
                    3.4177959e00,
                    5.7780758e01,
                    5.8610283e01,
                    2.7480240e00,
                    1.8454973e02,
                    2.2242123e02,
                    7.6335633e-01,
                ]
            )
            data_segment = (data_segment - X_mean_1457) / X_std_1457 
            


        elif scaling == "min-max":
            # min-max scaling in [-1, 1] range
            X_min_1457 = np.array(
                [
                    4.9921100e02,
                    1.0889111e03,
                    1.2294357e03,
                    9.1258221e02,
                    6.0976486e00,
                    4.4431176e00,
                    6.1691909e00,
                    7.8722315e00,
                    9.1395287e01,
                    9.3320122e01,
                    4.5537462e00,
                    1.4843353e03,
                    7.4330903e03,
                    7.4892521e-01,
                ]
            )

            X_max_1457 = np.array(
                [
                    6.3150879e02,
                    1.5320127e03,
                    1.9813180e03,
                    1.3461113e03,
                    2.0096666e01,
                    1.4716009e01,
                    2.0402708e01,
                    2.5905502e01,
                    4.4721826e02,
                    4.5370425e02,
                    1.6701591e01,
                    2.2791799e03,
                    8.8905791e03,
                    5.6173835e00,
                ]
            )
            data_segment = 2 * (data_segment - X_min_1457) / (X_max_1457 - X_min_1457) - 1
            #X = 2 * (X - X_min_1457) / (X_max_1457 - X_min_1457) - 1
        return data_segment