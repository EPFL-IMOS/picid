from typing import Any, Dict, Optional

import numpy as np
from numpy import ndarray

from lmetk.data.data_objects.data import OrderedInput
from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin
from lmetk.transforms.utils import _convert_from_numpy, _convert_to_numpy


class N_CMAPSSDescriptorsScaler(
    BaseTransform, ConcatinatedFitMultiSourceTransformMixin
):
    def __init__(self, scaling="standard"):

        super().__init__()
        self.scaling = scaling

    def fit_data(self, data: OrderedInput, metadata: Dict):
        """This transform do not require fitting."""
        return self

    def transform_data(self, data: OrderedInput, metadata: Dict) -> Any:

        descriptors = data.descriptors

        scaling = self.scaling
        if scaling == "standard":
            W_mean_1457 = np.array(
                [1.6362398e04, 5.4544175e-01, 6.1369926e01, 4.8835443e02]
            )
            W_std_1457 = np.array(
                [8.1254497e03, 1.2108228e-01, 1.8272049e01, 1.9934254e01]
            )
            descriptors = (descriptors - W_mean_1457) / W_std_1457
            # W = (W - W_mean_1457) / W_std_1457

        elif scaling == "min-max":
            # min-max scaling in [-1, 1] range
            W_min_1457 = np.array(
                [3.0020000e03, 2.0002499e-01, 2.3730299e01, 4.2319705e02]
            )
            W_max_1457 = np.array(
                [3.5011000e04, 7.3987198e-01, 8.7362656e01, 5.2488281e02]
            )
            descriptors = 2 * (descriptors - W_min_1457) / (W_max_1457 - W_min_1457) - 1
            # W = 2 * (W - W_min_1457) / (W_max_1457 - W_min_1457) - 1
        return descriptors


class N_CMAPSSFeaturesScaler(BaseTransform, ConcatinatedFitMultiSourceTransformMixin):
    def __init__(self, scaling="standard"):
        self.scaling = scaling

    def fit_data(self, data: OrderedInput, metadata: Dict):
        """This transform do not require fitting."""
        return self

    def transform_data(self, data: OrderedInput, metadata: Dict) -> Any:

        features = data.features

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
            features = (features - X_mean_1457) / X_std_1457

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
            features = 2 * (features - X_min_1457) / (X_max_1457 - X_min_1457) - 1
            # X = 2 * (X - X_min_1457) / (X_max_1457 - X_min_1457) - 1
        return features
