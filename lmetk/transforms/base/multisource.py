import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import copy
import numpy as np
from numpy import ndarray
import awkward as ak
import pandas as pd

from lmetk.data.data_objects.data import DatasetContainer, NamedTransformInput
from lmetk.transforms.base.base_transform import DenseTransform, RaggedTransform
from lmetk.transforms.utils import _convert_to_numpy
from lmetk.utils.awkward_utils import ak_find_var_dim

logger = logging.getLogger(__name__)


def find_singular_ragged_dim(arr: ak.Array):
    var_dims = [ak_find_var_dim(chunk) for chunk in arr]
    var_dims = set([x for row in var_dims for x in row])
    assert (
        len(var_dims) == 1
    ), f"Expected exactly one variable-length dimension, found {len(var_dims)}: {var_dims}"
    return var_dims.pop()


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
                "ConcatinatedFitMultiSourceTransformMixin requires the class "
                "to implement fit and transform methods."
            )

    def fit_function_to_data_segments(self, data_segments, fit_func, metadata):

        apply_to_keys = metadata["apply_to_keys"]
        # Check whenever keys list, if not make a list
        if not isinstance(apply_to_keys, list):
            apply_to_keys = [apply_to_keys]

        cls = self.__class__
        transform_supports_ragged = issubclass(cls, RaggedTransform)
        transform_supports_dense = issubclass(cls, DenseTransform)

        data_types = set([chunk.get_instance_cls() for chunk in data_segments])
        if len(data_types) != 1:
            raise ValueError(
                f"All data segments must have the same data type. Found: {data_types}"
            )
        data_is_ragged = data_types.pop() is ak.Array

        # Handle different cases based on data type and transform capabilities
        # --------------------------------------------------------------------

        if (not data_is_ragged) and transform_supports_dense:
            # Go over all data segments dict keys and concatenate the data from all sources
            train = {}
            for key in apply_to_keys:
                train[key] = np.concatenate(
                    [chunk[key] for chunk in data_segments], axis=0
                )

            fit_func(train, metadata=metadata)

        elif data_is_ragged and (not transform_supports_ragged):
            # Go over all data segments dict keys and concatenate the data from all sources
            train = {}
            var_dim = find_singular_ragged_dim(data_segments)

            for key in apply_to_keys:
                train[key] = ak.concatenate(
                    [chunk[key] for chunk in data_segments], axis=var_dim
                )

            train = {k: v.to_numpy() for k, v in train.items()}
            fit_func(train, metadata=metadata)

        elif data_is_ragged and transform_supports_ragged:
            # Go over all data segments dict keys and concatenate the data from all sources
            train = {}
            var_dim = find_singular_ragged_dim(data_segments)

            for key in apply_to_keys:
                train[key] = ak.concatenate(
                    [chunk[key] for chunk in data_segments], axis=var_dim
                )

            fit_func(train, metadata=metadata)

        else:
            raise ValueError(
                f"Transform {self.__class__.__name__} does not support the data type {data_types}"
            )

    def tranform_function_to_data_segments(
        self, data_segments, transform_func, metadata
    ):
        apply_to_keys = metadata["apply_to_keys"]
        # Check whenever keys list, if not make a list
        if not isinstance(apply_to_keys, list):
            apply_to_keys = [apply_to_keys]

        cls = self.__class__
        transform_supports_ragged = issubclass(cls, RaggedTransform)
        transform_supports_dense = issubclass(cls, DenseTransform)

        data_types = set([chunk.get_instance_cls() for chunk in data_segments])
        if len(data_types) != 1:
            raise ValueError(
                f"All data segments must have the same data type. Found: {data_types}"
            )
        data_is_ragged = data_types.pop() is ak.Array

        # Handle different cases based on data type and transform capabilities
        # --------------------------------------------------------------------
        # Construct current data segment from each source
        if (not data_is_ragged) and transform_supports_dense:
            out = []
            for data_segment in data_segments:
                out.append(transform_func(data_segment, metadata=metadata))

        elif data_is_ragged and (not transform_supports_ragged):
            # The logic is that we find the rugged dimension, ensure that there is only one,
            # flatten along that dimension, convert to numpy, apply the transform function,
            # and then reassemble the original structure.
            # Many loops, therefore I assume quite slow.

            out = []
            for data_segment in data_segments:
                var_dim = find_singular_ragged_dim(data_segment)

                ds = {k: ak.flatten(v, axis=var_dim) for k, v in data_segment.items()}

                # check if all first dimensions match in datasegment keys
                first_dims = set([ak.num(v, axis=0).item() for v in ds.values()])
                if len(first_dims) != 1:
                    raise ValueError(
                        f"All keys in the data segment must have the same first dimension. Found: {first_dims}"
                    )

                # we need to supply all subarray keys to the transform function
                # therefore we reasseble the dict

                n_subarrays = first_dims.pop()
                out_builder = {}
                for n in range(n_subarrays):
                    ds_sub = {k: v[n].to_numpy() for k, v in ds.items()}
                    ds_sub = transform_func(ds_sub, metadata=metadata)
                    for k in ds_sub.keys():
                        item = ds_sub[k]
                        if not isinstance(item, np.ndarray):
                            raise ValueError(
                                f"Transform function must return numpy arrays for key {k}. Found: {type(item)}"
                            )
                        if k not in out_builder:
                            out_builder[k] = []
                        out_builder[k].append(item)

                out.append({k: ak.Array(np.stack(v)) for k, v in out_builder.items()})

        elif data_is_ragged and transform_supports_ragged:
            out = []
            for data_segment in data_segments:
                out.append(transform_func(data_segment, metadata=metadata))

        else:
            raise ValueError(
                f"Transform {self.__class__.__name__} does not support the data type {data_types}"
            )

        return out

    def fit_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Get all train sources and fit the scalers to them.
        """
        self.fit_function_to_data_segments(
            data_segments, self.fit_data, copy.deepcopy(metadata)
        )

    def transform_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list[ndarray]:
        """
        Transform the data from all sources using the fitted scalers.
        """
        lengths = [len(chunk.keys()) for chunk in data_segments]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"The lengths of the data segments are not the same. Lengths: {lengths}"
            )

        out = self.tranform_function_to_data_segments(
            data_segments, self.transform_data, copy.deepcopy(metadata)
        )

        return out


class NoFitMultiSourceTransformMixin(ConcatinatedFitMultiSourceTransformMixin):
    """
    A stateless mixin for transforming data from multiple sources without fitting.
    This mixin is intended for use with transforms that do not require or support
    fitting on multiple data sources. Any attempt to fit the transform using
    `fit_multi_source` will raise a `NotImplementedError`.
    Inheritance:
        ConcatinatedFitMultiSourceTransformMixin
    Methods:
        fit_multi_source(data_segments, metadata=None):
            Raises NotImplementedError to indicate fitting is not supported.
    Attributes:
        None
    Usage:
        Use this mixin when implementing transforms that operate on multiple sources
        but do not require a fitting step.
    """

    def fit_multi_source(
        self,
        data_segments: list[ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError(
            "fit_multi_source is not implemented for this transform."
        )


class InverseTransformMixin(ABC):
    """
    Mixin for transforms that support inverse transformations.
    """

    def __init__(self):
        super().__init__()
        if not hasattr(self, "inverse_transform_data"):
            raise TypeError(
                "InverseTransformMixin requires the class to implement "
                "inverse_transform_data method."
            )

    def inverse_transform_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list[ndarray]:
        """
        Inverse transform the data from all sources using the fitted scalers.
        """
        out = []
        for data_segment in data_segments:
            out.append(
                self.inverse_transform_data(
                    data_segment, metadata=copy.deepcopy(metadata)
                )
            )
        return out

    @abstractmethod
    def inverse_transform(
        self,
        data: NamedTransformInput,
        metadata: dict,
    ) -> Any:
        raise NotImplementedError(
            "inverse_transform_data is not implemented for this transform."
        )
