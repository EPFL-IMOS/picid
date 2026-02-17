import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, override
import copy
import numpy as np
from numpy import ndarray
import awkward as ak
from collections import defaultdict
from sortedcontainers import SortedDict

from lmetk.data.data_objects.base import BaseDataObjectWithMetadata
from lmetk.data.data_objects.data import DatasetContainer, NamedTransformInput
from lmetk.transforms.base.base_transform import (
    BaseTransform,
    DenseTransform,
    RaggedOrDenseTransform,
    RaggedTransform,
)
from lmetk.transforms.base.strategy import postprocess_transformed_data
from lmetk.transforms.base.data_kind import (
    CAPABILITY_BOTH,
    CAPABILITY_DENSE,
    CAPABILITY_RAGGED,
    DATA_KIND_RAGGED,
    DATA_KIND_RAGGED_REGULAR,
    get_capability,
    infer_data_kind,
)
from lmetk.transforms.base.handlers import get_handler
from lmetk.transforms.utils import _convert_to_numpy
from lmetk.utils.awkward_utils import (
    ak_find_var_dims,
    ak_regularize_regular_axes,
    ragged_index_tuples,
    get_ak_shape,
)
from tqdm import tqdm
from typing import Optional

from lmetk.utils.rich_output import describe_data_type

logger = logging.getLogger(__name__)


def find_singular_ragged_dim(nti: NamedTransformInput) -> Optional[int]:
    """
    Find the single variable-length (ragged) dimension in a NamedTransformInput.

    This function iterates through all Awkward arrays in the input and identifies
    their variable-length dimensions. It expects to find at most one unique
    variable-length dimension across all arrays.

    Parameters
    ----------
    nti : NamedTransformInput
        A dictionary-like object containing Awkward arrays.

    Returns
    -------
    Optional[int]
        The singular variable-length dimension if found, otherwise None if all
        arrays are regular.

    Raises
    ------
    ValueError
        If more than one unique variable-length dimension is found.
    """
    var_dims = [ak_find_var_dims(arr) for arr in nti.values()]

    if any([len(dim) == 0 for dim in var_dims]):
        idx = [len(dim) == 0 for dim in var_dims].index(True)
        # TODO: this warning polutes a lot the outputs
        # logger.warning(
        #     f"One of the values of the {type(nti)} object has no ragge dim, key: {list(nti.keys())[idx]}, val: {list(nti.values())[idx].type}"
        # )
    unique_var_dims = set([x for row in var_dims for x in row])

    # TODO checks to warn
    if len(unique_var_dims) == 0:
        # No ragged dimensions found; the data is regular in structure.
        return None
    elif len(unique_var_dims) == 1:
        # Exactly one ragged dimension found, as expected.
        return unique_var_dims.pop()
    else:
        # More than one ragged dimension makes the concatenation axis ambiguous.
        raise ValueError(
            f"Expected at most one variable-length dimension, found {len(unique_var_dims)}: {unique_var_dims}"
        )


# recursively turn dicts into lists
def tolist(d):
    """
    Recursively convert a nested dictionary to a nested list.

    The conversion is done by sorting the dictionary keys at each level
    to ensure a consistent order.

    Parameters
    ----------
    d : dict or any
        The dictionary to convert. If not a dictionary, the object is
        returned as is.

    Returns
    -------
    list or any
        The nested list representation of the dictionary, or the original
        object if it was not a dictionary.
    """
    if isinstance(d, (SortedDict)):
        return [tolist(v) for v in d.values()]

    elif isinstance(d, dict):
        return [tolist(d[i]) for i in sorted(d)]
    return d  # already a numpy array


class MultiSourceTransformInterface(ABC):

    @abstractmethod
    def fit_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def transform_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list[ndarray]:
        pass  # pragma: no cover


class ConcatinatedFitMultiSourceTransformMixin(MultiSourceTransformInterface):
    """
    A mixin for transforms that fit on data concatenated from multiple sources.

    This mixin provides a `fit_multi_source` method that standardizes the fitting
    process for transforms that operate on data from multiple units or sources.
    It handles different data formats (dense NumPy arrays and ragged Awkward arrays)
    and transform capabilities.

    The core logic is in `fit_function_to_data_segments`, which prepares the data
    by concatenating it from all sources before passing it to the user-defined
    `fit_data` method of the transform.

    The transformation logic is handled by `tranform_function_to_data_segments`,
    which applies the `transform_data` method to each data segment individually.

    Case-specific behaviors for fitting:
    1.  **Dense Data (NumPy) & Dense-supporting Transform**:
        - Data from all sources for each key is concatenated along `axis=0`.
        - The resulting single large NumPy array is passed to `fit_data`.

    2.  **Ragged Data (Awkward) & Dense-only Transform**:
        - The single variable-length dimension across all sources is identified.
        - Data is first concatenated along the variable dimension within each source,
          then across sources, effectively flattening the ragged structure.
        - The result is converted to a dense NumPy array and passed to `fit_data`.

    3.  **Ragged Data (Awkward) & Ragged-supporting Transform**:
        - The single variable-length dimension is identified.
        - Data from all sources is concatenated along this variable dimension.
        - The resulting single large Awkward array is passed to `fit_data`.

    Attributes
    ----------
    fit_data : callable
        A method that must be implemented by the consuming class to fit the transform.
    transform_data : callable
        A method that must be implemented by the consuming class to apply the transform.
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

    def fit_function_to_data_segments(
        self,
        data_segments: list[NamedTransformInput],
        fit_func: Callable,
        metadata: Dict[str, Any],
    ):
        """
        Prepare data segments and apply a fitting function.

        This method orchestrates the fitting process by handling different
        data types (dense vs. ragged) and transform capabilities. It
        concatenates data from multiple sources before passing it to the
        provided `fit_func`.

        Parameters
        ----------
        data_segments : list[NamedTransformInput]
            A list of data chunks from multiple sources.
        fit_func : Callable
            The fitting function (e.g., `self.fit_data`) to be called with the
            prepared data.
        metadata : Dict[str, Any]
            A dictionary containing metadata for the transformation, including
            `apply_to_keys`.

        Raises
        ------
        ValueError
            If data segments have inconsistent data types or if the transform
            does not support the given data type.
        """

        apply_to_keys = metadata["apply_to_keys"]
        if not isinstance(apply_to_keys, list):
            apply_to_keys = [apply_to_keys]

        data_types = {
            v for chunk in data_segments for k, v in chunk.get_instance_cls().items()
        }
        if len(data_types) != 1:
            raise ValueError(
                f"All data segments must have the same data type. Found: {data_types}"
            )

        # Phase 2: registry-based dispatch
        data_kind = infer_data_kind(data_segments, apply_to_keys, find_singular_ragged_dim)
        capability = get_capability(self)
        try:
            handler = get_handler(data_kind, capability)
        except KeyError:
            raise ValueError(
                f"Transform {self.__class__.__name__} does not support the data type {data_types}"
            )
        handler.fit_prepare(data_segments, apply_to_keys, metadata, fit_func)

    def transform_function_to_data_segments(
        self,
        data_segments: list[NamedTransformInput],
        transform_func: Callable,
        metadata: Dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        Apply a transformation function to a list of data segments.

        This method handles the logic for applying a `transform_func` to each
        data segment, considering whether the data is dense or ragged and
        whether the transform supports ragged arrays.

        For dense data, it applies the function to each segment.
        For ragged data with a dense-only transform, it iterates through
        each sub-segment of the ragged array, converts it to NumPy, transforms
        it, and reassembles the ragged structure. This can be slow.
        For ragged data with a ragged-supporting transform, it applies the
        function directly to each ragged segment.

        Parameters
        ----------
        data_segments : list[NamedTransformInput]
            A list of data chunks from multiple sources.
        transform_func : Callable
            The transformation function (e.g., `self.transform_data`) to apply.
        metadata : Dict[str, Any]
            A dictionary containing metadata for the transformation.

        Returns
        -------
        list
            A list of transformed data segments.
        """
        transform_log = {}
        apply_to_keys = metadata["apply_to_keys"]
        if not isinstance(apply_to_keys, list):
            apply_to_keys = [apply_to_keys]

        data_types = set(
            [v for chunk in data_segments for v in chunk.get_instance_cls().values()]
        )
        if len(data_types) != 1:
            raise ValueError(
                f"All data segments must have the same data type. Found: {data_types}"
            )

        # Phase 2: registry-based dispatch (for exception logging we keep these)
        data_kind = infer_data_kind(data_segments, apply_to_keys, find_singular_ragged_dim)
        capability = get_capability(self)
        data_is_ragged = data_kind == DATA_KIND_RAGGED
        data_is_ragged_but_regular = data_kind == DATA_KIND_RAGGED_REGULAR
        transform_supports_ragged = capability in (CAPABILITY_RAGGED, CAPABILITY_BOTH)
        transform_supports_dense = capability in (CAPABILITY_DENSE, CAPABILITY_BOTH)

        try:
            handler = get_handler(data_kind, capability)
            out, transform_log = handler.transform_apply(
                data_segments, transform_func, metadata, self.__class__.__name__
            )
        except KeyError:
            raise ValueError(
                f"Transform {self.__class__.__name__} does not support the data type {data_types}"
            )

        except Exception:
            t_name = self.__class__.__name__
            logger.error("-" * 160)
            logger.error(
                f"Transform Error Details ({t_name}.transform_function_to_data_segments())"
            )
            logger.error("-" * 160)

            logger.error(
                f"Data type details: "
                f"data_is_ragged={data_is_ragged}, "
                f"data_is_ragged_but_regular={data_is_ragged_but_regular}, "
                f"transform_supports_ragged={transform_supports_ragged}, "
                f"transform_supports_dense={transform_supports_dense}",
            )

            logger.error("-" * 160)
            logger.error(f"Metadata: {metadata}")
            _seg = data_segments[0] if data_segments else None
            logger.error(f"Data segment: {_seg}")
            if _seg is not None:
                logger.error(f"Data segment keys: {list(_seg.keys())}")
                logger.error("Data segment contents:")
                for k, v in _seg.items():
                    logger.error(
                        f" > Key: {k}, Type: {type(v)}, Description: {describe_data_type(v, calculate_stat=True)}"
                    )

            logger.error("-" * 160)
            logger.error("Case analysis:")

            if (not data_is_ragged) and transform_supports_dense:
                logger.error("Case: Dense Data & Dense-supporting Transform.")
                logger.error(" > the data is directly passed to the transform.")
            elif data_is_ragged and (not transform_supports_ragged):
                logger.error("Case: Ragged Data & Dense-only Transform")
                logger.error(
                    " > the data is processed in regular blocks, converted to numpy, transformed, and reassembled."
                )
                logger.error(
                    " > Error occurred in handler (index_depth/subsegment not available when using registry)."
                )
            elif data_is_ragged and transform_supports_ragged:
                logger.error("Case: Ragged Data & Ragged-supporting Transform")
                logger.error(" > the data is directly passed to the transform.")
            else:
                logger.error("Unknown case encountered during transformation.")

            logger.error("-" * 160)
            logger.error("Error Details: ", exc_info=True)
            logger.error("-" * 160)

            raise

        transform_log["flags"] = {
            "data_is_ragged": data_is_ragged,
            "data_is_ragged_but_regular": data_is_ragged_but_regular,
            "transform_supports_ragged": transform_supports_ragged,
            "transform_supports_dense": transform_supports_dense,
        }

        return out, transform_log

    def fit_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Fit the transform on data concatenated from multiple sources.

        This method takes a list of data segments, prepares them by
        concatenating them according to the logic in
        `fit_function_to_data_segments`, and then calls the transform's
        `fit_data` method.

        Parameters
        ----------
        data_segments : list[NamedTransformInput]
            A list of data chunks, where each chunk comes from a different source.
        metadata : Optional[Dict[str, Any]], optional
            Metadata to guide the fitting process, by default None.
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
        Transform data from multiple sources individually.

        This method applies the fitted transformation to a list of data segments,
        processing each segment from each source separately.

        Parameters
        ----------
        data_segments : list[NamedTransformInput]
            A list of data chunks to be transformed.
        metadata : Optional[Dict[str, Any]], optional
            Metadata to guide the transformation process, by default None.

        Returns
        -------
        list[ndarray]
            A list containing the transformed data for each source.

        Transform the data from all sources using the fitted scalers.
        """
        lengths = [len(chunk.keys()) for chunk in data_segments]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"The lengths of the data segments are not the same. Lengths: {lengths}"
            )

        out, transform_log = self.transform_function_to_data_segments(
            data_segments, self.transform_data, copy.deepcopy(metadata)
        )

        return out, transform_log


class ConcatenateUnitsMixing(MultiSourceTransformInterface):
    """"""

    def __init__(self, axis: int = None):
        """
        Initialize the MultiSourceTransformMixin.
        """
        super().__init__()
        # Check if the class is a subclass of TransformerMixin
        self.axis = axis

        # assert axis is not None, "axis must be specified"
        assert self.axis is not None, (
            "axis must be specified for the transforms which inherit from the ConcatenateUnitsMixing."
            "Please choose along which axis you want to concatenate the data bafore it is passed to the transform."
        )

    def fit_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """ """
        raise NotImplementedError(
            f"fit_multi_source is not implemented for {self.__class__.__name__}"
        )
        # self.fit_data(data_segments, metadata=copy.deepcopy(metadata))

    def transform_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        go from list of n -> list of 1
        """

        lengths = [len(chunk.keys()) for chunk in data_segments]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"The lengths of the data segments are not the same. Lengths: {lengths}"
            )

        apply_to_keys = metadata["apply_to_keys"]

        # Get the types of the named transform input
        data_types = defaultdict(list)
        for data in data_segments:
            classes = data.get_instance_cls()
            for k, v in classes.items():
                data_types[k].append(v)

        # Check that for each key the data types are the same
        for k, v in data_types.items():
            if len(set(v)) != 1:
                raise ValueError(
                    f"The data types for key '{k}' are not the same across all data segments. Found: {v}"
                )
            data_types[k] = v[0]

        assert set(data_types.keys()) == set(
            apply_to_keys
        ), "The keys in the data segments do not match the keys to apply the transform to."

        # Concatenation logic in case the incomming data is ragged
        out = {}
        for key in apply_to_keys:
            collect = []

            if data_types[key] is ak.Array:
                for data_segment in data_segments:
                    assert (
                        key in data_segment
                    ), f"Key '{key}' not found in data segment."

                    assert isinstance(
                        data_segment[key], ak.Array
                    ), f"Key '{key}' is not an awkward array."
                    collect.append(data_segment[key])

                # Concatenate along the specified axis
                out[key] = ak.concatenate(collect, axis=self.axis)

            elif data_types[key] is np.ndarray:
                for data_segment in data_segments:
                    assert (
                        key in data_segment
                    ), f"Key '{key}' not found in data segment."

                    collect.append(data_segment[key])

                out[key] = np.concatenate(collect, axis=self.axis)

            else:
                raise ValueError(
                    f"Data type {data_types[key]} for key '{key}' is not supported."
                )

        if len(apply_to_keys) > 1:
            lengths = []
            for key in apply_to_keys:
                if data_types[key] is ak.Array:
                    # Check the final arrays have the same number of rows
                    lengths.append(get_ak_shape(out[key])[self.axis])

                elif data_types[key] is np.ndarray:
                    # Check the final arrays have the same number of rows
                    lengths.append(out[key].shape[self.axis])
                else:
                    raise ValueError(
                        f"Data type {data_types[key]} for key '{key}' is not supported."
                    )

            if len(set(lengths)) != 1:
                raise ValueError(
                    f"The lengths of the concatenated arrays are not the same. Lengths: {lengths}"
                )

        out = NamedTransformInput(**out)
        out = [self.transform_data(out, metadata=copy.deepcopy(metadata))]

        return out, {}


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
        metadata: dict = None,
    ) -> Any:
        raise NotImplementedError(
            "inverse_transform_data is not implemented for this transform."
        )
