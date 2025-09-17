import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, override
import copy
import numpy as np
from numpy import ndarray
import awkward as ak
import pandas as pd

from lmetk.data.data_objects.data import DatasetContainer, NamedTransformInput
from lmetk.transforms.base.base_transform import (
    BaseTransform,
    DenseTransform,
    RaggedOrDenseTransform,
    RaggedTransform,
)
from lmetk.transforms.base.strategy import postprocess_transformed_data
from lmetk.transforms.utils import _convert_to_numpy
from lmetk.utils.awkward_utils import (
    ak_find_var_dims,
    ak_regularize_regular_axes,
    ragged_index_tuples,
)
from tqdm import tqdm
from typing import Optional

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
        logger.warning(
            f"One of the values of the {type(nti)} object has no ragge dim, key: {list(nti.keys())[idx]}, val: {list(nti.values())[idx].type}"
        )
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
    if isinstance(d, dict):
        return [tolist(d[i]) for i in sorted(d)]
    return d  # already a numpy array


class MultiSourceTransformInterface(ABC):

    @abstractmethod
    def fit_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @abstractmethod
    def transform_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list[ndarray]:
        pass


class UnflattenUnitsTransform(MultiSourceTransformInterface, BaseTransform):

    @override
    def fit_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @override
    def transform_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list[ndarray]:
        pass

    @override
    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> Any:
        pass


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
        # Check whenever keys list, if not make a list
        if not isinstance(apply_to_keys, list):
            apply_to_keys = [apply_to_keys]

        # Check if the incomming transform supports ragged arrays or numpy arrays
        cls = self.__class__
        transform_supports_ragged = issubclass(cls, RaggedTransform)
        transform_supports_dense = issubclass(cls, DenseTransform)

        data_types = {
            v for chunk in data_segments for k, v in chunk.get_instance_cls().items()
        }
        if len(data_types) != 1:
            raise ValueError(
                f"All data segments must have the same data type. Found: {data_types}"
            )

        # Currently we assume that all the element under every key in the data chunk have the same data type. Note that is a strict rule.
        data_is_ragged = data_types.pop() is ak.Array

        if data_is_ragged:
            # Determine if the data is ragged but structurally regular (i.e., all dimensions are fixed).
            # This occurs if `find_singular_ragged_dim` returns `None` for all segments.
            ragged_dims = {find_singular_ragged_dim(ds) for ds in data_segments}
            # The data is regular if the only "ragged dimension" found is None.
            data_is_ragged_but_regular = ragged_dims == {None}
            if data_is_ragged_but_regular:
                data_is_ragged = False

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

            # Extract variable dimention
            var_dim = set(
                [
                    find_singular_ragged_dim(data_segment)
                    for data_segment in data_segments
                ]
            )
            assert (
                len(var_dim) == 1
            ), f"Expected exactly one variable-length dimension, found {var_dim}"
            var_dim = var_dim.pop()

            for key in apply_to_keys:
                train[key] = ak.concatenate(
                    [
                        ak.concatenate(chunk[key], axis=var_dim - 1)
                        for chunk in data_segments
                    ],
                    axis=0,
                )

            train = {k: v.to_numpy() for k, v in train.items()}
            fit_func(train, metadata=metadata)

        elif data_is_ragged and transform_supports_ragged:
            # Go over all data segments dict keys and concatenate the data from all sources
            train = {}

            # Extract variable dimention
            var_dim = set(
                [
                    find_singular_ragged_dim(data_segment)
                    for data_segment in data_segments
                ]
            )
            assert (
                len(var_dim) == 1
            ), f"Expected exactly one variable-length dimension, found {var_dim}"
            var_dim = var_dim.pop()

            for key in apply_to_keys:
                train[key] = ak.concatenate(
                    [chunk[key] for chunk in data_segments], axis=var_dim - 1
                )

            fit_func(train, metadata=metadata)

        else:
            raise ValueError(
                f"Transform {self.__class__.__name__} does not support the data type {data_types}"
            )

    def tranform_function_to_data_segments(
        self,
        data_segments: list[NamedTransformInput],
        transform_func: Callable,
        metadata: Dict[str, Any],
    ):
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

        apply_to_keys = metadata["apply_to_keys"]
        # Check whenever keys list, if not make a list
        if not isinstance(apply_to_keys, list):
            apply_to_keys = [apply_to_keys]

        cls = self.__class__
        transform_supports_ragged = issubclass(cls, RaggedTransform) or issubclass(
            cls, RaggedOrDenseTransform
        )
        transform_supports_dense = issubclass(cls, DenseTransform) or issubclass(
            cls, RaggedOrDenseTransform
        )

        data_types = set(
            [v for chunk in data_segments for v in chunk.get_instance_cls().values()]
        )
        if len(data_types) != 1:
            raise ValueError(
                f"All data segments must have the same data type. Found: {data_types}"
            )
        data_is_ragged = data_types.pop() is ak.Array

        # Determine if the data is ragged but structurally regular (i.e., all dimensions are fixed).
        # This occurs if `find_singular_ragged_dim` returns `None` for all segments.
        if data_is_ragged:
            ragged_dims = {find_singular_ragged_dim(ds) for ds in data_segments}
            # The data is regular if the only "ragged dimension" found is None.
            data_is_ragged_but_regular = ragged_dims == {None}
            data_is_ragged = not data_is_ragged_but_regular
        else:
            data_is_ragged_but_regular = False

        # Handle different cases based on data type and transform capabilities
        # --------------------------------------------------------------------
        # Construct current data segment from each source
        if (not data_is_ragged) and transform_supports_dense:
            out = []
            for data_segment in data_segments:
                # Handle the case when the data is ragged but structurally regular
                if data_is_ragged_but_regular:
                    for key, val in data_segment.items():
                        data_segment[key] = val.to_numpy()

                out_segment = transform_func(data_segment, metadata=metadata)

                if data_is_ragged_but_regular:
                    out_segment = ak.Array(out_segment)

                out.append(out_segment)

        elif data_is_ragged and (not transform_supports_ragged):
            # The logic is that we find the rugged dimension, ensure that there is only one,
            # flatten along that dimension, convert to numpy, apply the transform function,
            # and then reassemble the original structure.
            # Many loops, therefore I assume quite slow.
            logger.warning(
                f"Transform {self.__class__.__name__} does not support ragged arrays natively. "
                "This might be slow for large datasets."
            )
            out = []

            for data_segment in data_segments:
                # TODO the logic here still do not support the case when one feature is ragged and another one is array
                # in the apply to keys ['features','target'] and 'target' is an array then there is not check or logic that takes it into consideration for now.

                # Ensure that all the keys have same dim 0
                assert len(set([len(val) for val in data_segment.values()])) == 1, (
                    f"All keys in data_segment must have the same length at the first dimension. "
                    f"Found lengths: {[len(val) for val in data_segment.values()]}"
                )

                var_dim = find_singular_ragged_dim(data_segment)
                multi_index = ragged_index_tuples(
                    next(iter(data_segment.values())), var_dim
                )

                # Merge metadata from the segment with the overall metadata
                data_segment_metadata = copy.deepcopy(metadata)
                data_segment_metadata.update(data_segment.metadata or {})

                out_segment = {}
                use_tqdm = len(multi_index) > 100
                iterator = multi_index
                if use_tqdm:
                    iterator = tqdm(
                        multi_index,
                        desc=f"{self.__class__.__name__}: transforming ragged segments",
                    )
                for coord in iterator:
                    subsegment = {}
                    for k in data_segment.keys():
                        np_data = data_segment[k][coord].to_numpy()
                        subsegment[k] = np_data
                    out_subsegment = [transform_func(subsegment, metadata=metadata)]

                    out_subsegment = postprocess_transformed_data(
                        out_subsegment, metadata
                    )
                    assert (
                        len(out_subsegment) == 1
                    ), "Expected single output from transform."

                    # Assemble all the output in a 2 level dictionary
                    # first level is the array key,
                    # second level is the coord in the ragged dim

                    for k, block in out_subsegment[0].items():
                        d: dict = out_segment.setdefault(k, {})
                        for i in coord[:-1]:
                            d: dict = d.setdefault(i, {})
                        d[coord[-1]] = block  # keep as np.ndarray

                for k, v in out_segment.items():
                    aka = ak.Array(tolist(v))
                    out_segment[k] = ak_regularize_regular_axes(aka)

                out.append(out_segment)

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
        metadata: dict = None,
    ) -> Any:
        raise NotImplementedError(
            "inverse_transform_data is not implemented for this transform."
        )
