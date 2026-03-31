import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import awkward as ak

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform, RaggedTransform
from picid.transforms.base.multisource import (
    NoFitPerSegmentMixin,
    NoFitConcatAlongAxisMixin,
)
from picid.transforms.base.pipeline.unit_metadata import aggregate_unit_metadata
from picid.utils.awkward_utils import ak_find_var_dims

logger = logging.getLogger(__name__)


class ConcatenateTransform(NoFitPerSegmentMixin, DenseTransform):
    def __init__(self, dim: int = 1, **kwargs):
        """
        Initialize the transform.

        Parameters
        ----------
        dim : int, optional
            Axis along which to concatenate arrays.
        **kwargs
            Additional keyword arguments.
        """
        self.dim = dim
        assert dim in (
            1,
            2,
        ), "This transform currently supports concatination along dim 1 or 2 only."
        super().__init__(**kwargs)

    def _convert_to_numpy(self, value: Any, key: str) -> np.ndarray:
        """
        Convert a value to :class:`numpy.ndarray` if it is not already one.

        Parameters
        ----------
        value : Any
            Input value to convert.
        key : str
            Name of the source key, used only for error reporting.

        Returns
        -------
        numpy.ndarray
            Converted array.
        """
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value)
                logger.info(f"Converted key '{key}' to numpy array")
            except Exception as e:
                raise ValueError(f"Could not convert key '{key}' to numpy array: {e}")
        return value

    def _ensure_2d(self, value: np.ndarray, key: str) -> np.ndarray:
        """
        Ensure an array is 2D for concatenation.

        Parameters
        ----------
        value : numpy.ndarray
            Array to reshape if required.
        key : str
            Name of the source key, used only for error reporting.

        Returns
        -------
        numpy.ndarray
            A 2D array suitable for concatenation.
        """
        if value.ndim == 1:
            value = value.reshape(-1, 1)
            logger.info(f"Reshaped 1D array '{key}' to 2D: {value.shape}")
        elif value.ndim > 2:
            raise ValueError(
                f"Key '{key}' has unsupported dimensions: {value.ndim}. "
                f"Only 1D and 2D arrays are supported."
            )
        return value

    def _ensure_3d(self, value: np.ndarray, key: str) -> np.ndarray:
        """
        Ensure an array is 3D for concatenation.

        Parameters
        ----------
        value : numpy.ndarray
            Array to reshape if required.
        key : str
            Name of the source key, used only for error reporting.

        Returns
        -------
        numpy.ndarray
            A 3D array suitable for concatenation.
        """
        if value.ndim == 1:
            value = value.reshape(-1, 1, 1)
            logger.info(f"Reshaped 1D array '{key}' to 3D: {value.shape}")
        elif value.ndim == 2:
            value = value.reshape(value.shape[0], value.shape[1], 1)
            logger.info(f"Reshaped 2D array '{key}' to 3D: {value.shape}")
        elif value.ndim > 3:
            raise ValueError(
                f"Key '{key}' has unsupported dimensions: {value.ndim}. "
                f"Only 1D, 2D, and 3D arrays are supported."
            )
        return value

    def _check_shape_consistency(
        self, arrays: List[np.ndarray], keys: List[str], concat_dim: int
    ) -> Tuple[int, Tuple[int, ...]]:
        """
        Check that all arrays match in every non-concatenated dimension.

        Parameters
        ----------
        arrays : list[numpy.ndarray]
            Arrays to validate.
        keys : list[str]
            Source keys corresponding to ``arrays``.
        concat_dim : int
            Axis that is allowed to differ.

        Returns
        -------
        tuple[int, tuple[int, ...]]
            The concatenation length along ``concat_dim`` and the reference shape.
        """
        if not arrays:
            raise ValueError("No arrays to concatenate")

        ref_shape = arrays[0].shape
        for key, arr in zip(keys, arrays):
            if len(arr.shape) != len(ref_shape):
                raise ValueError(
                    f"Array '{key}' has {len(arr.shape)} dims, expected {len(ref_shape)}"
                )
            for dim, (s1, s2) in enumerate(zip(ref_shape, arr.shape)):
                if dim != concat_dim and s1 != s2:
                    raise ValueError(
                        f"Shape mismatch in dimension {dim} (excluding concat_dim={concat_dim}): "
                        f"{keys[0]}={ref_shape}, {key}={arr.shape}"
                    )

        return ref_shape[concat_dim], ref_shape

    def _check_for_nans(self, arrays: List[np.ndarray], keys: List[str]) -> None:
        """
        Check for NaN values and raise an error if any are found.

        Parameters
        ----------
        arrays : list[numpy.ndarray]
            Arrays to inspect.
        keys : list[str]
            Source keys corresponding to ``arrays``.
        """
        for arr, key in zip(arrays, keys):
            if np.any(np.isnan(arr)):
                raise ValueError(f"NaN values found in array '{key}'")

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Concatenate all input arrays along the configured axis.

        Parameters
        ----------
        data : dict
            Mapping of keys to arrays to concatenate.
        metadata : dict
            Auxiliary metadata passed through by the pipeline.

        Returns
        -------
        np.ndarray
            Concatenated numpy array.
        """
        if not data:
            raise ValueError("No data provided for concatenation")

        # Collect arrays to concatenate
        arrays = []
        keys = []

        max_dim = max([np.array(v).ndim for v in data.values()])
        assert max_dim <= 3, "This transform currently supports up to 3D arrays only."

        # plot the dimensions of each array
        logger.info(
            f"Concatenation input: {[ (k, np.array(v).shape) for k,v in data.items() ]}"
        )
        for key, value in data.items():
            # Convert to numpy array
            value = self._convert_to_numpy(value, key)

            if max_dim < 3:
                value = self._ensure_2d(value, key)
            else:
                value = self._ensure_3d(value, key)

            arrays.append(value)
            keys.append(key)
            logger.info(f"Added data '{key}' with shape {value.shape}")

        if not arrays:
            raise ValueError("No arrays to concatenate")

        # Check length consistency
        n_rows, ref_shape = self._check_shape_consistency(arrays, keys, self.dim)
        logger.info(f"All arrays have consistent shapes: {n_rows} {ref_shape}")

        # Check for NaN values (will raise error if found)
        self._check_for_nans(arrays, keys)

        # Final concatenation
        logger.info(
            f"Concatenating arrays with shapes: {[arr.shape for arr in arrays]}"
        )
        result = np.concatenate(arrays, axis=self.dim)
        logger.info(f"Final concatenated shape: {result.shape}")

        return result

    def __call__(self, data: Dict, metadata: Dict) -> np.ndarray:
        return self.transform_data(data, metadata)


class RuggedToDenseTransform(NoFitPerSegmentMixin, RaggedTransform):
    def __init__(self, **kwargs):
        """
        Initialize the transform.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Concatenate ragged arrays along their single ragged dimension.

        Parameters
        ----------
        data : dict
            Mapping of keys to ragged arrays.
        metadata : dict
            Auxiliary metadata passed through by the pipeline.

        Returns
        -------
        numpy.ndarray
            Dense arrays created from the ragged inputs.
        """
        for key, value in data.items():
            assert isinstance(value, ak.Array), f"Key '{key}' is not an awkward array."
            dims = ak_find_var_dims(value)
            if len(dims) != 1:
                raise ValueError(
                    f"Key '{key}' has {len(dims)} ragged dimensions, expected exactly one."
                )
            else:
                var_dim = dims[0]
                data[key] = ak.concatenate(value, axis=var_dim - 1).to_numpy()

        return data


class MultiDatasetRuggedToDenseTransform(NoFitConcatAlongAxisMixin, RaggedTransform):
    def __init__(self, axis: int, **kwargs):
        """
        Initialize the transform.

        Parameters
        ----------
        axis : int
            The axis along which to concatenate ragged arrays.
        **kwargs
            Additional keyword arguments.
        """
        RaggedTransform.__init__(self)
        NoFitConcatAlongAxisMixin.__init__(self, axis=axis, **kwargs)

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> np.ndarray:
        """
        Convert the selected ragged arrays to NumPy arrays.

        Parameters
        ----------
        data : dict
            Mapping of keys to ragged arrays.
        metadata : dict
            Pipeline metadata containing ``apply_to_keys``.

        Returns
        -------
        dict
            Mapping of the selected keys to NumPy arrays.
        """

        apply_to_keys = metadata["apply_to_keys"]
        out = {}
        for key in apply_to_keys:
            # convert to numpy
            out[key] = ak.to_numpy(data[key])

        return out

    def propagate_unit_metadata(
        self,
        *,
        unit_metadata_by_split: Dict[str, List[Dict[str, Any]]],
        transformed_results_for_new_key: Dict[str, Dict[str, List[Any]]],
        metadata: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        # This transform intentionally collapses many units into a single dense
        # block per split. We therefore aggregate the original unit metadata
        # into one summary record so the pipeline's 1:1 alignment check stays
        # truthful after the collapse.
        #
        # In practice this is the behavior needed by configurations such as the
        # Airbus statistics pipeline: multiple per-unit ragged arrays are
        # concatenated into one split-level dense matrix, so preserving the
        # original per-unit metadata would leave the container lying about how
        # many output units now exist.
        return aggregate_unit_metadata(
            unit_metadata_by_split=unit_metadata_by_split,
            transformed_results_for_new_key=transformed_results_for_new_key,
            metadata=metadata,
        )
