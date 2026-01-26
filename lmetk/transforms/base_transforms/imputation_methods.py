import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.data.data_objects.utils import convert_to_numpy
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin

logger = logging.getLogger(__name__)


class ImputationTransform(DenseTransform, ConcatinatedFitMultiSourceTransformMixin):
    """
    Imputes missing values (NaNs) in time-series data using configurable strategies.

    This transform processes input arrays to fill in `NaN` values, ensuring data completeness
    for downstream models. It supports both statistical imputation (e.g., mean) and
    time-series specific methods (e.g., Last Observation Carried Forward, Linear Interpolation).

    The transform handles 3D arrays of shape ``(Batch, Time, Channels)`` or 2D arrays
    of shape ``(Time, Channels)``. Imputation is applied independently per channel and,
    for time-series methods, independently per batch to prevent data leakage.

    Attributes:
        strategy (str): The imputation strategy to use. Must be one of:
            - ``"mean"``: Replaces NaNs with the global mean of the channel, computed
              during the ``fit`` phase. If a channel is all-NaN during fit, it falls back to 0.0.
            - ``"locf"``: Last Observation Carried Forward. Propagates the last valid observation
              forward to next missing values. Leading NaNs are filled via backfill (NOCB).
            - ``"linear"``: Applies linear interpolation between valid points. Edges (start/end)
              that cannot be interpolated are filled using forward/backward fill, defaulting to 0.0.
            - ``"zero"``: Replaces all NaNs with 0.0.
        apply_to (Optional[List[str]]): A list of keys in the input dictionary
            (e.g., ``["features"]``) to which the imputation should be applied.
            If ``None``, the transform is applied to all keys in the input data.
        fitted_means (Dict[str, np.ndarray]): Stores the computed means for each key
            when ``strategy="mean"``. Populated during ``fit_data``.

    Examples:
        >>> import numpy as np
        >>> # Setup: Data with missing values
        >>> data = {"features": np.array([[1.0, np.nan], [2.0, 3.0]])}
        >>>
        >>> # Strategy 1: Zero Imputation
        >>> imputer = ImputationTransform(strategy="zero", apply_to=["features"])
        >>> out_zero = imputer.transform_data(data, metadata={})
        >>> print(out_zero["features"])
        [[1. 0.]
         [2. 3.]]
        >>>
        >>> # Strategy 2: Mean Imputation (requires fit)
        >>> imputer_mean = ImputationTransform(strategy="mean")
        >>> imputer_mean.fit_data(data, metadata={})  # Fits mean: Ch0=1.5, Ch1=3.0
        >>> out_mean = imputer_mean.transform_data(data, metadata={})
        >>> print(out_mean["features"])
        [[1. 3.]
         [2. 3.]]
    """

    VALID_STRATEGIES = {"mean", "locf", "linear", "zero"}

    def __init__(
        self,
        strategy: str = "linear",
        apply_to: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initializes the ImputationTransform with a specific strategy.

        Args:
            strategy: The name of the imputation strategy. Defaults to ``"linear"``.
            apply_to: List of dictionary keys to transform. If ``None``, all keys
                are transformed. Defaults to ``None``.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If ``strategy`` is not one of the valid options.
        """
        super().__init__(**kwargs)
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Valid: {self.VALID_STRATEGIES}"
            )

        self.strategy = strategy
        self.apply_to = apply_to

        # Store learned parameters for the 'mean' strategy.
        # Structure: {key_name: np.ndarray(shape=(n_channels,))}
        self.fitted_means: Dict[str, np.ndarray] = {}

    def fit_data(self, data: NamedTransformInput, metadata: Dict):
        """
        Fits the transform to the data.

        For ``strategy="mean"``, this calculates the global mean for each channel across
        all batches and time steps, ignoring NaNs. For other strategies, this is a no-op.

        Args:
            data: A dictionary containing input arrays.
            metadata: Metadata dictionary (unused).

        Returns:
            self: Returns the instance itself.

        Raises:
            ValueError: If input data has unsupported dimensions (not 2D or 3D).
        """
        if self.strategy != "mean":
            return self

        keys_to_process = self.apply_to if self.apply_to is not None else data.keys()

        for key, value in data.items():
            if key not in keys_to_process:
                continue

            # Ensure 3D arrays are preserved (Batch, Time, Channel)
            np_val = convert_to_numpy(value, ensure_2d=False)

            # Compute channel-wise means, collapsing Batch and Time dimensions.
            # We use warnings context manager to suppress RuntimeWarnings when a channel is all-NaN.
            if np_val.ndim == 2:
                # Shape (Time, Channels) -> Mean along Time (axis 0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    means = np.nanmean(np_val, axis=0)
            elif np_val.ndim == 3:
                # Shape (Batch, Time, Channels) -> Mean along Batch & Time (axes 0, 1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    means = np.nanmean(np_val, axis=(0, 1))
            else:
                raise ValueError(
                    f"Data for key '{key}' has invalid dimensions: {np_val.ndim}"
                )

            # Replace NaNs in the computed means (resulting from all-NaN columns) with 0.0
            means = np.nan_to_num(means, nan=0.0)
            self.fitted_means[key] = means

        return self

    def _impute_array(self, X: np.ndarray, key: str) -> np.ndarray:
        """
        Applies the configured imputation strategy to a single NumPy array.

        This method standardizes input to 3D, applies the strategy, and restores original shape.

        Args:
            X: Input array with missing values.
            key: The key name of the data (used to retrieve fitted means).

        Returns:
            np.ndarray: Imputed array with the same shape as input.

        Raises:
            ValueError: If input dimensions are not 2D or 3D.
        """
        # Standardize input to (Batch, Time, Channels)
        original_shape = X.shape
        if X.ndim == 2:
            # (Time, Channels) -> (1, Time, Channels)
            X_working = X[np.newaxis, :, :]
        elif X.ndim == 3:
            X_working = X
        else:
            raise ValueError("Data must be 2D or 3D")

        # Create a copy to avoid modifying the input in-place
        X_imp = X_working.copy()
        B, T, C = X_imp.shape

        if self.strategy == "zero":
            # Replace all NaNs with 0.0
            X_imp = np.nan_to_num(X_imp, nan=0.0)

        elif self.strategy == "mean":
            # Retrieve fitted means or fallback to 0.0
            if key not in self.fitted_means:
                logger.warning(
                    f"Key '{key}' was not seen during fit(). Using 0.0 fallback."
                )
                means = np.zeros(C)
            else:
                means = self.fitted_means[key]

            # Apply channel-specific means
            for c in range(C):
                mask = np.isnan(X_imp[:, :, c])
                X_imp[:, :, c][mask] = means[c]

        elif self.strategy == "locf":
            # Last Observation Carried Forward
            # Applied independently per batch using Pandas for efficiency
            for b in range(B):
                df_batch = pd.DataFrame(X_imp[b, :, :])
                # ffill propagates valid values forward.
                # bfill handles leading NaNs (Next Observation Carried Backward).
                df_filled = df_batch.ffill().bfill()
                X_imp[b, :, :] = df_filled.values

        elif self.strategy == "linear":
            # Linear Interpolation
            for b in range(B):
                df_batch = pd.DataFrame(X_imp[b, :, :])
                # 'both' limit direction ensures interpolation attempts to fill edges if possible.
                df_filled = df_batch.interpolate(
                    method="linear", limit_direction="both"
                )
                # Fallback: Edges or full-NaN columns might remain NaN.
                # Use ffill/bfill and finally fillna(0.0) as a safe fallback.
                df_filled = df_filled.ffill().bfill().fillna(0.0)
                X_imp[b, :, :] = df_filled.values

        # Restore original dimensionality (squeeze batch dim if input was 2D)
        if len(original_shape) == 2:
            return X_imp[0]
        return X_imp

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Executes the imputation on the provided data dictionary.

        Args:
            data: Dictionary of input arrays (e.g., ``{'features': ...}``).
            metadata: Metadata dictionary (unused).

        Returns:
            Dict[str, np.ndarray]: Dictionary with imputed arrays.
        """
        output_data = {}
        keys_to_process = self.apply_to if self.apply_to is not None else data.keys()

        for key, value in data.items():
            if key in keys_to_process:
                # Ensure 3D structure is preserved
                np_val = convert_to_numpy(value, ensure_2d=False)
                output_data[key] = self._impute_array(np_val, key)
            else:
                output_data[key] = value

        return output_data
