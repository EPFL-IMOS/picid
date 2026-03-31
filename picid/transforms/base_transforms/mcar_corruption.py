import logging
from typing import Dict, List, Optional, Union

import numpy as np

from picid.data.data_objects import NamedTransformInput
from picid.data.data_objects.utils import convert_to_numpy
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin

logger = logging.getLogger(__name__)


class MCARCorruptorTransform(NoFitPerSegmentMixin, DenseTransform):
    """
    Inject Missing Completely At Random (MCAR) faults into time-series data.

    This transform simulates data quality issues—such as sensor failures, packet loss,
    or transmission outages—by artificially introducing ``NaN`` values into the input
    arrays. It supports two distinct mechanisms for generating missingness: independent
    point-wise drops ("jitter") and contiguous block outages ("outages").

    The transform operates on 3D arrays of shape ``(Batch, Time, Channels)`` or 2D
    arrays of shape ``(Time, Channels)``. It handles channel-specific corruption
    ratios, allowing for heterogeneous sensor reliability simulations.

    Parameters
    ----------
    ratios : list[float]
        Target missingness ratio for each channel.
    mode : str, default="block"
        Corruption mode, either ``"point"`` or ``"block"``.
    block_params : dict[str, int | float], optional
        Block-size configuration used when ``mode="block"``.
    seed : int, default=42
        Seed for the NumPy random number generator.
    apply_to : list[str], optional
        Keys in the input dictionary to corrupt. When omitted, all keys are
        processed.
    **kwargs : Any
        Additional keyword arguments forwarded to the parent class.

    Attributes
    ----------
    ratios : List[float]
        A list of target missingness ratios (between 0.0 and 1.0) for each
        channel. If the input data has more channels than elements in this
        list, the remaining channels are assumed to have a ratio of 0.0
        (i.e., they remain pristine).
    mode : str
        The mechanism used to generate missing values.
        - ``"point"``: Simulates independent random failures (e.g., packet loss).
          Each point is dropped independently with probability proportional to
          the ratio.
        - ``"block"``: Simulates sensor outages or reboots. Missingness occurs
          in contiguous chunks of time.
    block_params : Optional[Dict[str, Union[int, float]]]
        Configuration for block sizes when ``mode="block"``. Must contain keys:
        - ``"min_size"`` (int | float): Minimum duration of an outage block.
          If float (0.0 to 1.0), it represents a proportion of the total
          signal length. If int, it represents an absolute number of time steps.
        - ``"max_size"`` (int | float): Maximum duration of an outage block.
          If float, it represents a proportion. If int, it represents absolute
          steps. Defaults to ``{"min_size": 5, "max_size": 20}`` if not provided.
    seed : int
        Seed for the NumPy random number generator to ensure deterministic
        reproducibility of the corruption patterns.
    apply_to : Optional[List[str]]
        A list of keys in the input dictionary (e.g., ``["features"]``) to
        which the corruption should be applied. If ``None``, the transform is
        applied to all keys in the input data.

    Examples
    --------
        >>> import numpy as np
        >>> # Setup: 1 Batch, 100 Time steps, 2 Channels
        >>> data = {"features": np.ones((1, 100, 2))}
        >>>
        >>> # Scenario 1: Corrupt Channel 0 by 50% using random point drops
        >>> corruptor_point = MCARCorruptorTransform(
        ...     ratios=[0.5, 0.0],
        ...     mode="point",
        ...     seed=42,
        ...     apply_to=["features"]
        ... )
        >>> # Scenario 2: Corrupt Channel 0 by 30% using proportional block drops
        >>> # Blocks will be between 5% and 10% of the signal length
        >>> corruptor_block = MCARCorruptorTransform(
        ...     ratios=[0.3, 0.0],
        ...     mode="block",
        ...     block_params={"min_size": 0.05, "max_size": 0.1},
        ...     seed=42
        ... )
    """

    def __init__(
        self,
        ratios: List[float],
        mode: str = "block",
        block_params: Optional[Dict[str, Union[int, float]]] = None,
        seed: int = 42,
        apply_to: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the MCARCorruptorTransform with corruption parameters.

        Parameters
        ----------
        ratios : list[float]
            A list of floats defining the target missingness ratio for each
            channel (e.g., ``[0.1, 0.5]``).
        mode : str
            The corruption mode, either ``"point"`` or ``"block"``.
            Defaults to ``"block"``.
        block_params : dict[str, int | float], optional
            A dictionary defining ``"min_size"`` and ``"max_size"``
            for block outages. Values can be integers (absolute length) or
            floats (proportion of signal length). Required only if ``mode="block"``.
            Defaults to ``None``.
        seed : int
            Random seed for reproducibility. Defaults to 42.
        apply_to : list[str], optional
            List of dictionary keys to transform. If ``None``, all keys
            are transformed. Defaults to ``None``.
        **kwargs : Any
            Additional keyword arguments passed to the parent class.

        Raises
        ------
        ValueError
            If ``mode`` is not one of ``"point"`` or ``"block"``.
        ValueError
            If ``mode="block"`` and ``block_params`` is invalid or
            malformed.
        """
        super().__init__(**kwargs)
        self.ratios = ratios
        self.mode = mode
        self.block_params = (
            block_params
            if block_params is not None
            else {"min_size": 5, "max_size": 20}
        )
        self.seed = seed
        self.apply_to = apply_to
        self.rng = np.random.default_rng(seed)

        # Validate the corruption mode
        if self.mode not in ["point", "block"]:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Supported: 'point', 'block'."
            )

        # Validate block parameters if block mode is enabled
        if self.mode == "block":
            if (
                "min_size" not in self.block_params
                or "max_size" not in self.block_params
            ):
                raise ValueError("block_params must contain 'min_size' and 'max_size'.")

            # Check logical consistency if types allow straightforward comparison
            min_p = self.block_params["min_size"]
            max_p = self.block_params["max_size"]

            if isinstance(min_p, type(max_p)) and min_p > max_p:
                raise ValueError(
                    "block_params['min_size'] cannot be greater than ['max_size']."
                )

    def _resolve_block_size(
        self, size_param: Union[int, float], total_length: int
    ) -> int:
        """
        Resolve the block size parameter to an absolute integer length.

        This handles the logic for proportional sizing (float inputs) vs
        absolute sizing (integer inputs).

        Parameters
        ----------
        size_param : int or float
            The parameter value, either an int (absolute size) or
            float (proportion of total_length).
        total_length : int
            The length of the time series cycle.

        Returns
        -------
        int
            The resolved block size in time steps.

        Raises
        ------
        ValueError
            If a float parameter is not between 0.0 and 1.0.
        """
        if isinstance(size_param, float):
            if not 0.0 <= size_param <= 1.0:
                raise ValueError(
                    f"Proportion value {size_param} must be between 0.0 and 1.0"
                )
            return int(total_length * size_param)
        return int(size_param)

    def _corrupt_array(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the configured corruption logic to a single NumPy array.

        This method standardizes the input dimensions to 3D, ensures the data type
        supports NaNs, and iterates through channels/batches to apply the specific
        corruption pattern.

        Parameters
        ----------
        X : np.ndarray
            The input data array. Expected shapes are ``(Time, Channels)`` or
            ``(Batch, Time, Channels)``.

        Returns
        -------
        np.ndarray
            The corrupted array with the same shape as input ``X``,
            containing ``NaN`` values where data was dropped.

        Raises
        ------
        ValueError
            If the input array dimensions are not 2D or 3D.
        ValueError
            If a configured ratio exceeds 1.0.
        """

        # Standardize input to (Batch, Time, Channels) for uniform processing logic
        original_shape = X.shape
        if X.ndim == 2:
            # Assume (Time, Channels) -> unsqueeze to (1, T, C)
            X_working = X[np.newaxis, :, :]
        elif X.ndim == 3:
            X_working = X
        else:
            raise ValueError(f"Data must be 2D (T, C) or 3D (B, T, C), got {X.ndim}D")

        # Cast to float64 to ensure the array can hold NaN values (integers cannot)
        X_corrupted = X_working.astype(np.float64)
        B, T, C = X_corrupted.shape
        # 'Mask' tracks indices that have already been dropped to handle overlaps in block mode
        Mask = np.zeros_like(X_corrupted, dtype=bool)

        for c in range(C):
            # Determine the ratio for this specific channel.
            # Default to 0.0 if the ratios list is shorter than the number of channels.
            ratio = self.ratios[c] if c < len(self.ratios) else 0.0

            # Skip processing if no corruption is requested for this channel
            if ratio <= 0:
                continue

            # Sanity check for ratio bounds
            if ratio > 1:
                raise ValueError(f"Ratio must be <= 1.0, got {ratio}")

            target_missing = int(T * ratio)

            # Optimization: If ratio is 1.0, drop everything immediately
            if ratio == 1.0:
                X_corrupted[:, :, c] = np.nan
                Mask[:, :, c] = True
                continue

            # Apply corruption independently per batch
            for b in range(B):
                if self.mode == "point":
                    # --- POINT MODE ---
                    # Simply select 'target_missing' unique random indices
                    idx = self.rng.choice(T, size=target_missing, replace=False)
                    X_corrupted[b, idx, c] = np.nan
                    Mask[b, idx, c] = True

                elif self.mode == "block":
                    # --- BLOCK MODE ---
                    current_missing = 0
                    attempts = 0

                    # Resolve min/max sizes dynamically based on T (cycle length)
                    # This handles both int (absolute) and float (proportion) params
                    min_block_param = self.block_params["min_size"]
                    max_block_param = self.block_params["max_size"]

                    raw_min_size = self._resolve_block_size(min_block_param, T)
                    raw_max_size = self._resolve_block_size(max_block_param, T)

                    # Clamp block sizes to the actual time dimension
                    # Ensure min_size is at least 1, but not larger than T
                    min_block = max(1, min(raw_min_size, T))
                    max_block = max(1, min(raw_max_size, T))

                    # Ensure consistency after clamping/resolving
                    if min_block > max_block:
                        min_block = max_block

                    if min_block == 0:
                        continue

                    # Use a safety counter to prevent infinite loops if the array is nearly full
                    while current_missing < target_missing and attempts < 200:
                        # 1. Sample a random block size
                        if min_block == max_block:
                            raw_block_size = min_block
                        else:
                            raw_block_size = self.rng.integers(min_block, max_block)

                        remaining = target_missing - current_missing

                        # If the block is larger than what we have left, truncate it
                        # (Optional: or skip it if you strictly enforce min_size)
                        block_size = min(raw_block_size, remaining)

                        # Edge case: If clamping makes it smaller than 1, we are done
                        if block_size < 1:
                            break

                        # 2. Sample a valid start position
                        # Ensure we don't index out of bounds
                        if block_size >= T:
                            block_size = T
                            start_idx = 0
                        else:
                            # +1 because high is exclusive in random.integers
                            start_idx = self.rng.integers(0, T - block_size + 1)

                        end_idx = start_idx + block_size

                        # 3. Check for Overlap
                        # We calculate how many *new* points this block would drop.
                        existing_drops = np.sum(Mask[b, start_idx:end_idx, c])

                        # Heuristic: If more than 50% of this block is already missing,
                        # skip it and try to find a cleaner spot to avoid clumping.
                        if existing_drops > (block_size * 0.5):
                            attempts += 1
                            continue

                        # 4. Apply Corruption
                        new_drops = block_size - existing_drops
                        X_corrupted[b, start_idx:end_idx, c] = np.nan
                        Mask[b, start_idx:end_idx, c] = True
                        current_missing += new_drops

                        attempts += 1

        # Restore original shape
        if len(original_shape) == 2:
            return X_corrupted[0]
        return X_corrupted

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Execute the transformation on the provided data dictionary.

        Parameters
        ----------
        data : dict
            A dictionary where keys are signal names (e.g., ``'features'``) and
            values are input arrays.
        metadata : dict
            A metadata dictionary (unused by this specific transform but
            required by the interface).

        Returns
        -------
        dict
            A new dictionary containing the corrupted arrays.
            Keys not present in ``apply_to`` are returned unmodified.
        """
        output_data = {}

        # If 'apply_to' is not set, default to processing all keys in the dictionary
        keys_to_process = self.apply_to if self.apply_to is not None else data.keys()

        for key, value in data.items():
            if key in keys_to_process:
                # IMPORTANT: ensure_2d=False allows 3D (Batch, Time, Channel) arrays
                # to pass through without being flattened or rejected by the utility.
                np_val = convert_to_numpy(value, ensure_2d=False)
                corrupted_val = self._corrupt_array(np_val)
                output_data[key] = corrupted_val
            else:
                # Passthrough for non-targeted keys (e.g., 'target' or 'timestamps')
                output_data[key] = value

        return output_data
