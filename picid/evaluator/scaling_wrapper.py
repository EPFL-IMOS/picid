"""Utilities for handling data scaling and inverse scaling."""

# numpydoc ignore=GL08

from einops import rearrange
import torch
import numpy as np
from typing import Optional, Tuple

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.multisource import InverseTransformMixin


class ScalingWrapper:
    """
    Wrapper class to handle inverse scaling operations.

    Parameters
    ----------
    inverse_transform : Optional[InverseTransformMixin], default=None
        Transformer object providing ``inverse_transform``.
    apply_inverse_scaling : bool, default=False
        Whether to apply inverse scaling.
    task_mode : bool, default=False
        Task mode used to validate expected tensor shapes.
    """

    def __init__(
        self,
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        task_mode: bool = False,
    ):
        """
        Initialize the scaling wrapper.

        Parameters
        ----------
        inverse_transform : Optional[InverseTransformMixin], default=None
            Transformer object providing ``inverse_transform``.
        apply_inverse_scaling : bool, default=False
            Whether to apply inverse scaling.
        task_mode : bool, default=False
            Task mode used to validate expected tensor shapes.
        """
        self.inverse_transform = inverse_transform
        self.apply_inverse = apply_inverse_scaling
        self.task_mode = task_mode

    def inverse_transform_if_needed(
        self,
        predictions: Tuple[torch.Tensor, np.ndarray],
        targets: Tuple[torch.Tensor, np.ndarray],
        **kwargs,
    ) -> tuple:
        """
        Apply inverse transform to predictions and targets if needed.

        Parameters
        ----------
        predictions : Tuple[torch.Tensor, np.ndarray]
            Model predictions.
        targets : Tuple[torch.Tensor, np.ndarray]
            Ground-truth targets.
        **kwargs : dict
            Optional metadata forwarded to the inverse transform.

        Returns
        -------
        tuple
            Tuple of transformed predictions and targets.
        """
        metadata = kwargs.get("metadata", {})
        if not (self.inverse_transform and self.apply_inverse):
            return predictions, targets

        # Check if the incomming data is torch tensor, is so, then covert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().detach().numpy()

        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().detach().numpy()

        # Apply inverse transform while preserving original shapes
        pred_shape = predictions.shape
        target_shape = targets.shape
        # TODO: this logic should not be here! We have to define how does the output should look like
        if len(pred_shape) == 3:
            # predictions: (B, T, C).
            # For now we the T=1
            if self.task_mode != "forecasting":
                assert (
                    pred_shape[1] == 1
                ), f"Expected first dim (time dimentions) to be 1 for 3D predictions, got {pred_shape[1]}"
            elif self.task_mode == "forecasting":
                # 1. time dim is assumed to be first and can be greater than 1

                # 2. we do not work with multivariate for now
                assert (
                    pred_shape[2] == 1
                ), f"Expected last dim (feature dimentions) to be 1 for 3D predictions, got {pred_shape[2]}"

            # flatten time dimension into batch dimension
            predictions = rearrange(predictions, "b t c -> (b t) c")
            targets = rearrange(targets, "b t c -> (b t) c")

            # Apply inverse transform (assuming scaler expects 2D input)
            predictions_transformed = self.inverse_transform.inverse_transform(
                NamedTransformInput(**dict(predictions=predictions)),
                metadata=metadata,
            ).reshape(pred_shape)

            targets_transformed = self.inverse_transform.inverse_transform(
                NamedTransformInput(**dict(targets=targets)), metadata=metadata
            ).reshape(target_shape)

        elif len(pred_shape) == 2:
            # Apply inverse transform (assuming scaler expects 2D input)
            predictions_transformed = self.inverse_transform.inverse_transform(
                NamedTransformInput(**dict(predictions=predictions)), metadata=metadata
            )  # .reshape((1, -1))

            targets_transformed = self.inverse_transform.inverse_transform(
                NamedTransformInput(**dict(targets=targets)), metadata=metadata
            )  # .reshape((1, -1))
        else:
            raise ValueError(f"Unexpected shape: {pred_shape}")

        return predictions_transformed, targets_transformed


class MultivariateTimeseriesScalingWrapper:
    """
    Wrapper class to handle inverse scaling operations.

    Parameters
    ----------
    inverse_transform : Optional[InverseTransformMixin], default=None
        Transformer object providing ``inverse_transform``.
    apply_inverse_scaling : bool, default=False
        Whether to apply inverse scaling.
    task_mode : bool, default=False
        Task mode used to validate expected tensor shapes.
    """

    def __init__(
        self,
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        task_mode: bool = False,
    ):
        """
        Initialize the scaling wrapper.

        Parameters
        ----------
        inverse_transform : Optional[InverseTransformMixin], default=None
            Transformer object providing ``inverse_transform``.
        apply_inverse_scaling : bool, default=False
            Whether to apply inverse scaling.
        task_mode : bool, default=False
            Task mode used to validate expected tensor shapes.
        """
        self.inverse_transform = inverse_transform
        self.apply_inverse = apply_inverse_scaling
        self.task_mode = task_mode

        assert (
            task_mode == "multivariate"
        ), "MultivariateTimeseriesScalingWrapper only supports multivariate task_mode"

    def inverse_transform_if_needed(
        self,
        predictions: Tuple[torch.Tensor, np.ndarray],
        targets: Tuple[torch.Tensor, np.ndarray],
        **kwargs,
    ) -> tuple:
        """
        Apply inverse transform to predictions and targets if needed.

        Parameters
        ----------
        predictions : Tuple[torch.Tensor, np.ndarray]
            Model predictions.
        targets : Tuple[torch.Tensor, np.ndarray]
            Ground-truth targets.
        **kwargs : dict
            Optional ``metadata`` dict forwarded to ``inverse_transform`` (e.g. ``unit_id``).

        Returns
        -------
        tuple
            Tuple of transformed predictions and targets.
        """
        metadata = kwargs.get("metadata", {})
        if not (self.inverse_transform and self.apply_inverse):
            return predictions, targets

        # Check if the incomming data is torch tensor, is so, then covert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().detach().numpy()

        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().detach().numpy()

        # Apply inverse transform while preserving original shapes
        pred_shape = predictions.shape

        if len(pred_shape) == 3:
            assert (
                pred_shape[0] == 1 or self.task_mode
            ), f"Expected batch size of 1 for 3D predictions, got {pred_shape[0]}"

            preds_flattened = rearrange(predictions, "b t c -> (b t) c")
            targets_flattened = rearrange(targets, "b t c -> (b t) c")

            # Apply inverse transform (assuming scaler expects 2D input)
            predictions_transformed = self.inverse_transform.inverse_transform(
                NamedTransformInput(**dict(predictions=preds_flattened)),
                metadata=metadata,
            )

            targets_transformed = self.inverse_transform.inverse_transform(
                NamedTransformInput(**dict(targets=targets_flattened)),
                metadata=metadata,
            )

            targets_unflattened = rearrange(
                targets_transformed, "(b t) c -> b t c", b=pred_shape[0]
            )

            predictions_unflattened = rearrange(
                predictions_transformed, "(b t) c -> b t c", b=pred_shape[0]
            )

        else:
            raise ValueError(f"Unexpected shape: {pred_shape}, need to be 3D")

        return predictions_unflattened, targets_unflattened
